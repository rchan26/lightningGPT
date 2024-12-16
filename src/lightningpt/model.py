import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from lightningpt.attention import CausalSelfAttention
from lightningpt.logger import logger
from lightningpt.utils import ModelArgs


class MLP(nn.Module):
    def __init__(self, n_embd: int, ffn_dim_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(n_embd * ffn_dim_multiplier)
        self.c_fc = nn.Linear(n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(
            n_embd=config.n_embd,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            dropout=config.ffn_dropout,
        )

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False, start_pos: int = 0):
        x = x + self.attn(self.ln_1(x), use_kv_cache=use_kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)        

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Initialise a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialised minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai's model vocabulary
        config.max_seq_len = 1024  # openai's model max_seq_len
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]  # ignore these
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_kv_cache: bool = False,
        start_pos: int = 0,
    ):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.max_seq_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.max_seq_len}"
        if use_kv_cache:
            # need to add the correct position embedding for the token we're generating
            # shape (1, t)
            pos = torch.tensor(
                [start_pos + i for i in range(t)], dtype=torch.long, device=device
            ).unsqueeze(
                0
            )  
        else:
            # shape (1, t)
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
                0
            )  

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)  
        pos_emb = self.transformer.wpe(
            pos
        )

        # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, use_kv_cache=use_kv_cache, start_pos=start_pos)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        use_kv_cache: bool = False,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if use_kv_cache:
            prev_pos = 0
            assert idx.size(1) < self.max_seq_len
            if idx.size(1) + max_new_tokens > self.max_seq_len:
                max_new_tokens = self.max_seq_len - idx.size(1)
                logger.warning(
                    "Input length + max_new_tokens is larger than max_seq_len, truncating to max_seq_len. "
                    f"Only generating {max_new_tokens} tokens."
                )

            # pass in idx to start generating
            for cur_pos in range(idx.size(1), idx.size(1) + max_new_tokens):
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(
                    idx[:, prev_pos:cur_pos], use_kv_cache=True, start_pos=prev_pos
                )
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalised) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
                # update the previous position
                prev_pos = cur_pos
        else:
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at max_seq_len
                idx_cond = (
                    idx
                    if idx.size(1) <= self.max_seq_len
                    else idx[:, -self.max_seq_len :]
                )
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond, use_kv_cache=False)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalised) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

        return idx
