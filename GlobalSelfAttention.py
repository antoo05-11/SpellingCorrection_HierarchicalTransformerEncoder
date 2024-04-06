from BaseAttention import BaseAttention


class GlobalSelfAttention(BaseAttention):
    def call(self, inputs):
        x, mask = inputs
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
