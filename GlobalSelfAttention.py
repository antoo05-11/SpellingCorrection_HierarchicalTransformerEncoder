from BaseAttention import BaseAttention


class GlobalSelfAttention(BaseAttention):
    def call(self, inputs):
        attn_output = self.mha(query=inputs, value=inputs, key=inputs)
        x = self.add([inputs, attn_output])
        x = self.layernorm(x)
        return x
