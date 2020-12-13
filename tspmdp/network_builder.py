from tspmdp.modules.graph_encoder import CustomizableEncoder
from tspmdp.modules.decoder import CustomizableQDecoder


class CustomizableNetworkBuilder:
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        d_key: int,
        d_hidden: int,
        n_omega: int = 64,
        transformer: str = "preln",
        final_ln: bool = True,
        decoder_mha: str = "softmax",
        use_graph_context: bool = True,
    ):
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_hidden = d_hidden
        self.n_omega = n_omega
        self.transformer = transformer
        self.final_ln = final_ln
        self.decoder_mha = decoder_mha
        self.use_graph_context = use_graph_context

    def __call__(self):
        encoder = CustomizableEncoder(
            d_model=self.d_model,
            depth=self.depth,
            n_heads=self.n_heads,
            d_key=self.d_key,
            d_hidden=self.d_hidden,
            n_omega=self.n_omega,
            transformer=self.transformer,
            final_ln=self.final_ln
        )

        decoder = CustomizableQDecoder(
            n_heads=self.n_heads,
            d_key=self.d_key,
            n_omega=self.n_omega,
            mha=self.decoder_mha,
            use_graph_context=self.use_graph_context
        )

        return encoder, decoder
