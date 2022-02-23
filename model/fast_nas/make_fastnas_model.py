from model.fast_nas.encoders import create_encoder
from model.fast_nas.micro_decoders import MicroDecoder as Decoder
from torch import nn

class Segmenter(nn.Module):
    """Create Segmenter"""

    def __init__(self, encoder, decoder):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

def create_segmenter(encoder):
        decoder_config = [[0, [0, 0, 4, 0], [0, 1, 4, 8], [1, 4, 9, 1]], [[1, 0], [1, 1], [1, 1]]]
        decoder = Decoder(
            inp_sizes=encoder.out_sizes,
            num_classes=5,
            config=decoder_config,
            agg_size=48,
            aux_cell=True,
            repeats=1,
        )

        # Fuse encoder and decoder
        segmenter = Segmenter(encoder, decoder).cuda()

        return segmenter
def fastNas():
    encoder = create_encoder("cvpr")
    fastNas = create_segmenter(encoder)
    return fastNas