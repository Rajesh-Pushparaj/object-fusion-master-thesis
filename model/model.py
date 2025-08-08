import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.sinusoidal_embedding import SinusoidalEmbedding
from model.modules.decoder import Decoder, FusionDecoder
from model.modules.encoder import Encoder, SensorEncoder
from model.modules.predict import MLPModule, MLP

from einops import rearrange, repeat
from torch import Tensor
from typing import Optional


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=8,
        num_encoder_layers=2,
        max_seq_len=20,
        param_size=18,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        bbox_drop=0.0,
        cls_drop=0.0,
        num_of_sensors=5,
        num_classes=5,
    ):  # input with std_dev has 18 params, and without std_dev 11 params
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.posEmbed = SinusoidalEmbedding(
            2, 2
        )  # 2 inputs channels x and y, 2 frequencies.
        self.MLPInputChannelSize = (
            param_size + 8
        )  # 2 inputs channels x and y, 2 frequencies.
        # self.inputEmbed = MLPModule(input_size = 26, hidden_sizes= [16, 32, 64],    # inputsize = param_size + ( Inputchannels_sinEmbed(2*Num_Frequency))
        #                             output_size=self.d_model)
        self.inputEmbed = MLPModule(
            input_size=self.MLPInputChannelSize,
            hidden_sizes=[
                32,
                32,
                64,
            ],  # inputsize = param_size + ( Inputchannels_sinEmbed(2*Num_Frequency))
            output_size=self.d_model,
        )  # without bottleneck layer
        self.num_of_sensors = num_of_sensors
        self.query_embed = nn.Embedding(max_seq_len, self.d_model)
        self.nhead = nhead
        self.bboxHead = MLP(self.d_model, self.d_model, 5, 3, dropout=bbox_drop)
        self.class_head_dropout = nn.Dropout(cls_drop)
        self.classHead = nn.Linear(self.d_model, num_classes + 1)
        self.sensorEncoder = SensorEncoder(
            self.d_model, dim_feedforward, num_encoder_layers, nhead
        )
        self.fusiondecoder = FusionDecoder(
            self.d_model, dim_feedforward, num_decoder_layers, nhead
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, sensor_data):

        # stack all the sensor data into a tensor
        x = sensor_data.to("cuda")
        bs, no_of_sensors, no_of_obj, objParam = x.shape
        x = rearrange(x, "B S O C -> (B S O) C").to(torch.float32)
        # separate positon parameters (x,y) fom rest
        Position = x[:, :2]
        rest = x[:, 2:]
        # sinusoidal embedding the object positions
        posSinEmbed = self.posEmbed(Position)
        x = torch.cat([posSinEmbed, rest], dim=1)
        # # Input embedding the sensor data
        x = self.inputEmbed(
            x
        )  # .view(bs, no_of_sensors, no_of_obj,-1).permute(1,2,0,3)
        # sensor data  to tranformer encoder
        x = rearrange(
            x, "(B S O) C -> (S O) B C", B=bs, S=no_of_sensors, O=no_of_obj
        )  # convert to [100, bs, Cin] so self-attention works on all the sensors
        encoded = self.sensorEncoder(x)

        # Decoding
        query_embed = self.query_embed.weight
        query_embed = repeat(query_embed, "O C -> O B C", B=bs)
        # object_queries = torch.zeros_like(query_embed)  # learnable object queries
        # tgt_attn_mask = torch.triu(torch.ones(object_queries.size(0), object_queries.size(0)), diagonal=1).bool().to('cuda')  # Create attention mask for target
        decoded_output = self.fusiondecoder(query_embed, encoded)

        # Detection
        seq_len, bs, _ = decoded_output.shape
        decoded_output = rearrange(decoded_output, "O B C -> (O B) C")
        # BBox detection
        bbox = self.bboxHead(decoded_output)
        bbox = rearrange(bbox, "(O B) C -> B O C", B=bs, O=self.max_seq_len).sigmoid()
        # class detection
        objClass = self.classHead(self.class_head_dropout(decoded_output))
        objClass = rearrange(objClass, "(O B) C -> B O C", B=bs, O=self.max_seq_len)
        # motion param detection
        # motion = self.motionHead(decoded_output)
        # motion = rearrange(motion, '(O B) C -> B O C', B=bs, O=no_of_obj)

        output = {"BBox": bbox, "Object_class": objClass}  # , 'Motion_params': motion}
        return output


def create_model(cfg):
    model = Transformer(
        dropout=cfg["dropout"],
        bbox_drop=cfg["bbox_drop"],
        cls_drop=cfg["cls_drop"],
        max_seq_len=cfg["max_seq_len"],
    )
    return model
