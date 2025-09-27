from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
import torch


class TransformerModel(torch.nn.Module):

    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size) -> None:
        super().__init__()
        self.model = BertEncoder(
            BertConfig(
                vocab_size=2,  # not using it
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            )
        )
    
    def forward(self, hidden_states, hidden_mask):
        # hidden_states [batch, len, dim]
        # hidden_mask [batch, len]
        full_mask = (1.0 - hidden_mask[:, None, None, :]) * torch.finfo(torch.float).min if hidden_mask is not None else None
        out = self.model(
            hidden_states=hidden_states,
            attention_mask=full_mask
        ).last_hidden_state
        return out

if __name__ == "__main__":
    m = TransformerModel(32, 4, 4, 256)
    mask = torch.randint(0,2,[4, 10])
    print((m(torch.randn([4, 10, 32]), torch.randint(0,1,[4, 10]))*mask.unsqueeze(-1).float()))