import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, 2)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, 128)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, 256)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, 128)
        self.embedding_cluster_hour = nn.Embedding(self.args.n_cluster_hour + 1, 3)
        self.cate_embedding_dim = self.embedding_interaction.embedding_dim + \
                                    self.embedding_test.embedding_dim + \
                                    self.embedding_question.embedding_dim + \
                                    self.embedding_tag.embedding_dim + \
                                    self.embedding_cluster_hour.embedding_dim
        self.cate_layer_norm = nn.LayerNorm(self.cate_embedding_dim, eps=1e-12)

        # 수치형 Linear 
        self.linear_cont = nn.Linear(4, 4)
        self.cont_layer_norm = nn.LayerNorm(4, eps=1e-12)

        # embedding combination projection
        self.comb_proj = nn.Linear(self.cate_embedding_dim + 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=args.drop_out
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, Tagrate, answerrate, elapsed, cumAnswerRate, cluster_hour, mask, interaction = input


        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_cluster_hour = self.embedding_cluster_hour(cluster_hour)

        Tagrate = Tagrate.unsqueeze(2)
        answerrate = answerrate.unsqueeze(2)
        elapsed = elapsed.unsqueeze(2)
        cumAnswerRate = cumAnswerRate.unsqueeze(2)

        cat_cont = torch.cat([Tagrate, answerrate, elapsed, cumAnswerRate], 2)
        embed_cont = self.linear_cont(cat_cont)
        embed_cont_norm = self.cont_layer_norm(embed_cont)

        embed_cate = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cluster_hour,
            ],
            2,
        )
        embed_cate_norm = self.cate_layer_norm(embed_cate)
        embed = torch.cat([embed_cate_norm, embed_cont_norm], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, 2)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, 256)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, 256)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, 256)
        self.embedding_cluster_hour = nn.Embedding(self.args.n_cluster_hour + 1, 3)
        self.cate_embedding_dim = self.embedding_interaction.embedding_dim + \
                                    self.embedding_test.embedding_dim + \
                                    self.embedding_question.embedding_dim + \
                                    self.embedding_tag.embedding_dim + \
                                    self.embedding_cluster_hour.embedding_dim
        self.cate_layer_norm = nn.LayerNorm(self.cate_embedding_dim, eps=1e-12)

        # 수치형 Linear 
        self.linear_cont = nn.Linear(4, 4)
        self.cont_layer_norm = nn.LayerNorm(4, eps=1e-12)

        # embedding combination projection
        self.comb_proj = nn.Linear(self.cate_embedding_dim + 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        # test, question, tag, _, mask, interaction, _ = input
        test, question, tag, _, Tagrate, answerrate, elapsed, cumAnswerRate, cluster_hour, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_cluster_hour = self.embedding_cluster_hour(cluster_hour)
        
        Tagrate = Tagrate.unsqueeze(2)
        answerrate = answerrate.unsqueeze(2)
        elapsed = elapsed.unsqueeze(2)
        cumAnswerRate = cumAnswerRate.unsqueeze(2)

        cat_cont = torch.cat([Tagrate, answerrate, elapsed, cumAnswerRate], 2)
        embed_cont = self.linear_cont(cat_cont)
        embed_cont_norm = self.cont_layer_norm(embed_cont)

        embed_cate = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cluster_hour,
            ],
            2,
        )
        embed_cate_norm = self.cate_layer_norm(embed_cate)
        embed = torch.cat([embed_cate_norm, embed_cont_norm], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_cluster_hour = nn.Embedding(self.args.n_cluster_hour + 1, self.hidden_dim //3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 5, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # test, question, tag, _, mask, interaction, _ = input
        test, question, tag,cluster_hour, _, mask, interaction = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)
        embed_cluster_hour = self.embedding_cluster_hour(cluster_hour)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cluster_hour,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
