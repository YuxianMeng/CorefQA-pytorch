#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# author: xiaoy li
# description:
# corefqa model 


import torch
import torch.nn as nn

# from transformers.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class CorefQA(BertPreTrainedModel):
    def __init__(self, bert_config, config, device):
        super(CorefQA, self).__init__(bert_config)

        self.model_config = config
        self.bert = BertModel(bert_config)
        self.device = device 

        # other configs, todo(yuxian)
        self.pad_idx = self.model_config.pad_idx 
        self.max_span_width = 3
        self.span_ratio = 0.1
        self.max_candidate_num = 10
        self.max_antecedent_num = 5
        self.sliding_window_size = 50
        self.mention_start_idx = 7
        self.mention_end_idx = 70
        self.mention_loss_ratio = 0.1

        self.apply(self.init_bert_weights)
        # mention proposal 
        self.mention_start_ffnn = nn.Linear(self.config.hidden_size, 1)
        self.mention_end_ffnn = nn.Linear(self.config.hidden_size, 1)
        self.mention_span_ffnn = nn.Linear(self.config.hidden_size * 2, 1)

        # cluster todo(yuxian): check是否应该和上面的参数不share
        self.forward_qa_ffnn = nn.Linear(self.config.hidden_size * 2, 1)
        self.backward_qa_ffnn = nn.Linear(self.config.hidden_size * 2, 1)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, doc_idx, sentence_map, subtoken_map, input_ids, input_mask, gold_mention_span=None,
                token_type_ids=None, attention_mask=None, span_starts=None, span_ends=None, cluster_ids=None):
        """
        forward
        Args:
            doc_idx: [1]
            sentence_map: [num_tokens], each token's sentence index
            subtoken_map: [num_tokens], each token's ??? todo
            input_ids: [num_windows, window_size]
            input_mask: [num_windows, window_size]
            token_type_ids: [num_windows, window_size]
            attention_mask:
            span_start: [num_spans], span start indices
            span_end: [num_spans], span end indices
            cluster_ids: [num_spans], span to cluster indices
            gold_mention_span: [num_candidates], gold mention labels (0/1) todo(yuxian): 和span_start/end重复，没必要传

        Returns:

        """
        num_tokens = sentence_map.shape[0]
        flattened_input_ids = input_ids.view(-1)
        flattened_input_mask = input_mask.view(-1)
        mention_sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                               output_all_encoded_layers=False)
        # [num_candidates], [num_candidates]
        candidate_starts, candidate_ends = self.get_candidate_spans(sentence_map)
        num_candidate_mentions = int(num_tokens * self.span_ratio)
        k = min(self.max_candidate_num, num_candidate_mentions)
        c = min(self.max_antecedent_num, k)
        # [num_candidates]
        candidates_embeddings = self.get_span_embeddings(mention_sequence_output.view(-1, self.config.hidden_size),
                                                         candidate_starts,
                                                         candidate_ends)
        # [num_candidates]
        candidate_labels = self.get_candidate_labels(candidate_starts=candidate_starts,
                                                     candidate_ends=candidate_ends,
                                                     labeled_starts=span_starts,
                                                     labeled_ends=span_ends,
                                                     labels=cluster_ids)
        # get topk scores
        # [num_candidates]
        candidate_mention_scores = self.mention_span_ffnn(candidates_embeddings).squeeze(-1)
        # [k]
        top_mention_scores, top_mention_indices = torch.topk(candidate_mention_scores, k)
        topk_span_starts = candidate_starts[top_mention_indices]
        topk_span_ends = candidate_ends[top_mention_indices]
        topk_span_labels = candidate_labels[top_mention_indices]

        # QA linking(for each proposal in k)
        topc_starts = []
        topc_ends = []
        topc_labels = []
        topc_scores = []
        questions = []
        for mention_idx, (span_start, span_end) in enumerate(zip(topk_span_starts, topk_span_ends)):
            # [query_length]
            question_tokens = self.get_question_token_ids(sentence_map=sentence_map,
                                                          flattened_input_ids=flattened_input_ids,
                                                          flattened_input_mask=flattened_input_mask,
                                                          span_start=span_start,
                                                          span_end=span_end)
            questions.append(question_tokens)
        # [k, num_windows*window_size, embed_size]
        batch_forward_candidate_embeddings = self.get_batch_query_mention_embeddings(
            questions=questions,
            input_ids=input_ids
        )
        for mention_idx, (span_start, span_end) in enumerate(zip(topk_span_starts, topk_span_ends)):
            # [num_windows*window_size, embed_size]
            forward_candidate_embeddings = batch_forward_candidate_embeddings[mention_idx]
            # [num_candidates, 2*embed_size]
            flattened_forward_candidate_embeddings = self.get_span_embeddings(
                token_embeddings=forward_candidate_embeddings,
                span_starts=candidate_starts,
                span_ends=candidate_ends)
            # [num_candidates]
            forward_candidate_mention_scores = self.mention_span_ffnn(flattened_forward_candidate_embeddings).squeeze(-1)
            # [c]
            topc_forward_scores, topc_forward_indices = torch.topk(forward_candidate_mention_scores, c)
            topc_forward_starts = candidate_starts[topc_forward_indices]
            topc_forward_ends = candidate_ends[topc_forward_indices]
            topc_forward_labels = candidate_labels[topc_forward_indices]
            topc_mention_scores = candidate_mention_scores[topc_forward_indices]

            # compute backward score
            backward_topc_scores = []
            for tmp_start, tmp_end in zip(topc_forward_starts, topc_forward_ends):
                backward_question = self.get_question_token_ids(sentence_map=sentence_map,
                                                                flattened_input_ids=flattened_input_ids,
                                                                flattened_input_mask=flattened_input_mask,
                                                                span_start=tmp_start,
                                                                span_end=tmp_end)
                # [num_windows*window_size, embed_size]
                backward_candidate_embeddings = self.get_query_mention_embeddings(
                    question_tokens=backward_question,
                    input_ids=input_ids
                )
                # [num_candidates, 2*embed_size]
                flattened_backward_candidate_embeddings = self.get_span_embeddings(
                    token_embeddings=backward_candidate_embeddings,
                    span_starts=torch.LongTensor([span_start]).to(backward_candidate_embeddings.device),
                    span_ends=torch.LongTensor([span_end])).to(backward_candidate_embeddings.device)
                # [1]
                backward_score = self.mention_span_ffnn(flattened_backward_candidate_embeddings).squeeze(-1)
                backward_topc_scores.append(backward_score)
            # [c]
            backward_topc_scores = torch.cat(backward_topc_scores)
            # [c]
            cluster_mention_scores = ((topc_forward_scores + backward_topc_scores) / 2 +
                                      top_mention_scores[mention_idx] +
                                      topc_mention_scores)

            topc_starts.append(topc_forward_starts)
            topc_ends.append(topc_forward_ends)
            topc_labels.append(topc_forward_labels)
            topc_scores.append(cluster_mention_scores)

        # [k, c]
        top_link_starts = torch.stack(topc_starts)
        top_link_ends = torch.stack(topc_ends)
        top_link_scores = torch.stack(topc_scores)
        top_link_labels = torch.stack(topc_labels)

        # (k, c)
        same_cluster_indicator = top_link_labels == topk_span_labels.unsqueeze(-1)
        # (k, c)
        pairwise_labels = torch.logical_and(same_cluster_indicator, (topk_span_labels > 0).unsqueeze(-1))
        dummy_labels = torch.logical_not(torch.any(pairwise_labels, dim=1, keepdims=True))  # [k, 1]
        loss_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1).long()  # [k, c + 1]
        dummy_scores = torch.zeros([k, 1]).to(loss_antecedent_labels.device)
        loss_antecedent_scores = torch.cat([dummy_scores, top_link_scores], 1)  # [k, c + 1]
        loss = self.marginal_likelihood(loss_antecedent_scores, loss_antecedent_labels)
        loss += self.mention_loss_ratio * self.bce_loss(candidate_mention_scores,
                                                        (candidate_labels > 0).float())
                                                        # gold_mention_span)
        return loss
        # return prediction, loss

    def marginal_likelihood(self, antecedent_scores, antecedent_labels):
        """
        Desc:
            marginal likelihood of gold antecedent spans from coreference clusters.
        Args:
            antecedent_scores: [k, c+1] the predicted scores by the model
            antecedent_labels: [k, c+1] the gold-truth cluster labels
        Returns:
            a scalar of loss
        """
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [k]
        log_norm = torch.logsumexp(antecedent_scores, 1)  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return loss.sum()  # todo(yuxian): sum会不会导致与mention的mean不一致

    def get_candidate_spans(self, sentence_map):
        """
        Desc:
            get candidate spans based on: the length of candidate spans <= max_span_width
            each span is located in a single sentence
        Args:
            sentence_map: [num_tokens], each token's sentence index

        Returns:
            start and end indices w.r.t num_tokens (num_candidates, ), (num_candidates, )
        """
        num_tokens = sentence_map.shape[0]
        # candidate_span: every position can be span start, there are max_span_width kinds of end for each start
        # [num_tokens, max_span_width]
        candidate_starts = torch.range(0, num_tokens - 1, dtype=torch.int64).unsqueeze(1).expand(
            [-1, self.max_span_width]).contiguous()
        # [num_tokens, max_span_width]
        candidate_ends = candidate_starts + torch.range(0, self.max_span_width - 1, dtype=torch.int64).unsqueeze(
            0).expand([num_tokens, -1]).contiguous()
        # [num_tokens, max_span_width]
        candidate_starts = candidate_starts.view(-1).to(sentence_map.device)
        candidate_ends = candidate_ends.view(-1).to(sentence_map.device)
        # [num_tokens*max_span_width]，get sentence_id for each token indices
        candidate_start_sentence_indices = torch.gather(sentence_map, 0, candidate_starts)
        candidate_end_sentence_indices = torch.gather(sentence_map, 0,
                                                      torch.clamp(candidate_ends, min=0, max=num_tokens - 1))
        # [num_tokens*max_span_width]，legitimate spans should reside in a single sentence.
        candidate_mask = torch.logical_and(
            candidate_ends < num_tokens,
            (candidate_start_sentence_indices - candidate_end_sentence_indices) == 0
        )
        candidate_starts = candidate_starts[candidate_mask]
        candidate_ends = candidate_ends[candidate_mask]
        return candidate_starts, candidate_ends

    def get_span_embeddings(self, token_embeddings, span_starts, span_ends):
        """
        get span embeddings from span start embedding, span end embedding and optionally span width embedding
        Args:
            token_embeddings: [num_tokens, embed_size]
            span_starts: [num_candidates]
            span_ends: [num_candidates]
        Returns:
            span_embeddings: [num_candidates, embed_size*2]
        """
        span_start_emb = token_embeddings[span_starts]  # [num_candidates, embed_size]
        span_end_emb = token_embeddings[span_ends]  # [num_candidates, embed_size]

        span_emb = torch.cat([span_start_emb, span_end_emb], dim=-1)
        return span_emb

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        """
            method to get golden cluster id of candidate spans
        Args:
            candidate_starts: [num_candidates, ]
            candidate_ends: [num_candidates, ]
            labeled_starts: [num_mentions, ]
            labeled_ends: [num_mentions, ]
            labels: [num_mentions, ] gold truth cluster ids
        Returns:
            candidate_labels: [num_candidates]
        """
        # [num_mentions, num_candidates]
        same_start = labeled_starts.unsqueeze(1) == candidate_starts.unsqueeze(0)
        # [num_mentions, num_candidates]
        same_end = labeled_ends.unsqueeze(1) == candidate_ends.unsqueeze(0)
        # [num_mentions, num_candidates]
        same_span = torch.logical_and(same_start, same_end)
        # [1, num_candidates]
        candidate_labels = torch.matmul(labels.unsqueeze(0).float(), same_span.float())
        candidate_labels = candidate_labels.long()
        return candidate_labels.squeeze(0)

    def get_question_token_ids(self, sentence_map, flattened_input_ids, flattened_input_mask, span_start, span_end):
        """
        Desc:
            construct question based on the selected mention span
        Args:
            sentence_map: [num_tokens] tokens to sentence_id
            flattened_input_ids: [num_windows * window_size]
            flattened_input_mask: [num_windows * window_size]
            span_start: int
            span_end: int
        Returns:
            query tokens: [query_length, ]
        """
        sentence_idx = sentence_map[span_start]
        # [num_sent_token]
        sent_positions = torch.where(sentence_map == sentence_idx)[0]
        sent_start = torch.where(flattened_input_mask == sent_positions[0])[0][0]
        sent_end = torch.where(flattened_input_mask == sent_positions[-1])[0][0]
        origin_tokens = flattened_input_ids[sent_start: sent_end+1]

        mention_start = torch.where(flattened_input_mask == span_start)[0][0]
        mention_end = torch.where(flattened_input_mask == span_end)[0][0]
        mention_start_in_sentence = mention_start - sent_start
        mention_end_in_sentence = mention_end - sent_start

        question_token_ids = torch.cat([origin_tokens[: mention_start_in_sentence],
                                        torch.Tensor([self.mention_start_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_start_in_sentence: mention_end_in_sentence + 1],
                                        torch.Tensor([self.mention_end_idx]).long().to(origin_tokens.device),
                                        origin_tokens[mention_end_in_sentence + 1:],
                                        ], dim=0)
        return question_token_ids

    def get_query_mention_embeddings(self, question_tokens, input_ids):
        """
        get embedding of all candidates with respect to question
        Args:
            question_tokens: [query_length]
            input_ids: [num_windows, window_size]

        Returns:
            query mention embeddings, [num_window*window_size, hidden]

        """
        num_windows = input_ids.shape[0]
        query_length = question_tokens.shape[0]
        # [num_windows, query_length]
        windows_questions = question_tokens.unsqueeze(0).expand([num_windows, -1])
        question_ones = torch.ones_like(windows_questions)
        question_zeros = torch.zeros_like(windows_questions)
        actual_mask = (input_ids == self.pad_idx).long()  # (num_windows, window_size)
        # [num_windows, query_length + window_size]
        qa_input_ids = torch.cat([windows_questions, input_ids], 1)
        # [num_windows, query_length + window_size]
        qa_input_mask = torch.cat([question_ones, actual_mask], 1)
        token_type_ids = torch.cat([question_zeros, torch.ones_like(input_ids)], 1)
        # [num_windows, query_length + window_size, embed_size]
        forward_embeddings, _ = self.bert(input_ids=qa_input_ids, attention_mask=qa_input_mask,
                                          token_type_ids=token_type_ids, output_all_encoded_layers=False)
        # [num_windows, window_size, embed_size]
        forward_embeddings = forward_embeddings[:, query_length:]
        # [num_windows*window_size, embed_size]
        flattened_forward_embeddings = forward_embeddings.view(-1, self.config.hidden_size)
        return flattened_forward_embeddings

    def get_batch_query_mention_embeddings(self, questions, input_ids):
        """
        get embedding of all candidates with respect to question. concatenate all questions as batch
        Args:
            questions: list of different sizes tensor of shape [query_length]
            input_ids: [num_windows, window_size]
        Returns:
            flattened_embeddings: [num_questions, num_windws*window_size, embed_size]
        """
        num_questions = len(questions)
        num_windows, window_size = input_ids.shape
        batch_input_ids = []
        batch_input_mask = []
        batch_token_type_ids = []
        for question_tokens in questions:
            # [num_windows, query_length]
            windows_questions = question_tokens.unsqueeze(0).expand([num_windows, -1])
            question_ones = torch.ones_like(windows_questions)
            question_zeros = torch.zeros_like(windows_questions)
            actual_mask = (input_ids == self.pad_idx).long()  # (num_windows, window_size)
            # [num_windows, query_length + window_size]
            qa_input_ids = torch.cat([windows_questions, input_ids], 1)
            # [num_windows, query_length + window_size]
            qa_input_mask = torch.cat([question_ones, actual_mask], 1)
            # todo(yuxian): ones or zeros?
            token_type_ids = torch.cat([question_zeros, torch.ones_like(input_ids)], 1)
            batch_input_ids.append(qa_input_ids)
            batch_input_mask.append(qa_input_mask)
            batch_token_type_ids.append(token_type_ids)
        # [batch*num_windows, max_seq_len]
        batch_input_ids = self.pad_stack(batch_input_ids).view(num_questions*num_windows, -1)
        batch_input_mask = self.pad_stack(batch_input_mask).view(num_questions*num_windows, -1)
        # todo(yuxian): 1 or 0 ?
        batch_token_type_ids = self.pad_stack(batch_token_type_ids, 1).view(num_questions*num_windows, -1)

        # [batch*num_windows, max_seq_len, embed_size]
        forward_embeddings, _ = self.bert(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                                          token_type_ids=batch_token_type_ids, output_all_encoded_layers=False)
        # [batch, num_windows, max_seq_len, embed_size]
        forward_embeddings = forward_embeddings.view(num_questions, num_windows, -1, self.config.hidden_size)
        # [batch, num_windows, window_size, embed_size]
        truncated_embeddings = [
            embeddings[:, question.shape[0]: question.shape[0]+window_size:, ]
            for embeddings, question in zip(forward_embeddings, questions)
        ]
        truncated_embeddings = torch.stack(truncated_embeddings, 0)
        flattened_forward_embeddings = truncated_embeddings.view(num_questions, -1, self.config.hidden_size)
        return flattened_forward_embeddings

    @staticmethod
    def pad_stack(tensors, pad_idx: int = 0, dim: int = -1):
        """
        stack tensors and pad to same max length
        Args:
            tensors: List of tensor, has different length at dim
            pad_idx: pad index
            dim: pad dimension
        Returns:
            batch_tensor: [batch, tensor.shape]
        """
        max_length = max(t.shape[dim] for t in tensors)
        first_tensor_shape = list(tensors[0].shape)
        final_shape = first_tensor_shape
        final_shape[dim] = max_length
        tensor_num = len(tensors)
        output_tensor = torch.zeros([tensor_num] + final_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        output_tensor.fill_(pad_idx)
        for tensor_idx, tensor in enumerate(tensors):
            slices = [slice(None, size, None) for size in tensor.shape]
            output_tensor[tensor_idx][slices] = tensor
        return output_tensor
