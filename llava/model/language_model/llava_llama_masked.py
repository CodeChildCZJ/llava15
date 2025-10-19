#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch_masked import LlavaMetaModelMasked, LlavaMetaForCausalLMMasked


class LlavaConfigMasked(LlamaConfig):
    model_type = "llava_llama_masked"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mask相关配置
        self.mask_visual_token = kwargs.get('mask_visual_token', False)
        self.mask_ratio = kwargs.get('mask_ratio', 0.1)
        self.mask_strategy = kwargs.get('mask_strategy', 'random')
        self.mask_token_value = kwargs.get('mask_token_value', 0.0)


class LlavaLlamaModelMasked(LlavaMetaModelMasked, LlamaModel):
    config_class = LlavaConfigMasked

    def __init__(self, config: LlavaConfigMasked):
        super(LlavaLlamaModelMasked, self).__init__(config)


class LlavaLlamaForCausalLMMasked(LlamaForCausalLM, LlavaMetaForCausalLMMasked):
    config_class = LlavaConfigMasked

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModelMasked(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def generate(
        self,
        images,
        image_sizes=None,
        **kwargs,
    ):
        # 获取注意力权重用于mask策略
        attention_weights = kwargs.pop('attention_weights', None)
        
        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=kwargs.get('input_ids'),
                position_ids=kwargs.get('position_ids'),
                attention_mask=kwargs.get('attention_mask'),
                past_key_values=kwargs.get('past_key_values'),
                labels=kwargs.get('labels'),
                images=images,
                image_sizes=image_sizes,
                attention_weights=attention_weights
            )
            kwargs['inputs_embeds'] = inputs_embeds

        return super().generate(**kwargs)
