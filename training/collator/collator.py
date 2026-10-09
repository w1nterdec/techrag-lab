import torch


class TechRAGCollator:


    def __init__(
        self,
        tokenizer
    ):
        self.tokenizer = tokenizer



    def build_prompt(self, sample):

        prompt = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
"""

        response = sample["output"]

        return prompt, response



    def __call__(self, batch):

        input_ids = []

        labels = []


        for sample in batch:

            prompt, response = self.build_prompt(
                sample
            )


            prompt_ids = self.tokenizer.encode(
                prompt
            )


            response_ids = self.tokenizer.encode(
                response
            )


            ids = (
                prompt_ids
                +
                response_ids
            )


            target = (
                [-100] * len(prompt_ids)
                +
                response_ids
            )


            input_ids.append(ids)

            labels.append(target)



        max_length = max(
            len(x)
            for x in input_ids
        )


        padded_input_ids = []
        padded_labels = []
        attention_mask = []


        for ids, target in zip(
            input_ids,
            labels
        ):

            padding_length = (
                max_length - len(ids)
            )


            padded_input_ids.append(
                ids + [0] * padding_length
            )


            padded_labels.append(
                target + [-100] * padding_length
            )


            attention_mask.append(
                [1] * len(ids)
                +
                [0] * padding_length
            )


        return {

            "input_ids": torch.tensor(
                padded_input_ids
            ),

            "attention_mask": torch.tensor(
                attention_mask
            ),

            "labels": torch.tensor(
                padded_labels
            )
        }
