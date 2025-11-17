import torch
from transformers import StoppingCriteria

STOP_SEQUENCES = [
    "\n\n\n\n",
    "\n\n\n",
    "\n\n",
    "\n",
    "Question:",
    "Context:",
    "Answer:",
    "Q:",
    "A:",
    " Q ",
]


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to("cuda") for i in self.stops
            ]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == "text":
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length :], skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop) :]
            else:
                raise
            if match:
                return True
        return False
