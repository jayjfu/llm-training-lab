from math_verify import parse, verify

BOXED = "\\boxed{"


def extract_boxed(text):
    start = text.rfind(BOXED)
    if start == -1:
        return ""
    begin = start + len(BOXED)
    depth = 0
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            if depth == 0:
                return text[begin:index].strip()
            depth -= 1
    return ""


def is_equivalent(prediction, answer):
    if "".join(prediction.split()) == "".join(answer.split()):
        return True
    return verify(parse(f"${answer}$"), parse(f"${prediction}$"))


def completion_text(completion):
    return completion[-1]["content"] if isinstance(completion, list) else completion


def accuracy_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, answer in zip(completions, ground_truth):
        prediction = extract_boxed(completion_text(completion))
        rewards.append(1.0 if prediction and is_equivalent(prediction, answer) else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    return [0.5 if extract_boxed(completion_text(completion)) else 0.0 for completion in completions]


REWARD_FUNCS = [accuracy_reward, format_reward]