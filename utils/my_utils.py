def acc_reward(logits, labels):
    return (logits.argmax(dim=1) == labels).float()