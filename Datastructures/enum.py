from enum import Enum

class AttackMode(Enum):
    TARGETED = "targeted"
    NOISE_UNTARGETED = "noise-untargeted"
    UNTARGETED = "untargeted"