from .losses import (
    BinaryFocalDice, BinaryBatchWeightedCrossEntropy, WeightedLogDice,
    OfficialBinaryFocalLoss, BinaryFocalLoss, ImageBinaryLogDice,
    BinaryOnlyPositiveFocalDice, BinaryCrossEntropy
)


LOSSES = {
    "cross_entropy": BinaryCrossEntropy,
    "binary_focal": BinaryFocalLoss,                                # the best
    "binary_focal_dice": BinaryFocalDice,                           # was used
    "binary_positive_focal_dice": BinaryOnlyPositiveFocalDice,
    "binary_batch_weighted_bce": BinaryBatchWeightedCrossEntropy,
    "binary_weighted_log_dice": WeightedLogDice,                    # the best
    # "official_binary_focal": OfficialBinaryFocalLoss, # Bad impl
    # "image_binary_log_dice": ImageBinaryLogDice       # Just bad
}
