from .schedulers import PlateauLRDecay, WarmRestart, LinearDecay

SCHEDULERS = dict(plateau=PlateauLRDecay,
                  warmrestart=WarmRestart,
                  linear=LinearDecay)
