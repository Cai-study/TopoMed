config_GCN = {
    "gmn": {
        "opt": {
            "opt_name": 'SGD',
            "lr": 0.0025,
            "params": (0.9, 0.999),
            "weight_decay": 1e-4,
            'momentum': 0.9,
        },
        "sch": {
            "STEPS": (1600,),
            "GAMMA": 0.1,
            "WARMUP_FACTOR": 1 / 3,
            "WARMUP_ITERS": 1600,
            "WARMUP_METHOD": 'constant',
        },
    },
}