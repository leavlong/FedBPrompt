import torch

def make_optimizer_0stage(models,op_type):
    params = []
    keys = []
    for model in models:
        for key, value in model.named_parameters():
            if "text_encoder" in key:
                value.requires_grad_(False)
                continue
            if "prompt_learner" in key:
                value.requires_grad_(False)
                continue
            if not value.requires_grad:
                continue
            lr = 0.000005
            weight_decay = 0.0001
            if "bias" in key:
                lr = 0.000005 * 2
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    if op_type == 'SGD':
        optimizer = getattr(torch.optim, op_type)(params, momentum=0.95)
    elif op_type == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=0.000005, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)

    return optimizer


def make_optimizer_1stage(args, model, op_type):
        params = []#存储了参数组的信息
        keys = []#记录哪些参数被分到了这个组中
        for key, value in model.named_parameters():
            if "prompt_learner" in key:
                # lr = cfg.SOLVER.STAGE1.BASE_LR
                lr = args.lrs1
                # weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
                weight_decay = args.weight_decay
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                keys += [key]
                print(key)
        if op_type == 'SGD':
            # optimizer = getattr(torch.optim, op_type)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
            optimizer = getattr(torch.optim, op_type)(params, momentum=args.momentum)
        elif op_type == 'AdamW':
            # optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
            optimizer = torch.optim.AdamW(params, lr=args.lrs1, weight_decay=args.weight_decay)
        else:
            optimizer = getattr(torch.optim, op_type)(params)
        return optimizer


def make_optimizer_2stage(args, models, center_criterion, op_type , use_prompt = True):
    params = []
    keys = []
    models = models[0]
    models = models.module

    for param in models.text_encoder.parameters():
        param.requires_grad_(False)

    if use_prompt:
        for param in models.image_encoder.parameters():
            param.requires_grad_(False)

    for name, param in models.named_parameters():
        if "prompt" in name:
            param.requires_grad = True
        # if "classifier.weight" in name:
        #     param.requires_grad = False

    for key, value in models.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:  # 不更新文本prompt
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.lrs2
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.lrs2 * 2
            weight_decay = args.weight_decay
        if False:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    for name, param in models.named_parameters():
        if param.requires_grad:
            print(name)
    if op_type == 'SGD':
        optimizer = getattr(torch.optim, op_type)(params, momentum=args.momentum)
    elif op_type == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lrs2, weight_decay=args.weight_decay)
    else:
        optimizer = getattr(torch.optim, op_type)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    return optimizer, optimizer_center