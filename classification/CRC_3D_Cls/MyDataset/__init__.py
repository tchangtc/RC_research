from .dataset import build

def build_dataset(image_set, args):
    if args.dataset == 'ColoRectal':
        return build(image_set, args)
    raise ValueError(f'dataset {args.dataset} not supported')