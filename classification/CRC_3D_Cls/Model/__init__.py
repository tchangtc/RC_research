from .model_3D import build_YNetCls


def build_model(args):
    if args.model == 'YNetCls':
        return build_YNetCls(args)

    else:
        raise ValueError('Please Select a Supported Version Model! ')