def get_gan_wrapper(args, target=False):

    kwargs = {}
    for kw, arg in args:
        if kw != 'gan_type':
            if (not kw.startswith('source_')) and (not kw.startswith('target_')):
                kwargs[kw] = arg
            else:
                if target and kw.startswith('target_'):
                    final = kw[len('target_'):]
                    kwargs[f'source_{final}'] = arg
                elif (not target) and kw.startswith('source_'):
                    kwargs[kw] = arg

    if args.gan_type == "SDStochasticText":
        from .stable_diffusion_stochastic_text_wrapper import SDStochasticTextWrapper
        return SDStochasticTextWrapper(**kwargs)
    else:
        raise ValueError()

