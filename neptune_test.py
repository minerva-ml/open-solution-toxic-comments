import re

from deepsense import neptune


def get_correct_channel_name(ctx, name):
    channels_with_name = [channel for channel in ctx.job._channels if name in channel.name]
    if len(channels_with_name) == 0:
        return name
    else:
        channel_ids = [re.split('[^\d]', channel.name)[-1] for channel in channels_with_name]
        channel_ids = sorted([int(idx) if idx != '' else 0 for idx in channel_ids])
        last_id = channel_ids[-1]
        corrected_name = '{} {}'.format(name, last_id + 1)
        return corrected_name


ctx = neptune.Context()
new_channel_name = get_correct_channel_name(ctx, 'test channel')
ctx.channel_send(new_channel_name, 0, 1)
new_channel_name = get_correct_channel_name(ctx, 'test channel')
ctx.channel_send(new_channel_name, 0, 1)
new_channel_name = get_correct_channel_name(ctx, 'test channel')
ctx.channel_send(new_channel_name, 0, 1)
new_channel_name = get_correct_channel_name(ctx, 'test channel')
ctx.channel_send(new_channel_name, 0, 1)
