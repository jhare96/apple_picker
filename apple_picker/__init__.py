from gym.envs.registration import register
from .applepicker import ApplePicker, ApplePickerDeterministic

register(
    id='ApplePicker-v0',
    entry_point='apple_picker:ApplePicker',
)

register(
    id='ApplePickerDeterministic-v0',
    entry_point='apple_picker:ApplePickerDeterministic',
)