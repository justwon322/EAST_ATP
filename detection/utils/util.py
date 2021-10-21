import logging
import os
import random
import shutil

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, device):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.device = device

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
        
def make_epoch_description(history: dict, current: int, total: int, best: int, exclude: list = []):
    """Create description string for logging progress."""
    pfmt = f">{len(str(total))}d"
    desc = f" Epoch: [{current:{pfmt}}/{total:{pfmt}}] ({best:{pfmt}}) |"
    for metric_name, metric_dict  in history.items():
        if not isinstance(metric_dict, dict):
            raise TypeError("`history` must be a nested dictionary.")
        if metric_name in exclude:
            continue
        for k, v in metric_dict.items():
            desc += f" {k}_{metric_name}: {v:.3f} |"
    return desc


def get_rich_pbar(transient: bool = True, auto_refresh: bool = False):
    """A colorful progress bar based on the `rich` python library."""
    console = Console(color_system='256', force_terminal=True, width=160)
    return Progress(
        console=console,
        auto_refresh=auto_refresh,
        transient=transient
    )

def get_rich_logger(logfile: str = None, level=logging.INFO):
    """A colorful logger based on the `rich` python library."""

    myLogger = logging.getLogger()

    # File handler
    if logfile is not None:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s"))
        myLogger.addHandler(fileHandler)

    # Rich handler
    width, _ = shutil.get_terminal_size()
    console = Console(color_system='256', width=width)
    richHandler = RichHandler(console=console)
    richHandler.setFormatter(logging.Formatter("%(message)s"))
    myLogger.addHandler(richHandler)

    # Set level
    myLogger.setLevel(level)

    return myLogger


def get_logger(stream=False, logfile=None, level=logging.INFO):
    """
    Arguments:
        stream: bool, default False.
        logfile: str, path.
    """
    _format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logFormatter = logging.Formatter(_format)

    rootLogger = logging.getLogger()

    if logfile:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(logFormatter)
        rootLogger.addHandler(streamHandler)

    rootLogger.setLevel(level)

    return rootLogger

def touch(filepath: str, mode: str='w'):
    assert mode in ['a', 'w']
    directory, _ = os.path.split(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    open(filepath, mode).close()

