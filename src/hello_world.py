import time

from experimenter import experiment, run, e
from lib.event_listeners.web_console.web_console import EXPBoard


@experiment(
    """
    I'm just testing the base resnet 3D and the whole script. 
    """,
    {
    },
    event_listeners=lambda: [
        EXPBoard()
    ]
)
def main():
    i = 0
    while True:
        print('Hello World!')
        time.sleep(1)
        i = i + 1
        e.emit('epoch_end', {'epoch': i})


run(main, open_e=False)
# query()
