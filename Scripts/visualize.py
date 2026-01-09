from pycallgraph2 import PyCallGraph
from pycallgraph2 import Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.globbing_filter import GlobbingFilter

def run_viz():
    config = Config()
    config.trace_filter = GlobbingFilter(exclude=[
        'pycallgraph2.*', 'tqdm*', 'torch.*', 'numpy.*', 'os.*', 'logging.*'
    ])

    graphviz = GraphvizOutput()
    graphviz.output_file = 'clean_architecture_flow.png'

    with PyCallGraph(output=graphviz, config=config):
        from Scripts.adversarial_tts_classes import main
        main()

if __name__ == "__main__":
    run_viz()