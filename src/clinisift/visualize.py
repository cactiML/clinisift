import logging
import glob
from flask import Flask, request
from clinisift.doc import Doc
from clinisift.cliniparse import Parser
from os.path import basename


class Visualizer:
    vis_dir = None
    parser = None
    files = []
    host = "0.0.0.0"
    port = "5000"

    def __init__(self, vis_dir, parser):
        self.vis_dir = vis_dir
        self.parser = parser

        files = list(glob.glob(self.vis_dir + "/*.txt"))
        files.sort()
        self.files = files

        if len(self.files) == 0:
            logger.warning("No .txt files found in specified directory.")

    def visualize(self):
        app = Flask(__name__)

        header = f"""
        <html><head>
        <title>clinisift visualizer</title>
        <link rel="stylesheet" href="https://unpkg.com/bamboo.css">
        </head>

        <h1>⚗️ clinisift visualizer</h1>
        <h2>{self.vis_dir}</h2>
        """

        file_list = "<table>"
        for f in self.files:
            file_list += (
                f"<tr><td><a href=/file?filepath={f}>{basename(f)}</a></td></tr>"
            )
        file_list += "</table>"

        @app.route("/")
        def vis():
            return header + file_list

        @app.route("/file")
        def file_vis():
            filepath = request.args.get("filepath")
            doc = Doc(filepath, self.parser, is_file=True)
            doc.parse()
            return doc.visualize(return_html=True, header=filepath)

        app.run(host=self.host, port=self.port)


# default visualizer
if __name__ == "__main__":
    from sys import argv

    if len(argv) != 2:
        raise IOError(
            "Error: Directory must be passed as commandline argument. For example: python -m clinisift.visualize $PWD"
        )
        exit()
    include_ents = ["problem", "m", "do", "du", "f"]
    logging.warning(
        f"Using default settings -- only visualizing {include_ents}, and `sent_per_line`= True. Run visualizer programmatically to adjust settings."
    )
    parser = Parser(include_ents=include_ents, sent_per_line=True)
    vis_dir = str(argv[1])

    vis = Visualizer(vis_dir, parser)
    vis.visualize()
