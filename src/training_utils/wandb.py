import wandb
from datetime import datetime

class WandbLogger:
    """ Wandb Logger for logging metrics and images to Weights & Biases. """
    def __init__(self, project_name, api_key):
        self.project_name = project_name
        self.api_key = api_key
        self.init_run()

    def init_run(self):
        """ Initializes a new wandb run with the specified project name and API key. """
        wandb.login(key=self.api_key)
        self.run = wandb.init(project=self.project_name, name=str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def log(self, dict):
        """ Logs a dictionary of metrics to wandb. """
        self.run.log(dict)

    def log_image(self, plot, caption: str):
        """ Logs an image to wandb with a caption. """
        self.run.log({caption: wandb.Image(plot, caption)})
