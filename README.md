# pytorch-basics
### Just a simple repo with some basic examples from [Pytorch's 60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

#### Sometimes going back to the basics can be super useful 

### Quick start

#### There are some Docker options:
- ```docker compose --profile torch up ``` will run a classification training pipeline
- ``` docker compose --profile mlflow up ``` will run a classification training pipeline connecting to a tracking mlflow server. Note that mlflow.set_tracking_uri() may need to be changed and the mlflow server needs to be up and running
- ``` docker compose --profile develop up ``` will start the container by itself. This is super useful for attaching a shell to the container and changing the code to experiment new things

### Setting up the mlflow container
- ``` cd mlflow && docker compose up ```. This should start a mlflow server at [localhost:8686](localhost:8686) and you will be able to log new runs
