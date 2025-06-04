# Setup Ollama
## Adding Modelfile to configure context window for prompt
Copy "Modelfile" to the directory of the saved model on machine.

## Starting Ollama application as a container
Run the following command to start the ollama container:
```
docker compose up -d
```

Note: the current setup expects a system with 2 GPUs

## Deploying the model to Ollama container
After container is up, run the following command inside the ollama container:
```
ollama create <model-name-to-use> -f <path_to_mount_modelfile_in_container>/Modelfile
```
Note: Multiple models can be deployed simultaneously by adding additional lines in the docker-compose.yml file for each model. Each model must be saved on the machine local storage and the Modelfile copied to each models directory.
