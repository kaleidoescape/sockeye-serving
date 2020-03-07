# sockeye-serving (prefix constraints)

This repo was created for a hackathon event. USE THIS REPO AT YOUR OWN RISK.

This repo builds off of jameswoo's sockeye-serving (see below for original instructions). This fixes some errors with using constraints (currently only the cpu Dockerfile is updated for these fixes). In addition, this uses a custom SentencePiece BPE model to encode/decode the sentence. 

WARNING! This system is highly customized for a specific use case, and much of the code does not follow the principle of separation of concerns. We need to generalize much of the code, and to rewrite much of it. Cruciailly, the system enforces the inclusion of a start-of-sentence symbol in the constraints, requires the user to specific model files into specific directories, and requires the user to make specific config file updates so everything can build properly. 

## Quickstart

To build the Dockerfiles from scratch, you first need a sockeye model. Copy the model into the `docker/cpu` directory. Then copy the scripts inside `src/sockeye_serving` into the model directory (if they are not already there). 

TODO: Automate this portion of the build process.

```bash
cp -r mymodel docker/cpu/models/
cp -r src/sockeye_serving docker/cpu/models/mymodel/
```

Next, build the docker container, run it, and observe the logs:

```bash
cd docker/cpu
docker build -t sockeye .
docker run -itd --name sockeye -p 8090:8080 -p 8091:8081 sockeye
docker logs sockeye -f
```

In a new terminal, register your model:
```
curl -X POST "http://localhost:8091/models?synchronous=true&initial_workers=1&url=de2en_vanilla"
```

Now you are ready to request prefix-constrained translations:

```
curl -X POST "http://localhost:8090/predictions/de2en" -H "Content-Type: application/json"     -d '{ "text": "Ich gehe zum Laden", "constraints": ["I walk"]}'
```

To rebuild (e.g. when you add new models to the models directory), right now you need to delete this docker image, and start over with the docker build step. You can check if you have containers running with `docker container ls -a`, and check if you have images with `docker images`.
```
docker rm sockeye -f
docker rmi sockeye -f
```



# sockeye-serving (original instructions)
`sockeye-serving` is a containerized service for neural machine translation that uses Amazon's `sockeye` framework as the translation engine.
The web server makes use of `mxnet-model-server`, which provides a management API for loading models and a prediction API for requesting translations.

Any Sockeye model can be loaded via the management API.
Text preprocessing is built into the request pipeline and supports a wide variety of languages.
Specialized processing for specific languages can be implemented using custom handlers.

## Quickstart
This example shows how to serve an existing model for Chinese to English translation.
First, pull the latest Docker image:
```bash
docker pull jwoo11/sockeye-serving
```

Download the example model archive (MAR).
This is a ZIP archive containing the parameter files and scripts needed to run translation:
* https://www.dropbox.com/s/pk7hmp7a5zjcfcj/zh.mar?dl=0

Extract the MAR file to `/tmp/models`.
 This directory will be the source for a bind mount for Docker:
```bash
unzip -d /tmp/models/zh zh.mar
```

Start the server:
```bash
docker run -itd --name sockeye_serving -p 8080:8080 -p 8081:8081 -v /tmp/models:/opt/ml/model jwoo11/sockeye-serving
```

Now, load the model using the management API. Note that the URL of the model is relative to the bind mount:
```bash
curl -X POST "http://localhost:8081/models?synchronous=true&initial_workers=1&url=zh"
```
Get the status of the model with the following:
```bash
curl -X GET "http://localhost:8081/models/zh"
```
The response should look like this:
```json
{
  "modelName": "zh",
  "modelUrl": "zh",
  "runtime": "python3",
  "minWorkers": 1,
  "maxWorkers": 1,
  "batchSize": 1,
  "maxBatchDelay": 100,
  "workers": [
    {
      "id": "9000",
      "startTime": "2019-01-26T00:49:10.431Z",
      "status": "READY",
      "gpu": false,
      "memoryUsage": 601395200
    }
  ]
}
```

To translate text use the inference API. Notice that the port is different from above.
```bash
curl -X POST "http://localhost:8080/predictions/zh" -H "Content-Type: application/json" \
    -d '{ "text": "我的世界是一款開放世界遊戲，玩家沒有具體要完成的目標，即玩家有超高的自由度選擇如何玩遊戲" }'
```

The translation quality depends on the model. The provided model returns this translation:
```json
{
  "translation": "in my life was a life of a life of a public public, and a public, a time, a video, a play, which, it was a time of a time of a time."
}
```

A better model trained on more data returns this response:
```json
{
  "translation": "My world is an open world game, and players have no specific goal to accomplish, that is, players have a high degree of freedom to choose how to play."
}
```

## Installation
To install the command line clients for `sockeye-serving` run the following in a virtual environment:
```bash
pip install sockeye-serving
```
If you want to install from source, a `Pipfile` is provided.
Clone the repository and run `pipenv install`.

Installation places the command line interfaces `sockeye-serving` and `sockeye-client` on your virtual environment's path.

## Command Line Interfaces
You can use `sockeye-serving` to easily start Docker and to make REST calls to both the management and prediction APIs.
First, a configuration file must be placed in either the current directory or some place referenced by `SOCKEYE_SERVING_CONF`.
Example properties are located in `config/sockeye-serving.conf`.
Here's some basic usage:
```bash
# start the Docker container
sockeye-serving start

# deploy a model
sockeye-serving deploy zh

# list available models
sockeye-serving list

# translate text
sockeye-serving translate zh "my text"

# upload a file for translation
sockeye-serving upload zh "my_file.txt"
```
Run `sockeye-serving help` for a full list of commands.

The Python client takes a YAML configuration file.
An example configuration is in `config/sockeye-client.yml`.
This client does not support restarting Docker, however, it does exercise the full API provided by `mxnet-model-server`.
The commands which accept query parameters are below:
```bash
$ sockeye-client deploy -h
usage: sockeye-client deploy [-h] [-m MODEL_NAME] [-x HANDLER] [-r RUNTIME]
                             [-b BATCH_SIZE] [-d MAX_BATCH_DELAY]
                             [-i INITIAL_WORKERS] [-s] [-t RESPONSE_TIMEOUT]
                             url
...

$ sockeye-client list -h
usage: sockeye-client list [-h] [-l LIMIT] [-t NEXT_PAGE_TOKEN]
...

$ sockeye-client scale -h
usage: sockeye-client scale [-h] [-a MIN_WORKER] [-b MAX_WORKER]
                            [-n NUMBER_GPU] [-s] [-t TIMEOUT]
                            model_name
...
```
Run `sockeye-client -h` to show a full list of commands.
For more information on the API, see [additional documentation](#additional-documentation) for `mxnet-model-server`.

## Jupyter Notebook
If you want to translate text with Jupyter, you can use `notebooks/machine_translation.ipynb`.
Make sure `requests` is installed in your Python environment.

## Choosing between CPUs and GPUs
`sockeye-serving` provides different image tags for CPUs and GPUs.
You can set the desired tag in your `sockeye-serving.conf` file.
You'll also need to specify a Sockeye config file `sockeye-args.txt`.
This file contains arguments passed to the Sockeye translation engine.
Example files for both CPU and GPU configs are under `config/sockeye`.

To use GPUs, ensure `nvidia-docker` is installed on the host machine.
In `sockeye-serving.conf` set the image tag to one with "gpu" in its name, such as `latest-gpu`, and set `docker_exec="nvidia-docker"`.
Then run `sockeye-serving update MODEL_NAME config/sockeye/gpu/sockeye-args.txt`.

For CPUs, use a tag without "gpu" in its name, such as `latest`, and use the CPU version of the Sockeye config file.
The changes to `sockeye-serving.conf` will be picked up when you run `sockeye-serving start`.

## Initializing Models
Each model must be initialized with a `MANIFEST.json` file in order for `mxnet-model-server` to deploy it.
An easy way to initialize a model is to run `sockeye-serving archive MODEL_NAME HANDLER`, where `HANDLER` is the name of a Python handler module under `src/sockeye_serving`.
The provided handlers include `ko_handler` (Korean), `zh_handler` (Chinese), and `default_handler` (generic).
After running the archive command, your model directory should have a file `MAR-INF/MANIFEST.json` that looks like:
```json
{
  "runtime": "python3",
  "model": {
    "modelName": "zho",
    "handler": "sockeye_serving.zh_handler:handle"
  },
  "modelServerVersion": "1.0",
  "implementationVersion": "1.0",
  "specificationVersion": "1.0"
}
```

## Enabling TLS
The provided configuration instructs the server to use plain HTTP.
To enable TLS, you can either supply a Java keystore or a private key and certificate in PEM format.

Using `config/config.properties` as a starting point, create a new `config.properties` file and save it under `/tmp/models`:
```properties
model_store=/opt/ml/model
inference_address=https://0.0.0.0:8443
management_address=https://0.0.0.0:8444
```
Suppose you have a key pair residing on the host at `/path/to/certs`.
Set the properties for the keystore:
```properties
keystore=/path/to/certs/keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
```
Or provide the path to the server's private key and certificate:
```properties
private_key_file=/path/to/certs/private.key
certificate_file=/path/to/certs/cert.pem
```
Then start the container:
```bash
docker run -itd --name sockeye_serving -p 8443:8443 -p 8444:8444 \
    -v /path/to/certs:/path/to/certs \
    -v /tmp/models:/opt/ml/model jwoo11/sockeye-serving \
    mxnet-model-server --start --mms-config /opt/ml/model/config.properties
```

To make requests using `curl` you should ensure that you set `--cert`, `--key`, and `--cacert` as needed.

## <a name="additional-documentation"></a> Additional Documentation

For more information on `mxnet-model-server`, see:
* https://github.com/awslabs/mxnet-model-server/tree/master/docs
