# trt-mnist
This is a demo of how you can use tensorRT to speed up your tensorflow model.
It should work on the Nvidia xavier boards.
Tested on a `Nvidia Xavier NX`

## How it works
1. create the model
1. export as saved_model
1. import the model with the tensorRT converter
1. convert the model with options
1. reexport it as a saved_model
1. import the converted model and use it!

## How to use
this project uses poetry.
run the command below to install all the dependencies. This command took about 30min in my environment.

```bash
$ poetry install
```

to pretrain the model run the below command.

```bash
$ poetry run train
```

to run the network in tensorRT 


```bash
$ poetry run inference --percision-mode MODE
```

`MODE` can be one of thease
|MODE|description|
|---|---|
|native| no tensorRT|
|FP32|tensorRT FP32|
|FP16|tensorRT FP16|
|INT8|tensorRT INT8(needs calibration)|
