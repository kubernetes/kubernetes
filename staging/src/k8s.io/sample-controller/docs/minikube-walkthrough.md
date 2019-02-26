# Minikube walkthrough

This document will take you through setting up and trying the sample controller on a local minikube from a fresh clone of this repo.

## Pre requisites

- Go 1.7.x or later installed and setup. More infomration can be found at [go installation](https://golang.org/doc/install)
- Dockerhub account to push the image to [Dockerhub](https://hub.docker.com/)

## Install Minikube

Minikube is a single node Kubernetes cluster that runs on your local machine. The Minikube docs have installation instructions for your OS.
- [minikube installation](https://github.com/kubernetes/minikube#installation)

## Clone the repository

In order to build the sample controller image we will need to build the controller binary.

```
cd $GOPATH/src
mkdir -p k8s.io
cd k8s.io
git clone https://github.com/kubernetes/sample-controller.git
```

## Build the binary

Next we will want to create a new binary to both test we can build the controller and to use for the container image.

From the root of this repo, where ```main.go``` is located, run the following command:
```
export GOOS=linux; go build .
```
if everything went well, you should have a binary called ```sample-controller``` present in your current directory.

## Build the container image

Using the binary we just built, we will now create a Docker image so that we deploy it to our cluster.
There is a sample ```Dockerfile``` located in ```artifacts/simple-image``` we will use this to build our own image.

Again from the root of this repo run the following commands:
```
cp ./sample-controller ./artifacts/simple-image/kube-sample-controller
docker build -t kube-sample-controller:latest ./artifacts/simple-image
```

## Deploy to Minikube

In order to setup the sample controller, you need to ensure you have the ```kubectl``` tool installed. 
[Install kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

```
# create the sample controller deployment
kubectl apply -f artifacts/examples/sample-controller.yaml
```

## Test that your setup has worked

```
kubectl apply -f artifacts/examples/crd.yaml
# outputs customresourcedefinition.apiextensions.k8s.io/foos.samplecontroller.k8s.io created

kubectl apply -f artifacts/examples/example-foo.yaml
# foo.samplecontroller.k8s.io/example-foo created
```

You can then get this resource by running:

```
kubectl get foo
# outputs
NAME          CREATED AT
example-foo   3m


kubectl get deployment
# outputs
# NAME          DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
# example-foo   1         1         1            1           3m
```
