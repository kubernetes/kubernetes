# Minikube walkthrough

This document will take you through setting up and trying the sample apiserver on a local minikube from a fresh clone of this repo.
You can optionally run and develop the sample apiserver on your local machine and use minikube to host the etcd.

## Prerequisites

- Go 1.7.x or later installed and setup. More information can be found at [go installation](https://golang.org/doc/install)
- Dockerhub account to push the image to [Dockerhub](https://hub.docker.com/)
- Minikube to run a single node Kubernetes cluster on your local machine. The Minikube docs have [installation instructions](https://github.com/kubernetes/minikube#installation) for your OS.
- kubectl for running commands against the minikube Kubernetes cluster. [Install kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

## Clone the repository

In order to build or develop the sample apiserver image we'll need to clone the repository.

```bash
cd $GOPATH/src
mkdir -p k8s.io
cd k8s.io
git clone https://github.com/kubernetes/sample-apiserver.git
```

## Running on minikube or local development against minikube

Two options are available:

- [Running the sample apiserver on minikube](#running-on-minikube)
- [Developing and debugging the sample apiserver on your machine](#developing-on-your-machine)

## Running on minikube

This section will focus on running the sample apiserver in minikube.

### Build the binary

Next we will want to create a new binary to both test we can build the server and to use for the container image.

From the root of this repo, where `main.go` is located, run the following command:

```bash
export GOOS=linux; go build .
```

if everything went well, you should have a binary called `sample-apiserver` present in your current directory.

### Build the container image

Using the binary we just built, we will now create a Docker image and push it to our Dockerhub registry so that we deploy it to our cluster.
There is a sample `Dockerfile` located in `artifacts/simple-image` we will use this to build our own image.

Again from the root of this repo run the following commands:

```bash
cp ./sample-apiserver ./artifacts/simple-image/kube-sample-apiserver
docker build -t <YOUR_DOCKERHUB_USER>/kube-sample-apiserver:latest ./artifacts/simple-image
docker push <YOUR_DOCKERHUB_USER>/kube-sample-apiserver
```

### Modify the the replication controller

You need to modify the [artifacts/example/rc.yaml](/artifacts/example/rc.yaml) file to change the `imagePullPolicy` to `Always` or `IfNotPresent`.

You also need to change the image from `kube-sample-apiserver:latest` to `<YOUR_DOCKERHUB_USER>/kube-sample-apiserver:latest`. For example:

```yaml
...
      containers:
      - name: wardle-server
        image: <YOUR_DOCKERHUB_USER>/kube-sample-apiserver:latest
        imagePullPolicy: Always
...
```

Save this file and we are then ready to deploy and try out the sample apiserver.

### Deploy to Minikube

We will need to create several objects in order to setup the sample apiserver using `kubectl`.

```bash
# create the namespace to run the apiserver in
kubectl create ns wardle

# create the service account used to run the server
kubectl create -f artifacts/example/sa.yaml -n wardle

# create the rolebindings that allow the service account user to delegate authz back to the kubernetes master for incoming requests to the apiserver
kubectl create -f artifacts/example/auth-delegator.yaml -n kube-system
kubectl create -f artifacts/example/auth-reader.yaml -n kube-system

# create the service and replication controller
kubectl create -f artifacts/example/rc.yaml -n wardle
kubectl create -f artifacts/example/service.yaml -n wardle

# create the apiservice object that tells kubernetes about your api extension and where in the cluster the server is located
kubectl create -f artifacts/example/apiservice.yaml
```

You can skip to [test your setup](#test-that-your-setup-has-worked)

## Developing on your machine

This section will focus on how to develop and run the sample apisever on your machine and use minikube to host `etcd`.

### Deploying etcd and setting services

```bash

# you'll need the IP address of your machine, otherwise minikube's apiserver cannot talk to your sample-apiserver.
# On Linux:
hostname -I
# On OSX:
ipconfig getifaddr en0

MACHINE_IP=<YOUR_MACHINE_IP>
MINIKUBE_IP=$(minikube ip)

# create the namespace to run the apiserver in
kubectl create ns wardle

# create the etcd
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: ReplicationController
metadata:
  name: wardle-etcd
  namespace: wardle
  labels:
    etcd: "true"
spec:
  replicas: 1
  selector:
    etcd: "true"
  template:
    metadata:
      labels:
        etcd: "true"
    spec:
      containers:
      - image: quay.io/coreos/etcd:v3.1.10
        name: etcd
        command:
        - etcd
        - -advertise-client-urls=http://${MINIKUBE_IP}:2379
        - -listen-client-urls=http://0.0.0.0:2379
EOF

# create the etcd service and expose it on port 32379
kubectl create -f artifacts/example/local-development/etcd-service.yaml

# create the sample-apiserver service with external IP address pointing to our local machine
cat <<EOF | kubectl create -f -
kind: Service
apiVersion: v1
metadata:
  name: api
  namespace: wardle
spec:
  ports:
  - protocol: TCP
    port: 443
    targetPort: 8443
---
kind: Endpoints
apiVersion: v1
metadata:
  name: api
  namespace: wardle
subsets:
- addresses:
  - ip: ${MACHINE_IP}
  ports:
  - port: 8443
EOF

# create the apiservice object that tells kubernetes about your api extension and where outside of the cluster the server is located
kubectl create -f artifacts/example/apiservice.yaml
```

### Run the sample-apiserver locally

We'll reuse minikube's certificates and your local `kubeconfig`:

```bash
go run main.go \
 --etcd-servers=$(minikube ip):32379 \
 --tls-cert-file ~/.minikube/apiserver.crt \
 --tls-private-key-file ~/.minikube/apiserver.key \
 --secure-port=8443 \
 --kubeconfig ~/.kube/config \
 --authentication-kubeconfig ~/.kube/config \
 --authorization-kubeconfig ~/.kube/config
```

## Test that your setup has worked

You should now be able to create the resource type `Flunder` which is the resource type registered by the sample apiserver.

```bash
kubectl create -f artifacts/flunders/01-flunder.yaml
# outputs flunder "my-first-flunder" created
```

You can then get this resource by running:

```bash
kubectl get flunder my-first-flunder

#outputs
# NAME               KIND
# my-first-flunder   Flunder.v1alpha1.wardle.k8s.io
```
