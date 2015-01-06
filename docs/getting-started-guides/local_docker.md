## Getting started locally with docker

This method runs a local kubernetes cluster self hosted in Docker
itself. The Kubelet is started in a container with access to the
Docker API. It then launches a pod of containers that comprise the
rest of a local-only kubernetes cluster.

### Pre-requisites

#### With boot2docker
- Install [boot2docker](http://boot2docker.io/) 
```
boot2docker up
$(boot2docker shellinit)
export DOCKER_HOST_IP=$(boot2docker ip 2>/dev/null)
export KUBERNETES_MASTER=http://$DOCKER_HOST_IP:8080
```

#### With local docker daemon
```
export DOCKER_HOST_IP=127.0.0.1
export KUBERNETES_MASTER=http://$DOCKER_HOST_IP:8080
```

### Build the kubernetes docker images

```
./build/make-run-images.sh 
```

### Bootstrap the cluster

```
docker run -v /var/run/docker.sock:/var/run/docker.sock kubernetes-bootstrap
```

### Build kubernetes client

```
./hack/build-go.sh
# set $host_os and $host_arch to match your local host os and architecture.
host_os=linux host_arch=amd64
export PATH=$(readlink -f _output/build/$host_os/$host_arch):$PATH
```

### Manage your pods
```
kubecfg list /pods
kubecfg -p 8181:80 run nginx 1 kube-nginx
kubecfg list /pods
curl $DOCKER_HOST_IP:8181
```
