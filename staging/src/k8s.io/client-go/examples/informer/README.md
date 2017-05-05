# Informer Example

Informers provide a high-level API for creating custom controllers for Kubernetes resources.

This particular example demonstrates:

* How to write an Informer against a core resource type.
* How to handle add, update and delete events.

## Running

To run the example outside the Kubernetes cluster you need to supply the path to a Kubernetes config file.

```sh
go run main.go -kubeconfig=$HOME/.kube/config -logtostderr
```

By default `glog` logs to files.
Use the `-logtostderr` command line argument so that you can see the output on the console.

## Running Inside a Kubernetes Cluster

You can also run the example inside a Kubernetes cluster.
In this case you don't need to supply any configuration.
Here's how you might run the example inside a Minikube cluster.

* Build a static binary.

```sh
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o informer main.go
```

* Start Minikube and source its Docker environment

```sh
minikube start
eval $(minikube docker-env)
```

* Create a Dockerfile

```
FROM scratch
ADD informer /
VOLUME /tmp
ENTRYPOINT ["/informer"]
```

* Build a Docker image, with the static binary, on the Minikube virtual machine.

```sh
docker build --tag informer .
```

* Run the Docker image as a Pod in Minikube

```sh
kubectl run --image informer --restart Never informer -- -logtostderr
kubectl logs informer --follow
```

The output will look like this:

```sh
$ kubectl logs informer --follow
W0505 12:52:47.788509       1 client_config.go:517] Neither --kubeconfig nor --master was specified.  Using the inClusterConfig.  This might not work.
I0505 12:52:47.821582       1 main.go:114] POD CREATED:default/informer
I0505 12:52:47.821647       1 main.go:114] POD CREATED:kube-system/kube-addon-manager-minikube
I0505 12:52:47.821756       1 main.go:114] POD CREATED:kube-system/kube-dns-v20-w52kw
I0505 12:52:47.821853       1 main.go:114] POD CREATED:kube-system/kubernetes-dashboard-h1b2j
I0505 12:52:47.891095       1 main.go:97] listing pods from store:
I0505 12:52:47.891406       1 main.go:102] kube-dns-v20-w52kw
I0505 12:52:47.891718       1 main.go:102] kubernetes-dashboard-h1b2j
I0505 12:52:47.891772       1 main.go:102] informer
I0505 12:52:47.891961       1 main.go:102] kube-addon-manager-minikube
I0505 12:52:48.271966       1 main.go:121] POD UPDATED:
I0505 12:52:48.271994       1 main.go:122] old:default/informer
I0505 12:52:48.271999       1 main.go:123] new:default/informer
```

## Use Cases

* Capturing resource events for logging to external systems
  (e.g. monitor non-"Normal" events and publish metrics to a time series database)
* Creating lifecycle controllers for ThirdPartyResources
  (e.g. coordinate add / update / delete of an external datastore represented via a ThirdPartyResource type)
