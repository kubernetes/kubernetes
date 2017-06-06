## Kubernetes DNS example

This is a toy example demonstrating how to use kubernetes DNS.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://kubernetes.io/docs/getting-started-guides/). Make sure DNS is enabled in your setup, see [DNS doc](https://github.com/kubernetes/dns).

```sh
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

### Step One: Create two namespaces

We'll see how cluster DNS works across multiple [namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/), first we need to create two namespaces:

```sh
$ kubectl create -f examples/cluster-dns/namespace-dev.yaml
$ kubectl create -f examples/cluster-dns/namespace-prod.yaml
```

Now list all namespaces:

```sh
$ kubectl get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
```

For kubectl client to work with each namespace, we define two contexts:

```sh
$ kubectl config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
$ kubectl config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}
```

You can view your cluster name and user name in kubernetes config at ~/.kube/config.

### Step Two: Create backend replication controller in each namespace

Use the file [`examples/cluster-dns/dns-backend-rc.yaml`](dns-backend-rc.yaml) to create a backend server [replication controller](https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller/) in each namespace.

```sh
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
```

Once that's up you can list the pod in the cluster:

```sh
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

Now repeat the above commands to create a replication controller in prod namespace:

```sh
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

### Step Three: Create backend service

Use the file [`examples/cluster-dns/dns-backend-service.yaml`](dns-backend-service.yaml) to create
a [service](https://kubernetes.io/docs/concepts/services-networking/service/) for the backend server.

```sh
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
```

Once that's up you can list the service in the cluster:

```sh
$ kubectl get service dns-backend
NAME         CLUSTER_IP       EXTERNAL_IP       PORT(S)                SELECTOR          AGE
dns-backend  10.0.2.3         <none>            8000/TCP               name=dns-backend  1d
```

Again, repeat the same process for prod namespace:

```sh
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
$ kubectl get service dns-backend
NAME         CLUSTER_IP       EXTERNAL_IP       PORT(S)                SELECTOR          AGE
dns-backend  10.0.2.4         <none>            8000/TCP               name=dns-backend  1d
```

### Step Four: Create client pod in one namespace

Use the file [`examples/cluster-dns/dns-frontend-pod.yaml`](dns-frontend-pod.yaml) to create a client [pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/) in dev namespace. The client pod will make a connection to backend and exit. Specifically, it tries to connect to address `http://dns-backend.development.cluster.local:8000`.

```sh
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
```

Once that's up you can list the pod in the cluster:

```sh
$ kubectl get pods dns-frontend
NAME           READY     STATUS       RESTARTS   AGE
dns-frontend   0/1       ExitCode:0   0          1m
```

Wait until the pod succeeds, then we can see the output from the client pod:

```sh
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```

Please refer to the [source code](images/frontend/client.py) about the log. First line prints out the ip address associated with the service in dev namespace; remaining lines print out our request and server response.

If we switch to prod namespace with the same pod config, we'll see the same result, i.e. dns will resolve across namespace.

```sh
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```


#### Note about default namespace

If you prefer not using namespace, then all your services can be addressed using `default` namespace, e.g. `http://dns-backend.default.svc.cluster.local:8000`, or shorthand version `http://dns-backend:8000`


### tl; dr;

For those of you who are impatient, here is the summary of the commands we ran in this tutorial. Remember to set first `$CLUSTER_NAME` and `$USER_NAME` to the values found in `~/.kube/config`.

```sh
# create dev and prod namespaces
kubectl create -f examples/cluster-dns/namespace-dev.yaml
kubectl create -f examples/cluster-dns/namespace-prod.yaml

# create two contexts
kubectl config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
kubectl config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}

# create two backend replication controllers
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-backend-rc.yaml

# create backend services
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-backend-service.yaml
kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-backend-service.yaml

# create a pod in each namespace and get its output
kubectl config use-context dev
kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
kubectl logs dns-frontend

kubectl config use-context prod
kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
kubectl logs dns-frontend
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cluster-dns/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
