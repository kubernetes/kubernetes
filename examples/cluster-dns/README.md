## Kubernetes DNS example

This is a toy example demonstrating how to use kubernetes DNS.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides). Make sure DNS is enabled in your setup, see [DNS doc](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/cluster/addons/dns).

```shell
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

### Step One: Create two namespaces

We'll see how cluster DNS works across multiple [namespaces](../../docs/namespaces.md), first we need to create two namespaces:

```shell
$ kubectl create -f examples/cluster-dns/namespace-dev.yaml
$ kubectl create -f examples/cluster-dns/namespace-prod.yaml
```

Now list all namespaces:

```shell
$ kubectl get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
```

For kubectl client to work with each namespace, we define two contexts:

```shell
$ kubectl config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
$ kubectl config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}
```

You can view your cluster name and user name in kubernetes config at ~/.kube/config.

### Step Two: Create backend replication controller in each namespace

Use the file [`examples/cluster-dns/dns-backend-rc.yaml`](dns-backend-rc.yaml) to create a backend server [replication controller](../../docs/replication-controller.md) in each namespace.

```shell
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
```

Once that's up you can list the pod in the cluster:

```shell
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

Now repeat the above commands to create a replication controller in prod namespace:

```shell
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-rc.yaml
$ kubectl get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

### Step Three: Create backend service

Use the file [`examples/cluster-dns/dns-backend-service.yaml`](dns-backend-service.yaml) to create
a [service](../../docs/services.md) for the backend server.

```shell
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
```

Once that's up you can list the service in the cluster:

```shell
$ kubectl get service dns-backend
NAME          LABELS    SELECTOR           IP(S)          PORT(S)
dns-backend   <none>    name=dns-backend   10.0.236.129   8000/TCP
```

Again, repeat the same process for prod namespace:

```shell
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-backend-service.yaml
$ kubectl get service dns-backend
NAME          LABELS    SELECTOR           IP(S)         PORT(S)
dns-backend   <none>    name=dns-backend   10.0.35.246   8000/TCP
```

### Step Four: Create client pod in one namespace

Use the file [`examples/cluster-dns/dns-frontend-pod.yaml`](dns-frontend-pod.yaml) to create a client [pod](../../docs/pods.md) in dev namespace. The client pod will make a connection to backend and exit. Specifically, it tries to connect to address `http://dns-backend.development.cluster.local:8000`.

```shell
$ kubectl config use-context dev
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
```

Once that's up you can list the pod in the cluster:

```shell
$ kubectl get pods dns-frontend
NAME           READY     STATUS       RESTARTS   AGE
dns-frontend   0/1       ExitCode:0   0          1m
```

Wait until the pod succeeds, then we can see the output from the client pod:

```shell
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```

Please refer to the [source code](./images/frontend/client.py) about the log. First line prints out the ip address associated with the service in dev namespace; remaining lines print out our request and server response.

If we switch to prod namespace with the same pod config, we'll see the same result, i.e. dns will resolve across namespace.

```shell
$ kubectl config use-context prod
$ kubectl create -f examples/cluster-dns/dns-frontend-pod.yaml
$ kubectl logs dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.cluster.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```


#### Note about default namespace

If you prefer not using namespace, then all your services can be addressed using `default` namespace, e.g. `http://dns-backend.default.cluster.local:8000`, or shorthand version `http://dns-backend:8000`


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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cluster-dns/README.md?pixel)]()
