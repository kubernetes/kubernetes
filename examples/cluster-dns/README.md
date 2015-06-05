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
$ cluster/kubectl.sh create -f examples/cluster-dns/namespace-dev.yaml
$ cluster/kubectl.sh create -f examples/cluster-dns/namespace-prod.yaml
```

Now list all namespaces:

```shell
$ cluster/kubectl.sh get namespaces
NAME          LABELS             STATUS
default       <none>             Active
development   name=development   Active
production    name=production    Active
```

For kubectl client to work with each namespace, we define two contexts:

```shell
$ cluster/kubectl.sh config set-context dev --namespace=development --cluster=${CLUSTER_NAME} --user=${USER_NAME}
$ cluster/kubectl.sh config set-context prod --namespace=production --cluster=${CLUSTER_NAME} --user=${USER_NAME}
```

### Step Two: Create backend replication controller in each namespace

Use the file [`examples/cluster-dns/dns-backend-rc.yaml`](dns-backend-rc.yaml) to create a backend server [replication controller](../../docs/replication-controller.md) in each namespace.

```shell
$ cluster/kubectl.sh config use-context dev
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-backend-rc.yaml
```

Once that's up you can list the pod in the cluster:

```shell
$ cluster/kubectl.sh get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

Now repeat the above commands to create a replication controller in prod namespace:

```shell
$ cluster/kubectl.sh config use-context prod
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-backend-rc.yaml
$ cluster/kubectl.sh get rc
CONTROLLER    CONTAINER(S)   IMAGE(S)              SELECTOR           REPLICAS
dns-backend   dns-backend    ddysher/dns-backend   name=dns-backend   1
```

### Step Three: Create backend service

Use the file [`examples/cluster-dns/dns-backend-service.yaml`](dns-backend-service.yaml) to create
a [service](../../docs/services.md) for the backend server.

```shell
$ cluster/kubectl.sh config use-context dev
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-backend-service.yaml
```

Once that's up you can list the service in the cluster:

```shell
$ cluster/kubectl.sh get service dns-backend
NAME          LABELS    SELECTOR           IP(S)          PORT(S)
dns-backend   <none>    name=dns-backend   10.0.236.129   8000/TCP
```

Again, repeat the same process for prod namespace:

```shell
$ cluster/kubectl.sh config use-context prod
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-backend-service.yaml
$ cluster/kubectl.sh get service dns-backend
NAME          LABELS    SELECTOR           IP(S)         PORT(S)
dns-backend   <none>    name=dns-backend   10.0.35.246   8000/TCP
```

### Step Four: Create client pod in one namespace

Use the file [`examples/cluster-dns/dns-frontend-pod.yaml`](dns-frontend-pod.yaml) to create a client [pod](../../docs/pods.md) in dev namespace. The client pod will make a connection to backend and exit. Specifically, it tries to connect to address `http://dns-backend.development.kubernetes.local:8000`.

```shell
$ cluster/kubectl.sh config use-context dev
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-frontend-pod.yaml
```

Once that's up you can list the pod in the cluster:

```shell
$ cluster/kubectl.sh get pods dns-frontend
POD            IP           CONTAINER(S)   IMAGE(S)               HOST                                    LABELS              STATUS    CREATED     MESSAGE
dns-frontend   10.244.2.9                                         kubernetes-minion-sswf/104.154.55.211   name=dns-frontend   Running   3 seconds
                            dns-frontend   ddysher/dns-frontend                                                               Running   2 seconds
```

Wait until the pod succeeds, then we can see the output from the client pod:

```shell
$ cluster/kubectl.sh log dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.kubernetes.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```

Please refer to the [source code](./images/frontend/client.py) about the logs. First line prints out the ip address associated with the service in dev namespace; remaining lines print out our request and server response. If we switch to prod namespace with the same pod config, we'll see the same result, i.e. dns will resolve across namespace.

```shell
$ cluster/kubectl.sh config use-context prod
$ cluster/kubectl.sh create -f examples/cluster-dns/dns-frontend-pod.yaml
$ cluster/kubectl.sh log dns-frontend
2015-05-07T20:13:54.147664936Z 10.0.236.129
2015-05-07T20:13:54.147721290Z Send request to: http://dns-backend.development.kubernetes.local:8000
2015-05-07T20:13:54.147733438Z <Response [200]>
2015-05-07T20:13:54.147738295Z Hello World!
```


#### Note about default namespace

If you prefer not using namespace, then all your services can be addressed using `default` namespace, e.g. `http://dns-backend.default.kubernetes.local:8000`, or shorthand version `http://dns-backend:8000`


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cluster-dns/README.md?pixel)]()
