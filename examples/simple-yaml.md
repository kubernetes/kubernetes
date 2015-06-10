## Getting started with config files.

In addition to the imperative style commands described [elsewhere](simple-nginx.md), Kubernetes
supports declarative YAML or JSON configuration files.  Often times config files are preferable
to imperative commands, since they can be checked into version control and changes to the files
can be code reviewed, producing a more robust, reliable and archival system.

### Running a container from a pod configuration file

```bash
cd kubernetes
kubectl create -f pod.yaml
```

Where pod.yaml contains something like:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

You can see your cluster's pods:

```bash
kubectl get pods
```

and delete the pod you just created:

```bash
kubectl delete pods nginx
```

### Running a replicated set of containers from a configuration file
To run replicated containers, you need a [Replication Controller](../docs/replication-controller.md).
A replication controller is responsible for ensuring that a specific number of pods exist in the
cluster.

```bash
cd kubernetes
kubectl create -f replication.yaml
```

Where ```replication.yaml``` contains:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

To delete the replication controller (and the pods it created):
```bash
kubectl delete rc nginx
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/simple-yaml.md?pixel)]()
