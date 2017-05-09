<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/simple-nginx.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Running your first containers in Kubernetes

Once you've run one of the [getting started guides](../docs/getting-started-guides/) and you have
successfully turned up a Kubernetes cluster.  This guide will help you running your first containers on the cluster. 
The example is easy and simple enough, it is reliable and isn't involved other things.

### Create the nginx replicated pods

From this point onwards, it is assumed that `kubectl` is on your path from one of the getting started guides.

The [`kubectl create`](../docs/user-guide/kubectl/kubectl_create.md) line below will create two [nginx](https://registry.hub.docker.com/_/nginx/) [pods](../docs/user-guide/pods.md) listening on port 80. It will also create a [replication controller](../docs/user-guide/replication-controller.md) named `nginx-controller` to ensure that there are always two pods running.

```bash
kubectl create -f nginx-controller.yaml
```
Where nginx-controller.yaml contains something like:
<!-- BEGIN MUNGE: EXAMPLE pod.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx-controller
spec:
  replicas: 1
  selector:
    name: nginx
  template:
    metadata:
      labels:
        name: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80

```

[Download example](nginx-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE pod.yaml -->


Once the pods are created, you can list them to see what is up and running:

```bash
kubectl get pods
```

You can also see the replication controller that was created:

```bash
kubectl get rc
```

To stop the two replicated containers, stop the replication controller:

```bash
kubectl stop rc my-nginx
```

### Create the nginx service

Define [nginx-service-clusterip](nginx-service-clusterip.yaml):

```bash
kubectl create -f nginx-service-clusterip.yaml
```

This should print the service that has been created, and map an external IP address to the service. Where to find this external IP address will depend on the environment you run in. Another way is define nginx-service-nodeport.yaml.

```console
$ kubectl get services
NAME                      CLUSTER_IP      EXTERNAL_IP   PORT(S)    SELECTOR     AGE
kubernetes                192.168.3.1     <none>        443/TCP    <none>       1d
nginx-service-clusterip   192.168.3.232   <none>        8001/TCP   name=nginx   6h
nginx-service-nodeport    192.168.3.56    nodes         8000/TCP   name=nginx   6h
```

In order to access your nginx landing page, you also have to make sure that traffic from external IPs is allowed. Do this by opening a firewall to allow traffic on port 8001.

### Accessing the nginx site externally

On any node in cluster, you can use curl to confirm service and nginx is running. 
```console
$ curl  192.168.3.232:8001
  or execute
$ curl  192.168.3.56:8000

<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
    body {
        width: 35em;
        margin: 0 auto;
        font-family: Tahoma, Verdana, Arial, sans-serif;
    }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>

<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>

```

