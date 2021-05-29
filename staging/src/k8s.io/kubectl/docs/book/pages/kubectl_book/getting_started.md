{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Creating Resources
- Printing Resources
- Debugging Containers
{% endpanel %}

# Getting Started With Kubectl

**Note**: If you are already familiar with Kubectl, you can skip this section.

This section provides a brief overview of the most basic Kubectl commands, which are
described in more detail in later chapters.

For more background on the Kubernetes APIs themselves, see the docs at [k8s.io](k8s.io).

## Listing Kubernetes Resources

{% method %}

List the Kubernetes *Deployment* Resources that are in the kube-system namespace.

**Note:** Deployments are Resources which manage Pod replicas (Pods run Containers)

{% sample lang="yaml" %}
```bash
kubectl get deployments --namespace kube-system
```

```bash
NAME                     DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
event-exporter-v0.2.3    1         1         1            1           14d
fluentd-gcp-scaler       1         1         1            1           14d
kube-dns                 2         2         2            2           14d
kube-dns-autoscaler      1         1         1            1           14d
l7-default-backend       1         1         1            1           14d
metrics-server-v0.3.1    1         1         1            1           14d
```

{% endmethod %}

{% method %}

Print detailed information about the kube-dns Deployment in the kube-system namespace.

{% sample lang="yaml" %}
```bash
kubectl describe deployment kube-dns --namespace kube-system
```

```bash
Name:                   kube-dns
Namespace:              kube-system
CreationTimestamp:      Wed, 06 Mar 2019 17:36:05 -0800
Labels:                 addonmanager.kubernetes.io/mode=Reconcile
                        k8s-app=kube-dns
                        kubernetes.io/cluster-service=true
Annotations:            deployment.kubernetes.io/revision: 2
...
```
{% endmethod %}

## Creating a Resource from Config

{% method %}

Create or Update Kubernetes Resources from Remote Config.

{% sample lang="yaml" %}
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/kubectl/master/docs/book/examples/nginx/nginx.yaml
```

```bash
service/nginx created
deployment.apps/nginx-deployment created
```
{% endmethod %}

{% method %}

Create or Update Kubernetes Resources from Local Config.

{% sample lang="yaml" %}
```bash
kubectl apply -f ./examples/nginx/nginx.yaml
```

```bash
service/nginx created
deployment.apps/nginx-deployment created
```
{% endmethod %}

{% method %}

Print the Resources that were Applied.

{% sample lang="yaml" %}
```bash
kubectl get -f ./examples/nginx/nginx.yaml --show-labels
```

```bash
NAME            TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE   LABELS
service/nginx   ClusterIP   10.59.245.201   <none>        80/TCP    11m   <none>

NAME                               DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE   LABELS
deployment.apps/nginx-deployment   3         3         3            3           11m   app=nginx
```
{% endmethod %}

## Generating a Config from a Command

{% method %}

Generate Config for a Deployment Resource.  This could be Applied to a cluster by writing the output
to a file, and then running `kubectl apply -f <yaml-file>`

**Note:** The generated Config has extra boilerplate that users shouldn't include but exists
due to the serialization process of go objects.

{% sample lang="yaml" %}
```bash
kubectl create deployment nginx --dry-run=client -o yaml --image nginx
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null # delete this
  labels:
    app: nginx
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  strategy: {} # delete this
  template:
    metadata:
      creationTimestamp: null # delete this
      labels:
        app: nginx
    spec:
      containers:
      - image: nginx
        name: nginx
        resources: {} # delete this
status: {} # delete this
```
{% endmethod %}

## Viewing Pods Associated with Resources

{% method %}

View the Pods created by the Deployment using the Pod labels.

{% sample lang="yaml" %}
```bash
kubectl get pods -l app=nginx
```

```bash
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-5c689d88bb-b2xfk   1/1     Running   0          10m
nginx-deployment-5c689d88bb-rx569   1/1     Running   0          10m
nginx-deployment-5c689d88bb-s7xcv   1/1     Running   0          10m
```
{% endmethod %}

## Debugging Containers

{% method %}

Get the logs from all Pods managed by the Deployment.

{% sample lang="yaml" %}

```bash
kubectl logs -l app=nginx
```

{% endmethod %}

{% method %}

Get a shell into a specific Pod's Container

{% sample lang="yaml" %}

```bash
kubectl exec -i -t  nginx-deployment-5c689d88bb-s7xcv bash
```

```bash
root@nginx-deployment-5c689d88bb-s7xcv:/#
```

{% endmethod %}

## Extending kubectl

There is a plugin mechanism to adapt `kubectl` to your particular needs.

{% method %}

Show which plugins are currently available

{% sample lang="yaml" %}

```bash
kubectl plugin list
```

{% endmethod %}

The easiest way to discover and install plugins is via the kubernetes sub-project [krew.sigs.k8s.io](https://krew.sigs.k8s.io/docs/user-guide/setup/install/).
