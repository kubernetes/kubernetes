{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Print information about the Cluster and Client versions
- Print information about the Control Plane
- Print information about Nodes
- Print information about APIs
{% endpanel %}

# Cluster Info

## Motivation

It may be necessary to learn about the Kubernetes cluster itself, rather
than just the workloads running in it.  This can be useful for debugging
unexpected behavior.

## Versions

{% method %}

The `kubectl version` prints the client and server versions.  Note that
the client version may not be present for clients built locally from
source.

{% sample lang="yaml" %}

```bash
kubectl version
```

```bash
Client Version: version.Info{Major:"1", Minor:"9", GitVersion:"v1.9.5", GitCommit:"f01a2bf98249a4db383560443a59bed0c13575df", GitTreeState:"clean", BuildDate:"2018-03-19T19:38:17Z", GoVersion:"go1.9.4", Compiler:"gc", Platform:"darwin/amd64"}
Server Version: version.Info{Major:"1", Minor:"11+", GitVersion:"v1.11.6-gke.2", GitCommit:"04ad69a117f331df6272a343b5d8f9e2aee5ab0c", GitTreeState:"clean", BuildDate:"2019-01-04T16:19:46Z", GoVersion:"go1.10.3b4", Compiler:"gc", Platform:"linux/amd64"}
```
{% endmethod %}

{% panel style="warning", title="Version Skew" %}
Kubectl supports +/-1 version skew with the Kubernetes cluster.  Kubectl
versions that are more than 1 version ahead of or behind the cluster are
not guaranteed to be compatible.
{% endpanel %}

## Control Plane and Addons

{% method %}

The `kubectl cluster-info` prints information about the control plane and
add-ons.

{% sample lang="yaml" %}

```bash
kubectl cluster-info
```

```bash
  Kubernetes master is running at https://1.1.1.1
  GLBCDefaultBackend is running at https://1.1.1.1/api/v1/namespaces/kube-system/services/default-http-backend:http/proxy
  Heapster is running at https://1.1.1.1/api/v1/namespaces/kube-system/services/heapster/proxy
  KubeDNS is running at https://1.1.1.1/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
  Metrics-server is running at https://1.1.1.1/api/v1/namespaces/kube-system/services/https:metrics-server:/proxy
```

{% endmethod %}

{% panel style="info", title="Kube Proxy" %}
The URLs printed by `cluster-info` can be accessed at `127.0.0.1:8001` by
running `kubectl proxy`. 
{% endpanel %}

## Nodes


{% method %}

The `kubectl top node` and `kubectl top pod` print information about the
top nodes and pods.

{% sample lang="yaml" %}

```bash
kubectl top node
```

```bash
  NAME                                 CPU(cores)   CPU%      MEMORY(bytes)   MEMORY%   
  gke-dev-default-pool-e1e7bf6a-cc8b   37m          1%        571Mi           10%       
  gke-dev-default-pool-e1e7bf6a-f0xh   103m         5%        1106Mi          19%       
  gke-dev-default-pool-e1e7bf6a-jfq5   139m         7%        1252Mi          22%       
  gke-dev-default-pool-e1e7bf6a-x37l   112m         5%        982Mi           17%  
```

{% endmethod %}

## APIs

The `kubectl api-versions` and `kubectl api-resources` print information
about the available Kubernetes APIs.  This information is read from the
Discovery Service.

{% method %}

Print the Resource Types available in the cluster.

{% sample lang="yaml" %}

```bash
kubectl api-resources
```

```bash
NAME                              SHORTNAMES   APIGROUP                       NAMESPACED   KIND
bindings                                                                      true         Binding
componentstatuses                 cs                                          false        ComponentStatus
configmaps                        cm                                          true         ConfigMap
endpoints                         ep                                          true         Endpoints
events                            ev                                          true         Event
limitranges                       limits                                      true         LimitRange
namespaces                        ns                                          false        Namespace
...
```
{% endmethod %}

{% method %}

Print the API versions available in the cluster.

{% sample lang="yaml" %}

```bash
kubectl api-versions
```

```bash
  admissionregistration.k8s.io/v1beta1
  apiextensions.k8s.io/v1beta1
  apiregistration.k8s.io/v1
  apiregistration.k8s.io/v1beta1
  apps/v1
  apps/v1beta1
  apps/v1beta2
  ...
```

{% endmethod %}

{% panel style="info", title="Discovery" %}
The discovery information can be viewed at `127.0.0.1:8001/` by running
`kubectl proxy`.  The Discovery for specific API can be found under either
`/api/v1` or `/apis/<group>/<version>`, depending on the API group -
e.g. `127.0.0.1:8001/apis/apps/v1`
{% endpanel %}


{% method %}

The `kubectl explain` command can be used to print metadata about specific
Resource types.  This is useful for learning about the type.

{% sample lang="yaml" %}

```bash
kubectl explain deployment --api-version apps/v1
```

```bash
KIND:     Deployment
VERSION:  apps/v1

DESCRIPTION:
     Deployment enables declarative updates for Pods and ReplicaSets.

FIELDS:
   apiVersion	<string>
     APIVersion defines the versioned schema of this representation of an
     object. Servers should convert recognized schemas to the latest internal
     value, and may reject unrecognized values. More info:
     https://git.k8s.io/community/contributors/devel/api-conventions.md#resources

   kind	<string>
     Kind is a string value representing the REST resource this object
     represents. Servers may infer this from the endpoint the client submits
     requests to. Cannot be updated. In CamelCase. More info:
     https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds

   metadata	<Object>
     Standard object metadata.

   spec	<Object>
     Specification of the desired behavior of the Deployment.

   status	<Object>
     Most recently observed status of the Deployment.
```

{% endmethod %}

 
