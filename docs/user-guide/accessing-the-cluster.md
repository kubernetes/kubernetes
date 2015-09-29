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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/accessing-the-cluster.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# User Guide to Accessing the Cluster

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [User Guide to Accessing the Cluster](#user-guide-to-accessing-the-cluster)
  - [Accessing the cluster API](#accessing-the-cluster-api)
    - [Accessing for the first time with kubectl](#accessing-for-the-first-time-with-kubectl)
    - [Directly accessing the REST API](#directly-accessing-the-rest-api)
      - [Using kubectl proxy](#using-kubectl-proxy)
      - [Without kubectl proxy](#without-kubectl-proxy)
    - [Programmatic access to the API](#programmatic-access-to-the-api)
    - [Accessing the API from a Pod](#accessing-the-api-from-a-pod)
  - [Accessing services running on the cluster](#accessing-services-running-on-the-cluster)
    - [Ways to connect](#ways-to-connect)
    - [Discovering builtin services](#discovering-builtin-services)
      - [Manually constructing apiserver proxy URLs](#manually-constructing-apiserver-proxy-urls)
        - [Examples](#examples)
      - [Using web browsers to access services running on the cluster](#using-web-browsers-to-access-services-running-on-the-cluster)
  - [Requesting redirects](#requesting-redirects)
  - [So Many Proxies](#so-many-proxies)

<!-- END MUNGE: GENERATED_TOC -->

## Accessing the cluster API

### Accessing for the first time with kubectl

When accessing the Kubernetes API for the first time, we suggest using the
Kubernetes CLI, `kubectl`.

To access a cluster, you need to know the location of the cluster and have credentials
to access it.  Typically, this is automatically set-up when you work through
though a [Getting started guide](../getting-started-guides/README.md),
or someone else setup the cluster and provided you with credentials and a location.

Check the location and credentials that kubectl knows about with this command:

```console
$ kubectl config view
```

Many of the [examples](../../examples/) provide an introduction to using
kubectl and complete documentation is found in the [kubectl manual](kubectl/kubectl.md).

### Directly accessing the REST API

Kubectl handles locating and authenticating to the apiserver.
If you want to directly access the REST API with an http client like
curl or wget, or a browser, there are several ways to locate and authenticate:
  - Run kubectl in proxy mode.
    - Recommended approach.
    - Uses stored apiserver location.
    - Verifies identity of apiserver using self-signed cert.  No MITM possible.
    - Authenticates to apiserver.
    - In future, may do intelligent client-side load-balancing and failover.
  - Provide the location and credentials directly to the http client.
    - Alternate approach.
    - Works with some types of client code that are confused by using a proxy.
    - Need to import a root cert into your browser to protect against MITM.

#### Using kubectl proxy

The following command runs kubectl in a mode where it acts as a reverse proxy.  It handles
locating the apiserver and authenticating.
Run it like this:

```console
$ kubectl proxy --port=8080 &
```

See [kubectl proxy](kubectl/kubectl_proxy.md) for more details.

Then you can explore the API with curl, wget, or a browser, like so:

```console
$ curl http://localhost:8080/api/
{
  "versions": [
    "v1"
  ]
}
```

#### Without kubectl proxy

It is also possible to avoid using kubectl proxy by passing an authentication token
directly to the apiserver, like this:

```console
$ APISERVER=$(kubectl config view | grep server | cut -f 2- -d ":" | tr -d " ")
$ TOKEN=$(kubectl config view | grep token | cut -f 2 -d ":" | tr -d " ")
$ curl $APISERVER/api --header "Authorization: Bearer $TOKEN" --insecure
{
  "versions": [
    "v1"
  ]
}
```

The above example uses the `--insecure` flag.  This leaves it subject to MITM
attacks.  When kubectl accesses the cluster it uses a stored root certificate
and client certificates to access the server.  (These are installed in the
`~/.kube` directory).  Since cluster certificates are typically self-signed, it
make take special configuration to get your http client to use root
certificate.

On some clusters, the apiserver does not require authentication; it may serve
on localhost, or be protected by a firewall.  There is not a standard
for this.  [Configuring Access to the API](../admin/accessing-the-api.md)
describes how a cluster admin can configure this.  Such approaches may conflict
with future high-availability support.

### Programmatic access to the API

There are [client libraries](../devel/client-libraries.md) for accessing the API
from several languages.  The Kubernetes project-supported
[Go](http://releases.k8s.io/HEAD/pkg/client/)
client library can use the same [kubeconfig file](kubeconfig-file.md)
as the kubectl CLI does to locate and authenticate to the apiserver.

See documentation for other libraries for how they authenticate.

### Accessing the API from a Pod

When accessing the API from a pod, locating and authenticating
to the api server are somewhat different.

The recommended way to locate the apiserver within the pod is with
the `kubernetes` DNS name, which resolves to a Service IP which in turn
will be routed to an apiserver.

The recommended way to authenticate to the apiserver is with a
[service account](service-accounts.md) credential.  By kube-system, a pod
is associated with a service account, and a credential (token) for that
service account is placed into the filesystem tree of each container in that pod,
at `/var/run/secrets/kubernetes.io/serviceaccount/token`.

From within a pod the recommended ways to connect to API are:
  - run a kubectl proxy as one of the containers in the pod, or as a background
    process within a container.  This proxies the
    Kubernetes API to the localhost interface of the pod, so that other processes
    in any container of the pod can access it.  See this [example of using kubectl proxy
    in a pod](../../examples/kubectl-container/).
  - use the Go client library, and create a client using the `client.NewInCluster()` factory.
    This handles locating and authenticating to the apiserver.
In each case, the credentials of the pod are used to communicate securely with the apiserver.


## Accessing services running on the cluster

The previous section was about connecting the Kubernetes API server.  This section is about
connecting to other services running on Kubernetes cluster.  In Kubernetes, the
[nodes](../admin/node.md), [pods](pods.md) and [services](services.md) all have
their own IPs.  In many cases, the node IPs, pod IPs, and some service IPs on a cluster will not be
routable, so they will not be reachable from a machine outside the cluster,
such as your desktop machine.

### Ways to connect

You have several options for connecting to nodes, pods and services from outside the cluster:
  - Access services through public IPs.
    - Use a service with type `NodePort` or `LoadBalancer` to make the service reachable outside
      the cluster.  See the [services](services.md) and
      [kubectl expose](kubectl/kubectl_expose.md) documentation.
    - Depending on your cluster environment, this may just expose the service to your corporate network,
      or it may expose it to the internet.  Think about whether the service being exposed is secure.
      Does it do its own authentication?
    - Place pods behind services.  To access one specific pod from a set of replicas, such as for debugging,
      place a unique label on the pod it and create a new service which selects this label.
    - In most cases, it should not be necessary for application developer to directly access
      nodes via their nodeIPs.
  - Access services, nodes, or pods using the Proxy Verb.
    - Does apiserver authentication and authorization prior to accessing the remote service.
      Use this if the services are not secure enough to expose to the internet, or to gain
      access to ports on the node IP, or for debugging.
    - Proxies may cause problems for some web applications.
    - Only works for HTTP/HTTPS.
    - Described [here](#discovering-builtin-services).
  - Access from a node or pod in the cluster.
    - Run a pod, and then connect to a shell in it using [kubectl exec](kubectl/kubectl_exec.md).
      Connect to other nodes, pods, and services from that shell.
    - Some clusters may allow you to ssh to a node in the cluster.  From there you may be able to
      access cluster services.  This is a non-standard method, and will work on some clusters but
      not others.  Browsers and other tools may or may not be installed.  Cluster DNS may not work.

### Discovering builtin services

Typically, there are several services which are started on a cluster by kube-system. Get a list of these
with the `kubectl cluster-info` command:

```console
$ kubectl cluster-info

  Kubernetes master is running at https://104.197.5.247
  elasticsearch-logging is running at https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging
  kibana-logging is running at https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/kibana-logging
  kube-dns is running at https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/kube-dns
  grafana is running at https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/monitoring-grafana
  heapster is running at https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster
```

This shows the proxy-verb URL for accessing each service.
For example, this cluster has cluster-level logging enabled (using Elasticsearch), which can be reached
at `https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging/` if suitable credentials are passed, or through a kubectl proxy at, for example:
`http://localhost:8080/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging/`.
(See [above](#accessing-the-cluster-api) for how to pass credentials or use kubectl proxy.)

#### Manually constructing apiserver proxy URLs

As mentioned above, you use the `kubectl cluster-info` command to retrieve the service's proxy URL. To create proxy URLs that include service endpoints, suffixes, and parameters, you simply append to the service's proxy URL:
`http://`*`kubernetes_master_address`*`/`*`service_path`*`/`*`service_name`*`/`*`service_endpoint-suffix-parameter`*
<!--- TODO: update this part of doc because it doesn't seem to be valid. What
about namespaces? 'proxy' verb? -->

##### Examples

 * To access the Elasticsearch service endpoint `_search?q=user:kimchy`, you would use:   `http://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging/_search?q=user:kimchy`
 * To access the Elasticsearch cluster health information `_cluster/health?pretty=true`, you would use:   `https://104.197.5.247/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging/_cluster/health?pretty=true`

  ```json
  {
	 "cluster_name" : "kubernetes_logging",
	 "status" : "yellow",
	 "timed_out" : false,
	 "number_of_nodes" : 1,
	 "number_of_data_nodes" : 1,
	 "active_primary_shards" : 5,
	 "active_shards" : 5,
	 "relocating_shards" : 0,
	 "initializing_shards" : 0,
	 "unassigned_shards" : 5
  }
  ```

#### Using web browsers to access services running on the cluster

You may be able to put an apiserver proxy url into the address bar of a browser. However:
  - Web browsers cannot usually pass tokens, so you may need to use basic (password) auth.  Apiserver can be configured to accept basic auth,
    but your cluster may not be configured to accept basic auth.
  - Some web apps may not work, particularly those with client side javascript that construct urls in a
    way that is unaware of the proxy path prefix.

## Requesting redirects

The redirect capabilities have been deprecated and removed.  Please use a proxy (see below) instead.

## So Many Proxies

There are several different proxies you may encounter when using Kubernetes:
  1. The [kubectl proxy](#directly-accessing-the-rest-api):
    - runs on a user's desktop or in a pod
    - proxies from a localhost address to the Kubernetes apiserver
    - client to proxy uses HTTP
    - proxy to apiserver uses HTTPS
    - locates apiserver
    - adds authentication headers
  1. The [apiserver proxy](#discovering-builtin-services):
    - is a bastion built into the apiserver
    - connects a user outside of the cluster to cluster IPs which otherwise might not be reachable
    - runs in the apiserver processes
    - client to proxy uses HTTPS (or http if apiserver so configured)
    - proxy to target may use HTTP or HTTPS as chosen by proxy using available information
    - can be used to reach a Node, Pod, or Service
    - does load balancing when used to reach a Service
  1. The [kube proxy](services.md#ips-and-vips):
    - runs on each node
    - proxies UDP and TCP
    - does not understand HTTP
    - provides load balancing
    - is just used to reach services
  1. A Proxy/Load-balancer in front of apiserver(s):
    - existence and implementation varies from cluster to cluster (e.g. nginx)
    - sits between all clients and one or more apiservers
    - acts as load balancer if there are several apiservers.
  1. Cloud Load Balancers on external services:
    - are provided by some cloud providers (e.g. AWS ELB, Google Cloud Load Balancer)
    - are created automatically when the Kubernetes service has type `LoadBalancer`
    - use UDP/TCP only
    - implementation varies by cloud provider.



Kubernetes users will typically not need to worry about anything other than the first two types.  The cluster admin
will typically ensure that the latter types are setup correctly.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/accessing-the-cluster.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
