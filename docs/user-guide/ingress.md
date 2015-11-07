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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/ingress.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Ingress

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Ingress](#ingress)
  - [What is Ingress?](#what-is-ingress)
  - [The Ingress Resource](#the-ingress-resource)
  - [Ingress controllers](#ingress-controllers)
  - [Types of Ingress](#types-of-ingress)
    - [Single Service Ingress](#single-service-ingress)
    - [Simple fanout](#simple-fanout)
    - [Name based virtual hosting](#name-based-virtual-hosting)
    - [Loadbalancing](#loadbalancing)
  - [Updating an Ingress](#updating-an-ingress)
  - [Future Work](#future-work)
  - [Alternatives](#alternatives)

<!-- END MUNGE: GENERATED_TOC -->

__Terminology__

Throughout this doc you will see a few terms that are sometimes used interchangably elsewhere, that might cause confusion. This section attempts to clarify them.

* Node: A single virtual or physical machine in a Kubernetes cluster.
* Cluster: A group of nodes firewalled from the internet, that are the primary compute resources managed by Kubernetes.
* Edge router: A router that enforces the firewall policy for your cluster. This could be a gateway managed by a cloudprovider or a physical piece of hardware.
* Cluster network: A set of links, logical or physical, that facilitate communication within a cluster according to the [Kubernetes networking model](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/admin/networking.md). Examples of a Cluster network include Overlays such as [flannel](https://github.com/coreos/flannel#flannel) or SDNs such as [OVS](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/admin/ovs-networking.md).
* Service: A Kubernetes [Service](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/user-guide/services.md) that identifies a set of pods using label selectors. Unless mentioned otherwise, Services are assumed to have virtual IPs only routable within the cluster network.

## What is Ingress?

Typically, services and pods have IPs only routable by the cluster network. All traffic that ends up at an edge router is either dropped or forwarded elsewhere. Conceptually, this might look like:

```
    internet
        |
  ------------
  [ Services ]
```

An Ingress is a collection of rules that allow inbound connections to reach the cluster services.

```
    internet
        |
   [ Ingress ]
   --|-----|--
   [ Services ]
```

It can be configured to give services externally-reachable urls, load balance traffic, terminate SSL, offer name based virtual hosting etc. Users request ingress by POSTing the Ingress resource to the API server. An [Ingress controller](#ingress-controllers) is responsible for fulfilling the Ingress, usually with a loadbalancer, though it may also configure your edge router or additional frontends to help handle the traffic in an HA manner.

## The Ingress Resource

A minimal Ingress might look like:

```yaml
01. apiVersion: extensions/v1beta1
02. kind: Ingress
03. metadata:
04.  name: test-ingress
05. spec:
06.  rules:
07.  - http:
08.      paths:
09.      - path: /testpath
10.        backend:
11.          serviceName: test
12.          servicePort: 80
```

*POSTing this to the API server will have no effect if you have not configured an [Ingress controller](#ingress-controllers).*

__Lines 1-4__: As with all other Kubernetes config, an Ingress needs `apiVersion`, `kind`, and `metadata` fields.  For general information about working with config files, see [here](simple-yaml.md), [here](configuring-containers.md), and [here](working-with-resources.md).

__Lines 5-7__: Ingress [spec](../devel/api-conventions.md#spec-and-status) has all the information needed to configure a loadbalancer or proxy server. Most importantly, it contains a list of rules matched against all incoming requests. Currently the Ingress resource only supports http rules.

__Lines 8-9__: Each http rule contains the following information: A host (eg: foo.bar.com, defaults to * in this example), a list of paths (eg: /testpath) each of which has an associated backend (test:80). Both the host and path must match the content of an incoming request before the loadbalancer directs traffic to the backend.

__Lines 10-12__: A backend is a service:port combination as described in the [services doc](services.md). Ingress traffic is typically sent directly to the endpoints matching a backend.

__Global Parameters__: For the sake of simplicity the example Ingress has no global parameters, see the [api-reference](../../pkg/apis/extensions/v1beta1/types.go) for a full definition of the resource. One can specify a global default backend in the absence of which requests that don't match a path in the spec are sent to the default backend of the Ingress controller. Though the Ingress resource doesn't support HTTPS yet, security configs would also be global.

## Ingress controllers

In order for the Ingress resource to work, the cluster must have an Ingress controller running. This is unlike other types of controllers, which typically run as part of the `kube-controller-manager` binary, and which are typically started automatically as part of cluster creation. You need to choose the ingress controller implementation that is the best fit for your cluster, or implement one.  Examples and instructions can be found [here](https://github.com/kubernetes/contrib/tree/master/Ingress).

## Types of Ingress

### Single Service Ingress

There are existing Kubernetes concepts that allow you to expose a single service (see [alternatives](#alternatives)), however you can do so through an Ingress as well, by specifying a *default backend* with no rules.

<!-- BEGIN MUNGE: EXAMPLE ingress.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: test-ingress
spec:
  backend:
    serviceName: testsvc
    servicePort: 80
```

[Download example](ingress.yaml?raw=true)
<!-- END MUNGE: EXAMPLE ingress.yaml -->

If you create it using `kubectl -f` you should see:

```shell
$ kubectl get ing
NAME                RULE          BACKEND        ADDRESS
test-ingress        -             testsvc:80     107.178.254.228
```

Where `107.178.254.228` is the IP allocated by the Ingress controller to satisfy this Ingress. The `RULE` column shows that all traffic send to the IP is directed to the Kubernetes Service listed under `BACKEND`.

### Simple fanout

As described previously, pods within kubernetes have ips only visible on the cluster network, so we need something at the edge accepting ingress traffic and proxying it to the right endpoints. This component is usually a highly available loadbalancer/s. An Ingress allows you to keep the number of loadbalancers down to a minimum, for example, a setup like:

```
foo.bar.com -> 178.91.123.132 -> / foo    s1:80
                                 / bar    s2:80
```

would require an Ingress such as:

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: test
spec:
  rules:
  - host: foo.bar.com
    http:
      paths:
      - path: /foo
        backend:
          serviceName: s1
          servicePort: 80
      - path: /bar
        backend:
          serviceName: s2
          servicePort: 80
```

When you create the Ingress with `kubectl create -f`:

```
$ kubectl get ing
NAME      RULE          BACKEND   ADDRESS
test      -
          foo.bar.com
          /foo          s1:80
          /bar          s2:80
```

The Ingress controller will provision an implementation specific loadbalancer that satisfies the Ingress, as long as the services (s1, s2) exist. When it has done so, you will see the address of the loadbalancer under the last column of the Ingress.

### Name based virtual hosting

Name-based virtual hosts use multiple host names for the same IP address.

```

foo.bar.com --|                 |-> foo.bar.com s1:80
              | 178.91.123.132  |
bar.foo.com --|                 |-> bar.foo.com s2:80
```

The following Ingress tells the backing loadbalancer to route requests based on the [Host header](https://tools.ietf.org/html/rfc7230#section-5.4).

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: test
spec:
  rules:
  - host: foo.bar.com
    http:
      paths:
      - backend:
          serviceName: s1
          servicePort: 80
  - host: bar.foo.com
    http:
      paths:
      - backend:
          serviceName: s2
          servicePort: 80
```


__Default Backends__: An Ingress with no rules, like the one shown in the previous section, sends all traffic to a single default backend. You can use the same technique to tell a loadbalancer where to find your website's 404 page, by specifying a set of rules *and* a default backend. Traffic is routed to your default backend if none of the Hosts in your Ingress match the Host in the request header, and/or none of the paths match the url of the request.

### Loadbalancing

An Ingress controller is bootstrapped with some loadbalancing policy settings that it applies to all Ingress, such as the loadbalancing algorithm, backend weight scheme etc. More advanced loadbalancing concepts (eg: persistent sessions, dynamic weights) are not yet exposed through the Ingress. You can still get these features through the [service loadbalancer](https://github.com/kubernetes/contrib/tree/master/service-loadbalancer). With time, we plan to distil loadbalancing patterns that are applicable cross platform into the Ingress resource.

It's also worth noting that even though health checks are not exposed directly through the Ingress, there exist parallel concepts in Kubernetes such as [readiness probes](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/user-guide/production-pods.md#liveness-and-readiness-probes-aka-health-checks) which allow you to achieve the same end result.

## Updating an Ingress

Say you'd like to add a new Host to an existing Ingress, you can update it by editing the resource:

```shell
$ kubectl get ing
NAME      RULE          BACKEND   ADDRESS
test      -                       178.91.123.132
          foo.bar.com
          /foo          s1:80
$ kubectl edit ing test
```

This should pop up an editor with the existing yaml, modify it to include the new Host.

```yaml
spec:
  rules:
  - host: foo.bar.com
    http:
      paths:
      - backend:
          serviceName: s1
          servicePort: 80
        path: /foo
  - host: bar.baz.com
    http:
      paths:
      - backend:
          serviceName: s2
          servicePort: 80
        path: /foo
..
```

saving it will update the resource in the API server, which should tell the Ingress controller to reconfigure the loadbalancer.

```shell
$ kubectl get ing
NAME      RULE          BACKEND   ADDRESS
test      -                       178.91.123.132
          foo.bar.com
          /foo          s1:80
          bar.baz.com
          /foo          s2:80
```

You can achieve the same by invoking `kubectl replace -f` on a modified Ingress yaml file.

## Future Work

* Various modes of HTTPS/TLS support (edge termination, sni etc)
* Requesting an IP or Hostname via claims
* Combining L4 and L7 Ingress
* More Ingress controllers

Please track the [L7 and Ingress proposal](https://github.com/kubernetes/kubernetes/pull/12827) for more details on the evolution of the resource, and the [Ingress sub-repository](https://github.com/kubernetes/contrib/tree/master/Ingress) for more details on the evolution of various Ingress controllers.

## Alternatives

You can expose a Service in multiple ways that don't directly involve the Ingress resource:
* Use [Service.Type=LoadBalancer](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/user-guide/services.md#type-loadbalancer)
* Use [Service.Type=NodePort](https://github.com/kubernetes/kubernetes/blob/release-1.0/docs/user-guide/services.md#type-nodeport)
* Use a [Port Proxy] (https://github.com/kubernetes/contrib/tree/master/for-demos/proxy-to-service)
* Deploy the [Service loadbalancer](https://github.com/kubernetes/contrib/tree/master/service-loadbalancer). This allows you to share a single IP among multiple Services and achieve more advanced loadbalancing through Service Annotations.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/ingress.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
