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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# L4 Ingress

## Abstract

A proposal to add support for TCP and UDP to the Ingress Resource.

Existing issue regarding Ingress and TCP
* Ingress and TCP [#23291](https://github.com/kubernetes/kubernetes/issues/23291)


## Use Cases

1. Be able to accept and route traffic for arbitrary TCP ports
1. Be able to accept and route traffic for arbitrary UDP ports


## Motivation

When deploying and exposing non-http(s) based services, Kubernetes users are
limited to NodePort and service type=LoadBalancer as methods for exposing these
services using interfaces supported by the Kubernetes API.

Other methods exist, via contributed code, for exposing non-http(s) based services:

* [keepalived-vip](https://github.com/kubernetes/contrib/tree/master/keepalived-vip)
* NGINX Ingress Controller
  [nginx-ingress](https://github.com/kubernetes/contrib/tree/master/ingress/controllers/nginx)

However these implementations rely on a ConfigMap being populated with a
mapping of destination endpoint to service.  A single common interface for
describing all Ingress would is highly desirable.


## Design

### TCP and UDP support

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
    name: test-ingress
spec:
    rules:
    - tcp:
        port: 53
      backend:
        serviceName: testdns
        servicePort: 53
    - udp:
        port: 53
      backend:
        serviceName: testdns
        servicePort: 53
```

While HTTP rules have host specifiers, in TCP and UDP rules they would not be
allowed as hostname handling is specific to L5 or higher layer protocol being
used over TCP or UDP.  TCP and UDP as protocol specifiers should be agnostic to
higher layer protocols.

### TCP with TLS

Given the following secret

```yaml
apiVersion: v1
kind: Secret
data:
    tls.crt: base64 encoded cert
    tls.key: base64 encoded key
metadata:
    name: testsecret
type: Opaque
```

TCP with TLS would be almost identical to the existing TLS Ingress
specification with the addition of the tcp attribute for the Ingress rule

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
    name: test-ingress
spec:
    rules:
    - tcp:
        port: 3306
      host: www.foo.com
      tls:
        - secretName: testsecret
      backend:
        serviceName: testmysql
        servicePort: 3306
```

TLS as a protocol does support hostnames via SNI, thus allowing a host
specifier seems to make sense and opens the possibility of supporting multiple
TCP services on a single IP.

### Ingress type specifier

The current Ingress API assumes a protocol based on the presence or absence
of a key (tls, http).  This leaves open the possibility of a spec that defines
all protocols.

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
    name: test-ingress
spec:
    rules:
    - tcp:
        port: 3306
      udp:
        port: 3306
      host: www.foo.com
      tls:
        - secretName: testsecret
      backend:
        serviceName: testmysql
        servicePort: 3306
      http:
        paths:
        - path: /bar
          backend:
              serviceName: bar
              servicePort: 80
```

A specification like the one above is arguably too complex and should be
written as separate rules for each protocol being supported.  However, the
loose syntax of the specification allows construction of a complex
specification.  Using a 'type' or 'protocol' specifier would remove this
ambiguity in the specification.

```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
    name: test-ingress
spec:
    rules:
    - protocol: tcp
      port: 3306
      backend:
        serviceName: testmysql
        servicePort: 3306

    - protocol: udp
      port: 3306
      backend:
        serviceName: testmysql
        servicePort: 3306

    - protocol: tls
      host: www.foo.com
      tls:
        - secretName: testsecret
      http:
        paths:
        - path: /bar
          backend:
              serviceName: bar
              servicePort: 80
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/l4-ingress.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
