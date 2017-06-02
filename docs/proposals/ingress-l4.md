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

# Ingress L4 Proposal - TCP

## Abstract

A proposal to add support for TCP to the Ingress Resource.

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


## Design Considerations

### State of L4 Protocol Support in Cloud Providers

L4 support is not uniform across all Cloud Providers, specifically TLS
termination is lacking in several Cloud Providers.

|           | TCP | UDP | TLS |
| --------- | --- | --- | --- |
| AWS       |  x  |     |  x  |
| GCE       |  x  |  x  | (1) |
| Mesos     |  ?  |  ?  |  ?  |
| Openstack |  x  |  x  |  x  |
| Rackspace |  x  |  x  |     |
| vSphere   |  x  |  x  |  x  |

(1) GCE has TLS as an alpha service

### TCP support

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
```

While HTTP rules have host specifiers, in TCP rules hostnames would not be
as hostname handling is specific to L5 or higher layer protocol being used over
TCP.  TCP as a protocol specifier should be agnostic to higher layer protocols.

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

### L4 TLS Termination

While it would be very useful to be able to express TLS protected L4 Ingress,
such that TLS can be terminated by the Ingress implementation, not all cloud
providers support TLS termination, thus more design is required to add L4 TLS
for Ingress in a manner that would work for all cloud providers.

## Proposed Addition/Change

 - Add a protocol specifier for rules to explicitly declare the desired protocol
   for Ingress
 - Add TCP protocol type to allow definition of TCP Ingress

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

    - protocol: http
      host: www.foo.com
      paths:
      - path: /bar
        backend:
            serviceName: bar
            servicePort: 80

    - protocol: https
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

Although the proposal is to use a protocol specifier for http/https traffic
it does not preclude supporting the http and tls ingress types from the
existing Ingress spec.

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

    - host: www.foo.com
      http:
        paths:
        - path: /bar
          backend:
              serviceName: bar
              servicePort: 80

    - tls:
        - secretName: testsecret
      backend:
        serviceName: bar
        servicePort: 80
```

Despite this the proposal is to eventually deprecate the existance of a http or
tls key as the indication of which protocol is desired for Ingress, and use an
explicit protocol specifier.

## Aditional notes about Cloud Providers' supported protocols

### AWS

AWS supports TCP

#### Protocols

 - HTTP
 - HTTPS
 - TCP
 - TLS

#### Security Groups

Required for each LB instance

#### Static IP

No support for static IP

### GCE

#### Protocols

 - HTTP
 - HTTPS
 - TCP
 - UDP

### Mesos

### Openstack

 - TCP
 - UDP

### Rackspace

 - TCP
 - UDP

The following protocols are advertised but they seem like convenience options
that map to a TCP or UDP port as appropriate, rather than actual support for
the underlying protocol.

 - DNS
 - FTP
 - HTTP
 - HTTPS
 - IMPAS
 - IMPAv2
 - IMPAv3
 - IMPAv4
 - LDAP
 - LDAPS
 - MYSQL
 - POP3
 - POP3S
 - SFTP
 - SMTP

### vSphere

Supported protocols for vSphere vary based on the load balancer appliance used
with vSphere.  An in-cluster (k8s) software load balancer is currently
available in contrib that use NGINX as the software load balancing solution.

 - HTTP
 - HTTPS
 - TCP
 - TLS
 - UDP


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/ingress-l4.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
