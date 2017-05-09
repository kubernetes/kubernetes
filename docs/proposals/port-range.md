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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/proposals/pod-security-context.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for allow port ranges in services and pod specifications. 

## Motivation

Currently, it's possible to specify ports in the pod and services specifications but only
one port each time, specifying also the protocol and the name, and targetPort in the case of 
services. When you want to specify a full range of ports, the only way is specify each port
individually. As result, the specification is large and error-prone.

It affects systems using Real Time Communication protocols where it's needed a port for each open
connection with an user. 

Users should be able to:

1.  Use a range of ports in their pod specification.
2.  Use a range of ports in their services specification.

This proposal is a dependency for other changes related to port ranges and RTC:

1.  [Implement multi-port Services](https://github.com/kubernetes/kubernetes/pull/6182)
2.  [Implement multi-port endpoints](https://github.com/kubernetes/kubernetes/pull/5939)
3.  [fix expose multi protocols issue](https://github.com/kubernetes/kubernetes/pull/24090)

Goals of this design:

1.  Describe the use cases for which a port range is necessary.
2.  Thoroughly describe the backward compatibility issues that arise from the introduction of
    port ranges.
3.  Describe all implementation changes necessary for the feature.

## Constraints and assumptions

1.  This feature isn't critical, there is a workaround which it's usable.
2.  API shouldn't be modified in any way to develop this feature to avoid backward
    compatibility issues.

## Use Cases

1.  As a user, I want to be able to specify port ranges instead of ports individually in a
    pod specification.
2.  As a user, I want to be able to specify port ranges instead of ports individually in a
    service specification.
    
## Proposed Design

To avoid backward compatibility issues, API will not be modified or extended. Instead of that,
annotations will be used. In this way, the final user will have the functionality desired but
core components will not modified assuring any problem shouldn't occur because of this change.
 
The result of this change shouldn't be the long term solution. Use an annotation and keep the
actual port specification as integer doesn't fix the core problem and the current limitation to
extend Kubernetes to new use cases. But it provides the functionality to the user, limit the
risk of unexpected issues and it's a good start to implement the change. A more general issue
may be open to address the limitation of the API v1 regarding the usage of ports once the 
annotation solution demonstrates the functionality is provided in a safe way.

Kubernetes allows the creation of Service and Pod Annotations. Here we propose the use of the 
following standard annotations:

* `portrange.alpha.kubernetes.io/port-end-portname` 

`portname` will be replaced with the name of the port. Internally, k8s will use the annotation 
 and the port and name values to build the original structure combining the name and the position 
 of the port in the range so it will be easy to map the port inside the range.

#### Examples

1. Pod specification example with example ports http (80/tcp), sip (5060-5064/tcp) and rtp (65000-65000/udp).  

A client creates a pod with the new annotations:
    
```yaml
 
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
      annotations:
        portrange.alpha.kubernetes.io/port-end-sip: "5064"
        portrange.alpha.kubernetes.io/port-end-rtp: "65050"
    spec:
      containers:
      - name: a
        image: a:a
        ports:
          - name: http
            containerPort: 80
            protocol: TCP
          - name: sip
            containerPort: 5060
            protocol: TCP
          - name: rtp
            containerPort: 65000
            protocol: UDP
```

The equivalent pod without the new annotations:

```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: a
        image: a:a
        ports:
          - name: http
            containerPort: 80
            protocol: TCP
          - name: sip0
            containerPort: 5060
            protocol: TCP
          - name: sip1
            containerPort: 5061
            protocol: TCP
          - name: sip2
            containerPort: 5062
            protocol: TCP
          - name: sip3
            containerPort: 5063
            protocol: TCP
          - name: sip4
            containerPort: 5064
            protocol: TCP
          - name: rtp0
            containerPort: 65000
            protocol: UDP
          - name: rtp1
            containerPort: 65001
            protocol: UDP
          - ...
          - name: rtp51
            containerPort: 65051
            protocol: UDP
```


2. Service specification example with example ports http (80/tcp), sip (5060-5064/tcp) and rtp (65000-65000/udp).  

A client creates a service with the new annotations:
    
```json
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "a-service",
        "annotations": {
          "portrange.alpha.kubernetes.io/port-end-sip": "5064",
          "portrange.alpha.kubernetes.io/port-end-rtp": "65050"
        }
    },
    "spec": {
        "selector": {
            "app": "a-core"
        },
        "ports": [
            {
                "name": "http",
                "protocol": "TCP",
                "port": 80,
                "targetPort": 80 
            },
            {
                "name": "sip",
                "protocol": "TCP",
                "port": 5060,
                "targetPort": 5060
            },
             {
                "name": "rtp",
                "protocol": "UDP",
                "port": 65000,
                "targetPort": 65000
            }
      }      
```

The equivalent service without the new annotations:

```json
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "a-service"
    },
    "spec": {
        "selector": {
            "app": "a-core"
        },
        "ports": [
            {
                "name": "http",
                "protocol": "TCP",
                "port": 80,
                "targetPort": 80 
            },
            {
                "name": "sip0",
                "protocol": "TCP",
                "port": 5060,
                "targetPort": 5060
            },
            {
                "name": "sip1",
                "protocol": "TCP",
                "port": 5061,
                "targetPort": 5061
            },
            {
                "name": "sip2",
                "protocol": "TCP",
                "port": 5062,
                "targetPort": 5062
            },
            {
                "name": "sip3",
                "protocol": "TCP",
                "port": 5063,
                "targetPort": 5063
            },
            {
                "name": "sip4",
                "protocol": "TCP",
                "port": 5064,
                "targetPort": 5064
            },
             {
                "name": "rtp0",
                "protocol": "UDP",
                "port": 65000,
                "targetPort": 65000
            },
                ....,
             {
                "name": "rtp50",
                "protocol": "UDP",
                "port": 65050,
                "targetPort": 65050
            }
      }      
```

#### Testing

The test suite will verify compatibility by converting objects into the internal API and 
examining the results.

All of the examples here will be used as test-cases.  As more test cases are added, the proposal will
be updated.

An example of a test like this can be found for the [RestComm package](https://gist.github.com/antonmry/91394adc7dec2e816c525c1f655f5018).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/port-range.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
