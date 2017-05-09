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

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [API server HA proposal](#api-server-ha-proposal)
    - [Introduction](#introduction)
    - [Motivation](#motivation)
    - [Design](#design)
    - [Initional Implementation step](#initional-implementation-step)
      - [1. Introduce a knob to enable quorum read](#1-introduce-a-knob-to-enable-quorum-read)
      - [2. Default kubernetes service/endpoints](#2-default-kubernetes-serviceendpoints)
      - [3. API server component Health check](#3-api-server-component-health-check)
      - [4. Client behavior](#4-client-behavior)

<!-- END MUNGE: GENERATED_TOC -->

# API server HA proposal

**Status**: Design & Implementation in progress.

> Contact @mqliang or @dalanlan for questions & suggestions.

### Introduction

This document serves as a proposal for high availability of the API server
component in Kubernetes. This proposal is intended to provide a simple high
availability solution for API server component.

### Motivation

Currently, high availability solution for scheduler and controller components
in Kubernetes has been proposed, and are under heavy development. The HA
solution for scheduler and controller components are Warm Standby, which
mean there is only one active component acting as the master and additional
components running but not providing service or responding to requests.

API server should also support HA so that k8s will be a truly reliable, highly
available distributed system. As a matter fact, API server is already highly
available in some sense since it's stateless, and we could build up a k8s cluster
with replicated API servers easily(see
[high-availability](../../docs/admin/high-availability.md) ).
However, there are still some problems
need to address. This proposal aims at address those problems.

### Design

```
  |--------|            |--------|            |--------|              
  | etcd-1 |            | etcd-2 |            | etcd-3 |        
  |--------|            |--------|            |--------|
      |                     |                     |
      |                     |                     |
|-------------|      |-------------|        |-------------|             
| apiserver-1 |      | apiserver-2 |        | apiserver-3 |         
|-------------|      |-------------|        |-------------|   
      |                     |                     |
      |                     |                     |
|---------------------------------------------------------|             
|  Nginx/HAProxy, or client randomly select an endpoint   |         
|---------------------------------------------------------|
       |                     |                     |
       |                     |                     |
 |------------|       |------------|        |-------------|              
 |  scheduler |       | controller |        |    node     |      
 |------------|       |------------|        |-------------|          
```

### Initional Implementation step

#### 1. Introduce a knob to enable quorum read

There are two cases where we may read old data if we have multi API server:

1) When a etcd member is isolated from etcd cluster, we can still read from
it(but all the write request sent to the isolated member will failed). If
unfortunately nginx forward some read request to the isolated member,
those read request will get old data.

2) Consider the following case:

A write request(e.g. ?key=hello&value=world) was forward to apiserver-1
and then forward to etcd-1, etcd-1 will return 200 indicating the write
request was succeeded if the data has been written to a majority of members.
(for example, written to etcd-1 and etcd-2) Almost immediately, a read
request(?key=hello) was forward to etcd-3, if API server doesn't enable quorum
read, this read request will failed(key not found) since the data has not been
synced to etcd-3.

So, we must introduce a knob to enable quorum read if we have multi API server.
Since quorum read will make read request less effective, user could disenable it
if they just want a single API server.


#### 2. Default kubernetes service/endpoints

When API server start, it will create a kubernetes service, and will periodically
reconcile it. Currently, API server assume there is only one endpoint(benckend)
for the service, every apiserver will constantly overwrites the etcd key
“/registry/services/endpoints/default/kubernetes". We'd fix this by writing
multiple endpoints in there.

#### 3. API server component Health check

Every ApiServer will fetch the default kubernetes service/endpoints periodically
and check each other's health, if health check failed, remove the failed endpoint
from the EndpointSubset.

#### 4. Client behavior

1) Ramdomly select a API server endpoint

By defaut, client of scheduler/controller/kubelet would accept a API server
addresses list, and try to connect one, if connect succeed, fetch the default
kubernetes service/endpoints. When a request failed, try to connect another
endpoint. Request will failed iff all the API server endpoints failed.

2) Pin to a single API server address

Introduce a knob to allow user pin to a single API server address, in suce a case,
user could pass their niginx/haproxy address and let nginx/haproxy forward all the
request.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/apiserver-ha.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
