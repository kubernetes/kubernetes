# Kubernetes Proposal - Admission Control

**Related PR:** 

| Topic | Link |
| ----- | ---- |

## Background

High level goals:

* Enable an easy-to-use mechanism to provide admission control to cluster
* Enable a provider to support multiple admission control strategies or author their own
* Ensure any rejected request can propagate errors back to the caller with why the request failed
* Enable usage of cluster resources to satisfy admission control criteria
* Enable admission controller criteria to change without requiring restart of kube-apiserver

Policy is focused on answering if a user is authorized to perform an action.

Admission Control is focused on if the system will accept an authorized action.

The Kubernetes cluster may choose to dismiss an authorized action based on any number of admission control strategies they choose to author and deploy:

1. Quota enforcement of allocated desired usage
2. Pod black-lister to restrict running specific images on the cluster
3. Privileged container checker
4. Host port reservation
5. Volume validation - e.g. may or may not use hostDir, etc.
6. Min/max constraint checker for pod requested resources
7. ...

This proposal therefore attempts to enumerate the basic design, and describe how any number of admission controllers could be injected.

## kube-apiserver

The kube-apiserver takes the following OPTIONAL arguments to enable admission control

| Option | Behavior |
| ------ | -------- |
| admission_controllers | List of addresses (ip:port, dns name) to invoke for admission control |
| admission_controller_service | Service label selector to resolve for admission control (namespace/labelKey/labelValue) |

If the list of addresses to invoke for admission control are provided as a label selector, the kube-apiserver will update the list 
of admission control services at a regular interval. 

Upon an incoming request, the kube-apiserver performs the following basic flow:

1. Authorize the request, if authorized, continue
2. Invoke the Admission Control REST API for each defined address, if all return true, continue
3. RESTStorage processes request
4. Data is persisted in store

If there is no configured admission control address, then by default, all requests are admitted.

Admission control is enforced on POST/PUT operations, but is ignored on GET/DELETE operations.

## Admission Control REST API

An admission controller satisfies a stable REST API invoked by the kube-apiserver to satisfy requests.

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /admissionController | Send a request for admission to evaluate for admittance or denial |

The message body to the admissionController includes the following:

1. requesting user identity
2. action to perform
3. proposed resource to create/modify (if any)

If the request for admission is satisfied, return a HTTP 200.

If the request for admission is denied, return a HTTP 403, the response must include a reason for why the response failed.

## System Design

The following demonstrates potential cluster setups using an external list of admission control endpoints.

                                 Request
                                    +
                                    |
                                    |
                    +---------------|----------+
                    | API Server    |          |
                    |---------------|----------|
                    |               v          |
                    |        +--------+        |
                    |        | Policy |        |       +---------------------+---+
                    |        ++-------+        |       |Endpoints                |
    +---------+     |         |                |       |-------------------------|
    |Scheduler|<---+|         v                |       |E1. Quota Enforcer       |
    +---------+     | +----------------------+ |       |E2. Capacity Planner     |
                    | | Admission Controller +-------->|E3. White-lister         |
                    | +----+-----------------+ |       |...                      |
                    |      |                   |       +-------------------------+
                    |     +v-------------+     |
                    |     | REST Storage |     |
                    |     +--------------+     |
                    +----------------+---------+
                                     |
                                     v
                          +--------------+
                          |  Data Store  |
                          |--------------|
                          |              |
                          |              |
                          |              |
                          +--------------+

The following demonstrates potential cluster setup that uses services to fulfill admission control.

In this context, the cluster itself is used to provide HA admission control, and pods may choose to
invoke the API Server to determine if a request is or is not admissible.


                                 Request                    +--------+
                                    +                       |Pods... |
                                    |                       |--------|
                                    |                       |        <-------+
                    +---------------|----------+            |        |       |
                    | API Server    |          |            |        |       |
                    |---------------|----------|            +---+----+       |
                    |               v          |                |            |
                    |        +--------+        |<---------------+            |
                    |        | Policy |        |       +-------------------------------------+
                    |        ++-------+        |       |Service (ns=infra, labelKey=admitter)|
    +---------+     |         |                |       |-------------------------------------|
    |Scheduler|<---+|         v                |       |Service1                             |
    +---------+     | +----------------------+ |       |Service2                             |
                    | | Admission Controller +-------->|Service3                             |
                    | +----+-----------------+ |       |...                                  |
                    |      |                   |       +-------------------------------------+
                    |     +v-------------+     |
                    |     | REST Storage |     |
                    |     +--------------+     |
                    +----------------+---------+
                                     |
                                     v
                          +--------------+
                          |  Data Store  |
                          |--------------|
                          |              |
                          |              |
                          |              |
                          +--------------+