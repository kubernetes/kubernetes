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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/application-troubleshooting.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Application Troubleshooting

This guide is to help users debug applications that are deployed into Kubernetes and not behaving correctly.
This is *not* a guide for people who want to debug their cluster.  For that you should check out
[this guide](../admin/cluster-troubleshooting.md)

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Application Troubleshooting](#application-troubleshooting)
  - [FAQ](#faq)
  - [Diagnosing the problem](#diagnosing-the-problem)
    - [Debugging Pods](#debugging-pods)
      - [My pod stays pending](#my-pod-stays-pending)
      - [My pod stays waiting](#my-pod-stays-waiting)
      - [My pod is crashing or otherwise unhealthy](#my-pod-is-crashing-or-otherwise-unhealthy)
    - [Debugging Replication Controllers](#debugging-replication-controllers)
    - [Debugging Services](#debugging-services)
      - [My service is missing endpoints](#my-service-is-missing-endpoints)
      - [Network traffic is not forwarded](#network-traffic-is-not-forwarded)
      - [More information](#more-information)

<!-- END MUNGE: GENERATED_TOC -->

## FAQ

Users are highly encouraged to check out our [FAQ](https://github.com/GoogleCloudPlatform/kubernetes/wiki/User-FAQ)

## Diagnosing the problem

The first step in troubleshooting is triage.  What is the problem?  Is it your Pods, your Replication Controller or
your Service?
   * [Debugging Pods](#debugging-pods)
   * [Debugging Replication Controllers](#debugging-replication-controllers)
   * [Debugging Services](#debugging-services)

### Debugging Pods

The first step in debugging a Pod is taking a look at it.  Check the current state of the Pod and recent events with the following command:

```console
$ kubectl describe pods ${POD_NAME}
```

Look at the state of the containers in the pod.  Are they all `Running`?  Have there been recent restarts?

Continue debugging depending on the state of the pods.

#### My pod stays pending

If a Pod is stuck in `Pending` it means that it can not be scheduled onto a node.  Generally this is because
there are insufficient resources of one type or another that prevent scheduling.  Look at the output of the
`kubectl describe ...` command above.  There should be messages from the scheduler about why it can not schedule
your pod.  Reasons include:

* **You don't have enough resources**:  You may have exhausted the supply of CPU or Memory in your cluster, in this case
you need to delete Pods, adjust resource requests, or add new nodes to your cluster. See [Compute Resources document](compute-resources.md#my-pods-are-pending-with-event-message-failedscheduling) for more information.

* **You are using `hostPort`**:  When you bind a Pod to a `hostPort` there are a limited number of places that pod can be
scheduled.  In most cases, `hostPort` is unnecessary, try using a Service object to expose your Pod.  If you do require
`hostPort` then you can only schedule as many Pods as there are nodes in your Kubernetes cluster.


#### My pod stays waiting

If a Pod is stuck in the `Waiting` state, then it has been scheduled to a worker node, but it can't run on that machine.
Again, the information from `kubectl describe ...` should be informative.  The most common cause of `Waiting` pods is a failure to pull the image.  There are three things to check:
* Make sure that you have the name of the image correct
* Have you pushed the image to the repository?
* Run a manual `docker pull <image>` on your machine to see if the image can be pulled.

#### My pod is crashing or otherwise unhealthy

First, take a look at the logs of
the current container:

```console
$ kubectl logs ${POD_NAME} ${CONTAINER_NAME}
```

If your container has previously crashed, you can access the previous container's crash log with:

```console
$ kubectl logs --previous ${POD_NAME} ${CONTAINER_NAME}
```

Alternately, you can run commands inside that container with `exec`:

```console
$ kubectl exec ${POD_NAME} -c ${CONTAINER_NAME} -- ${CMD} ${ARG1} ${ARG2} ... ${ARGN}
```

Note that `-c ${CONTAINER_NAME}` is optional and can be omitted for Pods that only contain a single container.

As an example, to look at the logs from a running Cassandra pod, you might run

```console
$ kubectl exec cassandra -- cat /var/log/cassandra/system.log
```


If none of these approaches work, you can find the host machine that the pod is running on and SSH into that host,
but this should generally not be necessary given tools in the Kubernetes API. Therefore, if you find yourself needing to ssh into a machine, please file a
feature request on GitHub describing your use case and why these tools are insufficient.

### Debugging Replication Controllers

Replication controllers are fairly straightforward.  They can either create Pods or they can't.  If they can't
create pods, then please refer to the [instructions above](#debugging-pods) to debug your pods.

You can also use `kubectl describe rc ${CONTROLLER_NAME}` to introspect events related to the replication
controller.

### Debugging Services

Services provide load balancing across a set of pods.  There are several common problems that can make Services
not work properly.  The following instructions should help debug Service problems.

First, verify that there are endpoints for the service. For every Service object, the apiserver makes an `endpoints` resource available.

You can view this resource with:

```console
$ kubectl get endpoints ${SERVICE_NAME}
```

Make sure that the endpoints match up with the number of containers that you expect to be a member of your service.
For example, if your Service is for an nginx container with 3 replicas, you would expect to see three different
IP addresses in the Service's endpoints.

#### My service is missing endpoints

If you are missing endpoints, try listing pods using the labels that Service uses.  Imagine that you have
a Service where the labels are:

```yaml
...
spec:
  - selector:
     name: nginx
     type: frontend
```

You can use:

```console
$ kubectl get pods --selector=name=nginx,type=frontend
```

to list pods that match this selector.  Verify that the list matches the Pods that you expect to provide your Service.

If the list of pods matches expectations, but your endpoints are still empty, it's possible that you don't
have the right ports exposed.  If your service has a `containerPort` specified, but the Pods that are
selected don't have that port listed, then they won't be added to the endpoints list.

Verify that the pod's `containerPort` matches up with the Service's `containerPort`

#### Network traffic is not forwarded

If you can connect to the service, but the connection is immediately dropped, and there are endpoints
in the endpoints list, it's likely that the proxy can't contact your pods.

There are three things to
check:
   * Are your pods working correctly?  Look for restart count, and [debug pods](#debugging-pods)
   * Can you connect to your pods directly?  Get the IP address for the Pod, and try to connect directly to that IP
   * Is your application serving on the port that you configured?  Kubernetes doesn't do port remapping, so if your application serves on 8080, the `containerPort` field needs to be 8080.

#### More information

If none of the above solves your problem, follow the instructions in [Debugging Service document](debugging-services.md) to make sure that your `Service` is running, has `Endpoints`, and your `Pods` are actually serving; you have DNS working, iptables rules installed, and kube-proxy does not seem to be misbehaving.

You may also visit [troubleshooting document](../troubleshooting.md) for more information.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/application-troubleshooting.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
