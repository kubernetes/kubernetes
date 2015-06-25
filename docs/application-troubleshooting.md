# Application Troubleshooting.

This guide is to help users debug applications that are deployed into Kubernetes and not behaving correctly.
This is *not* a guide for people who want to debug their cluster.  For that you should check out
[this guide](cluster-troubleshooting.md)

## FAQ
Users are highly encouraged to check out our [FAQ](https://github.com/GoogleCloudPlatform/kubernetes/wiki/User-FAQ)

## Diagnosing the problem
The first step in troubleshooting is triage.  What is the problem?  Is it your Pods, your Replication Controller or
your Service?
   * [Debugging Pods](#debugging-pods)
   * [Debugging Replication Controllers](#debugging-replication-controllers)
   * [Debugging Services](#debugging-services)

### Debugging Pods
The first step in debugging a Pod is taking a look at it.  For the purposes of example, imagine we have a pod
```my-pod``` which holds two containers ```container-1``` and ```container-2```

First, describe the pod.  This will show the current state of the Pod and recent events.

```sh
export POD_NAME=my-pod
kubectl describe pods ${POD_NAME}
```

Look at the state of the containers in the pod.  Are they all ```Running```?  Have there been recent restarts?

Depending on the state of the pod, you may want to:
   * [Debug a pending pod](#debugging-pending-pods)
   * [Debug a waiting pod](#debugging-waiting-pods)
   * [Debug a crashing pod](#debugging-crashing-pods-or-otherwise-unhealthy-pods)

#### Debuging Pending Pods
If a Pod is stuck in ```Pending``` it means that it can not be scheduled onto a node.  Generally this is because
there are insufficient resources of one type or another that prevent scheduling.  Look at the output of the
```kubectl describe ...``` command above.  There should be messages from the scheduler about why it can not schedule
your pod.  Reasons include:

You don't have enough resources.  You may have exhausted the supply of CPU or Memory in your cluster, in this case
you need to delete Pods, adjust resource requests, or add new nodes to your cluster.

You are using ```hostPort```.  When you bind a Pod to a ```hostPort``` there are a limited number of places that pod can be
scheduled.  In most cases, ```hostPort``` is unnecesary, try using a Service object to expose your Pod.  If you do require
```hostPort``` then you can only schedule as many Pods as there are nodes in your Kubernetes cluster.


#### Debugging Waiting Pods
If a Pod is stuck in the ```Waiting``` state, then it has been scheduled to a worker node, but it can't run on that machine.
Again, the information from ```kubectl describe ...``` should be informative.  The most common cause of ```Waiting``` pods
is a failure to pull the image.  Make sure that you have the name of the image correct.  Have you pushed it to the repository?
Does it work if you run a manual ```docker pull <image>``` on your machine?

#### Debugging Crashing or otherwise unhealthy pods

Let's suppose that ```container-2``` has been crash looping and you don't know why, you can take a look at the logs of
the current container:

```sh
kubectl logs ${POD_NAME} ${CONTAINER_NAME}
```

If your container has previously crashed, you can access the previous container's crash log with:
```sh
kubectl logs --previous ${POD_NAME} ${CONTAINER_NAME}
```

Alternately, you can run commands inside that container with ```exec```:

```sh
kubectl exec ${POD_NAME} -c ${CONTAINER_NAME} -- ${CMD} ${ARG1} ${ARG2} ... ${ARGN}
```

Note that ```-c ${CONTAINER_NAME}``` is optional and can be omitted for Pods that only contain a single container.

As an example, to look at the logs from a running Cassandra pod, you might run
```sh
kubectl exec cassandra -- cat /var/log/cassandra/system.log
```


If none of these approaches work, you can find the host machine that the pod is running on and SSH into that host,
but this should generally not be necessary given tools in the Kubernetes API. Indeed if you find yourself needing to ssh into a machine, please file a
feature request on GitHub describing your use case and why these tools are insufficient.

### Debugging Replication Controllers
Replication controllers are fairly straightforward.  They can either create Pods or they can't.  If they can't
create pods, then please refer to the [instructions above](#debugging-pods)

You can also use ```kubectl describe rc ${CONTROLLER_NAME}``` to introspect events related to the replication
controller.

### Debugging Services
Services provide load balancing across a set of pods.  There are several common problems that can make Services
not work properly.  The following instructions should help debug Service problems.

#### Verify that there are endpoints for the service
For every Service object, the apiserver makes an ```endpoints`` resource available.

You can view this resource with:

```
kubectl get endpoints ${SERVICE_NAME}
```

Make sure that the endpoints match up with the number of containers that you expect to be a member of your service.
For example, if your Service is for an nginx container with 3 replicas, you would expect to see three different
IP addresses in the Service's endpoints.

#### Missing endpoints
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
```
kubectl get pods --selector=name=nginx,type=frontend
```

to list pods that match this selector.  Verify that the list matches the Pods that you expect to provide your Service.

If the list of pods matches expectations, but your endpoints are still empty, it's possible that you don't
have the right ports exposed.  If your service has a ```containerPort``` specified, but the Pods that are
selected don't have that port listed, then they won't be added to the endpoints list.

Verify that the pod's ```containerPort``` matches up with the Service's ```containerPort```

#### Network traffic isn't forwarded
If you can connect to the service, but the connection is immediately dropped, and there are endpoints
in the endpoints list, it's likely that the proxy can't contact your pods.

There are three things to
check:
   * Are your pods working correctly?  Look for restart count, and [debug pods](#debugging-pods)
   * Can you connect to your pods directly?  Get the IP address for the Pod, and try to connect directly to that IP
   * Is your application serving on the port that you configured?  Kubernetes doesn't do port remapping, so if your application serves on 8080, the ```containerPort``` field needs to be 8080.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/application-troubleshooting.md?pixel)]()
