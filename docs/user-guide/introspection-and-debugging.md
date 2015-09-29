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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/introspection-and-debugging.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Application Introspection and Debugging

Once your application is running, you’ll inevitably need to debug problems with it.
Earlier we described how you can use `kubectl get pods` to retrieve simple status information about
your pods. But there are a number of ways to get even more information about your application.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications: Application Introspection and Debugging](#kubernetes-user-guide-managing-applications-application-introspection-and-debugging)
  - [Using `kubectl describe pod` to fetch details about pods](#using-kubectl-describe-pod-to-fetch-details-about-pods)
  - [Example: debugging Pending Pods](#example-debugging-pending-pods)
  - [Example: debugging a down/unreachable node](#example-debugging-a-downunreachable-node)
  - [What's next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

## Using `kubectl describe pod` to fetch details about pods

For this example we’ll use a ReplicationController to create two pods, similar to the earlier example.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
        ports:
        - containerPort: 80
```

```console
$ kubectl create -f ./my-nginx-rc.yaml
replicationcontrollers/my-nginx
```

```console
$ kubectl get pods
NAME             READY     REASON    RESTARTS   AGE
my-nginx-gy1ij   1/1       Running   0          1m
my-nginx-yv5cn   1/1       Running   0          1m
```

We can retrieve a lot more information about each of these pods using `kubectl describe pod`. For example:

```console
$ kubectl describe pod my-nginx-gy1ij
Name:				my-nginx-gy1ij
Image(s):			nginx
Node:				kubernetes-minion-y3vk/10.240.154.168
Labels:				app=nginx
Status:				Running
Reason:				
Message:			
IP:				10.244.1.4
Replication Controllers:	my-nginx (2/2 replicas created)
Containers:
  nginx:
    Image:	nginx
    Limits:
      cpu:		500m
      memory:		128Mi
    State:		Running
      Started:		Thu, 09 Jul 2015 15:33:07 -0700
    Ready:		True
    Restart Count:	0
Conditions:
  Type		Status
  Ready 	True 
Events:
  FirstSeen				LastSeen			Count	From					SubobjectPath				Reason		Message
  Thu, 09 Jul 2015 15:32:58 -0700	Thu, 09 Jul 2015 15:32:58 -0700	1	{scheduler }									scheduled	Successfully assigned my-nginx-gy1ij to kubernetes-minion-y3vk
  Thu, 09 Jul 2015 15:32:58 -0700	Thu, 09 Jul 2015 15:32:58 -0700	1	{kubelet kubernetes-minion-y3vk}	implicitly required container POD	pulled		Pod container image "gcr.io/google_containers/pause:0.8.0" already present on machine
  Thu, 09 Jul 2015 15:32:58 -0700	Thu, 09 Jul 2015 15:32:58 -0700	1	{kubelet kubernetes-minion-y3vk}	implicitly required container POD	created		Created with docker id cd1644065066
  Thu, 09 Jul 2015 15:32:58 -0700	Thu, 09 Jul 2015 15:32:58 -0700	1	{kubelet kubernetes-minion-y3vk}	implicitly required container POD	started		Started with docker id cd1644065066
  Thu, 09 Jul 2015 15:33:06 -0700	Thu, 09 Jul 2015 15:33:06 -0700	1	{kubelet kubernetes-minion-y3vk}	spec.containers{nginx}			pulled		Successfully pulled image "nginx"
  Thu, 09 Jul 2015 15:33:06 -0700	Thu, 09 Jul 2015 15:33:06 -0700	1	{kubelet kubernetes-minion-y3vk}	spec.containers{nginx}			created		Created with docker id 56d7a7b14dac
  Thu, 09 Jul 2015 15:33:07 -0700	Thu, 09 Jul 2015 15:33:07 -0700	1	{kubelet kubernetes-minion-y3vk}	spec.containers{nginx}			started		Started with docker id 56d7a7b14dac
```

Here you can see configuration information about the container(s) and Pod (labels, resource requirements, etc.), as well as status information about the container(s) and Pod (state, readiness, restart count, events, etc.)

The container state is one of Waiting, Running, or Terminated. Depending on the state, additional information will be provided -- here you can see that for a container in Running state, the system tells you when the container started.

Ready tells you whether the container passed its last readiness probe. (In this case, the container does not have a readiness probe configured; the container is assumed to be ready if no readiness probe is configured.)

Restart Count tells you how many times the container has restarted; this information can be useful for detecting crash loops in containers that are configured with a restart policy of “always.”

Currently the only Condition associated with a Pod is the binary Ready condition, which indicates that the pod is able to service requests and should be added to the load balancing pools of all matching services.

Lastly, you see a log of recent events related to your Pod. The system compresses multiple identical events by indicating the first and last time it was seen and the number of times it was seen. "From" indicates the component that is logging the event, "SubobjectPath" tells you which object (e.g. container within the pod) is being referred to, and "Reason" and "Message" tell you what happened.

## Example: debugging Pending Pods

A common scenario that you can detect using events is when you’ve created a Pod that won’t fit on any node. For example, the Pod might request more resources than are free on any node, or it might specify a label selector that doesn’t match any nodes. Let’s say we created the previous Replication Controller with 5 replicas (instead of 2) and requesting 600 millicores instead of 500, on a four-node cluster where each (virtual) machine has 1 CPU. In that case one of the Pods will not be able to schedule. (Note that because of the cluster addon pods such as fluentd, skydns, etc., that run on each node, if we requested 1000 millicores then none of the Pods would be able to schedule.)

```console
$ kubectl get pods
NAME             READY     REASON    RESTARTS   AGE
my-nginx-9unp9   0/1       Pending   0          8s
my-nginx-b7zs9   0/1       Running   0          8s
my-nginx-i595c   0/1       Running   0          8s
my-nginx-iichp   0/1       Running   0          8s
my-nginx-tc2j9   0/1       Running   0          8s
```

To find out why the my-nginx-9unp9 pod is not running, we can use `kubectl describe pod` on the pending Pod and look at its events:

```console
$ kubectl describe pod my-nginx-9unp9 
Name:				my-nginx-9unp9
Image(s):			nginx
Node:				/
Labels:				app=nginx
Status:				Pending
Reason:				
Message:			
IP:				
Replication Controllers:	my-nginx (5/5 replicas created)
Containers:
  nginx:
    Image:	nginx
    Limits:
      cpu:		600m
      memory:		128Mi
    State:		Waiting
    Ready:		False
    Restart Count:	0
Events:
  FirstSeen				LastSeen			Count	From		SubobjectPath	Reason			Message
  Thu, 09 Jul 2015 23:56:21 -0700	Fri, 10 Jul 2015 00:01:30 -0700	21	{scheduler }			failedScheduling	Failed for reason PodFitsResources and possibly others
```

Here you can see the event generated by the scheduler saying that the Pod failed to schedule for reason `PodFitsResources` (and possibly others). `PodFitsResources` means there were not enough resources for the Pod on any of the nodes. Due to the way the event is generated, there may be other reasons as well, hence "and possibly others."

To correct this situation, you can use `kubectl scale` to update your Replication Controller to specify four or fewer replicas. (Or you could just leave the one Pod pending, which is harmless.)

Events such as the ones you saw at the end of `kubectl describe pod` are persisted in etcd and provide high-level information on what is happening in the cluster. To list all events you can use

```
kubectl get events
```

but you have to remember that events are namespaced. This means that if you're interested in events for some namespaced object (e.g. what happened with Pods in namespace `my-namespace`) you need to explicitly provide a namespace to the command:

```
kubectl get events --namespace=my-namespace
```

To see events from all namespaces, you can use the `--all-namespaces` argument.

In addition to `kubectl describe pod`, another way to get extra information about a pod (beyond what is provided by `kubectl get pod`) is to pass the `-o yaml` output format flag to `kubectl get pod`. This will give you, in YAML format, even more information than `kubectl describe pod`--essentially all of the information the system has about the Pod. Here you will see things like annotations (which are key-value metadata without the label restrictions, that is used internally by Kubernetes system components), restart policy, ports, and volumes.

```yaml
$ kubectl get pod my-nginx-i595c -o yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    kubernetes.io/created-by: '{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"ReplicationController","namespace":"default","name":"my-nginx","uid":"c555c14f-26d0-11e5-99cb-42010af00e4b","apiVersion":"v1","resourceVersion":"26174"}}'
  creationTimestamp: 2015-07-10T06:56:21Z
  generateName: my-nginx-
  labels:
    app: nginx
  name: my-nginx-i595c
  namespace: default
  resourceVersion: "26243"
  selfLink: /api/v1/namespaces/default/pods/my-nginx-i595c
  uid: c558e44b-26d0-11e5-99cb-42010af00e4b
spec:
  containers:
  - image: nginx
    imagePullPolicy: IfNotPresent
    name: nginx
    ports:
    - containerPort: 80
      protocol: TCP
    resources:
      limits:
        cpu: 600m
        memory: 128Mi
    terminationMessagePath: /dev/termination-log
    volumeMounts:
    - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      name: default-token-zkhkk
      readOnly: true
  dnsPolicy: ClusterFirst
  nodeName: kubernetes-minion-u619
  restartPolicy: Always
  serviceAccountName: default
  volumes:
  - name: default-token-zkhkk
    secret:
      secretName: default-token-zkhkk
status:
  conditions:
  - status: "True"
    type: Ready
  containerStatuses:
  - containerID: docker://9506ace0eb91fbc31aef1d249e0d1d6d6ef5ebafc60424319aad5b12e3a4e6a9
    image: nginx
    imageID: docker://319d2015d149943ff4d2a20ddea7d7e5ce06a64bbab1792334c0d3273bbbff1e
    lastState: {}
    name: nginx
    ready: true
    restartCount: 0
    state:
      running:
        startedAt: 2015-07-10T06:56:28Z
  hostIP: 10.240.112.234
  phase: Running
  podIP: 10.244.3.4
  startTime: 2015-07-10T06:56:21Z
```

## Example: debugging a down/unreachable node

Sometimes when debugging it can be useful to look at the status of a node -- for example, because you've noticed strange behavior of a Pod that’s running on the node, or to find out why a Pod won’t schedule onto the node. As with Pods, you can use `kubectl describe node` and `kubectl get node -o yaml` to retrieve detailed information about nodes. For example, here's what you'll see if a node is down (disconnected from the network, or kubelet dies and won't restart, etc.). Notice the events that show the node is NotReady, and also notice that the pods are no longer running (they are evicted after five minutes of NotReady status).

```console
$ kubectl get nodes
NAME                     LABELS                                          STATUS
kubernetes-minion-861h   kubernetes.io/hostname=kubernetes-minion-861h   NotReady
kubernetes-minion-bols   kubernetes.io/hostname=kubernetes-minion-bols   Ready
kubernetes-minion-st6x   kubernetes.io/hostname=kubernetes-minion-st6x   Ready
kubernetes-minion-unaj   kubernetes.io/hostname=kubernetes-minion-unaj   Ready

$ kubectl describe node kubernetes-minion-861h
Name:			kubernetes-minion-861h
Labels:			kubernetes.io/hostname=kubernetes-minion-861h
CreationTimestamp:	Fri, 10 Jul 2015 14:32:29 -0700
Conditions:
  Type		Status		LastHeartbeatTime			LastTransitionTime			Reason					Message
  Ready 	Unknown 	Fri, 10 Jul 2015 14:34:32 -0700 	Fri, 10 Jul 2015 14:35:15 -0700 	Kubelet stopped posting node status. 	
Addresses:	10.240.115.55,104.197.0.26
Capacity:
 cpu:		1
 memory:	3800808Ki
 pods:		100
Version:
 Kernel Version:		3.16.0-0.bpo.4-amd64
 OS Image:			Debian GNU/Linux 7 (wheezy)
 Container Runtime Version:	docker://Unknown
 Kubelet Version:		v0.21.1-185-gffc5a86098dc01
 Kube-Proxy Version:		v0.21.1-185-gffc5a86098dc01
PodCIDR:			10.244.0.0/24
ExternalID:			15233045891481496305
Pods:				(0 in total)
  Namespace			Name
Events:
  FirstSeen				LastSeen			Count	From					SubobjectPath	Reason		Message
  Fri, 10 Jul 2015 14:32:28 -0700	Fri, 10 Jul 2015 14:32:28 -0700	1	{kubelet kubernetes-minion-861h}			NodeNotReady	Node kubernetes-minion-861h status is now: NodeNotReady
  Fri, 10 Jul 2015 14:32:30 -0700	Fri, 10 Jul 2015 14:32:30 -0700	1	{kubelet kubernetes-minion-861h}			NodeNotReady	Node kubernetes-minion-861h status is now: NodeNotReady
  Fri, 10 Jul 2015 14:33:00 -0700	Fri, 10 Jul 2015 14:33:00 -0700	1	{kubelet kubernetes-minion-861h}			starting	Starting kubelet.
  Fri, 10 Jul 2015 14:33:02 -0700	Fri, 10 Jul 2015 14:33:02 -0700	1	{kubelet kubernetes-minion-861h}			NodeReady	Node kubernetes-minion-861h status is now: NodeReady
  Fri, 10 Jul 2015 14:35:15 -0700	Fri, 10 Jul 2015 14:35:15 -0700	1	{controllermanager }					NodeNotReady	Node kubernetes-minion-861h status is now: NodeNotReady


$ kubectl get node kubernetes-minion-861h -o yaml
apiVersion: v1
kind: Node
metadata:
  creationTimestamp: 2015-07-10T21:32:29Z
  labels:
    kubernetes.io/hostname: kubernetes-minion-861h
  name: kubernetes-minion-861h
  resourceVersion: "757"
  selfLink: /api/v1/nodes/kubernetes-minion-861h
  uid: 2a69374e-274b-11e5-a234-42010af0d969
spec:
  externalID: "15233045891481496305"
  podCIDR: 10.244.0.0/24
  providerID: gce://striped-torus-760/us-central1-b/kubernetes-minion-861h
status:
  addresses:
  - address: 10.240.115.55
    type: InternalIP
  - address: 104.197.0.26
    type: ExternalIP
  capacity:
    cpu: "1"
    memory: 3800808Ki
    pods: "100"
  conditions:
  - lastHeartbeatTime: 2015-07-10T21:34:32Z
    lastTransitionTime: 2015-07-10T21:35:15Z
    reason: Kubelet stopped posting node status.
    status: Unknown
    type: Ready
  nodeInfo:
    bootID: 4e316776-b40d-4f78-a4ea-ab0d73390897
    containerRuntimeVersion: docker://Unknown
    kernelVersion: 3.16.0-0.bpo.4-amd64
    kubeProxyVersion: v0.21.1-185-gffc5a86098dc01
    kubeletVersion: v0.21.1-185-gffc5a86098dc01
    machineID: ""
    osImage: Debian GNU/Linux 7 (wheezy)
    systemUUID: ABE5F6B4-D44B-108B-C46A-24CCE16C8B6E
```

## What's next?

Learn about additional debugging tools, including:
* [Logging](logging.md)
* [Monitoring](monitoring.md)
* [Getting into containers via `exec`](getting-into-containers.md)
* [Connecting to containers via proxies](connecting-to-applications-proxy.md)
* [Connecting to containers via port forwarding](connecting-to-applications-port-forward.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/introspection-and-debugging.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
