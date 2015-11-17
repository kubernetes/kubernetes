#scheduler dispatcher proposal

**Status**: Design & Implementation in progress.
> Contact @combk8s or @mqliang for questions & suggestions.

###Motivation

In current Kubernetes design, there is only one default scheduler in a Kubernetes cluster, and the default scheduler must schedule
pod sequentially, since it should take all the nodes into consideration and make an optimal decision, schedule concurrently is racy.
If a lot of users create a majority of pods in a short time, a serially running scheduler would become the bottleneck of the whole
cluster.  

A multiple schedulers framework is undoutedly much more helpful, several Multi-Scheduler and Scheduler Extension proposals have been 
came forward [#11793](https://github.com/kubernetes/kubernetes/issues/11793), [#13580](https://github.com/kubernetes/kubernetes/pull/13580). However, IMHO
the Multi-Scheduler proposal has it's own drawbacks: scheduling conflicts is inevitable and hard to solve. Personally, letting Kubelet to
solve the conflict problem is not a good solution, since it can make cluster into an uncertain state, especially when a lot of pods be scheduled concurrently, conflicts would happen very likely, and many pod would remain Pendding for a long time.

This proposal describes how kubernetes is going to support multi-scheduler by introducting a new component ***Dispatcher*** to 
dispatch pods and nodes to different schedulers, so that Kubernets can support multiple schedulers and ***avoid conflicts***, users
could even be able to run their own scheduler(s) to enable some customized scheduling behavior as they need. 


###Challenges in multiple schedulers

####Separating the pods
Each pod should be scheduled by only one scheduler. As for implementation, a pod should have an additional field to indicate by which 
scheduler it wants to be scheduled. To aviod adding new field to the API, our solution is that Dispatcher add an annotation 
to pods indicating which scheduler would scheduler it. If a user want to manually specify which scheduler to schedule a certain pod,
he can set the annotation in the pod template. 

####Dealing with conflicts: Group Nodes
Multiple schedulers is racy, which is hard to deal with in api-server and etcd. One solution is to let Kubelet to do the conflict
check and if the conflict happens, effected pods would be put back to scheduler and waiting to be scheduled again 
[#11797](https://github.com/kubernetes/kubernetes/pull/17197). 
Our solution is grouping nodes: component Dispathcer is responsible for node grouping by adding an annotation to nodes. 
Then, every scheduler will just take a sunset of all nodes(the node has a matching annotation) into consideration when schedule a pod, 
and make a suboptimal scheduling decision. Implementation details are in the later sections.

#### Design Overview
The architecture of our architecture proposal is like this:

![Dispatcher Diagram](scheduler-dispatcher-design.png?raw=true "Dispatcher overview")

We introduce a component ***Dispatcher***  to dispatch pods to different schedulers and a controller ***Group controller*** to group 
nodes. ***Group controller*** will try its best to arrange nodes to different groups and keep all groups balance. Since all groups 
are kept balanced by ***Group controller***, ***Dispatcher*** can be implemented as simple/fast as possible, to avoid
the ***Dispatcher*** become the bottleneck of whole cluster. 


#### Implementation details
#####1. First, ervery scheduler should have its own name or id.

#####2. Separate pods:

***Dispatcher*** will listwatch all unscheduled(`pod.nodeName == ""`) and undispatched(without a `k8s.io/scheduler-XXX: true`
annotation) pods, and dispatch those pods to different schedulers, for example add an annotation`k8s.io/scheduler-a: true`, 
to indicate that `scheduler-a` will schedule this pod. Dispatcher can dispatch pods to schedulers randomly or base on some 
dispatching algorithm, such as Round Robin Dispatching. If a user want to manually specify which scheduler to schedule a certain pod,
he can set the annotation in the pod template.

It is worth noting that those dispatching algorithm should be as simple as possible so that Dispatcher wouldn't becomes 
the bottleneck of whole cluster. Just like load balancer dispatching network traffic to server, Dispatcher in Kubernetes is
responsible for dispatching pods to schedulers. 

#####3. Separate nodes:
In order to make the ***Dispatche*** stateless, ***Group controller*** is introduced to arrange nodes to different groups and keep
all groups balance.

***Group controller*** will listwatch all schedulable(`node.Unscheduleable == false`) and not grouped nodes, and arrange those
nodes to different groups by adding an annotation to those nodes, for example add an annotation `k8s.io/scheduler-a: true`, 
to indicate that when scheduler named `scheduler-a` schedule a pod, it should take the nodes which have the annotation 
`k8s.io/scheduler-a: true` into consideration. ***Group controller*** can group nodes randomly or base on some grouping algorithm.

***Group controller*** should try its best to keep all the groups balanced, so it need statistical data of each group, and rearrange
nodes when groups become unbalanced. Usually, nodes need to be rearrange in following case:

* A large number of nodes in one group becomes unavailable or their resources have been exhausted
* Some new nodes join in cluster

Usually rearrange is not frequent.


#####4. Scheduler:

A scheduler can add a pod to its scheduling queue if and only if: 

* The pod has not been binded(`pod.nodeName==""`), **AND** 
* The scheduler-name specified in the pod's annotation `k8s.io/scheduler-name: true` matches the `scheduler-name` of the scheduler.
    
When a scheduler schedule a pod to a node, it will just take those nodes into consideration: 

* node is schedulerable(`node.Unschedulerable == false`), **AND**
* node has an annotation `k8s.io/scheduler-name:true` matches the `scheduler-name` of the scheduler


#####5. Binding/Schedule Error handling：
When configure Dispatcher to group nodes, it is possible when a pod dispatched to a scheduler `scheduler-a`, but all the nodes 
which have the annotation `scheduler-a:true` is not fit due to port conflict or some other reasons.
In such a case, `scheduler-a` will replace the `scheduler-a:true` annotation to `scheduler-a:failed` and update the pod, 
when ***Dispatcher*** dispatch pods to schedulers, it will ignore the failed one. If all the annotations becomes `failed`, 
it suggests that all the schedulers can not schedule this pod to a node, in such a case, we can just keep retrying or send an event.


#####6. Scheduler health check：
Just like load balancer would check servers' health status. ***Dispatcher*** or ***Group controller*** should check all schedulers' 
health status periodly, if a scheduler becomes unavailable, ***Group controller*** should rearrange the nodes to other group
and ***Dispatcher*** should redispatch the pods which have not been binded and arranged to the unavailabled scheduler to other 
schedulers.


#####7. Annotation contradiction
When pods' annotation contradict, such as a pod has a annotation `k8s.io/scheduler-a:true` as well as `k8s.io/scheduler-b:true`,
apiserver will reject the create/update request at validation time.



