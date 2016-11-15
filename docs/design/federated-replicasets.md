# Federated ReplicaSets

# Requirements & Design Document

This document is a markdown version converted from a working [Google Doc](https://docs.google.com/a/google.com/document/d/1C1HEHQ1fwWtEhyl9JYu6wOiIUJffSmFmZgkGta4720I/edit?usp=sharing). Please refer to the original for extended commentary and discussion.

Author: Marcin Wielgus [mwielgus@google.com](mailto:mwielgus@google.com)
Based on discussions with
Quinton Hoole [quinton@google.com](mailto:quinton@google.com), Wojtek Tyczyński [wojtekt@google.com](mailto:wojtekt@google.com)

## Overview

### Summary & Vision

When running a global application on a federation of Kubernetes
clusters the owner currently has to start it in multiple clusters and
control whether he has both enough application replicas running
locally in each of the clusters (so that, for example, users are
handled by a nearby cluster, with low latency) and globally (so that
there is always enough capacity to handle all traffic). If one of the
clusters has issues or hasn’t enough capacity to run the given set of
replicas the replicas should be automatically moved to some other
cluster to keep the application responsive.

In single cluster Kubernetes there is a concept of ReplicaSet that
manages the replicas locally. We want to expand this concept to the
federation level.

### Goals

+ Win large enterprise customers who want to easily run applications
   across multiple clusters
+ Create a reference controller implementation to facilitate bringing
   other Kubernetes concepts to Federated Kubernetes.

## Glossary

Federation Cluster - a cluster that is a member of federation.

Local ReplicaSet (LRS) - ReplicaSet defined and running on a cluster
that is a member of federation.

Federated ReplicaSet (FRS) - ReplicaSet defined and running inside of Federated K8S server.

Federated ReplicaSet Controller (FRSC) - A controller running inside
of Federated K8S server that controlls FRS.

## User Experience

### Critical User Journeys

+ [CUJ1] User wants to create a ReplicaSet in each of the federation
   cluster. They create a definition of federated ReplicaSet on the
   federated master and (local) ReplicaSets are automatically created
   in each of the federation clusters. The number of replicas is each
   of the Local ReplicaSets is (perheps indirectly) configurable by
   the user.
+ [CUJ2] When the current number of replicas in a cluster drops below
   the desired number and new replicas cannot be scheduled then they
   should be started in some other cluster.

### Features Enabling Critical User Journeys

Feature #1 -> CUJ1:
A component which looks for newly created Federated ReplicaSets and
creates the appropriate Local ReplicaSet definitions in the federated
clusters.

Feature #2 -> CUJ2:
A component that checks how many replicas are actually running in each
of the subclusters and if the number matches to the
FederatedReplicaSet preferences (by default spread replicas evenly
across the clusters but custom preferences are allowed - see
below). If it doesn’t and the situation is unlikely to improve soon
then the replicas should be moved to other subclusters.

### API and CLI

All interaction with FederatedReplicaSet will be done by issuing
kubectl commands pointing on the Federated Master API Server. All the
commands would behave in a similar way as on the regular master,
however in the next versions (1.5+) some of the commands may give
slightly different output. For example kubectl describe on federated
replica set should also give some information about the subclusters.

Moreover, for safety, some defaults will be different. For example for
kubectl delete federatedreplicaset cascade will be set to false.

FederatedReplicaSet would have the same object as local ReplicaSet
(although it will be accessible in a different part of the
api). Scheduling preferences (how many replicas in which cluster) will
be passed as annotations.

### FederateReplicaSet preferences

The preferences are expressed by the following structure, passed as a
serialized json inside annotations.

```
type FederatedReplicaSetPreferences struct {  
    // If set to true then already scheduled and running replicas may be moved to other clusters to
    // in order to bring cluster replicasets towards a desired state. Otherwise, if set to false,
    // up and running replicas will not be moved.
    Rebalance bool `json:"rebalance,omitempty"`

    // Map from cluster name to preferences for that cluster. It is assumed that if a cluster   
    // doesn’t have a matching entry then it should not have local replica. The cluster matches   
    // to "*" if there is no entry with the real cluster name.   
    Clusters map[string]LocalReplicaSetPreferences  
}

// Preferences regarding number of replicas assigned to a cluster replicaset within a federated replicaset.
type ClusterReplicaSetPreferences struct {
    // Minimum number of replicas that should be assigned to this Local ReplicaSet. 0 by default.
    MinReplicas int64 `json:"minReplicas,omitempty"`

    // Maximum number of replicas that should be assigned to this Local ReplicaSet. Unbounded if no value provided (default).
    MaxReplicas *int64 `json:"maxReplicas,omitempty"`

    // A number expressing the preference to put an additional replica to this LocalReplicaSet. 0 by default.
    Weight int64
}
```

How this works in practice:

**Scenario 1**. I want to spread my 50 replicas evenly across all available clusters. Config:

```
FederatedReplicaSetPreferences {
   Rebalance : true
   Clusters : map[string]LocalReplicaSet {
     "*" : LocalReplicaSet{ Weight: 1}
   } 
}
```

Example:

+  Clusters A,B,C,  all have capacity.
   Replica layout: A=16 B=17 C=17.
+  Clusters A,B,C and C has capacity for 6 replicas.
   Replica layout: A=22 B=22 C=6
+  Clusters A,B,C. B and C are offline:
   Replica layout: A=50

**Scenario 2**. I want to have only 2 replicas in each of the clusters.

```
FederatedReplicaSetPreferences {
   Rebalance : true
   Clusters : map[string]LocalReplicaSet {
     "*" : LocalReplicaSet{ MaxReplicas: 2; Weight: 1}
   } 
}
```

Or

```
FederatedReplicaSetPreferences {
   Rebalance : true
   Clusters : map[string]LocalReplicaSet {
     "*" : LocalReplicaSet{ MinReplicas: 2; Weight: 0 }
	 }
 }

```

Or

```
FederatedReplicaSetPreferences {  
   Rebalance : true
   Clusters : map[string]LocalReplicaSet {  
     "*" : LocalReplicaSet{ MinReplicas: 2; MaxReplicas: 2}   
   }  
}
```

There is a global target for 50, however if there are 3 clusters there will be only 6 replicas running.

**Scenario 3**. I want to have 20 replicas in each of 3 clusters.

```
FederatedReplicaSetPreferences {  
   Rebalance : true
   Clusters : map[string]LocalReplicaSet {  
     "*" : LocalReplicaSet{ MinReplicas: 20; Weight: 0}  
   }  
}
```

There is a global target for 50, however clusters require 60. So some clusters will have less replicas.
 Replica layout: A=20 B=20 C=10.

**Scenario 4**. I want to have equal number of replicas in clusters A,B,C, however don’t put more than 20 replicas to cluster C.

```
FederatedReplicaSetPreferences { 
   Rebalance : true 
   Clusters : map[string]LocalReplicaSet {  
     "*" : LocalReplicaSet{ Weight: 1}  
     “C” : LocalReplicaSet{ MaxReplicas: 20,  Weight: 1}  
   }  
}
```

Example:

+  All have capacity.
   Replica layout: A=16 B=17 C=17.
+  B is offline/has no capacity
   Replica layout: A=30 B=0 C=20
+  A and B are offline:
   Replica layout: C=20

**Scenario 5**. I want to run my application in cluster A, however if there are troubles FRS can also use clusters B and C, equally.

```
FederatedReplicaSetPreferences {  
   Clusters : map[string]LocalReplicaSet {  
     “A” : LocalReplicaSet{ Weight: 1000000}  
     “B” : LocalReplicaSet{ Weight: 1}  
     “C” : LocalReplicaSet{ Weight: 1}  
   }  
}
```

Example:

+  All have capacity.
   Replica layout: A=50 B=0 C=0.
+  A has capacity for only 40 replicas
   Replica layout: A=40 B=5 C=5

**Scenario 6**. I want to run my application in clusters A, B and C. Cluster A gets twice the QPS than other clusters.

```
FederatedReplicaSetPreferences {  
   Clusters : map[string]LocalReplicaSet {  
     “A” : LocalReplicaSet{ Weight: 2}  
     “B” : LocalReplicaSet{ Weight: 1}  
     “C” : LocalReplicaSet{ Weight: 1}  
   }  
}
```

**Scenario 7**. I want to spread my 50 replicas evenly across all available clusters, but if there
are already some replicas, please do not move them. Config:

```
FederatedReplicaSetPreferences {
   Rebalance : false
   Clusters : map[string]LocalReplicaSet {
     "*" : LocalReplicaSet{ Weight: 1}
   } 
}
```

Example:

+  Clusters A,B,C, all have capacity, but A already has 20 replicas
   Replica layout: A=20 B=15 C=15.
+  Clusters A,B,C and C has capacity for 6 replicas, A has already 20 replicas.
   Replica layout: A=22 B=22 C=6
+  Clusters A,B,C and C has capacity for 6 replicas, A has already 30 replicas.
   Replica layout: A=30 B=14 C=6

## The Idea

A new federated controller - Federated Replica Set Controller (FRSC)
will be created inside federated controller manager. Below are
enumerated the key idea elements:

+ [I0] It is considered OK to have slightly higher number of replicas
   globally for some time.

+ [I1] FRSC starts an informer on the FederatedReplicaSet that listens
   on FRS being created, updated or deleted. On each create/update the
   scheduling code will be started to calculate where to put the
   replicas. The default behavior is to start the same amount of
   replicas in each of the cluster. While creating LocalReplicaSets
   (LRS) the following errors/issues can occur:

   + [E1] Master rejects LRS creation (for known or unknown
      reason). In this case another attempt to create a LRS should be
      attempted in 1m or so. This action can be tied with
      [[I5]](#heading=h.ififs95k9rng). Until the the LRS is created
      the situation is the same as [E5]. If this happens multiple
      times all due replicas should be moved elsewhere and later moved
      back once the LRS is created.

   + [E2] LRS with the same name but different configuration already
      exists. The LRS is then overwritten and an appropriate event
      created to explain what happened. Pods under the control of the
      old LRS are left intact and the new LRS may adopt them if they
      match the selector.

   + [E3] LRS is new but the pods that match the selector exist. The
      pods are adopted by the RS (if not owned by some other
      RS). However they may have a different image, configuration
      etc. Just like with regular LRS.

+ [I2] For each of the cluster FRSC starts a store and an informer on
   LRS that will listen for status updates. These status changes are
   only interesting in case of troubles. Otherwise it is assumed that
   LRS runs trouble free and there is always the right number of pod
   created but possibly not scheduled.


   + [E4] LRS is manually deleted from the local cluster. In this case
      a new LRS should be created. It is the same case as
      [[E1]](#heading=h.wn3dfsyc4yuh). Any pods that were left behind
      won’t be killed and will be adopted after the LRS is recreated.

   + [E5] LRS fails to create (not necessary schedule) the desired
      number of pods due to master troubles, admission control
      etc. This should be considered as the same situation as replicas
      unable to schedule (see [[I4]](#heading=h.dqalbelvn1pv)).

   + [E6] It is impossible to tell that an informer lost connection
      with a remote cluster or has other synchronization problem so it
      should be handled by cluster liveness probe and deletion
      [[I6]](#heading=h.z90979gc2216).

+ [I3] For each of the cluster start an store and informer to monitor
   whether the created pods are eventually scheduled and what is the
   current number of correctly running ready pods. Errors:

   + [E7] It is impossible to tell that an informer lost connection
      with a remote cluster or has other synchronization problem so it
      should be handled by cluster liveness probe and deletion
      [[I6]](#heading=h.z90979gc2216)

+ [I4] It is assumed that a not scheduled pod is a normal situation
and can last up to X min if there is a huge traffic on the
cluster. However if the replicas are not scheduled in that time then
FRSC should consider moving most of the unscheduled replicas
elsewhere. For that purpose FRSC will maintain a data structure
where for each FRS controlled LRS we store a list of pods belonging
to that LRS along with their current status and status change timestamp.

+ [I5] If a new cluster is added to the federation then it doesn’t
   have a LRS and the situation is equal to
   [[E1]](#heading=h.wn3dfsyc4yuh)/[[E4]](#heading=h.vlyovyh7eef).

+ [I6] If a cluster is removed from the federation then the situation
   is equal to multiple [E4]. It is assumed that if a connection with
   a cluster is lost completely then the cluster is removed from the
   the cluster list (or marked accordingly) so
   [[E6]](#heading=h.in6ove1c1s8f) and [[E7]](#heading=h.37bnbvwjxeda)
   don’t need to be handled.

+ [I7] All ToBeChecked FRS are browsed every 1 min (configurable),
   checked against the current list of clusters, and all missing LRS
   are created. This will be executed in combination with [I8].

+ [I8] All pods from ToBeChecked FRS/LRS are browsed every 1 min
   (configurable) to check whether some replica move between clusters
   is needed or not.

+  FRSC never moves replicas to LRS that have not scheduled/running
pods or that has pods that failed to be created.

   + When FRSC notices that a number of pods are not scheduler/running
      or not_even_created in one LRS for more than Y minutes it takes
      most of them from LRS, leaving couple still waiting so that once
      they are scheduled FRSC will know that it is ok to put some more
      replicas to that cluster.

+   [I9] FRS becomes ToBeChecked if:
   +  It is newly created
   +  Some replica set inside changed its status
   +  Some pods inside cluster changed their status
   +  Some cluster is added or deleted.
> FRS stops ToBeChecked if is in desired configuration (or is stable enough).

## (RE)Scheduling algorithm

To calculate the (re)scheduling moves for a given FRS:

1. For each cluster FRSC calculates the number of replicas that are placed
(not necessary up and running) in the cluster and the number of replicas that
failed to be scheduled. Cluster capacity is the difference between the
the placed and failed to be scheduled.

2. Order all clusters by their weight and hash of the name so that every time
we process the same replica-set we process the clusters in the same order.
Include federated replica set name in the cluster name hash so that we get
slightly different ordering for different RS. So that not all RS of size 1
end up on the same cluster.

3. Assign minimum prefered number of replicas to each of the clusters, if
there is enough replicas and capacity.

4. If rebalance = false, assign the previously present replicas to the clusters,
remember the number of extra replicas added (ER). Of course if there
is enough replicas and capacity.

5. Distribute the remaining replicas with regard to weights and cluster capacity.
In multiple iterations calculate how many of the replicas should end up in the cluster.
For each of the cluster cap the number of assigned replicas by max number of replicas and
cluster capacity. If there were some extra replicas added to the cluster in step
4, don't really add the replicas but balance them gains ER from 4.

## Goroutines layout

+ [GR1] Involved in FRS informer (see
   [[I1]]). Whenever a FRS is created and
   updated it puts the new/updated FRS on FRS_TO_CHECK_QUEUE with
   delay 0.

+ [GR2_1...GR2_N] Involved in informers/store on LRS (see
   [[I2]]). On all changes the FRS is put on
   FRS_TO_CHECK_QUEUE with delay 1min.

+ [GR3_1...GR3_N] Involved in informers/store on Pods
   (see [[I3]] and [[I4]]). They maintain the status store
   so that for each of the LRS we know the number of pods that are
   actually running and ready in O(1) time. They also put the
   corresponding FRS on FRS_TO_CHECK_QUEUE with delay 1min.

+ [GR4] Involved in cluster informer (see
   [[I5]] and [[I6]] ). It puts all FRS on FRS_TO_CHECK_QUEUE
   with delay 0.

+ [GR5_*] Go routines handling FRS_TO_CHECK_QUEUE that put FRS on
   FRS_CHANNEL after the given delay (and remove from
   FRS_TO_CHECK_QUEUE). Every time an already present FRS is added to
   FRS_TO_CHECK_QUEUE the delays are compared and updated so that the
   shorter delay is used.

+ [GR6] Contains a selector that listens on a FRS_CHANNEL. Whenever
   a FRS is received it is put to a work queue. Work queue has no delay
   and makes sure that a single replica set is process is processed by
   only one goroutine.

+ [GR7_*] Goroutines related to workqueue. They fire DoFrsCheck on the FRS.
   Multiple replica set can be processed in parallel. Two Goroutines cannot
   process the same FRS at the same time.


## Func DoFrsCheck

The function does [[I7]] and[[I8]]. It is assumed that it is run on a
single thread/goroutine so we check and evaluate the same FRS on many
goroutines (however if needed the function can be parallelized for
different FRS). It takes data only from store maintained by GR2_* and
GR3_*. The external communication is only required to:

+ Create LRS. If a LRS doesn’t exist it is created after the
   rescheduling, when we know how much replicas should it have.

+ Update LRS replica targets.

If FRS is not in the desired state then it is put to
FRS_TO_CHECK_QUEUE with delay 1min (possibly increasing).

## Monitoring and status reporting

FRCS should expose a number of metrics form the run, like

+ FRSC -> LRS communication latency
+ Total times spent in various elements of DoFrsCheck

FRSC should also expose the status of FRS as an annotation on FRS and
as events.

## Workflow

Here is the sequence of tasks that need to be done in order for a
typical FRS to be split into a number of LRS’s and to be created in
the underlying federated clusters.

Note a: the reason the workflow would be helpful at this phase is that
for every one or two steps we can create PRs accordingly to start with
the development.

Note b: we assume that the federation is already in place and the
federated clusters are added to the federation.

Step 1. the client sends an RS create request to the
federation-apiserver

Step 2. federation-apiserver persists an FRS into the federation etcd

Note c: federation-apiserver populates the clusterid field in the FRS
before persisting it into the federation etcd

Step 3: the federation-level “informer” in FRSC watches federation
etcd for new/modified FRS’s, with empty clusterid or clusterid equal
to federation ID, and if detected, it calls the scheduling code

Step 4.

Note d: scheduler populates the clusterid field in the LRS with the
IDs of target clusters

Note e: at this point let us assume that it only does the even
distribution, i.e., equal weights for all of the underlying clusters

Step 5. As soon as the scheduler function returns the control to FRSC,
the FRSC starts a number of cluster-level “informer”s, one per every
target cluster, to watch changes in every target cluster etcd
regarding the posted LRS’s and if any violation from the scheduled
number of replicase is detected the scheduling code is re-called for
re-scheduling purposes.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/federated-replicasets.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
