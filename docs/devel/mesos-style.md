<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Building Mesos/Omega-style frameworks on Kubernetes

## Introduction

We have observed two different cluster management architectures, which can be
categorized as "Borg-style" and "Mesos/Omega-style." In the remainder of this
document, we will abbreviate the latter as "Mesos-style." Although out-of-the
box Kubernetes uses a Borg-style architecture, it can also be configured in a
Mesos-style architecture, and in fact can support both styles at the same time.
This document describes the two approaches and describes how to deploy a
Mesos-style architecture on Kubernetes.

As an aside, the converse is also true: one can deploy a Borg/Kubernetes-style
architecture on Mesos.

This document is NOT intended to provide a comprehensive comparison of Borg and
Mesos. For example, we omit discussion of the tradeoffs between scheduling with
full knowledge of cluster state vs. scheduling using the "offer" model. That
issue is discussed in some detail in the Omega paper.
(See [references](#references) below.)


## What is a Borg-style architecture?

A Borg-style architecture is characterized by:

* a single logical API endpoint for clients, where some amount of processing is
done on requests, such as admission control and applying defaults

* generic (non-application-specific) collection abstractions described
declaratively,

* generic controllers/state machines that manage the lifecycle of the collection
abstractions and the containers spawned from them

* a generic scheduler

For example, Borg's primary collection abstraction is a Job, and every
application that runs on Borg--whether it's a user-facing service like the GMail
front-end, a batch job like a MapReduce, or an infrastructure service like
GFS--must represent itself as a Job. Borg has corresponding state machine logic
for managing Jobs and their instances, and a scheduler that's responsible for
assigning the instances to machines.

The flow of a request in Borg is:

1. Client submits a collection object to the Borgmaster API endpoint

1. Admission control, quota, applying defaults, etc. run on the collection

1. If the collection is admitted, it is persisted, and the collection state
machine creates the underlying instances

1. The scheduler assigns a hostname to the instance, and tells the Borglet to
start the instance's container(s)

1. Borglet starts the container(s)

1. The instance state machine manages the instances and the collection state
machine manages the collection during their lifetimes

Out-of-the-box Kubernetes has *workload-specific* abstractions (ReplicaSet, Job,
DaemonSet, etc.) and corresponding controllers, and in the future may have
[workload-specific schedulers](../../docs/proposals/multiple-schedulers.md),
e.g. different schedulers for long-running services vs. short-running batch. But
these abstractions, controllers, and schedulers are not *application-specific*.

The usual request flow in Kubernetes is very similar, namely

1. Client submits a collection object (e.g. ReplicaSet, Job, ...) to the API
server

1. Admission control, quota, applying defaults, etc. run on the collection

1. If the collection is admitted, it is persisted, and the corresponding
collection controller creates the underlying pods

1. Admission control, quota, applying defaults, etc. runs on each pod; if there
are multiple schedulers, one of the admission controllers will write the
scheduler name as an annotation based on a policy

1. If a pod is admitted, it is persisted

1. The appropriate scheduler assigns a nodeName to the instance, which triggers
the Kubelet to start the pod's container(s)

1. Kubelet starts the container(s)

1. The controller corresponding to the collection manages the pod and the
collection during their lifetime

In the Borg model, application-level scheduling and cluster-level scheduling are
handled by separate components. For example, a MapReduce master might request
Borg to create a job with a certain number of instances with a particular
resource shape, where each instance corresponds to a MapReduce worker; the
MapReduce master would then schedule individual units of work onto those
workers.

## What is a Mesos-style architecture?

Mesos is fundamentally designed to support multiple application-specific
"frameworks." A framework is composed of a "framework scheduler" and a
"framework executor." We will abbreviate "framework scheduler" as "framework"
since "scheduler" means something very different in Kubernetes (something that
just assigns pods to nodes).

Unlike Borg and Kubernetes, where there is a single logical endpoint that
receives all API requests (the Borgmaster and API server, respectively), in
Mesos every framework is a separate API endpoint. Mesos does not have any
standard set of collection abstractions, controllers/state machines, or
schedulers; the logic for all of these things is contained in each
[application-specific framework](http://mesos.apache.org/documentation/latest/frameworks/)
individually. (Note that the notion of application-specific does sometimes blur
into the realm of workload-specific, for example
[Chronos](https://github.com/mesos/chronos) is a generic framework for batch
jobs. However, regardless of what set of Mesos frameworks you are using, the key
properties remain: each framework is its own API endpoint with its own
client-facing and internal abstractions, state machines, and scheduler).

A Mesos framework can integrate application-level scheduling and cluster-level
scheduling into a single component.

Note: Although Mesos frameworks expose their own API endpoints to clients, they
consume a common infrastructure via a common API endpoint for controlling tasks
(launching, detecting failure, etc.) and learning about available cluster
resources. More details
[here](http://mesos.apache.org/documentation/latest/scheduler-http-api/).

## Building a Mesos-style framework on Kubernetes

Implementing the Mesos model on Kubernetes boils down to enabling
application-specific collection abstractions, controllers/state machines, and
scheduling. There are just three steps:

* Use API plugins to create API resources for your new application-specific
collection abstraction(s)

* Implement controllers for the new abstractions (and for managing the lifecycle
of the pods the controllers generate)

* Implement a scheduler with the application-specific scheduling logic

Note that the last two can be combined: a Kubernetes controller can do the
scheduling for the pods it creates, by writing node name to the pods when it
creates them.

Once you've done this, you end up with an architecture that is extremely similar
to the Mesos-style--the Kubernetes controller is effectively a Mesos framework.
The remaining differences are:

* In Kubernetes, all API operations go through a single logical endpoint, the
API server (we say logical because the API server can be replicated). In
contrast, in Mesos, API operations go to a particular framework. However, the
Kubernetes API plugin model makes this difference fairly small.

* In Kubernetes, application-specific admission control, quota, defaulting, etc.
rules can be implemented in the API server rather than in the controller. Of
course you can choose to make these operations be no-ops for your
application-specific collection abstractions, and handle them in your controller.

* On the node level, Mesos allows application-specific executors, whereas
Kubernetes only has executors for Docker and rkt containers.

The end-to-end flow is:

1. Client submits an application-specific collection object to the API server

2. The API server plugin for that collection object forwards the request to the
API server that handles that collection type

3. Admission control, quota, applying defaults, etc. runs on the collection
object

4. If the collection is admitted, it is persisted

5. The collection controller sees the collection object and in response creates
the underlying pods and chooses which nodes they will run on by setting node
name

6. Kubelet sees the pods with node name set and starts the container(s)

7. The collection controller manages the pods and the collection during their
lifetimes

*Note: if the controller and scheduler are separated, then step 5 breaks
down into multiple steps:*

(5a) collection controller creates pods with empty node name.

(5b) API server admission control, quota, defaulting, etc. runs on the
pods; one of the admission controller steps writes the scheduler name as an
annotation on each pods (see pull request `#18262` for more details).

(5c) The corresponding application-specific scheduler chooses a node and
writes node name, which triggers the Kubelet to start the pod's container(s).

As a final note, the Kubernetes model allows multiple levels of iterative
refinement of runtime abstractions, as long as the lowest level is the pod. For
example, clients of application Foo might create a `FooSet` which is picked up
by the FooController which in turn creates `BatchFooSet` and `ServiceFooSet`
objects, which are picked up by the BatchFoo controller and ServiceFoo
controller respectively, which in turn create pods. In between each of these
steps there is an opportunity for object-specific admission control, quota, and
defaulting to run in the API server, though these can instead be handled by the
controllers.

## References

Mesos is described [here](https://www.usenix.org/legacy/event/nsdi11/tech/full_papers/Hindman_new.pdf).
Omega is described [here](http://research.google.com/pubs/pub41684.html).
Borg is described [here](http://research.google.com/pubs/pub43438.html).






<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/mesos-style.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
