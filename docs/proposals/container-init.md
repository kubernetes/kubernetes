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

# Pod initialization

@smarterclayton

March 2016

## Proposal and Motivation

Within a pod there is a need to initialize local data or adapt to the current
cluster environment that is not easily achieved in the current container model.
Containers start in parallel after volumes are mounted, leaving no opportunity
for coordination between containers without specialization of the image. If
two containers need to share common initialization data, both images must
be altered to cooperate using filesystem or network semantics, which introduces
coupling between images. Likewise, if an image requires configuration in order
to start and that configuration is environment dependent, the image must be
altered to add the necessary templating or retrieval.

This proposal introduces the concept of an **init container**, one or more
containers started in sequence before the pod's normal containers are started.
These init containers may share volumes, perform network operations, and perform
computation prior to the start of the remaining containers. They may also, by
virtue of their sequencing, block or delay the startup of application containers
until some precondition is met. In this document we refer to the existing pod
containers as **app containers**.

## Design Points

* Init containers should be able to:
  * Perform initialization of shared volumes
  * Delay the startup of application containers until preconditions are met
  * Download binaries that will be used in app containers as execution targets
  * Inject configuration or extension capability to generic images at startup
  * Perform complex templating of information avaliable in the local environment
  * Register the pod with other components of the system
* Reduce coupling between application images and reduce the need to customize
  those images for Kubernetes generally or specific roles
* Reduce the footprint of individual images by specializing which containers
  perform which tasks (install git into init container, use filesystem contents
  in web container)
* The order init containers start should be predictable and allow users to easily
  reason about the startup of a container
* Complex ordering and failure is out of scope - all complex workflows can if
  necessary be implemented inside of a single init container, and this proposal
  aims to enable that ordering without adding undue complexity to the system
* Both run-once and run-forever pods should be able to use init containers
* As much as possible, an init container should behave like an app container
  to reduce complexity for end users, for clients, and for divergent use cases.
  An init container is a container with the minimum alterations to accomplish
  its goal.

## Alternatives

* Any mechanism that runs user code on a node before regular pod containers
  should itself be a container and modeled as such - we explicitly reject
  creating new mechanisms for running user processes.
* The container pre-start hook (not yet implemented) requires execution within
  the container's image and so cannot adapt existing images. It also cannot
  block startup of containers
* Running a "pre-pod" would defeat the purpose of the pod being an atomic
  unit of scheduling.


## Design

Each pod may have 0..N init containers defined along with the existing
1..N app containers.

On startup of the pod, after the network and volumes are initialized, the
init containers are started in order. Each container must exit successfully
before the next is invoked. If a container fails to start, it is retried
according to the pod RestartPolicy. RestartPolicyNever pods will immediately
fail and exit. RestartPolicyAlways pods will retry the failing init container
with increasing backoff until it succeeds. Future revisions of this spec may
add the ability for an init container to define a limited container restart
policy distinct from the pod RestartPolicy.

A pod cannot be ready until all init containers have succeeded. The ports
on an init container are not aggregated under a service. A pod that is
being initialized is in the `Pending` phase but should have a distinct
condition.

If the pod is "restarted" all containers stopped and started due to
a node restart, change to the pod definition, or admin interaction, all
init containers must execute again.

Each init container has all of the fields of an app container. The following
fields are prohibited from being used on init containers by validation:

* `readinessProbe` - init containers must exit for pod startup to continue,
  are not included in rotation, and so cannot define readiness distinct from
  completion.

Because init containers are semantically different in lifecycle from app
containers (they are run serially, rather than in parallel), for backwards
compatibility with existing clients they must be identified as distinct
fields in the API:

    pod:
      spec:
        containers: ...
        initContainers:
        - name: init-container1
          image: ...
          ...
        - name: init-container2
        ...
      status:
        containerStatuses: ...
        initContainerStatuses:
        - name: init-container1
          ...
        - name: init-container2
          ...

This separation also serves to make the order of container initialization
clear - init containers are executed in the order that they appear, then all
app containers are started at once.

The name of each app and init container in a pod must be unique - it is a
validation error for any container to share a name.


### Resources

Given the ordering and execution for init containers, the following rules
for resource usage apply:

* The highest of any particular resource request or limit defined on all init
  containers is the **effective init request/limit**
* The pod's **effective request/limit** for a resource is the higher of:
  * sum of all app containers request/limit for a resource
  * effective init request/limit for a resource
* The highest QoS tier of init containers is the **effective init QoS tier**

So the following pod:

    pod:
      spec:
        initContainers:
        - limits:
				    cpu: 100m
				    memory: 1GiB
        - limits:
				    cpu: 50m
				    memory: 2GiB
			  containers:
        - limits:
				    cpu: 10m
				    memory: 1100MiB
        - limits:
				    cpu: 10m
				    memory: 1100MiB

has an effective pod limit of `cpu: 100m`, `memory: 2200MiB` (highest init
container cpu is larger than sum of all app containers, sum of container
memory is larger than the max of all init containers). The scheduler, node,
and quota must respect the effective pod request/limit.


### Kubelet and container runtime details

Container runtimes should treat the set of init and app containers as one
large pool. An individual init container execution should be identical to
an app container.

All app container operations are permitted on init containers. The
logs for an init container should be available for the duration of the pod
lifetime or until the pod is restarted.

During initialization, app container status should be shown with the reason
PodInitializing if any init containers are present. Each init container
should show appropriate container status, and all init containers that are
waiting for earlier init containers to finish should have the `reason`
PendingInitialization.

The container runtime should aggressively prune failed init containers.
The container runtime should record whether all init containers have
succeeded internally, and only invoke new init containers if a pod
restart is needed (for Docker, if all containers terminate or if the pod
infra container terminates). Init containers should follow backoff rules
as necessary.


### API Behavior

All APIs that access containers by name should operate on both init and
app containers. Because names are unique the addition of the init container
should be transparent to use cases.

A client with no knowledge of init containers should see appropriate
container status `reason` and `message` fields while the pod is in the
`Pending` phase, and so be able to communicate that to end users.


### Example init containers

* Wait for a service to be created

        pod:
          spec:
            initContainers:
            - name: wait
              image: centos:centos7
              command: ["/bin/sh", "-c", "for i in {1..100}; do sleep 1; if dig myservice; then exit 0; fi; exit 1"]
            containers:
            - name: run
              image: application-image
              command: ["/my_application_that_depends_on_myservice"]

* Wait for an arbitrary period of time

        pod:
          spec:
            initContainers:
            - name: wait
              image: centos:centos7
              command: ["/bin/sh", "-c", "sleep 60"]
            containers:
            - name: run
              image: application-image
              command: ["/static_binary_without_sleep"]

* Clone a git repository into a volume:

        pod:
          spec:
            initContainers:
            - name: download
              image: image-with-git
              command: ["git", "clone", "https://github.com/myrepo/myrepo.git", "/var/lib/data"]
              volumeMounts:
              - mountPath: /var/lib/data
                volumeName: git
            containers:
            - name: run
              image: centos:centos7
              command: ["/var/lib/data/binary"]
              volumeMounts:
              - mountPath: /var/lib/data
                volumeName: git
            volumes:
            - emptyDir: {}
              name: git

* Execute a template transformation based on environment

        pod:
          spec:
            initContainers:
            - name: copy
              image: application-image
              command: ["/bin/cp", "mytemplate.j2", "/var/lib/data/"]
              volumeMounts:
              - mountPath: /var/lib/data
                volumeName: data
            - name: transform
              image: image-with-jinja
              command: ["/bin/sh", "-c", "jinja /var/lib/data/mytemplate.j2 > /var/lib/data/mytemplate.conf"]
              volumeMounts:
              - mountPath: /var/lib/data
                volumeName: data
            containers:
            - name: run
              image: application-image
              command: ["/myapplication", "-conf", "/var/lib/data/mytemplate.conf"]
              volumeMounts:
              - mountPath: /var/lib/data
                volumeName: data
            volumes:
            - emptyDir: {}
              name: data

* Perform a container build

        pod:
          spec:
            initContainers:
            - name: copy
              image: base-image
              workingDir: /home/user/source-tree
              command: ["make"]
            containers:
            - name: commit
              image: image-with-docker
              command:
              - /bin/sh
              - -c
              - docker commit $(complex_bash_to_get_container_id_of_copy) \
                docker push $(commit_id) myrepo:latest
              volumesMounts:
              - mountPath: /var/run/docker.sock
                volumeName: dockersocket


## Open Questions

* We could support readiness on init containers, in which case there is no difference
  between app and init containers except order.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-init.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
