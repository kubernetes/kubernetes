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

This proposal also provides a high level design of **volume containers**, which
initialize a particular volume, as a feature that specializes some of the tasks
defined for init containers. The init container design anticipates the existence
of volume containers and highlights where they will take future work

## Design Points

* Init containers should be able to:
  * Perform initialization of shared volumes
    * Download binaries that will be used in app containers as execution targets
    * Inject configuration or extension capability to generic images at startup
    * Perform complex templating of information available in the local environment
    * Initialize a database by starting a temporary execution process and applying
      schema info.
  * Delay the startup of application containers until preconditions are met
  * Register the pod with other components of the system
* Reduce coupling:
  * Between application images, eliminating the need to customize those images for
    Kubernetes generally or specific roles
  * Inside of images, by specializing which containers perform which tasks
    (install git into init container, use filesystem contents
    in web container)
  * Between initialization steps, by supporting multiple sequential init containers
* Init containers allow simple start preconditions to be implemented that are
  decoupled from application code
  * The order init containers start should be predictable and allow users to easily
    reason about the startup of a container
  * Complex ordering and failure will not be supported - all complex workflows can
    if necessary be implemented inside of a single init container, and this proposal
    aims to enable that ordering without adding undue complexity to the system.
    Pods in general are not intended to support DAG workflows.
* Both run-once and run-forever pods should be able to use init containers
* As much as possible, an init container should behave like an app container
  to reduce complexity for end users, for clients, and for divergent use cases.
  An init container is a container with the minimum alterations to accomplish
  its goal.
* Volume containers should be able to:
  * Perform initialization of a single volume
  * Start in parallel
  * Perform computation to initialize a volume, and delay start until that
    volume is initialized successfully.
  * Using a volume container that does not populate a volume to delay pod start
    (in the absence of init containers) would be an abuse of the goal of volume
    containers.
* Container pre-start hooks are not sufficient for all initialization cases:
  * They cannot easily coordinate complex conditions across containers
  * They can only function with code in the image or code in a shared volume,
    which would have to be statically linked (not a common pattern in wide use)
  * They cannot be implemented with the current Docker implementation - see
    [#140](https://github.com/kubernetes/kubernetes/issues/140)



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
1..M app containers.

On startup of the pod, after the network and volumes are initialized, the
init containers are started in order. Each container must exit successfully
before the next is invoked. If a container fails to start (due to the runtime)
or exits with failure, it is retried according to the pod RestartPolicy.
RestartPolicyNever pods will immediately fail and exit. RestartPolicyAlways
pods will retry the failing init container with increasing backoff until it
succeeds. To align with the design of application containers, init containers
will only support "infinite retries" (RestartPolicyAlways) or "no retries"
(RestartPolicyNever).

A pod cannot be ready until all init containers have succeeded. The ports
on an init container are not aggregated under a service. A pod that is
being initialized is in the `Pending` phase but should have a distinct
condition. Each app container and all future init containers should have
the reason `PodInitializing`. The pod should have a condition `Initializing`
set to `false` until all init containers have succeeded, and `true` thereafter.
If the pod is restarted, the `Initializing` condition should be set to `false.

If the pod is "restarted" all containers stopped and started due to
a node restart, change to the pod definition, or admin interaction, all
init containers must execute again. Restartable conditions are defined as:

* An init container image is changed
* The pod infrastructure container is restarted (shared namespaces are lost)
* The Kubelet detects that all containers in a pod are terminated AND
  no record of init container completion is available on disk (due to GC)

Changes to the init container spec are limited to the container image field.
Altering the container image field is equivalent to restarting the pod.

Because init containers can be restarted, retried, or reexecuted, container
authors should make their init behavior idempotent by handling volumes that
are already populated or the possibility that this instance of the pod has
already contacted a remote system.

Each init container has all of the fields of an app container. The following
fields are prohibited from being used on init containers by validation:

* `readinessProbe` - init containers must exit for pod startup to continue,
  are not included in rotation, and so cannot define readiness distinct from
  completion.

Init container authors may use `activeDeadlineSeconds` on the pod and
`livenessProbe` on the container to prevent init containers from failing
forever. The active deadline includes init containers.

Because init containers are semantically different in lifecycle from app
containers (they are run serially, rather than in parallel), for backwards
compatibility and design clarity they will be identified as distinct fields
in the API:

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

While pod containers are in alpha state, they will be serialized as an annotation
on the pod with the name `pod.alpha.kubernetes.io/init-containers` and the status
of the containers will be stored as `pod.alpha.kubernetes.io/init-container-statuses`.
Mutation of these annotations is prohibited on existing pods.


### Resources

Given the ordering and execution for init containers, the following rules
for resource usage apply:

* The highest of any particular resource request or limit defined on all init
  containers is the **effective init request/limit**
* The pod's **effective request/limit** for a resource is the higher of:
  * sum of all app containers request/limit for a resource
  * effective init request/limit for a resource
* Scheduling is done based on effective requests/limits, which means
  init containers can reserve resources for initialization that are not used
  during the life of the pod.
* The lowest QoS tier of init containers per resource is the **effective init QoS tier**,
  and the highest QoS tier of both init containers and regular containers is the
  **effective pod QoS tier**.

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

In the absence of a defined request or limit on a container, the effective
request/limit will be applied. For example, the following pod:

    pod:
      spec:
        initContainers:
        - limits:
            cpu: 100m
            memory: 1GiB
        containers:
        - request:
            cpu: 10m
            memory: 1100MiB

will have an effective request of `10m / 1100MiB`, and an effective limit
of `100m / 1GiB`, i.e.:

    pod:
      spec:
        initContainers:
        - request:
            cpu: 10m
            memory: 1GiB
        - limits:
            cpu: 100m
            memory: 1100MiB
        containers:
        - request:
            cpu: 10m
            memory: 1GiB
        - limits:
            cpu: 100m
            memory: 1100MiB

and thus have the QoS tier **Burstable** (because request is not equal to
limit).

Quota and limits will be applied based on the effective pod request and
limit.

Pod level cGroups will be based on the effective pod request and limit, the
same as the scheduler.


### Kubelet and container runtime details

Container runtimes should treat the set of init and app containers as one
large pool. An individual init container execution should be identical to
an app container, including all standard container environment setup
(network, namespaces, hostnames, DNS, etc).

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
as necessary. The Kubelet *must* preserve at least the most recent instance
of an init container to serve logs and data for end users and to track
failure states. The Kubelet *should* prefer to garbage collect completed
init containers over app containers, as long as the Kubelet is able to
track that initialization has been completed. In the future, container
state checkpointing in the Kubelet may remove or reduce the need to
preserve old init containers.

For the initial implementation, the Kubelet will use the last termination
container state of the highest indexed init container to determine whether
the pod has completed initialization. During a pod restart, initialization
will be restarted from the beginning (all initializers will be rerun).


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

* Register this pod with a remote server

        pod:
          spec:
            initContainers:
            - name: register
              image: centos:centos7
              command: ["/bin/sh", "-c", "curl -X POST http://$MANAGEMENT_SERVICE_HOST:$MANAGEMENT_SERVICE_PORT/register -d 'instance=$(POD_NAME)&ip=$(POD_IP)'"]
              env:
              - name: POD_NAME
                valueFrom:
                  field: metadata.name
              - name: POD_IP
                valueFrom:
                  field: status.podIP
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

* Clone a git repository into a volume (can be implemented by volume containers in the future):

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

* Execute a template transformation based on environment (can be implemented by volume containers in the future):

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

## Backwards compatibilty implications

Since this is a net new feature in the API and Kubelet, new API servers during upgrade may not
be able to rely on Kubelets implementing init containers. The management of feature skew between
master and Kubelet is tracked in issue [#4855](https://github.com/kubernetes/kubernetes/issues/4855).


## Future work

* Unify pod QoS class with init containers
* Implement container / image volumes to make composition of runtime from images efficient


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-init.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
