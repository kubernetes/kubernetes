Kubernetes Proposal - Build Plugin
==================================

Problem/Rationale
-----------------

Kubernetes creates Docker containers from images that were built elsewhere and pushed to a Docker
registry.  Building Docker images is a foundational use-case in Docker-based workflows for
application development and deployment.  Without support for builds in Kubernetes, if a system
administrator wanted a system that could build images, he or she would have to select a pre-existing
build system or write a new one, and then figure out how to deploy and maintain it on or off
Kubernetes. However, in most cases operators would wish to leverage the ability of Kubernetes to
schedule task execution into a pool of available resources, and most build systems would want to
take advantage of that mechanism.   Offering an API for builds also makes Kubernetes a viable
backend for arbitrary third-party Docker image build systems which require resource constrainment
and scheduling capabilities, and allows organizations to orchestrate docker builds from their
existing continuous integration processes.  This is not a core component of Kubernetes, but would
have significant value as a plugin to enable CI/CD flows around Docker images.

Most build jobs share common characteristics - a set of build context parameters that define the job,
the need to run a certain process to completion, the capture of the logs from that build process,
publishing resources from successful builds, and the final “status” of the build.  In addition, the
image-driven deployment flow that Kubernetes advocates depends on having images available.

Builds should take advantage of resource restrictions – specifying limitations on things such as CPU
usage, memory usage, and build (pod) execution time – once support for this exists in Kubernetes.
Additionally, builds would become repeatable and consistent (same inputs = same output).

There are potentially several different types of builds that produce other types of output as well.
This proposal is for adding functionality to Kubernetes to build Docker images.

Here are some possible user scenarios for builds in Kubernetes:

1.   As a user of Kubernetes, I want to build an image from a source URL and push it to a registry
     (for eventual deployment in Kubernetes).
2.   As a user of Kubernetes, I want to build an image from a binary input (docker context, artifact)
     and push it to a registry (for eventual deployment in Kubernetes).
3.   As a provider of a service that involves building docker images, I want to offload the resource
     allocation, scheduling, and garbage collection associated with that activity to Kubernetes 
     instead of solving those problems myself.
4.   As a developer of a system which involves building docker images, I want to take advantage of
     Kubernetes to perform the build, but orchestrate from an existing CI in order to integrate with
     my organization’s devops SOPs.

Example Use: Cloud IDE
----------------------

Company X offers a docker-based cloud IDE service and needs to build docker images at scale for 
their customers’ hosted projects.  Company X wants a turn-key solution for this that handles
scheduling, resource allocation, and garbage collection.  Using the build API, Company X can 
leverage Kubernetes for the build work and concentrate on solving their core business problems.

Example Use: Enterprise Devops
------------------------------

Company Y wants to leverage Kubernetes to build docker images, but their Devops SOPs mandate the
use of a third-party CI server in order to facilitate things like triggering builds when an
upstream project is built and promoting builds when the result is signed off on in the CI server.
Using the build API, company Y implements workflows in the CI server that orchestrate building in
Kubernetes which integrating with their organization’s SOPs.

Proposed Design
---------------

Note: The proposed solution requires that run-once containers be implemented in Kubernetes.

**BuildConfig**

Add a new BuildConfig type that will be used to record the inputs to a Build. Its fields could include:

1.  Source URI
2.  Source ref (e.g. git branch)
3.  Image to use to perform the build
4.  Desired image tag
5.  Docker registry URL

Add appropriate registries and storage for BuildConfig and register /buildConfigs with the apiserver.

**Build**

Add a new Build type that will be used to record a build for historical purposes. A Build includes:

1.  A copy of a BuildConfig (as the standalone BuildConfig could be updated over time and should not
    affect a specific build)
2.  A status field (new, pending, running, complete, failed)
3.  The ID of the Pod associated with this Build

Add appropriate registries and storage for Build and register /builds with the apiserver.

**BuildController**

Add a new BuildController that runs a sync loop to execute builds.

For newly created builds, the BuildController will assign a pod ID to the build and set the build’s
state to pending. This way, the assignment of the pod ID and pending status is idempotent and won’t
result in two BuildControllers potentially scheduling two different pods for the same build.

For pending builds, the BuildController will attempt to create a pod to perform the build. If the
creation succeeds, it sets the build’s status to pending. If the pod already exists, that means
another BuildController already processed this build in a pending state, resulting in a no-op. Any
other pod creation error would result in the build’s status being set to failed.

It may be desirable to support variations in the pod descriptor used to create the build pod. As
such, it could be possible for plugins/extensions to register additional build pod definitions.
Examples of variations include a builder that runs `docker build` as well as a builder that uses
the Source-To-Images (sti) tool (https://github.com/openshift/geard/tree/master/cmd/sti).

For running builds, the BuildController will monitor the status of the pod. If the pod is still
running and the build has exceeded its allotted execution time, the BuildController will consider
it failed. If the pod is terminated, the BuildController will examine the exit codes for each of
the pod’s containers. If any exit code is non-zero, the build is marked as failed. Otherwise, it
is considered complete (successful).

Once the build has reached a terminal state (complete or failed), the BuildController will delete
the pod associated with the build. In the future, it will be desirable to keep a record of the
pod’s containers’ logs but that is out of scope of this proposal.

Docker Daemon Location: Use the minion’s Docker socket
------------------------------------------------------

With this approach, a pod containing a single container–a build container–would be created. The
minion’s Docker socket would be bind mounted into the build container. The build container would
execute the build command (e.g. `docker build`) and all interaction with Docker would be using the
host’s (minion’s) Docker daemon.

**Pros**

1.  Reduces number of Docker daemons required
2.  Minimizes image storage requirements

**Cons**

1.  Not possible to constrain resources per-user
2.  Containers created during the build are created outside the scope of / not managed by Kubernetes
3.  Containers created during the build don’t have the build container as their parent process, making
    container cleanup more difficult

Docker Daemon Location: Docker-in-Docker
----------------------------------------

With this approach, a pod containing a single container–a build container–would be created. The
build container would launch its own Docker daemon in the background, and then it would execute
the build command (e.g. `docker build`) and all interaction with Docker would be using the
container’s own (private) Docker daemon.

**Pros**

1.  Build process resources can be constrained to the user’s acceptable limits (cgroups)
2.  Containers created during the build have the build container as their parent process, making
    container cleanup trivial

**Cons**

1.  Requires a privileged container as running the Docker daemon (even as Docker-in-Docker) requires
    more privileges than a non-privileged container offers
2.  No easy way to share storage of images/layers among build containers, requiring each 
    Docker-in-Docker instance to store its own unique, full copy of any image(s) downloaded during
    the build process.  A caching proxy running on the minion could at least minimize the number of
    times an image is pulled from a remote registry, but that doesn’t eliminate the need for each 
    build container to have its own copy of the images.
