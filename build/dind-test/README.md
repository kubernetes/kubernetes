# A dind test cluster for Kubernetes CI

This is a docker-in-docker-in-docker cluster for integration testing. The nodes are
run in containers, and configured with kubeadm. The cluster is held inside a
top-level container. A single docker-run can spin up and expose an entire Kubernetes
cluster.

# host docker vs dind vs dindind

Using the host docker means a container talks to the root docker image. This can
be done by talking to the tcp port, or by mounting the socket inside the
container.

Docker-in-docker (dind) runs an instance of docker inside a container. This
instance creates nested containers. This introduces a few well-known complicates.

1. Nested overlay file-systems is hard on storage, kernel resources, and access.
   This is solved by putting /var/lib/docker on a volume.
1. The dind container must be privileged. This is a consequence of docker not
   namespacing cgroups.
1. The docker cache cannot be shared, because docker is a monolithic program
   with no inter-process concurrency control.
1. The kernel is still shared, so loading modules, or changing kernel config,
   will affect the whole system.
1. The inner docker may try applying security profiles for SELinux, AppArmor,
   or other Linux Security Module (LSM), which conflict with the outer
   profile.

Docker-in-docker-in-docker (dindind) is like dind, but with another layer of
nesting. This introduces a couple quirks of its own.

1. By default, docker doesn't propagate bind mounts, so file-systems from the
   top-level host cannot be passed in. This can be specified when creating the
   mount, but is unsupported prior to docker v1.10.

# Objective: integ tests

The purpose of this dind cluster is to run Kubernetes CI faster and more
reliably, and to give develops a fairly consistent way of replicating our CI
pipeline's behavior.

This isn't truly an e2e test. Most deployments have a cloud provider, or other
infrastructure. Furthermore, this deployment shares a kernel, which creates
complications for some applications (e.g., cAdvisor, and many storage modules).

## What tests can be run against these clusters

Most conformance tests are applicable for dind test clusters. This deployment
model is meant to aid core kubernetes development.

## What this does not currently support

1. Tests that manipulate kernel modules, because they cross-talk, and we don't
   currently clean up after ourselves. These could be supported in the future.
   1. NFS storage tests use kernel NFS, which requires specific modules
   1. Testing CNI implementations, because they require various kernel resources
      and/or permissions
1. Cloud provider tests, because those tests rely on a specific cloud
   environment
1. Most storage provider tests, because they rely on storage providers that
   either require configuring the kernel, or physical hardware we don't have
   access to
1. Upgrade and restart tests are highly coupled to a specific deployment
   environment.
1. The node e2e tests are meant to be run against a specific environment.

# Artifacts produced

The purpose of this directory is to produce testing artifacts, but not to run
tests themselves.

## dind-base

This is a base image we use that is simply a dind node. It's based on the
bare-bones debian image kubernetes uses to publish containerized components.

### systemd in docker

Most docker deployments are intended to be a single process. So docker runs the
target process as PID 1. This consequences for more complex deployments, because
many systems rely on init system features.

1. Reaping zombie processes
1. Logging for daemons
1. Creates D-Bus, which is expected by many Kubernetes components (e.g.,
   kube-proxy w/iptables)

So instead of setting an entrypoint to kubelet or docker, we run systemd.

This creates some additional complications, because systemd expects many
resources. Notably, systemd writes cgroups, and needs to mount /sys/fs/cgroup.
But so does docker, and for the same reason, so it's not a big deal for dind.

## dind-node

The docker-in-docker node is created by packaging the build artifacts needed for
a kubernetes node.

Some resources are expected to be on the host itself. Since we're running in a
debian environment under systemd, consuming the deb packages for these
components is trivial.

1. kubelet.deb
1. kubectl.deb
1. kubeadm.deb
1. kubernetes-cni.deb

The kubeadm tool expects many master components present as docker images, which
then get run as static pods or daemonsets. These docker images are produced by
the build, and placed directly onto the node's file-system.

### Master

Starting a node as a master requires loading the master component docker images,
and running the `kubeadm init` command.

Although it would be nice to preload the docker images during the build, this is
infeasible for two reasons:
1. Loading docker images requires a running dockerd on the container. This isn't
   available during the build.
1. The images cache must be placed onto a volume, but volumes cannot be created
   at build time. If specified, the data is placed onto the overlay, and moved
   onto a volume for each container creation.

### Worker

Starting a node as a worker requires loading docker images for universal
components (e.g., kube-proxy), and running the `kubeadm join` command.

## dind-cluster

A dind cluster requires several resources (e.g., a docker network, several
containers, mounts, permissions, and capabilities). These are created in a
top-level container to simplify resource tracking (so nodes are dindind). This
also simplifies running multiple instances.

There are two notable cleanup exceptions:
1. Because cgroups aren't namespaced, these can be leaked. But the leak consumes
   few resources, and is cleaned on boot.
1. Although tracked, Docker doesn't garbage collect dangling volumes. It may
   possible to eliminate the problem by using a tmpfs everywhere we use a
   volume. These can be manually deleted with the following commands:
   1. `docker system prune` (v1.25+)
   1. `docker volume rm $(docker volume ls -q -f dangling=true)`

The top-level image's file-system contains a copy of the dind-node image, and
the relevant testing binaries (e.g., e2e.test). This means that the appropriate
version of tests is always available to the cluster.

# Other Kubernetes dind projects

The [kube-spawn](https://github.com/kinvolk/kube-spawn) project attempts to
create a dind environment for testing applications on top of Kubernetes.

The [Mirantis k-d-c (kubeadm docker-in-docker cluster)](https://github.com/Mirantis/kubeadm-dind-cluster) is a similar project, but with a more general scope.
