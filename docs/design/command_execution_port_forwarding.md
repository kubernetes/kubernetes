# Container Command Execution & Port Forwarding in Kubernetes

## Abstract

This document describes how to use Kubernetes to execute commands in containers,
with stdin/stdout/stderr streams attached and how to implement port forwarding
to the containers.

## Background

See the following related issues/PRs:

- [Support attach](http://issue.k8s.io/1521)
- [Real container ssh](http://issue.k8s.io/1513)
- [Provide easy debug network access to services](http://issue.k8s.io/1863)
- [OpenShift container command execution proposal](https://github.com/openshift/origin/pull/576)

## Motivation

Users and administrators are accustomed to being able to access their systems
via SSH to run remote commands, get shell access, and do port forwarding.

Supporting SSH to containers in Kubernetes is a difficult task. You must
specify a "user" and a hostname to make an SSH connection, and `sshd` requires
real users (resolvable by NSS and PAM). Because a container belongs to a pod,
and the pod belongs to a namespace, you need to specify namespace/pod/container
to uniquely identify the target container. Unfortunately, a
namespace/pod/container is not a real user as far as SSH is concerned. Also,
most Linux systems limit user names to 32 characters, which is unlikely to be
large enough to contain namespace/pod/container. We could devise some scheme to
map each namespace/pod/container to a 32-character user name, adding entries to
`/etc/passwd` (or LDAP, etc.) and keeping those entries fully in sync all the
time. Alternatively, we could write custom NSS and PAM modules that allow the
host to resolve a namespace/pod/container to a user without needing to keep
files or LDAP in sync.

As an alternative to SSH, we are using a multiplexed streaming protocol that
runs on top of HTTP. There are no requirements about users being real users,
nor is there any limitation on user name length, as the protocol is under our
control. The only downside is that standard tooling that expects to use SSH
won't be able to work with this mechanism, unless adapters can be written.

## Constraints and Assumptions

- SSH support is not currently in scope.
- CGroup confinement is ultimately desired, but implementing that support is not
currently in scope.
- SELinux confinement is ultimately desired, but implementing that support is
not currently in scope.

## Use Cases

- A user of a Kubernetes cluster wants to run arbitrary commands in a
container with local stdin/stdout/stderr attached to the container.
- A user of a Kubernetes cluster wants to connect to local ports on his computer
and have them forwarded to ports in a container.

## Process Flow

### Remote Command Execution Flow

1. The client connects to the Kubernetes Master to initiate a remote command
execution request.
2. The Master proxies the request to the Kubelet where the container lives.
3. The Kubelet executes nsenter + the requested command and streams
stdin/stdout/stderr back and forth between the client and the container.

### Port Forwarding Flow

1. The client connects to the Kubernetes Master to initiate a remote command
execution request.
2. The Master proxies the request to the Kubelet where the container lives.
3. The client listens on each specified local port, awaiting local connections.
4. The client connects to one of the local listening ports.
4. The client notifies the Kubelet of the new connection.
5. The Kubelet executes nsenter + socat and streams data back and forth between
the client and the port in the container.

## Design Considerations

### Streaming Protocol

The current multiplexed streaming protocol used is SPDY. This is not the
long-term desire, however. As soon as there is viable support for HTTP/2 in Go,
we will switch to that.

### Master as First Level Proxy

Clients should not be allowed to communicate directly with the Kubelet for
security reasons. Therefore, the Master is currently the only suggested entry
point to be used for remote command execution and port forwarding. This is not
necessarily desirable, as it means that all remote command execution and port
forwarding traffic must travel through the Master, potentially impacting other
API requests.

In the future, it might make more sense to retrieve an authorization token from
the Master, and then use that token to initiate a remote command execution or
port forwarding request with a load balanced proxy service dedicated to this
functionality. This would keep the streaming traffic out of the Master.

### Kubelet as Backend Proxy

The kubelet is currently responsible for handling remote command execution and
port forwarding requests. Just like with the Master described above, this means
that all remote command execution and port forwarding streaming traffic must
travel through the Kubelet, which could result in a degraded ability to service
other requests.

In the future, it might make more sense to use a separate service on the node.

Alternatively, we could possibly inject a process into the container that only
listens for a single request, expose that process's listening port on the node,
and then issue a redirect to the client such that it would connect to the first
level proxy, which would then proxy directly to the injected process's exposed
port. This would minimize the amount of proxying that takes place.

### Scalability

There are at least 2 different ways to execute a command in a container:
`docker exec` and `nsenter`. While `docker exec` might seem like an easier and
more obvious choice, it has some drawbacks.

#### `docker exec`

We could expose `docker exec` (i.e. have Docker listen on an exposed TCP port
on the node), but this would require proxying from the edge and securing the
Docker API. `docker exec` calls go through the Docker daemon, meaning that all
stdin/stdout/stderr traffic is proxied through the Daemon, adding an extra hop.
Additionally, you can't isolate 1 malicious `docker exec` call from normal
usage, meaning an attacker could initiate a denial of service or other attack
and take down the Docker daemon, or the node itself.

We expect remote command execution and port forwarding requests to be long
running and/or high bandwidth operations, and routing all the streaming data
through the Docker daemon feels like a bottleneck we can avoid.

#### `nsenter`

The implementation currently uses `nsenter` to run commands in containers,
joining the appropriate container namespaces. `nsenter` runs directly on the
node and is not proxied through any single daemon process.

### Security

Authentication and authorization hasn't specifically been tested yet with this
functionality. We need to make sure that users are not allowed to execute
remote commands or do port forwarding to containers they aren't allowed to
access.

Additional work is required to ensure that multiple command execution or port
forwarding connections from different clients are not able to see each other's
data. This can most likely be achieved via SELinux labeling and unique process
 contexts.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/command_execution_port_forwarding.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
