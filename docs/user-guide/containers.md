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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/containers.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Containers with Kubernetes

## Containers and commands

So far the Pods we've seen have all used the `image` field to indicate what process Kubernetes
should run in a container.  In this case, Kubernetes runs the image's default command.  If we want
to run a particular command or override the image's defaults, there are two additional fields that
we can use:

1.  `Command`: Controls the actual command run by the image
2.  `Args`: Controls the arguments passed to the command

### How docker handles command and arguments

Docker images have metadata associated with them that is used to store information about the image.
The image author may use this to define defaults for the command and arguments to run a container
when the user does not supply values.  Docker calls the fields for commands and arguments
`Entrypoint` and `Cmd` respectively.  The full details for this feature are too complicated to
describe here, mostly due to the fact that the docker API allows users to specify both of these
fields as either a string array or a string and there are subtle differences in how those cases are
handled.  We encourage the curious to check out [docker's documentation]() for this feature.

Kubernetes allows you to override both the image's default command (docker `Entrypoint`) and args
(docker `Cmd`) with the `Command` and `Args` fields of `Container`.  The rules are:

1.  If you do not supply a `Command` or `Args` for a container, the defaults defined by the image
    will be used
2.  If you supply a `Command` but no `Args` for a container, only the supplied `Command` will be
    used; the image's default arguments are ignored
3.  If you supply only `Args`, the image's default command will be used with the arguments you
    supply
4.  If you supply a `Command` **and** `Args`, the image's defaults will be ignored and the values
    you supply will be used

Here are examples for these rules in table format

| Image `Entrypoint` |    Image `Cmd`   | Container `Command` |  Container `Args`  |    Command Run   |
|--------------------|------------------|---------------------|--------------------|------------------|
|     `[/ep-1]`      |   `[foo bar]`    |   &lt;not set&gt;   |   &lt;not set&gt;  | `[ep-1 foo bar]` |
|     `[/ep-1]`      |   `[foo bar]`    |      `[/ep-2]`      |   &lt;not set&gt;  |     `[ep-2]`     |
|     `[/ep-1]`      |   `[foo bar]`    |   &lt;not set&gt;   |     `[zoo boo]`    | `[ep-1 zoo boo]` |
|     `[/ep-1]`      |   `[foo bar]`    |      `[/ep-2]`      |     `[zoo boo]`    | `[ep-2 zoo boo]` |


## Capabilities

By default, Docker containers are "unprivileged" and cannot, for example, run a Docker daemon inside a Docker container. We can have fine grain control over the capabilities using cap-add and cap-drop.More details [here](https://docs.docker.com/reference/run/#runtime-privilege-linux-capabilities-and-lxc-configuration).

The relationship between Docker's capabilities and [Linux capabilities](http://man7.org/linux/man-pages/man7/capabilities.7.html)

| Docker's capabilities | Linux capabilities |
| ---- | ---- |
| SETPCAP |  CAP_SETPCAP |
| SYS_MODULE |  CAP_SYS_MODULE |
| SYS_RAWIO |  CAP_SYS_RAWIO |
| SYS_PACCT |  CAP_SYS_PACCT |
| SYS_ADMIN |  CAP_SYS_ADMIN |
| SYS_NICE |  CAP_SYS_NICE |
| SYS_RESOURCE |  CAP_SYS_RESOURCE |
| SYS_TIME |  CAP_SYS_TIME |
| SYS_TTY_CONFIG |  CAP_SYS_TTY_CONFIG |
| MKNOD |  CAP_MKNOD |
| AUDIT_WRITE |  CAP_AUDIT_WRITE |
| AUDIT_CONTROL |  CAP_AUDIT_CONTROL |
| MAC_OVERRIDE |  CAP_MAC_OVERRIDE |
| MAC_ADMIN |  CAP_MAC_ADMIN |
| NET_ADMIN |  CAP_NET_ADMIN |
| SYSLOG |  CAP_SYSLOG |
| CHOWN |  CAP_CHOWN |
| NET_RAW |  CAP_NET_RAW |
| DAC_OVERRIDE |  CAP_DAC_OVERRIDE |
| FOWNER |  CAP_FOWNER |
| DAC_READ_SEARCH |  CAP_DAC_READ_SEARCH |
| FSETID |  CAP_FSETID |
| KILL |  CAP_KILL |
| SETGID |  CAP_SETGID |
| SETUID |  CAP_SETUID |
| LINUX_IMMUTABLE |  CAP_LINUX_IMMUTABLE |
| NET_BIND_SERVICE |  CAP_NET_BIND_SERVICE |
| NET_BROADCAST |  CAP_NET_BROADCAST |
| IPC_LOCK |  CAP_IPC_LOCK |
| IPC_OWNER |  CAP_IPC_OWNER |
| SYS_CHROOT |  CAP_SYS_CHROOT |
| SYS_PTRACE |  CAP_SYS_PTRACE |
| SYS_BOOT |  CAP_SYS_BOOT |
| LEASE |  CAP_LEASE |
| SETFCAP |  CAP_SETFCAP |
| WAKE_ALARM |  CAP_WAKE_ALARM |
| BLOCK_SUSPEND |  CAP_BLOCK_SUSPEND |


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/containers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
