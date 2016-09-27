<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
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

# Linux Capabilities in Container Runtime Interface

The following proposal summarizes the current state and problems for runtime to implement Linux Capabilities.
And propose solutions to solve the problems

## Problems

Today's [Linux Capabilities Interface](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/kubelet/api/v1alpha1/runtime/api.proto#L350-L355) in CRI is derived from the Docker runtime implementation, which also maps
to the [k8s API](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/api/v1/types.go#L1164-L1170).

If a user wants to launch a container with required capabilities, he will go through such steps:

- Get a list of capabilities he would require
- Look up Docker's docs about the [default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities)
- Calculate the `Add` and `Drop` capability set based on the his requirement and the default capabilities list.
- Specify the `Add` and `Drop` set for the containers in a k8s pod spec.
- Launch the pod/container.

There are several issues here:

- The default capability set heavily depends on Docker's implementation, Kubernetes has no control over that.
When Docker changes the default capability set across versions (hopefull it should not happen), users of Docker get surprise,
users of Kubernetes get surprise, too.
- Even worse, to implement the same semantics as today's Docker runtime, it will require other runtimes (e.g. rkt) to depend
on the default capability set as well. And if Docker changes the default capability set, then consistency is not guaranteed.
- It is arguable that `Add` and `Drop` set are the best semantics for capability set compared to a [whitelist based](https://github.com/appc/spec/blob/af31bda9a474bf3a3e83144f6f0d264cfedc813f/spec/ace.md#oslinuxcapabilities-retain-set) semantics,
which explicitly sets the bounding capabilities that a container could have, and has less security implications (users don't guess what
is the final capability set).

## Solutions

As mentioned above, an explicit whitelist type of capability-retain-set could make things very consistent across runtimes and
version updates.
But as we can see, it's not very practical for system admins and operators to put a long [default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities) on every
container he wants to launch.
Besides, changing the k8s API could take several release cycles.

So in the proposed solution here. We will:
- Keep the k8s API as is today. Users will not need to change anything.
- Define a default capability list inside k8s, and document about it.
- Make CRI expose the whitelist based capability interface, and put the burden on the runtime to implement that.
- Make the final capability set **visible** through the container status, for example:

```shell
$ kubectl get pod -o yaml
apiVersion: v1
items:
- apiVersion: v1
  kind: Pod
  metadata:
  ...
  spec:
    containers:
    - command:
      - sleep
      - "1000"
      image: busybox
      imagePullPolicy: Always
      name: my-busybox
      securityContext:
        capabilities:
          add:
          - NET_ADMIN
      ...
  status:
    containerStatuses:
    - image: busybox
    capabilities:
      SETPCAP, MKNOD, AUDIT_WRITE, ...
      name: my-busybox
      state:
        running:
          startedAt: 2016-09-27T22:33:38Z
    ...
```

## Open questions:

**Q:** What the default capability list should be?

**A:** According to @smarterclayton, there has been questions and arguments about what the default capability list should be, and it's not settled down yet.
       So for now, in order to keep backward compatibility, we'd better just define it as [Docker's default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities) today.

**Q:** How wide does the default capability list apply? Does it apply to one node, or the whole cluster?

**A:** I don't really have a use case for support per-node-default default capability list, if someone could give one, it might be very helpful.
       Other than that, for today's use case, a cluster-wide default capability list is enough.

**Q:** How does runtimes implement the whitelist capability set?

**A:** For Docker runtime, this adds a burden to Docker runtime maintainers to convert the capabilities in the whitelist to `Add` and `Drop` set
       based on the Docker's default capability list.
       For rkt runtime, this can be done by using the `--caps-retain` flag.

## Changes need to make:

- Add a `DefaultCapabilities` field in the [PodSecurityPolicySpec](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/apis/extensions/types.go#L653), and make it default to today's default capability list.
- Modify CRI to replace the `Add`, `Drop` capability list with a capability whitelist .
- Modify runtimes to support the new interface.
- Make the container status to reflect the final capability set.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-runtime-interface/capabilities.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
