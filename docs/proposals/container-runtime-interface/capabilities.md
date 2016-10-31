# Linux Capabilities in Container Runtime Interface

The following proposal summarizes the current state and problems for runtimes to implement Linux Capabilities.

## Problems

Today's [Linux Capabilities Interface](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/kubelet/api/v1alpha1/runtime/api.proto#L350-L355) in CRI is derived from the Docker runtime implementation, which also maps
to the [k8s API](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/api/v1/types.go#L1164-L1170).

If users want to launch a container with ad-hoc capabilities, they will go through the following steps:

- Get a list of capabilities he would require
- Look up Docker's docs about the [default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities).
- Calculate the `Add` and `Drop` capability set based on his requirements and the default capabilities list.
- Specify the `Add` and `Drop` set for the containers in a k8s pod spec.
- Launch the pod/container.

There are several issues here:

- The default capability set heavily depends on Docker's implementation, Kubernetes has no control over that.
When Docker changes the default capability set across versions, the effective set of default capabilities can change suddenly.
- To implement the same semantics as today's Docker runtime, other runtimes (e.g. rkt) are required to depend on the implicit Docker default capability set as well. And if Docker changes the default capability set, then consistency is not guaranteed.
- From a security point of view, an explicit capability set which does not rely on undefined defaults is preferable for auditing purposes.

## Solution

The proposed solution will:
- Keep the k8s API as is today. Users will not need to change anything.
- Define a default capability whitelist inside k8s, and document it.
- Add a `DefaultCapabilities` field to the `PodSecurityPolicySpec` which overrides the default capability whitelist inside k8s.
- Set the default capability whitelist in the `PodSpec`'s `SecurityContext`.
- Make CRI expose the whitelist based capability interface, and put the burden on the runtime to implement that.
- Make the default capability set **visible** through the container spec, for example:

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
          default:
          - SETPCAP, MKNOD, AUDIT_WRITE, ...
          add:
          - NET_ADMIN
          drop:
          - MKNOD
      ...
  status:
    containerStatuses:
    - image: busybox
      name: my-busybox
      state:
        running:
          startedAt: 2016-09-27T22:33:38Z
    ...
```

The `PodSecurityPolicySpec` gets a new `DefaultCapabilities` field. The existing fields `DefaultAddCapabilities`, and `RequiredDropCapabilities` will be marked as deprecated.

```
type PodSecurityPolicySpec struct {
...
        // DefaultAddCapabilities is the default set of capabilities that will be added to the container
        // unless the pod spec specifically drops the capability. You may not list a capability in both
        // DefaultAddCapabilities and RequiredDropCapabilities.
        // Note: this field is deprecated in favor of the DefaultCapabilities field.
        DefaultAddCapabilities []api.Capability `json:"defaultAddCapabilities,omitempty"`
        // RequiredDropCapabilities are the capabilities that will be dropped from the container.  These
        // are required to be dropped and cannot be added.
        // Note: this field is deprecated in favor of the DefaultCapabilities field.
        RequiredDropCapabilities []api.Capability `json:"requiredDropCapabilities,omitempty"`
        // AllowedCapabilities is a list of capabilities that can be requested to add to the container.
        // Capabilities in this field may be added at the pod author's discretion.
        // You must not list a capability in both AllowedCapabilities and RequiredDropCapabilities.
        AllowedCapabilities []api.Capability `json:"allowedCapabilities,omitempty"`
        // DefaultCapabilities is a list of capabilities that will always be applied to the container.
        // In addition to the capabilities listed here, the pod author can add additional capabilities
        // specified in the AllowedCapabilities field.
        DefaultCapabilities []api.Capability `json:"defaultCapabilities,omitempty"`
...
}
```

The `Capabilities` struct in `pkg/api/types.go`, and `pkg/api/v1/types.go` gets a new `Default` field:

```
// Capabilities represent POSIX capabilities that can be added or removed to a running container to/from a default set.
type Capabilities struct {
        // Added capabilities to the default capabilities
        Add []Capability `json:"add,omitempty"`
        // Removed capabilities from the default capabilities
        Drop []Capability `json:"drop,omitempty"`
        // Default capabilities
        Default []Capability `json:"default,omitempty"`
}
```

With the changes above the runtime gets a complete capabilities execution context in the `PodSpec` with a default whitelist of capabilities, and a list of capabilities to be added/dropped.

## Open questions:

**Q:** What the default capability list should be?

**A:** According to @smarterclayton, there has been questions and arguments about what the default capability list should be, and it's not settled down yet.
       So for now, in order to keep backward compatibility, we'd better just define it as current [Docker's default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities).

**Q:** How wide does the default capability list apply? Does it apply to one node, or the whole cluster?

**A:** A cluster-wide default capability list is proposed in this document. More specific per-node defaults can be discussed at a later point.

**Q:** How does runtimes implement the whitelist capability set?

**A:** For Docker runtime, this adds a burden to Docker runtime maintainers to convert the capabilities in the whitelist to `Add` and `Drop` set based on the Docker's default capability list. For rkt runtime, this can be done by using the `--caps-retain` flag.

## Changes need to make:

- Modify the structs as described above.
- Modify CRI to replace the `Add`, `Drop` capability list with a capability whitelist .
- Modify runtimes to support the new interface.
- Make the container spec reflect the final capability set.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-runtime-interface/capabilities.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
