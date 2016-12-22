# Linux Capabilities in Container Runtime Interface

The following proposal summarizes the current state and problems for runtimes to implement Linux Capabilities.

## Problems

Today's [Linux Capabilities Interface](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/kubelet/api/v1alpha1/runtime/api.proto#L350-L355) in CRI is derived from the Docker runtime implementation, which also maps to the [k8s API](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/api/v1/types.go#L1164-L1170).

If users want to launch a container with ad-hoc capabilities, they will go through the following steps:

- Get a list of capabilities he would require
- Look up Docker's docs about the [default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities).
- Calculate the `Add` and `Drop` capability set based on his requirements and the default capabilities list.
- Specify the `Add` and `Drop` set for the containers in a k8s pod spec.
- Launch the pod/container.

There are several issues here:

- The default capability set heavily depends on Docker's implementation; Kubernetes has no control over that.
When Docker changes the default capability set across versions, the effective set of default capabilities can change suddenly.
- To implement the same semantics as today's Docker runtime, other runtimes (e.g. rkt) are required to depend on the implicit Docker default capability set as well. If Docker changes the default capability set, then consistency is not guaranteed.
- From a security point of view, an explicit capability set which does not rely on undefined defaults is preferable for auditing purposes.

## Solution

The proposed solution will:

- Keep the Kubernetes API backwards compatible.
- Add a new `RequestedSet` field to the `PodSpec.SecurityContext.Capabilities` struct, to allow users to request a specific set of capabilities.
- Add a new `DefaultCapabilities` field to the `PodSecurityPolicySpec` struct, to allow administrators to define a default set of capabilities.
- Define a default capability whitelist inside the following Kubernetes components:
  1. the *API server*: Capability defaults are being applied to `PodSpec.SecurityContext.Capabilities.RequestedSet` either through the `PodSecurityPolicySpec` or, if admission control is disabled, using a default capability whitelist.
  2. the *Kubelet*: If the kubelet receives a `PodSpec.SecurityContext.Capabilities.RequestedSet` that does not specify any capabilities, the kubelet applies capability defaults to retain backwards compatibility. This could occur for example when a kubelet is newer than the API server.
- Make CRI expose the whitelist based capability interface, remove the existing `add_capabilities`, and `drop_capabilities` fields, and put the burden on the runtime to implement that.
- Make the final requested capability set **visible** through the container spec, for example:

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
          requestedSet:
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

## Struct changes

The `PodSecurityPolicySpec` in `pkg/apis/extensions/types.go` gains a new `DefaultCapabilities` field. 
The existing fields `DefaultAddCapabilities`, and `RequiredDropCapabilities` will be marked as deprecated.

```
type PodSecurityPolicySpec struct {
...
        // DefaultAddCapabilities is the default set of capabilities that will be added to the container
        // unless the pod spec specifically drops the capability or the pod spec uses a requested-set.
        // You may not list a capability in both DefaultAddCapabilities and RequiredDropCapabilities.
        // Note: this field is deprecated in favor of the DefaultCapabilities field.
        DefaultAddCapabilities []api.Capability `json:"defaultAddCapabilities,omitempty"`

        // RequiredDropCapabilities are the capabilities that will be dropped from the container. These
        // are required to be dropped and cannot be added or present in the requested-set.
        RequiredDropCapabilities []api.Capability `json:"requiredDropCapabilities,omitempty"`

        // AllowedCapabilities is a list of capabilities that can be requested to add to the container.
        // Capabilities in this field may be added at the pod author's discretion via the add
        // or requested-set in the pod spec.
        // You must not list a capability in both AllowedCapabilities and RequiredDropCapabilities.
        AllowedCapabilities []api.Capability `json:"allowedCapabilities,omitempty"`

        // DefaultCapabilities is a set of default capabilities that will be used for pods
        // that do not have a requested-set specified in their pod spec.
        // The pod spec's add/drop capabilities are applied to this set.
        // In addition to the capabilities listed here, the pod author can add additional capabilities
        // specified in the AllowedCapabilities field.
        DefaultCapabilities []api.Capability `json:"defaultCapabilities,omitempty"`
...
}
```

The `Capabilities` struct in `pkg/api/types.go`, and `pkg/api/v1/types.go` gets a new `RequestedSet` field:

```
// Capabilities represent POSIX capabilities that can be added or removed to a running container to/from a default set.
type Capabilities struct {
        // Capabilities added to the default capabilities
        Add []Capability `json:"add,omitempty"`
        // Capabilities removed from the default capabilities
        Drop []Capability `json:"drop,omitempty"`
        // Requested set of capabilities
        RequestedSet []Capability `json:"requestedSet,omitempty"`
}
```

The `Capability` struct in `pkg/kubelet/api/v1alpha1/runtime/api.proto` replaces the existing `add_capabilities` and `drop_capabilities` fields with a new `requested_capabilities` field:

```
message Capability {
    // List of capabilities to retain.
    repeated string requested_capabilities = 1;
}
```

## Use Cases

### User specifies `requestedSet` only in the pod spec:

The simplest use case is when a user only sets the `requestedSet` field in a new pod spec:

```
  spec:
    containers:
      securityContext:
        capabilities:
          requestedSet:
          - MKNOD, NET_ADMIN, SETPCAP, AUDIT_WRITE, ...
```

The resolved requested set equals `requestedSet`.

### User specifies `add`/`drop` but no `requestedSet` in the pod spec:

The second primary use case will be dealing with backwards-compatibility with existing pod specs that may specify `add` or `remove`:

```
  spec:
    containers:
      securityContext:
        capabilities:
          add:
          - NET_ADMIN
          drop:
          - MKNOD
```

The resolved requested set is calculated in the API server based on a _default set_.

The _default set_ is determined as follows:

- The API server has a hard-coded set of capabilities (for illustrative purposes, called `HardcodedDefaultCapabilities`)
- If admission control is not enabled, or admission control is enabled but there are no capabilities-related fields set on the `PodSecurityPolicySpec`, then `HardcodedDefaultCapabilities` is used as the default set.
- Else (i.e. if admission control is enabled and the `PodSecurityPolicySpec` (PSP) has any capabilities-related fields set), then:
   - If the `DefaultCapabilities` field is set, that is merged with any `DefaultAddCapabilities` and `RequiredDropCapabilities` set on the same `PodSecurityPolicySpec`.
   - Else (i.e. the `DefaultCapabilities` field is not set), then:
       - If the submitted pod's `SecurityContext` has a `requested-set` field, that is merged with any `DefaultAddCapabilities` and `RequiredDropCapabilities` set on the same `PodSecurityPolicySpec`.
       - Else, `HardcodedDefaultCapabilities` is merged with any `DefaultAddCapabilities` and `RequiredDropCapabilities` set on the same `PodSecurityPolicySpec`.

The _resolved requested set_ is then calculated by merging the _default set_ with the `add` and `drop` fields on the submitted pod's `SecurityContext` (i.e. "default set" + `add` - `drop`).

Finally, the `requestedSet` field is set to "default set" - `add` - `drop` in the pod spec stored in the API server (which converts this pod spec into the form in the use case below).
This ensures that all pod spec objects returned from the API server have a populated `requestedSet` field that preserves the immutability invariant.


### User specifies both `requestedSet`, and `add`/`drop` in the pod spec:

This use case is a bit contrived and primarily presented to demonstrate backwards-compatibility and the converted form from the previous use case:

```
  spec:
    containers:
      securityContext:
        capabilities:
          requestedSet:
          - SETPCAP, AUDIT_WRITE, ...
          add:
          - NET_ADMIN
          drop:
          - MKNOD
```

The `add`/`drop` sets MUST be able to be idempotently applied to the requested-set, e.g.`MKNOD` must not be in `requestedSet` and `add`/`drop` on the same pod spec. 
This invariant is validated in the API server.

The resolved set is calculated by merging the `requestedSet` with the `add` and `drop` fields (i.e. `requestedSet`+`add`-`drop`).

## Open questions:

**Q:** What should the default capability list should be?

**A:** According to @smarterclayton, there has been questions and arguments about what the default capability list should be, and it's not settled down yet.
       So for now, in order to keep backward compatibility, we'd better just define it as current [Docker's default capability list](https://docs.docker.com/engine/reference/run/#/runtime-privilege-and-linux-capabilities).

**Q:** How do runtimes implement the whitelist capability set?

**A:** For the Docker runtime, this adds a burden to Docker runtime maintainers to convert the capabilities in the whitelist to `Add` and `Drop` set based on the Docker's default capability list. For rkt runtime, this can be done by using the `--caps-retain` flag.

## Changes need to make:

- Modify the structs as described above.
- Modify CRI to replace the `Add`, `Drop` capability list with a capability whitelist.
- Modify runtimes to support the new interface.
- Make the container spec reflect the final capability set.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/container-runtime-interface/capabilities.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
