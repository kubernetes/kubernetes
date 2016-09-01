<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Overview](#overview)
  - [Motivation](#motivation)
  - [Related work](#related-work)
- [Alpha Design](#alpha-design)
  - [Overview](#overview-1)
  - [Prerequisites](#prerequisites)
  - [API Changes](#api-changes)
    - [Pod Security Policy](#pod-security-policy)
  - [Deploying profiles](#deploying-profiles)
  - [Testing](#testing)
- [Beta Design](#beta-design)
  - [API Changes](#api-changes-1)
- [Future work](#future-work)
  - [System component profiles](#system-component-profiles)
  - [Deploying profiles](#deploying-profiles-1)
  - [Custom app profiles](#custom-app-profiles)
  - [Security plugins](#security-plugins)
  - [Container Runtime Interface](#container-runtime-interface)
  - [Alerting](#alerting)
  - [Profile authoring](#profile-authoring)
- [Appendix](#appendix)

<!-- END MUNGE: GENERATED_TOC -->

# Overview

AppArmor is a [mandatory access control](https://en.wikipedia.org/wiki/Mandatory_access_control)
(MAC) system for Linux that supplements the standard Linux user and group based
permissions. AppArmor can be configured for any application to reduce the potential attack surface
and provide greater [defense in depth](https://en.wikipedia.org/wiki/Defense_in_depth_(computing)).
It is configured through profiles tuned to whitelist the access needed by a specific program or
container, such as Linux capabilities, network access, file permissions, etc. Each profile can be
run in either enforcing mode, which blocks access to disallowed resources, or complain mode, which
only reports violations.

AppArmor is similar to SELinux. Both are MAC systems implemented as a Linux security module (LSM),
and are mutually exclusive. SELinux offers a lot of power and very fine-grained controls, but is
generally considered very difficult to understand and maintain. AppArmor sacrifices some of that
flexibility in favor of ease of use. Seccomp-bpf is another Linux kernel security feature for
limiting attack surface, and can (and should!) be used alongside AppArmor.

## Motivation

AppArmor can enable users to run a more secure deployment, and / or provide better auditing and
monitoring of their systems. Although it is not the only solution, we should enable AppArmor for
users that want a simpler alternative to SELinux, or are already maintaining a set of AppArmor
profiles. We have heard from multiple Kubernetes users already that AppArmor support is important to
them. The [seccomp proposal](../../docs/design/seccomp.md#use-cases) details several use cases that
also apply to AppArmor.

## Related work

Much of this design is drawn from the work already done to support seccomp profiles in Kubernetes,
which is outlined in the [seccomp design doc](../../docs/design/seccomp.md). The designs should be
kept close to apply lessons learned, and reduce cognitive and maintenance overhead.

Docker has supported AppArmor profiles since version 1.3, and maintains a default profile which is
applied to all containers on supported systems.

AppArmor was upstreamed into the Linux kernel in version 2.6.36. It is currently maintained by
[Canonical](http://www.canonical.com/), is shipped by default on all Ubuntu and openSUSE systems,
and is supported on several
[other distributions](http://wiki.apparmor.net/index.php/Main_Page#Distributions_and_Ports).

# Alpha Design

This section describes the proposed design for
[alpha-level](../../docs/devel/api_changes.md#alpha-beta-and-stable-versions) support, although
additional features are described in [future work](#future-work). For AppArmor alpha support
(targeted for Kubernetes 1.4) we will enable:

- Specifying a pre-loaded profile to apply to a pod container
- Restricting pod containers to a set of profiles (admin use case)

We will also provide a reference implementation of a pod for loading profiles on nodes, but an
official supported mechanism for deploying profiles is out of scope for alpha.

## Overview

An AppArmor profile can be specified for a container through the Kubernetes API with a pod
annotation. If a profile is specified, the Kubelet will verify that the node meets the required
[prerequisites](#prerequisites) (e.g. the profile is already configured on the node) before starting
the container, and will not run the container if the profile cannot be applied. If the requirements
are met, the container runtime will configure the appropriate options to apply the profile. Profile
requirements and defaults can be specified on the
[PodSecurityPolicy](security-context-constraints.md).

## Prerequisites

When an AppArmor profile is specified, the Kubelet will verify the prerequisites for applying the
profile to the container. In order to [fail
securely](https://www.owasp.org/index.php/Fail_securely), a container **will not be run** if any of
the prerequisites are not met. The prerequisites are:

1. **Kernel support** - The AppArmor kernel module is loaded. Can be checked by
   [libcontainer](https://github.com/opencontainers/runc/blob/4dedd0939638fc27a609de1cb37e0666b3cf2079/libcontainer/apparmor/apparmor.go#L17).
2. **Runtime support** - For the initial implementation, Docker will be required (rkt does not
   currently have AppArmor support). All supported Docker versions include AppArmor support. See
   [Container Runtime Interface](#container-runtime-interface) for other runtimes.
3. **Installed profile** - The target profile must be loaded prior to starting the container. Loaded
   profiles can be found in the AppArmor securityfs \[1\].

If any of the prerequisites are not met an event will be generated to report the error and the pod
will be
[rejected](https://github.com/kubernetes/kubernetes/blob/cdfe7b7b42373317ecd83eb195a683e35db0d569/pkg/kubelet/kubelet.go#L2201)
by the Kubelet.

*[1] The securityfs can be found in `/proc/mounts`, and defaults to `/sys/kernel/security` on my
Ubuntu system. The profiles can be found at `{securityfs}/apparmor/profiles`
([example](http://bazaar.launchpad.net/~apparmor-dev/apparmor/master/view/head:/utils/aa-status#L137)).*

## API Changes

The intial alpha support of AppArmor will follow the pattern
[used by seccomp](https://github.com/kubernetes/kubernetes/pull/25324) and specify profiles through
annotations. Profiles can be specified per-container through pod annotations. The annotation format
is a key matching the container, and a profile name value:

```
container.apparmor.security.alpha.kubernetes.io/<container_name>=<profile_name>
```

The profiles can be specified in the following formats (following the convention used by [seccomp](../../docs/design/seccomp.md#api-changes)):

1. `runtime/default` - Applies the default profile for the runtime. For docker, the profile is
   generated from a template
   [here](https://github.com/docker/docker/blob/master/profiles/apparmor/template.go). If no
   AppArmor annotations are provided, this profile is enabled by default if AppArmor is enabled in
   the kernel. Runtimes may define this to be unconfined, as Docker does for privileged pods.
2. `localhost/<profile_name>` - The profile name specifies the profile to load.

*Note: There is no way to explicitly specify an "unconfined" profile, since it is discouraged. If
 this is truly needed, the user can load an "allow-all" profile.*

### Pod Security Policy

The [PodSecurityPolicy](security-context-constraints.md) allows cluster administrators to control
the security context for a pod and its containers. An annotation can be specified on the
PodSecurityPolicy to restrict which AppArmor profiles can be used, and specify a default if no
profile is specified.

The annotation key is `apparmor.security.alpha.kubernetes.io/allowedProfileNames`.  The value is a
comma delimited list, with each item following the format described [above](#api-changes). If a list
of profiles are provided and a pod does not have an AppArmor annotation, the first profile in the
list will be used by default.

Enforcement of the policy is standard. See the
[seccomp implementation](https://github.com/kubernetes/kubernetes/pull/28300) as an example.

## Deploying profiles

We will provide a reference implementation of a DaemonSet pod for loading profiles on nodes, but
there will not be an official mechanism or API in the initial version (see
[future work](#deploying-profiles-1)).  The reference container will contain the `apparmor_parser`
tool and a script for using the tool to load all profiles in a set of (configurable)
directories. The initial implementation will poll (with a configurable interval) the directories for
additions, but will not update or unload existing profiles. The pod can be run in a DaemonSet to
load the profiles onto all nodes. The pod will need to be run in privileged mode.

This simple design should be sufficient to deploy AppArmor profiles from any volume source, such as
a ConfigMap or PersistentDisk. Users seeking more advanced features should be able extend this
design easily.

## Testing

Our e2e testing framework does not currently run nodes with AppArmor enabled, but we can run a node
e2e test suite on an AppArmor enabled node. The cases we should test are:

- *PodSecurityPolicy* - These tests can be run on a cluster even if AppArmor is not enabled on the
  nodes.
  - No AppArmor policy allows pods with arbitrary profiles
  - With a policy a default is selected
  - With a policy arbitrary profiles are prevented
  - With a policy allowed profiles are allowed
- *Node AppArmor enforcement* - These tests need to run on AppArmor enabled nodes, in the node e2e
  suite.
  - A valid container profile gets applied
  - An unloaded profile will be rejected

# Beta Design

The only part of the design that changes for beta is the API, which is upgraded from
annotation-based to first class fields.

## API Changes

AppArmor profiles will be specified in the container's SecurityContext, as part of an
`AppArmorOptions` struct. The options struct makes the API more flexible to future additions.

```go
type SecurityContext struct {
    ...
    // The AppArmor options to be applied to the container.
    AppArmorOptions *AppArmorOptions `json:"appArmorOptions,omitempty"`
    ...
}

// Reference to an AppArmor profile loaded on the host.
type AppArmorProfileName string

// Options specifying how to run Containers with AppArmor.
type AppArmorOptions struct {
    // The profile the Container must be run with.
    Profile AppArmorProfileName `json:"profile"`
}
```

The `AppArmorProfileName` format matches the format for the profile annotation values describe
[above](#api-changes).

The `PodSecurityPolicySpec` receives a similar treatment with the addition of an
`AppArmorStrategyOptions` struct. Here the `DefaultProfile` is separated from the `AllowedProfiles`
in the interest of making the behavior more explicit.

```go
type PodSecurityPolicySpec struct {
    ...
    AppArmorStrategyOptions *AppArmorStrategyOptions `json:"appArmorStrategyOptions,omitempty"`
    ...
}

// AppArmorStrategyOptions specifies AppArmor restrictions and requirements for pods and containers.
type AppArmorStrategyOptions struct {
    // If non-empty, all pod containers must be run with one of the profiles in this list.
    AllowedProfiles []AppArmorProfileName `json:"allowedProfiles,omitempty"`
    // The default profile to use if a profile is not specified for a container.
    // Defaults to "runtime/default". Must be allowed by AllowedProfiles.
    DefaultProfile AppArmorProfileName `json:"defaultProfile,omitempty"`
}
```

# Future work

Post-1.4 feature ideas. These are not fully-fleshed designs.

## System component profiles

We should publish (to GitHub) AppArmor profiles for all Kubernetes system components, including core
components like the API server and controller manager, as well as addons like influxDB and
Grafana. `kube-up.sh` and its successor should have an option to apply the profiles, if the AppArmor
is supported by the nodes. Distros that support AppArmor and provide a Kubernetes package should
include the profiles out of the box.

## Deploying profiles

We could provide an official supported solution for loading profiles on the nodes. One option is to
extend the reference implementation described [above](#deploying-profiles) into a DaemonSet that
watches the directory sources to sync changes, or to watch a ConfigMap object directly.  Another
option is to add an official API for this purpose, and load the profiles on-demand in the Kubelet.

## Custom app profiles

[Profile stacking](http://wiki.apparmor.net/index.php/AppArmorStacking) is an AppArmor feature
currently in development that will enable multiple profiles to be applied to the same object. If
profiles are stacked, the allowed set of operations is the "intersection" of both profiles
(i.e. stacked profiles are never more permissive). Taking advantage of this feature, the cluster
administrator could restrict the allowed profiles on a PodSecurityPolicy to a few broad profiles,
and then individual apps could apply more app specific profiles on top.

## Security plugins

AppArmor, SELinux, TOMOYO, grsecurity, SMACK, etc. are all Linux MAC implementations with similar
requirements and features. At the very least, the AppArmor implementation should be factored in a
way that makes it easy to add alternative systems. A more advanced approach would be to extract a
set of interfaces for plugins implementing the alternatives. An even higher level approach would be
to define a common API or profile interface for all of them. Work towards this last option is
already underway for Docker, called
[Docker Security Profiles](https://github.com/docker/docker/issues/17142#issuecomment-148974642).

## Container Runtime Interface

Other container runtimes will likely add AppArmor support eventually, so the
[Container Runtime Interface](container-runtime-interface-v1.md) (CRI) needs to be made compatible
with this design. The two important pieces are a way to report whether AppArmor is supported by the
runtime, and a way to specify the profile to load (likely through the `LinuxContainerConfig`).

## Alerting

Whether AppArmor is running in enforcing or complain mode it generates logs of policy
violations. These logs can be important cues for intrusion detection, or at the very least a bug in
the profile. Violations should almost always generate alerts in production systems. We should
provide reference documentation for setting up alerts.

## Profile authoring

A common method for writing AppArmor profiles is to start with a restrictive profile in complain
mode, and then use the `aa-logprof` tool to build a profile from the logs. We should provide
documentation for following this process in a Kubernetes environment.

# Appendix

- [What is AppArmor](https://askubuntu.com/questions/236381/what-is-apparmor)
- [Debugging AppArmor on Docker](https://github.com/docker/docker/blob/master/docs/security/apparmor.md#debug-apparmor)
- Load an AppArmor profile with `apparmor_parser` (required by Docker so it should be available):

  ```
  $ apparmor_parser --replace --write-cache /path/to/profile
  ```

- Unload with:

  ```
  $ apparmor_parser --remove /path/to/profile
  ```



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/apparmor.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
