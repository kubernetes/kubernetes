# CRI: the Container Runtime Interface

## What is CRI?

CRI (_Container Runtime Interface_) consists of a
[protobuf API](../../pkg/kubelet/api/v1alpha1/runtime/api.proto),
specifications/requirements (to-be-added),
and [libraries] (https://github.com/kubernetes/kubernetes/tree/master/pkg/kubelet/server/streaming)
for container runtimes to integrate with kubelet on a node. CRI is currently in Alpha.

In the future, we plan to add more developer tools such as the CRI validation
tests.

## Why develop CRI?

Prior to the existence of CRI, container runtimes (e.g., `docker`, `rkt`) were
integrated with kubelet through implementing an internal, high-level interface
in kubelet. The entrance barrier for runtimes was high because the integration
required understanding the internals of kubelet and contributing to the main
Kubernetes repository. More importantly, this would not scale because every new
addition incurs a significant maintenance overhead in the main kubernetes
repository.

Kubernetes aims to be extensible. CRI is one small, yet important step to enable
pluggable container runtimes and build a healthier ecosystem.

## How to use CRI?

1. Start the image and runtime services on your node. You can have a single
   service acting as both image and runtime services.
2. Set the kubelet flags
   - Pass the unix socket(s) to which your services listen to kubelet:
     `--container-runtime-endpoint` and `--image-service-endpoint`.
   - Enable CRI in kubelet by`--experimental-cri=true`.
   - Use the "remote" runtime by `--container-runtime=remote`.

Please see the [Status Update](#status-update) section for known issues for
each release.

Note that CRI is still in its early stages. We are actively incorporating
feedback from early developers to improve the API. Developers should expect
occasional API breaking changes.

## Does Kubelet use CRI today?

No, but we are working on it.

The first step is to switch kubelet to integrate with Docker via CRI by
default. The current [Docker CRI implementation](https://github.com/kubernetes/kubernetes/blob/release-1.5/pkg/kubelet/dockershim)
already passes most end-to-end tests, and has mandatory PR builders to prevent
regressions. While we are expanding the test coverage gradually, it is
difficult to test on all combinations of OS distributions, platforms, and
plugins. There are also many experimental or even undocumented features relied
upon by some users. We would like to **encourage the community to help test
this Docker-CRI integration and report bugs and/or missing features** to
smooth the transition in the near future. Please file a Github issue and
include @kubernetes/sig-node for any CRI problem.

### How to test the new Docker CRI integration?

Start kubelet with the following flags:
  - Use the Docker container runtime by `--container-runtime=docker`(the default).
  - Enable CRI in kubelet by`--experimental-cri=true`.

Please also see the [known issues](#docker-cri-1.5-known-issues) before trying
out.

## Design docs and proposals

We plan to add CRI specifications/requirements in the near future. For now,
these proposals and design docs are the best sources to understand CRI
besides discussions on Github issues.

  - [Original proposal](https://github.com/kubernetes/kubernetes/blob/release-1.5/docs/proposals/container-runtime-interface-v1.md)
  - [Exec/attach/port-forward streaming requests](https://docs.google.com/document/d/1OE_QoInPlVCK9rMAx9aybRmgFiVjHpJCHI9LrfdNM_s/edit?usp=sharing)
  - [Container stdout/stderr logs](https://github.com/kubernetes/kubernetes/blob/release-1.5/docs/proposals/kubelet-cri-logging.md)
  - Networking: The CRI runtime handles network plugins and the
    setup/teardown of the pod sandbox.

## Work-In-Progress CRI runtimes

 - [cri-o](https://github.com/kubernetes-incubator/cri-o)
 - [rktlet](https://github.com/kubernetes-incubator/rktlet)
 - [frakti](https://github.com/kubernetes/frakti)

## [Status update](#status-update)

### Kubernetes v1.5 release (CRI v1alpha1)

  - [v1alpha1 version](https://github.com/kubernetes/kubernetes/blob/release-1.5/pkg/kubelet/api/v1alpha1/runtime/api.proto) of CRI is released.

#### [CRI known issues](#cri-1.5-known-issues):

  - [#27097](https://github.com/kubernetes/kubernetes/issues/27097): Container
    metrics are not yet defined in CRI.
  - [#36401](https://github.com/kubernetes/kubernetes/issues/36401): The new
     container log path/format is not yet supported by the logging pipeline
    (e.g., fluentd, GCL).
  - CRI may not be compatible with other experimental features (e.g., Seccomp).
  - Streaming server needs to be hardened.
     - [#36666](https://github.com/kubernetes/kubernetes/issues/36666):
       Authentication.
     - [#36187](https://github.com/kubernetes/kubernetes/issues/36187): Avoid
       including user data in the redirect URL.

#### [Docker CRI integration known issues](#docker-cri-1.5-known-issues)

  - Docker compatibility: Support only Docker v1.11 and v1.12.
  - Network:
     - [#35457](https://github.com/kubernetes/kubernetes/issues/35457): Does
       not support host ports.
     - [#37315](https://github.com/kubernetes/kubernetes/issues/37315): Does
       not support bandwidth shaping.
  - Exec/attach/port-forward (streaming requests):
     - [#35747](https://github.com/kubernetes/kubernetes/issues/35747): Does
       not support `nsenter` as the exec handler (`--exec-handler=nsenter`).
     - Also see (#cri-1.5-known-issues) for limitations on CRI streaming.

## Contacts

  - Email: sig-node (kubernetes-sig-node@googlegroups.com)
  - Slack: https://kubernetes.slack.com/messages/sig-node


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/container-runtime-interface.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
