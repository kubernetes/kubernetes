Next generation rkt runtime integration
=======================================

Authors: Euan Kemp (@euank), Yifan Gu (@yifan-gu)

## Abstract

This proposal describes the design and road path for integrating rkt with kubelet with the new container runtime interface.

## Background

Currently, the Kubernetes project supports rkt as a container runtime via an implementation under [pkg/kubelet/rkt package](https://github.com/kubernetes/kubernetes/tree/v1.5.0-alpha.0/pkg/kubelet/rkt).

This implementation, for historical reasons, has required implementing a large amount of logic shared by the original Docker implementation.

In order to make additional container runtime integrations easier, more clearly defined, and more consistent, a new [Container Runtime Interface](https://github.com/kubernetes/kubernetes/blob/v1.5.0-alpha.0/pkg/kubelet/api/v1alpha1/runtime/api.proto) (CRI) is being designed.
The existing runtimes, in order to both prove the correctness of the interface and reduce maintenance burden, are incentivized to move to this interface.

This document proposes how the rkt runtime integration will transition to using the CRI.

## Goals

### Full-featured

The CRI integration must work as well as the existing integration in terms of features.

Until that's the case, the existing integration will continue to be maintained.

### Easy to Deploy

The new integration should not be any more difficult to deploy and configure than the existing integration.

### Easy to Develop

This iteration should be as easy to work and iterate on as the original one.

It will be available in an initial usable form quickly in order to validate the CRI.

## Design

In order to fulfill the above goals, the rkt CRI integration will make the following choices:

### Remain in-process with Kubelet

The current rkt container runtime integration is able to be deployed simply by deploying the kubelet binary.

This is, in no small part, to make it *Easy to Deploy*.

Remaining in-process also helps this integration not regress on performance, one axis of being *Full-Featured*.

### Communicate through gRPC

Although the kubelet and rktlet will be compiled together, the runtime and kubelet will still communicate through gRPC interface for better API abstraction.

For the near short term, they will still talk through a unix socket before we implement a custom gRPC connection that skips the network stack.

### Developed as a Separate Repository

Brian Grant's discussion on splitting the Kubernetes project into [separate repos](https://github.com/kubernetes/kubernetes/issues/24343) is a compelling argument for why it makes sense to split this work into a separate repo.

In order to be *Easy to Develop*, this iteration will be maintained as a separate repository, and re-vendored back in.

This choice will also allow better long-term growth in terms of better issue-management, testing pipelines, and so on.

Unfortunately, in the short term, it's possible that some aspects of this will also cause pain and it's very difficult to weight each side correctly.

### Exec the rkt binary (initially)

While significant work on the rkt [api-service](https://coreos.com/rkt/docs/latest/subcommands/api-service.html) has been made,
it has also been a source of problems and additional complexity,
and was never transitioned to entirely.

In addition, the rkt cli has historically been the primary interface to the rkt runtime.

The initial integration will execute the rkt binary directly for app creation/start/stop/removal, as well as image pulling/removal.

The creation of pod sanbox is also done via rkt command line, but it will run under `systemd-run` so it's monitored by the init process.

In the future, some of these decisions are expected to be changed such that rkt is vendored as a library dependency for all operations, and other init systems will be supported as well.


## Roadmap and Milestones

1. rktlet integrate with kubelet to support basic pod/container lifecycle (pod creation, container creation/start/stop, pod stop/removal) [[Done]](https://github.com/kubernetes-incubator/rktlet/issues/9)
2. rktlet integrate with kubelet to support more advanced features:
   - Support kubelet networking, host network
   - Support mount / volumes [[#33526]](https://github.com/kubernetes/kubernetes/issues/33526)
   - Support exposing ports
   - Support privileged containers
   - Support selinux options [[#33139]](https://github.com/kubernetes/kubernetes/issues/33139)
   - Support attach [[#29579]](https://github.com/kubernetes/kubernetes/issues/29579)
   - Support exec [[#29579]](https://github.com/kubernetes/kubernetes/issues/29579)
   - Support logging [[#33111]](https://github.com/kubernetes/kubernetes/pull/33111)

3. rktlet integrate with kubelet, pass 100% e2e and node e2e tests, with nspawn stage1.
4. rktlet integrate with kubelet, pass 100% e2e and node e2e tests, with kvm stage1.
5. Revendor rktlet into `pkg/kubelet/rktshim`, and start deprecating the `pkg/kubelet/rkt` package.
6. Eventually replace the current `pkg/kubelet/rkt` package.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-rkt-runtime.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
