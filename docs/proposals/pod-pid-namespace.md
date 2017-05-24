# Shared PID Namespace for the Docker Runtime

Pods share many namespaces, but the ability to share a PID namespace was not
supported by Docker until version 1.12. This document proposes how to roll out
support for sharing the PID namespace in the docker runtime.

## Motivation

Sharing a PID namespace is discussed in #1615, and enables:

  1. signaling between containers, which is useful for side cars (e.g. for
     signaling a daemon process after rotating logs).
  2. easier troubleshooting of pods.
  3. addressing [Docker's zombie problem][1] by reaping orphaned zombies in the
     infra container.

## Goals and Non-Goals

Goals include:
  - Change default behavior in the Kubernetes Docker runtime

Non-goals include:
  - Creating an init solution that works for all runtimes
  - Supporting isolated PID namespace indefinitely

## Rollout Plan

Sharing the PID namespace changes an implicit behavior of the Docker runtime
whereby the command run by the container image is always PID 1. This is a side
effect of isolated namespaces rather than intentional behavior, but users may
have built upon this assumption so we should change the default behavior over
the course of multiple releases.

  1. Release 1.6: Enable the shared PID namespace for pods annotated with
     `docker.kubernetes.io/shared-pid: true` (i.e. opt-in) when running with
     Docker >= 1.12. Pods with this annotation will fail to start with older
     Docker versions rather than failing to meet a user's expectation.
  2. Release 1.7: Enable the shared PID namespace for pods unless annotated
     with `docker.kubernetes.io/shared-pid: false` (i.e. opt-out) when running
     with Docker >= 1.12.
  3. Release 1.8: Remove the annotation. All pods receive a shared PID
     namespace when running with Docker >= 1.12.

With each step we will add a release note that clearly describes the change.
After each release we will poll kubernetes-users to determine what, if any,
applications were impacted by this change. If we discover a use case which
cannot be accommodated by a shared PID namespace, we will abort step 3 and
instead formalize a shared-pid field into the pod spec.

## Alternatives Considered

Changing this behavior over the course of 6 months is a bit conservative. We
could instead change the behavior in 2 releases by omitting the first step, but
the opt-in phase allows users to test the change with fewer surprises.

[1]: https://blog.phusion.nl/2015/01/20/docker-and-the-pid-1-zombie-reaping-problem/


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-pid-namespace.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
