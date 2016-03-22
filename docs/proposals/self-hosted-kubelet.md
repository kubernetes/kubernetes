# Proposal: Self-hosted Kubelet

## Abstract

In a self-hosted Kubernetes deployment, we have the initial bootstrap problem. This proposal presents a solution to the Kubelet bootstrap, and assumes a functioning "bootstrap" control plane, and a Kubelet that can securely contact the API server.

## Background

Our current approach to a self-hosted Kubelet is a "pivot" style installation. This procedure assumes a short-lived “bootstrap” Kubelet will run and start a long-running “self-hosted” Kubelet. Once the self-hosted Kubelet is running the bootstrap Kubelet will exit. Briefly, the following steps summarize a generic outline for this pivot:

1. "bootstrap" Kubelet run by $init_system starts with `--runonce`, `--lock-file`, `--api-servers`, repeatedly until it pulls down Kubelet pod and starts it.
1. "bootstrap" Kubelet exits.
1. "managed" Kubelet is now ready to begin functioning in the cluster.

Additionally, this process could be made simpler via [Taints and Tolerations](https://github.com/kubernetes/kubernetes/blob/master/docs/design/taint-toleration-dedicated.md), where the “bootstrap” Kubelet would register itself with a given taint, which would be tolerated by the “self-hosted” Kubelet.

## Proposal

We propose adding a patch to the Kubelet that would allow it to contact an API server while in “runonce” mode (e.g. running with the `--runonce` flag set). Additionally, we intend to remove the hard-coded timeout for a Kubelet in “runonce” mode, and introduce another flag `--runonce-timeout` which will control the duration of time to wait before forcing the Kubelet to exit. This work has already been started (**Issue:** https://github.com/kubernetes/kubernetes/issues/23073, **PR:** https://github.com/kubernetes/kubernetes/pull/23074).

Additionally, [Taints and Tolerations](https://github.com/kubernetes/kubernetes/blob/master/docs/design/taint-toleration-dedicated.md), whose design has already been accepted, would make the overall Kubelet bootstrap more deterministic. With this, we would also need the ability for a Kubelet to register itself with a given taint when it first contacts the API server. Given that, a Kubelet could register itself with a given taint such as “component=kubelet”, and a Kubelet pod could exist that has a toleration to that taint, ensuring it is the only pod the “bootstrap” Kubelet runs.

This will allow a "self-hosted" Kubelet with minimal new concepts introduced into the core Kubernetes code base, and remains flexible enough to work well with future [bootstrapping services](https://github.com/kubernetes/kubernetes/issues/5754).

To expand on this, we envision a flow similar to the following:

1. Systemd (or $init_system) continually runs “bootstrap” Kubelet in “runonce” mode with a file lock until it pulls down a “self-hosted” Kubelet pod and runs it.
1. “Self-hosted” Kubelet starts up and acquires file lock.
1. Systemd (or $init_system) loop sees that the file lock has been acquired and does nothing.

In this process, the init system would be responsible for ensuring a Kubelet is always running on the node.


## Other discussion

Various similar approaches have been discussed [here](https://github.com/kubernetes/kubernetes/issues/246#issuecomment-64533959) and [here](https://github.com/kubernetes/kubernetes/issues/23073#issuecomment-198478997). This also relies on the Kubelet being able to be run inside a container, more discussion on that is [here](https://github.com/kubernetes/kubernetes/issues/4869).

