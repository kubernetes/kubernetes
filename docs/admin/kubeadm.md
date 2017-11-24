
kubeadm: easily bootstrap a secure Kubernetes cluster

### Synopsis



kubeadm: easily bootstrap a secure Kubernetes cluster.

    ┌──────────────────────────────────────────────────────────┐
    │ KUBEADM IS CURRENTLY IN BETA                             │
    │                                                          │
    │ But please, try it out and give us feedback at:          │
    │ https://github.com/kubernetes/kubeadm/issues             │
    │ and at-mention @kubernetes/sig-cluster-lifecycle-bugs    │
    │ or @kubernetes/sig-cluster-lifecycle-feature-requests    │
    └──────────────────────────────────────────────────────────┘

Example usage:

    Create a two-machine cluster with one master (which controls the cluster),
    and one node (where your workloads, like Pods and Deployments run).

    ┌──────────────────────────────────────────────────────────┐
    │ On the first machine:                                    │
    ├──────────────────────────────────────────────────────────┤
    │ master# kubeadm init                                     │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │ On the second machine:                                   │
    ├──────────────────────────────────────────────────────────┤
    │ node# kubeadm join <arguments-returned-from-init>        │
    └──────────────────────────────────────────────────────────┘

    You can then repeat the second step on as many other machines as you like.



