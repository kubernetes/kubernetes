## DISCLAIMER
- Sig-Node community has reached a general consensus, as a best practice, to
avoid introducing any new checkpointing support. We reached this understanding
after struggling with some hard-to-debug issues in the production environments
caused by the checkpointing.
- Any changes to the checkpointed data structure would be considered incompatible and a component should add its own handling if it needs to ensure backward compatibility of reading old-format checkpoint files.

## Introduction
This folder contains a framework & primitives, Checkpointing Manager, which is
used by several other Kubelet submodules, `devicemanager`, `pods`,
`cpumanager`, `memorymanager`, `dra` and `allocation`, to implement 
checkpointing at each submodule level. As already explained in above `Disclaimer`
section, think twice before introducing any further checkpointing in Kubelet.
If still checkpointing is required, then this folder provides the common APIs
and the framework for implementing checkpointing. Using same APIs across all
the submodules will help maintaining consistency at Kubelet level.

Below is the history of checkpointing support in Kubelet.

| Package | First checkpointing support merged on | PR link | Status |
| ------- | ------------------------------------- | ------- | ------ |
| kubelet/dockershim | Feb 3, 2017 | [[CRI] Implement Dockershim Checkpoint](https://github.com/kubernetes/kubernetes/pull/39903) | Removed
| devicemanager | Sep 6, 2017 | [Deviceplugin checkpoint](https://github.com/kubernetes/kubernetes/pull/51744) | In use
| kubelet/pod | Nov 22, 2017 | [Initial basic bootstrap-checkpoint support](https://github.com/kubernetes/kubernetes/pull/50984) | In use
| cpumanager | Oct 27, 2017 |[Add file backed state to cpu manager](https://github.com/kubernetes/kubernetes/pull/54408) | In use
| memorymanager| Feb 9, 2021 |[Memory manager](https://github.com/kubernetes/kubernetes/pull/95479) | In use
| dra | Mar 12, 2023 |[kubelet dra: Add checkpointing mechanism in the DRA Manager ](https://github.com/kubernetes/kubernetes/pull/115912) | In use
| dra | Jul 24, 2024 |[DRA:refactor checkpointing](https://github.com/kubernetes/kubernetes/pull/126303) | In use
| allocation | Feb 28, 2025 |[Move pod resource allocation management out of the status manager](https://github.com/kubernetes/kubernetes/pull/130254) | In use

Note: Dockershim has been removed from the Kubernetes project as of release 1.24, as a consequence dockershim checkpoint support has also been removed.
