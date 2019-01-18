##### Deprecation Notice: cloud providers in this directory are deprecated and will be removed in favor of external (a.k.a out-of-tree) providers. Existing providers in this directory (a.k.a in-tree providers) should only make small incremental changes as needed and avoid large refactors or new features. New providers seeking to support Kubernetes should follow the out-of-tree model as specified in the [Running Kubernetes Cloud Controller Manager docs](https://kubernetes.io/docs/tasks/administer-cluster/running-cloud-controller/). For more information on the future of Kubernetes cloud providers see [KEP-0002](https://github.com/kubernetes/community/blob/master/keps/sig-cloud-provider/0002-cloud-controller-manager.md) and [KEP-0013](https://github.com/kubernetes/community/blob/master/keps/sig-cloud-provider/0013-build-deploy-ccm.md).

Cloud Providers in this directory will continue to be actively developed or maintained and supported at their current level of support as a longer-term solution evolves.

## Overview:
The mechanism for supporting cloud providers is currently in transition:  the original method of implementing cloud provider-specific functionality within the main kubernetes tree (here) is no longer advised; however, the proposed solution is still in development.

#### Guidance for potential cloud providers:
* Support for cloud providers is currently in a state of flux. Background information on motivation and the proposal for improving is in the github [proposal](https://git.k8s.io/community/contributors/design-proposals/cloud-provider/cloud-provider-refactoring.md).
* In support of this plan, a new cloud-controller-manager binary was added in 1.6. This was the first of several steps (see the proposal for more information).
* Attempts to contribute new cloud providers or (to a lesser extent) persistent volumes to the core repo will likely meet with some pushback from reviewers/approvers.
* It is understood that this is an unfortunate situation in which 'the old way is no longer supported but the new way is not ready yet', but the initial path is unsustainable, and contributors are encouraged to participate in the implementation of the proposed long-term solution, as there is risk that PRs for new cloud providers here will not be approved.
* Though the fully productized support envisioned in the proposal is still 2 - 3 releases out, the foundational work is underway, and a motivated cloud provider could accomplish the work in a forward-looking way. Contributors are encouraged to assist with the implementation of the design outlined in the proposal.

#### Some additional context on status / direction:
* 1.6 added a new cloud-controller-manager binary that may be used for testing the new out-of-core cloudprovider flow.
* Setting cloud-provider=external allows for creation of a separate controller-manager binary
* 1.7 adds [extensible admission control](https://git.k8s.io/community/contributors/design-proposals/api-machinery/admission_control_extension.md), further enabling topology customization.
