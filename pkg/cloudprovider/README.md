##### Deprecation Notice: This directory has entered maintenance mode and will not be accepting new providers. Cloud Providers in this directory will continue to be actively developed or maintained and supported at their current level of support as a longer-term solution evolves. 
 
## Overview: 
The mechanism for supporting cloud providers is currently in transition:  the original method of implementing cloud provider-specific functionality within the main kubernetes tree (here) is no longer advised; however, the proposed solution is still in development.
 
#### Guidance for potential cloud providers: 
* Support for cloud providers is currently in a state of flux. Background information on motivation and the proposal for improving is in the github [proposal](https://git.k8s.io/community/contributors/design-proposals/cloud-provider-refactoring.md). 
* In support of this plan, a new cloud-controller-manager binary was added in 1.6. This was the first of several steps (see the proposal for more information). 
* Attempts to contribute new cloud providers or (to a lesser extent) persistent volumes to the core repo will likely meet with some pushback from reviewers/approvers. 
* It is understood that this is an unfortunate situation in which 'the old way is no longer supported but the new way is not ready yet', but the initial path is unsustainable, and contributors are encouraged to participate in the implementation of the proposed long-term solution, as there is risk that PRs for new cloud providers here will not be approved. 
* Though the fully productized support envisioned in the proposal is still 2 - 3 releases out, the foundational work is underway, and a motivated cloud provider could accomplish the work in a forward-looking way. Contributors are encouraged to assist with the implementation of the design outlined in the proposal. 
 
#### Some additional context on status / direction: 
* 1.6 added a new cloud-controller-manager binary that may be used for testing the new out-of-core cloudprovider flow.
* Setting cloud-provider=external allows for creation of a separate controller-manager binary
* 1.7 adds [extensible admission control](https://git.k8s.io/community/contributors/design-proposals/admission_control_extension.md), further enabling topology customization. 
