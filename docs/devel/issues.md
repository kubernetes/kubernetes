GitHub Issues for the Kubernetes Project
========================================

A list quick overview of how we will review and prioritize incoming issues at https://github.com/GoogleCloudPlatform/kubernetes/issues

Priorities
----------

We will use GitHub issue labels for prioritization.  The absence of a priority label means the bug has not been reviewed and prioritized yet.

Priorities are "moment in time" labels, and what is low priority today, could be high priority tomorrow, and vice versa.  As we move to v1.0, we may decide certain bugs aren't actually needed yet, or that others really do need to be pulled in.

Here we define the priorities for up until v1.0.  Once the Kubernetes project hits 1.0, we will revisit the scheme and update as appropriate.

Definitions
-----------
* P0 - something broken for users, build broken, or critical security issue.  Someone must drop everything and work on it.
* P1 - must fix for earliest possible OSS binary release (every two weeks)
* P2 - must fix for v1.0 release - will block the release
* P3 - post v1.0
* untriaged - anything without a Priority/PX label will be considered untriaged