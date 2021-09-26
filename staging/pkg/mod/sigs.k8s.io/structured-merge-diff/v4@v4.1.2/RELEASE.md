# Release Process

This repo will follow go library versioning semantics.

Currently, it is not stable (version 0.0.0) and interfaces may change without
warning.

Once it looks like this code will be used in a Kubernetes release, we will mark
it v1.0.0 and any interface changes will begin accumulating in a v2 candidate.

We will publish versions in a way that's conformant with the new "go modules".

Reviewers / owners are expected to be vigilant about preventing
interface-breaking changes in stable versions.

When a candidate version is ready to be promoted to stable, the process is as follows:

1. An issue is proposing a new release with a changelog since the last release
1. All [OWNERS](OWNERS) must LGTM this release
1. An OWNER changes the name from vX-candidate to vX and starts a v(X+1)-candidate directory/module (details TBD when we first do this)
1. The release issue is closed
1. An announcement email is sent to `kubernetes-dev@googlegroups.com` with the subject `[ANNOUNCE] kubernetes-template-project $VERSION is released`

(This process is currently intended to be a hint and will be refined once we declare our first stable release.)
