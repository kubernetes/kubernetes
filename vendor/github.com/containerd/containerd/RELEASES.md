# Versioning and Release

This document details the versioning and release plan for containerd. Stability
is a top goal for this project and we hope that this document and the processes
it entails will help to achieve that. It covers the release process, versioning
numbering, backporting, API stability and support horizons.

If you rely on containerd, it would be good to spend time understanding the
areas of the API that are and are not supported and how they impact your
project in the future.

This document will be considered a living document. Supported timelines,
backport targets and API stability guarantees will be updated here as they
change.

If there is something that you require or this document leaves out, please
reach out by [filing an issue](https://github.com/containerd/containerd/issues).

## Releases

Releases of containerd will be versioned using dotted triples, similar to
[Semantic Version](http://semver.org/). For the purposes of this document, we
will refer to the respective components of this triple as
`<major>.<minor>.<patch>`. The version number may have additional information,
such as alpha, beta and release candidate qualifications. Such releases will be
considered "pre-releases".

### Major and Minor Releases

Major and minor releases of containerd will be made from master. Releases of
containerd will be marked with GPG signed tags and announced at
https://github.com/containerd/containerd/releases. The tag will be of the
format `v<major>.<minor>.<patch>` and should be made with the command `git tag
-s v<major>.<minor>.<patch>`.

After a minor release, a branch will be created, with the format
`release/<major>.<minor>` from the minor tag. All further patch releases will
be done from that branch. For example, once we release `v1.0.0`, a branch
`release/1.0` will be created from that tag. All future patch releases will be
done against that branch.

### Pre-releases

Pre-releases, such as alphas, betas and release candidates will be conducted
from their source branch. For major and minor releases, these releases will be
done from master. For patch releases, these pre-releases should be done within
the corresponding release branch.

While pre-releases are done to assist in the stabilization process, no
guarantees are provided.

### Upgrade Path

The upgrade path for containerd is such that the 0.0.x patch releases are
always backward compatible with its major and minor version. Minor (0.x.0)
version will always be compatible with the previous minor release. i.e. 1.2.0
is backwards compatible with 1.1.0 and 1.1.0 is compatible with 1.0.0. There is
no compatibility guarantees for upgrades that span multiple, _minor_ releases.
For example, 1.0.0 to 1.2.0 is not supported. One should first upgrade to 1.1,
then 1.2.

There are no compatibility guarantees with upgrades to _major_ versions. For
example, upgrading from 1.0.0 to 2.0.0 may require resources to migrated or
integrations to change. Each major version will be supported for at least 1
year with bug fixes and security patches.

### Next Release

The activity for the next release will be tracked in the
[milestones](https://github.com/containerd/containerd/milestones). If your
issue or PR is not present in a milestone, please reach out to the maintainers
to create the milestone or add an issue or PR to an existing milestone.

### Support Horizon

Support horizons will be defined corresponding to a release branch, identified
by `<major>.<minor>`. Releases branches will be in one of several states:

- __*Next*__: The next planned release branch.
- __*Active*__: The release is currently supported and accepting patches.
- __*End of Life*__: The release branch is no longer support and no new patches will be accepted.

Releases will be supported up to one year after a _minor_ release. This means that
we will accept bug reports and backports to release branches until the end of
life date. If no new _minor_ release has been made, that release will be
considered supported until the next _minor_ is released or one year, whichever
is longer.

The current state is available in the following table:

| Release | Status      | Start            | End of Life       |
|---------|-------------|------------------|-------------------|
| [0.0](https://github.com/containerd/containerd/releases/tag/0.0.5)  | End of Life | Dec 4, 2015  | - |
| [0.1](https://github.com/containerd/containerd/releases/tag/v0.1.0) | End of Life | Mar 21, 2016 | - |
| [0.2](https://github.com/containerd/containerd/tree/v0.2.x)         | Active | Apr 21, 2016      | Upon 1.0 release |
| 1.0     | Next   | TBD  | max(TBD+1 year, release of 1.1.0) |

Note that branches and release from before 1.0 may not follow these rules.

This table should be updated as part of the release preparation process.

### Backporting

Backports in containerd are community driven. As maintainers, we'll try to
ensure that sensible bugfixes make it into _active_ release, but our main focus
will be features for the next _minor_ or _major_ release. For the most part,
this process is straightforward and we are here to help make it as smooth as
possible.

If there are important fixes that need to be backported, please let use know in
one of three ways:

1. Open an issue.
2. Open a PR with cherry-picked change from master.
3. Open a PR with a ported fix.

__If you are reporting a security issue, please reach out discreetly at security@containerd.io__.
Remember that backported PRs must follow the versioning guidelines from this document.

Any release that is "active" can accept backports. Opening a backport PR is
fairly straightforward. The steps differ depending on whether you are pulling
a fix from master or need to draft a new commit specific to a particular
branch.

To cherry pick a straightforward commit from master, simply use the cherry pick
process:

1. Pick the branch to which you want backported, usually in the format
   `release/<minor>.<major>`. The following will create a branch you can
   use to open a PR:

	```console
	$ git checkout -b my-backport-branch release/<major>.<minor>.
	```

2. Find the commit you want backported.
3. Apply it to the release branch:

	```console
	$ git cherry-pick -xsS <commit>
	```
4. Push the branch and open up a PR against the _release branch_:

	```
	$ git push -u stevvooe my-backport-branch
	```

   Make sure to replace `stevvooe` with whatever fork you are using to open
   the PR. When you open the PR, make sure to switch `master` with whatever
   release branch you are targeting with the fix.

If there is no existing fix in master, you should first fix the bug in master,
or ask us a maintainer or contributor to do it via an issue. Once that PR is
completed, open a PR using the process above.

Only when the bug is not seen in master and must be made for the specific
release branch should you open a PR with new code.

## Public API Stability

The following table provides an overview of the components covered by
containerd versions:


| Component     | Status   | Stablized Version | Links         |
|---------------|----------|-------------------|---------------|
| GRPC API      | Beta     | 1.0               | [api/](api) |
| Metrics API   | Beta     | 1.0               | -
| Go client API | Unstable | 1.1 tentative     | [godoc](https://godoc.org/github.com/containerd/containerd) |
| `ctr` tool    | Unstable | Out of scope      | -             |

From the version stated in the above table, that component must adhere to the
stability constraints expected in release versions.

Unless explicitly stated here, components that are called out as unstable or
not covered may change in a future minor version. Breaking changes to
"unstable" components will be avoided in patch versions.

### GRPC API

The primary product of containerd is the GRPC API. As of the 1.0.0 release, the
GRPC API will not have any backwards incompatible changes without a _major_
version jump.

To ensure compatibility, we have collected the entire GRPC API symbol set into
a single file. At each _minor_ release of containerd, we will move the current
`next.pb.txt` file to a file named for the minor version, such as `1.0.pb.txt`,
enumerating the support services and messages. See [api/](api) for details.

Note that new services may be added in _minor_ releases. New service methods
and new fields on messages may be added if they are optional.

### Metrics API

The metrics API that outputs prometheus style metrics will be versioned independently,
prefixed with the API version. i.e. `/v1/metrics`, `/v2/metrics`.

The metrics API version will be incremented when breaking changes are made to the prometheus
output. New metrics can be added to the output in a backwards compatible manner without
bumping the API version.

#### Error Codes

Error codes will not change in a patch release, unless a missing error code
causes a blocking bug. Error codes of type "unknown" may change to more
specific types in the future. Any error code that is not "unknown" that is
currently returned by a service will not change without a _major_ release or a
new version of the service.

If you find that an error code that is required by your application is not
well-documented in the protobuf service description or tested explicitly,
please file and issue and we will clarify.

#### Opaque Fields

Unless explicitly stated, the formats of certain fields may not be covered by
this guarantee and should be treated opaquely. For example, don't rely on the
format details of a URL field unless we explicitly say that the field will
follow that format.

### Go client API

The Go client API, documented in
[godoc](https://godoc.org/github.com/containerd/containerd), is currently
considered unstable. It is recommended to vendor the necessary components to
stabilize your project build. Note that because the Go API interfaces with the
GRPC API, clients written against a 1.0 Go API should remain compatible with
future 1.x series releases.

We intend to stabilize the API in a future release when more integrations have
been carried out.

Any changes to the API should be detectable at compile time, so upgrading will
be a matter of fixing compilation errors and moving from there.

### `ctr` tool

The `ctr` tool provides the ability to introspect and understand the containerd
API. At this time, it is not considered a primary offering of the project. It
may be completely refactored or have breaking changes in _minor_ releases.

We will try not break the tool in _patch_ releases.

### Not Covered

As a general rule, anything not mentioned in this document is not covered by
the stability guidelines and may change in any release. Explicitly, this
pertains to this non-exhaustive list of components:

- File System layout
- Storage formats
- Snapshot formats

Between upgrades of subsequent, _minor_ versions, we may migrate these formats.
Any outside processes relying on details of these file system layouts may break
in that process. Container root file systems will be maintained on upgrade.

### Exceptions

We may make exceptions in the interest of __security patches__. If a break is
required, it will be communicated clearly and the solution will be considered
against total impact.
