# Maintaining openshift/kubernetes

OpenShift is based on upstream Kubernetes. With every release of Kubernetes that is
intended to be shipped as OCP, it is necessary to incorporate the upstream changes
while ensuring that our downstream customizations are maintained.

## Rebasing for releases < 4.6

The instructions in this document apply to OpenShift releases 4.6 and
above. For previous releases, please see the [rebase
enhancement](https://github.com/openshift/enhancements/blob/master/enhancements/rebase.md).

## Getting started

Before incorporating upstream changes you may want to:

- Read this document
- Get familiar with tig (text-mode interface for git)
- Find the best tool for resolving merge conflicts
- Use diff3 conflict resolution strategy
   (https://blog.nilbus.com/take-the-pain-out-of-git-conflict-resolution-use-diff3/)
- Teach Git to remember how you’ve resolved a conflict so that the next time it can
  resolve it automatically (https://git-scm.com/book/en/v2/Git-Tools-Rerere)

## Preparing the local repo clone

Clone from a personal fork of kubernetes via a pushable (ssh) url:

```
git clone git@github.com:<user id>/kubernetes
```

Add a remote for upstream and fetch its branches:

```
git remote add --fetch upstream https://github.com/kubernetes/kubernetes
```

Add a remote for the openshift fork and fetch its branches:

```
git remote add --fetch openshift https://github.com/openshift/kubernetes
```

## Creating a new local branch for the new rebase

- Branch the target `k8s.io/kubernetes` release tag (e.g. `v1.20.0`) to a new
  local branch

```
git checkout -b rebase-1.20.0 v1.20.0
```

- Merge the targeted `openshift/kubernetes` branch (e.g. `master`) with
  strategy `ours` to reset the the branch to the targeted release tag without
  involving manual conflict resolution.

```
git merge -s ours openshift/master
```

## Creating a spreadsheet of carry commits from the previous release

Given the upstream tag (e.g. `v1.19.2`) of the most recent rebase and the name
of the branch that is targeted for rebase (e.g. `master`), generate a csv file
containing the set of carry commits that need to be considered for picking:

```
git log $( git merge-base master v1.19.2 )..master \
 --pretty=format:',%H,%s,https://github.com/openshift/kubernetes/commit/%H' | \
  grep -v 'Merge pull request' | \
  sed 's#,UPSTREAM: \([0-9]*\)\(:.*\)#,UPSTREAM: \1\2,https://github.com/kubernetes/kubernetes/pull/\1#' > \
  v1.19.2.csv
```

This csv file can be imported into a google sheets spreadsheet to track the
progress of picking commits to the new rebase branch. The spreadsheet can also
be a way of communicating with rebase reviewers. For an example of this
communication, please see the [the spreadsheet used for the 1.19
rebase](https://docs.google.com/spreadsheets/d/10KYptJkDB1z8_RYCQVBYDjdTlRfyoXILMa0Fg8tnNlY/edit).

## Picking commits from the previous rebase branch to the new branch

Commits carried on rebase branches have commit messages prefixed as follows:

- `UPSTREAM: <carry>:`
  - A persistent carry that should probably be picked for the subsequent rebase branch.
  - In general, these commits are used to modify behavior for consistency or
    compatibility with openshift.
- `UPSTREAM: <drop>:`
  - A carry that should probably not be picked for the subsequent rebase branch.
  - In general, these commits are used to maintain the codebase in ways that are
    branch-specific, like the update of generated files or dependencies.
- `UPSTREAM: 77870:`
  - The number identifies a PR in upstream kubernetes
    (i.e. `https://github.com/kubernetes/kubernetes/pull/<pr id>`)
  - A commit with this message should only be picked into the subsequent rebase branch
    if the commits of the referenced PR are not included in the upstream branch.
  - To check if a given commit is included in the upstream branch, open the referenced
    upstream PR and check any of its commits for the release tag (e.g. `v.1.20.0`)
    targeted by the new rebase branch. For example:
    - <img src="openshift-hack/commit-tag.png">

With these guidelines in mind, pick the appropriate commits from the previous rebase
branch into the new rebase branch. As per the example of previous rebase spreadsheets,
color each commit in the spreadsheet to indicate to reviewers whether or not a commit
was picked and the rationale for your choice.

Where it makes sense to do so, squash carried changes that are tightly coupled to
simplify future rebases. If the commit message of a carry does not conform to
expectations, feel free to revise and note the change in the spreadsheet row for the
commit.

## Update the hyperkube image version to the release tag

The [hyperkube dockerfile](openshift-hack/images/hyperkube/Dockerfile.rhel)
hard-codes the Kubernetes version in an image label. It's necessary to manually
set this label to the new release tag. Prefix the commit summary with
`UPSTREAM: <drop>:` since a future rebase will need to add its own commit.

## Updating dependencies

Once the commits are all picked from the previous rebase branch, each of the
following repositories need to be updated to depend on the upstream tag
targeted by the rebase:

- https://github.com/openshift/api
- https://github.com/openshift/apiserver-library-go
- https://github.com/openshift/client-go
- https://github.com/openshift/library-go

Often these repositories are updated in parallel by other team members, so make
sure to ask around before starting the work of bumping their dependencies.

Once the above repos have been updated to the target release, it will be necessary to
update go.mod to point to the appropriate revision of these repos by running
`hack/pin-dependency.sh` for each of them and then running `hack/update-vendor.sh` (as
per the [upstream
documentation](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/vendor.md#adding-or-updating-a-dependency)).

Make sure to commit the result of a vendoring update with `UPSTREAM: <drop>: bump(*)`.

### Updating dependencies for pending bumps

The upstream `hack/pin-dependency.sh` script only supports setting dependency
for the original repository. To pin to a fork branch that has not yet been
merged (i.e. to test a rebase ahead of shared library bumps having merged), the
following `go mod` invovations are suggested:

```
go mod edit -replace github.com/openshift/<lib>=github.com/<username>/<lib>@SHA
go mod tidy
```

## Review test annotation rules

The names of upstream e2e tests are annotated according to the a set of
[declarative rules](openshift-hack/e2e/annotate/rules.go). These annotations
are used to group tests into suites and to skip tests that are known not to be
incompatible with some or all configurations of OpenShift.

When performing a rebase, it is important to review the rules to
ensure they are still relevant:

- [ ] Ensure that `[Disabled:Alpha]` rules are appropriate for the current kube
      level. Alpha features that are not enabled by default should be targeted
      by this annotation to ensure that tests of those features are skipped.
- [ ] Add new skips (along with a bz to track resolution) where e2e tests fail
      consistently.

Test failures representing major issues affecting cluster capability will
generally need to be addressed before merge of the rebase PR, but minor issues
(e.g. tests that fail to execute correctly but don't appear to reflect a
regression in behavior) can often be skipped and addressed post-merge.

## Updating generated files

- Update generated files by running `make update`
  - This step depends on etcd being installed in the path, which can be
    accomplished by running `hack/install-etcd.sh`.
- Commit the resulting changes as `UPSTREAM: <drop>: make update`.

## Building and testing

- Build the code with `make`
- Test the code with `make test`
  - Where test failures are encountered and can't be trivially resolved, the
    spreadsheet can be used to to track those failures to their resolution. The
    example spreadsheet should have a sheet that demonstrates this tracking.
  - Where a test failure proves challenging to fix without specialized knowledge,
    make sure to coordinate with the team(s) responsible for areas of focus
    exhibiting test failure. If in doubt, ask for help!
- Verify the code with `make verify`

### Rebase Checklists

In preparation for submitting a PR to the [openshift fork of
kubernetes](https://github.com/openshift/kubernetes), the following
should be true:

- [ ] The new rebase branch has been created from the upstream tag
- [ ] The new rebase branch includes relevant carries from target branch
- [ ] Dependencies have been updated
- [ ] Hyperkube dockerfile version has been updated
- [ ] `make update` has been invoked and the results committed
- [ ] `make` executes without error
- [ ] `make verify` executes without error
- [ ] `make test` executes without error
- [ ] Upstream tags are pushed to `openshift/kubernetes` to ensure that
      build artifacts are versioned correctly
      - Upstream tooling uses the value of the most recent tag (e.g. `v1.20.0`)
        in the branch history as the version of the binaries it builds.

Details to include in the description of the PR:

- [ ] A link to the rebase spreadsheet for the benefit for reviewers
- [ ] A comment reminding reviewers of the need for manual upgrade testing
      along with a `/hold` command to prevent merge until such testing is
      completed.

In addition to the standard requirement that all CI jobs be passing, the rebase
PR should not be merged until additional upgrade testing initiated with
cluster-bot is passing:

- [ ] `test upgrade [previous release e.g. 4.6] openshift/kubernetes#[PR#] [aws|azure]`
       - Only gcp upgrades are tested automatically via presubmit
         (`e2e-gcp-upgrade`) and it's necessary to manually test aws and azure.
- [ ] `test upgrade openshift/kubernetes#[PR#] openshift/kubernetes#[PR#]`
      - This 'self-upgrade' ensures that it is possible to upgrade _from_ the
        rebased release. The other upgrade testing validates that it's possible
        to upgrade _to_ the rebased release.

After the rebase PR has merged to `openshift/kubernetes`, vendor the changes
into `origin` to ensure that the openshift-tests binary reflects the upstream
test changes introduced by the rebase:

- [ ] Find the SHA of `openshift/kubernetes` branch after merge of the rebase
      PR
- [ ] Run `hack/update-kube-vendor.sh <o/k SHA>` in a clone of the `origin`
      repo and commit the results
- [ ] Run `make update` and commit the results
- [ ] Submit as a PR to `origin`

As a final step, send an email to the aos-devel mailing list announcing the
rebase. Make sure to include:

- [ ] The new version of upstream Kubernetes that OpenShift is now based on
- [ ] Link(s) to upstream changelog(s) detailing what has changed since the last rebase landed
- [ ] A reminder to component maintainers to bump their dependencies
- [ ] Relevent details of the challenges involved in landing the rebase that
      could benefit from a wider audience.
