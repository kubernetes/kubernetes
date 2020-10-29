# OpenShift's fork of k8s.io/kubernetes

This respository contains core Kubernetes components with OpenShift-specific patches.

## Cherry-picking an upstream commit into openshift/kubernetes: Why, how, and when.

`openshift/kubernetes` carries patches on top of each rebase in one of two ways:

1. *periodic rebases* against an upstream Kubernetes tag.  Eventually,
any code you have in upstream Kubernetes will land in Openshift via
this mechanism.

2. Cherry-picked patches for important *bug fixes*.  We really try to
limit feature back-porting entirely. Unless there are exceptional circumstances, your backport should at least be merged in kubernetes master branch. With every carry patch (not included in upstream) you are introducing a maintenance burden for the team managing rebases.

### For Openshift newcomers: Pick my Kubernetes fix into Openshift vs. wait for the next rebase?

Assuming you read the bullets above... If your patch is really far behind, for
example, if there have been 5 commits modifying the directory you care about,
cherry picking will be increasingly difficult and you should consider waiting
for the next rebase, which will likely include the commit you care about or at
least decrease the amount of cherry picks you need to do to merge.

To really know the answer, you need to know *how many commits behind you are in
a particular directory*, often.

To do this, just use git log, like so (using pkg/scheduler/ as an example).

```
MYDIR=pkg/scheduler/algorithm git log --oneline --
  ${MYDIR} | grep UPSTREAM | cut -d' ' -f 4-10 | head -1
```

The commit message printed above will tell you:

- what the LAST commit in Kubernetes was (which effected
"/pkg/scheduler/algorithm")
- directory, which will give you an intuition about how "hot" the code you are
cherry picking is.  If it has changed a lot, recently, then that means you
probably will want to wait for a rebase to land.

### Cherry-picking an upstream change

Since `openshift/kubernetes` closely resembles `k8s.io/kubernetes`,
cherry-picking largely involves proposing upstream commits in a PR to our
downstream fork. Other than the usual potential for merge conflicts, the
commit messages for all commits proposed to `openshift/kubernetes` must
reflect the following:

- `UPSTREAM: <UPSTREAM PR ID>:` The prefix for upstream commits to ensure
  correct handling during a future rebase. The person performing the rebase
  will know to omit a commit with this prefix if the referenced PR is already
  present in the new base history.
- `UPSTREAM: <drop>:` The prefix for downstream commits of code that is
  generated (i.e. via `make update`) or that should not be retained by the
  next rebase.
- `UPSTREAM: <carry>:` The prefix for downstream commits that maintain
  downstream-specific behavior (i.e. to ensure an upstream change is
  compatible with OpenShift). Commits with this are usually retained across
  rebases.

## Updating openshift/kubernetes to a new upstream release

Instructions for rebasing `openshift/kubernetes` are maintained in a [separate
document](REBASE.openshift.md).

## RPM Packaging

A specfile is included in this repo which can be used to produce RPMs
including the openshift binary. While the specfile will be kept up to
date with build requirements the version is not updated. Building the
rpm with the `openshift-hack/build-rpms.sh` helper script will ensure
that the version is set correctly.
