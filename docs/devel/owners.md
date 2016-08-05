<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/owners.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Owners files

_Note_: This is a design for a feature that is not yet implemented.

## Overview

We want to establish owners for different parts of the code in the Kubernetes codebase.  These owners
will serve as the approvers for code to be submitted to these parts of the repository.  Notably, owners
are not necessarily expected to do the first code review for all commits to these areas, but they are
required to approve changes before they can be merged.

## High Level flow

### Step One: A PR is submitted

After a PR is submitted, the automated kubernetes PR robot will append a message to the PR indicating the owners
that are required for the PR to be submitted.

Subsequently, a user can also request the approval message from the robot by writing:

```
@k8s-bot approvers
```

into a comment.

In either case, the automation replies with an annotation that indicates
the owners required to approve.  The annotation is a comment that is applied to the PR.
This comment will say:

```
Approval is required from <owner-a> OR <owner-b>, AND <owner-c> OR <owner-d>, AND ...
```

The set of required owners is drawn from the OWNERS files in the repository (see below).  For each file
there should be multiple different OWNERS, these owners are listed in the `OR` clause(s). Because
it is possible that a PR may cover different directories, with disjoint sets of OWNERS, a PR may require
approval from more than one person, this is where the `AND` clauses come from.

`<owner-a>` should be the github user id of the owner _without_ a leading `@` symbol to prevent the owner
from being cc'd into the PR by email.

### Step Two: A PR is LGTM'd

Once a PR is reviewed and LGTM'd it is eligible for submission.  However, for it to be submitted
an owner for all of the files changed in the PR have to 'approve' the PR.  A user is an owner for a
file if they are included in the OWNERS hierarchy (see below) for that file.

Owner approval comes in two forms:

   * An owner adds a comment to the PR saying "I approve" or "approved"
   * An owner is the original author of the PR

In the case of a comment based approval, the same rules as for the 'lgtm' label apply.  If the PR is
changed by pushing new commits to the PR, the previous approval is invalidated, and the owner(s) must
approve again.  Because of this is recommended that PR authors squash their PRs prior to getting approval
from owners.

### Step Three: A PR is merged

Once a PR is LGTM'd and all required owners have approved, it is eligible for merge.  The merge bot takes care of
the actual merging.

## Design details

We need to build new features into the existing github munger in order to accomplish this.  Additionally
we need to add owners files to the repository.

### Approval Munger

We need to add a munger that adds comments to PRs indicating whose approval they require.  This munger will
look for PRs that do not have approvers already present in the comments, or where approvers have been
requested, and add an appropriate comment to the PR.


### Status Munger

GitHub has a [status api](https://developer.github.com/v3/repos/statuses/), we will add a status munger that pushes a status onto a PR of approval status.  This status will only be approved if the relevant
approvers have approved the PR.

### Requiring approval status

Github has the ability to [require status checks prior to merging](https://help.github.com/articles/enabling-required-status-checks/)

Once we have the status check munger described above implemented, we will add this required status check
to our main branch as well as any release branches.

### Adding owners files

In each directory in the repository we may add an OWNERS file.  This file will contain the github OWNERS
for that directory.  OWNERSHIP is hierarchical, so if a directory does not container an OWNERS file, its
parent's OWNERS file is used instead.  There will be a top-level OWNERS file to back-stop the system.

Obviously changing the OWNERS file requires OWNERS permission.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/owners.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
