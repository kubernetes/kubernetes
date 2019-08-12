<!--  
Thanks for submitting a pull request!

If this is your first time submitting a PR, please familiarise yourself with:

  Contributor Guidelines 
  https://git.k8s.io/community/contributors/guide#your-first-contribution

  Developer Guide 
  https://git.k8s.io/community/contributors/devel/development.md#development-guide

The following are additional useful links for PRs:

  Best Practices for Faster Reviews 
  https://git.k8s.io/community/contributors/guide/pull-requests.md#best-practices-for-faster-reviews

  Marking Unfinished PRs for Review or as a Work in Progress (WIP)
  https://git.k8s.io/community/contributors/guide/pull-requests.md#marking-unfinished-pull-requests
-->

**Which is the sponsoring or primary SIG associated with this PR?**
<!--
Label the sponsoring or primary SIG associated with this PR.

For reference on available SIGs, you can find more details at:
https://github.com/kubernetes/test-infra/blob/master/label_sync/labels.md#labels-that-apply-to-all-repos-for-both-issues-and-prs

Add only ONE sig here via a /sig-primary <name> entry.
-->
/sig-primary ???

> DISCUSSION REQUIRED
>
> Should we use the above pattern and update the label prow plugin ? https://github.com/kubernetes/test-infra/tree/master/prow/plugins/label. Logic would have to ensure that there is only a single primary sig associated. This metadata can be leveraged by other processes outside of the release notes.
> 
> or, if this is ONLY relevant to release notes then, we can add here as a `` ```primary-sig`` code block.

**What type of PR is this?**
<!-- 
Label this pull request according to what type of issue you are addressing, especially if this is a release targeted pull request. 

For reference on required PR labels, you can find more details at:
https://git.k8s.io/community/contributors/devel/sig-release/release.md#issuepr-kind-label

Add only ONE of the following /kind <type> entries.

/kind api-change
/kind bug
/kind cleanup
/kind design
/kind documentation
/kind failing-test
/kind feature
/kind flake
-->
/kind ???

> DISCUSSION REQUIRED
>
> Should we include a kind type of `deprecation`? And should we then include a `` ```depracation-notes`` code block to provide specific detailed notes around the depracation.
> 
> Should we only allow a single kind to be specified? There are a number of examples where multiple kind values make sense.

**What this PR does / why we need it**:
<!--
Provide a description on what this PR does and/or why it is required.
-->
Description here ...

**Which issue(s) this PR fixes**:
<!--
Add references to issues that are fixed by this PR by using one or more of the following templates:

Fixes #<issue number>
Fixes <issue url>

NOTE: The linked issue will automatically be closed when the PR is merged.

If the PR is about failing-tests or flakes, please post the related issues/tests in a comment and do not use the Fixes templates above.
-->
Fixes #???

**Have you added or run tests for your PR?**
<!--
Indicate whether you have added or run the appropriate tests for your PR by using ONE of the following responses:

N/A
Yes
No

More details can be found in the Testing Guide:
https://git.k8s.io/community/contributors/devel/sig-testing/testing.md
-->
No

**Special notes for your reviewer**:
<!--
Provide any additional notes that may be helpful to the PR reviewers.
-->
None

**What user facing changes are introduced by this PR?**:
<!--
Indicate whether there are any user facing changes that should be added to the Release Notes. 

If there are no user facing changes, leave the "None" text in the release-note block below. If your PR does introduce user facing changes, then a release note is required. Add your extended release note in the release-note block below.

If the PR requires additional action from users switching to the new release, start the release-notes block with the text "Action required".

Familiarise yourself with the instructions for writing a release note: 
https://git.k8s.io/community/contributors/guide/release-notes.md
-->
```release-note
None
```

**Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.**:
<!--
Provide links to any additional relevant documentation in the docs block below ONLY if you have added a release note, otherwise leave it blank.

The Kubernetes Release Notes website (https://relnotes.k8s.io) provides additional documentation to the end users about the changes in this PR.

If you decide to add links which point to resources within git repositories, please ensure that the appropriate revision is included in the link and not generic ones like `master`. The same applies to external documentation, which may be not available any more because it got updated to a more recent version.

Use one or more of the following templates in the docs block for linking to documentation:
- [KEP]: <url>
- [Usage]: <url>
- [Other doc]: <url>
-->
```docs

```
