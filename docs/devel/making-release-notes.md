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
[here](http://releases.k8s.io/release-1.3/docs/devel/making-release-notes.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Making release notes

This documents the process for making release notes for a release.

### 1) Note the PR number of the previous release

Find the most-recent PR that was merged with the previous .0 release. Remember
this as $LASTPR.

- _TODO_: Figure out a way to record this somewhere to save the next
release engineer time.

Find the most-recent PR that was merged with the current .0 release. Remember
this as $CURRENTPR.

### 2) Run the release-notes tool

```bash
${KUBERNETES_ROOT}/build/make-release-notes.sh $LASTPR $CURRENTPR
```

### 3) Trim the release notes

This generates a list of the entire set of PRs merged since the last minor
release. It is likely long and many PRs aren't worth mentioning. If any of the
PRs were cherrypicked into patches on the last minor release, you should exclude
them from the current release's notes.

Open up `candidate-notes.md` in your favorite editor.

Remove, regroup, organize to your hearts content.


### 4) Update CHANGELOG.md

With the final markdown all set, cut and paste it to the top of `CHANGELOG.md`

### 5) Update the Release page

   * Switch to the [releases](https://github.com/kubernetes/kubernetes/releases)
page.

   * Open up the release you are working on.

   * Cut and paste the final markdown from above into the release notes

   * Press Save.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/making-release-notes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
