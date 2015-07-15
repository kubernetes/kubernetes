<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
## Making release notes
This documents the process for making release notes for a release.

### 1) Note the PR number of the previous release
Find the most-recent PR that was merged with the previous .0 release.  Remember this as $LASTPR.
_TODO_: Figure out a way to record this somewhere to save the next release engineer time.

Find the most-recent PR that was merged with the current .0 release.  Remeber this as $CURRENTPR.

### 2) Run the release-notes tool
```bash
${KUBERNETES_ROOT}/build/make-release-notes.sh $LASTPR $CURRENTPR
```

### 3) Trim the release notes
This generates a list of the entire set of PRs merged since the last minor
release.  It is likely long and many PRs aren't worth mentioning.  If any of the
PRs were cherrypicked into patches on the last minor release, you should exclude
them from the current release's notes.

Open up ```candidate-notes.md``` in your favorite editor.

Remove, regroup, organize to your hearts content.


### 4) Update CHANGELOG.md
With the final markdown all set, cut and paste it to the top of ```CHANGELOG.md```

### 5) Update the Release page
   * Switch to the [releases](https://github.com/GoogleCloudPlatform/kubernetes/releases) page.
   * Open up the release you are working on.
   * Cut and paste the final markdown from above into the release notes
   * Press Save.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/making-release-notes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
