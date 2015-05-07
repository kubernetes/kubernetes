## Making release notes
This documents the process for making release notes for a release.

### 1) Note the PR number of the previous release
Find the PR that was merged with the previous release.  Remember this number
_TODO_: Figure out a way to record this somewhere to save the next release engineer time.

### 2) Build the release-notes tool
```bash
${KUBERNETES_ROOT}/build/make-release-notes.sh <pr-number-from-1>
```

### 3) Trim the release notes
This generates a list of the entire set of PRs merged since the last release.  It is likely long
and many PRs aren't worth mentioning.

Open up ```candidate-notes.md``` in your favorite editor.

Remove, regroup, organize to your hearts content.


### 4) Update CHANGELOG.md
With the final markdown all set, cut and paste it to the top of ```CHANGELOG.md```

### 5) Update the Release page
   * Switch to the [releases](https://github.com/GoogleCloudPlatform/kubernetes/releases) page.
   * Open up the release you are working on.
   * Cut and paste the final markdown from above into the release notes
   * Press Save.

