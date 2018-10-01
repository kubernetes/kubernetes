# How to contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult [GitHub Help] for more
information on using pull requests.

[GitHub Help]: https://help.github.com/articles/about-pull-requests/

## Instructions

Fork the repo, checkout the upstream repo to your GOPATH by:

```
$ go get -d go.opencensus.io
```

Add your fork as an origin:

```
cd $(go env GOPATH)/src/go.opencensus.io
git remote add fork git@github.com:YOUR_GITHUB_USERNAME/opencensus-go.git
```

Run tests:

```
$ go test ./...
```

Checkout a new branch, make modifications and push the branch to your fork:

```
$ git checkout -b feature
# edit files
$ git commit
$ git push fork feature
```

Open a pull request against the main opencensus-go repo.
