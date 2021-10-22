# Setup from scratch

1. [Install Go](https://golang.org/dl/).
    1. Ensure that your `GOBIN` directory (by default `$(go env GOPATH)/bin`)
    is in your `PATH`.
    1. Check it's working by running `go version`.
        * If it doesn't work, check the install location, usually
        `/usr/local/go`, is on your `PATH`.

1. Sign one of the
[contributor license agreements](#contributor-license-agreements) below.

1. Run `go get golang.org/x/review/git-codereview && go install golang.org/x/review/git-codereview`
to install the code reviewing tool.

    1. Ensure it's working by running `git codereview` (check your `PATH` if
    not).

    1. If you would like, you may want to set up aliases for `git-codereview`,
    such that `git codereview change` becomes `git change`. See the
    [godoc](https://pkg.go.dev/golang.org/x/review/git-codereview) for details.

        * Should you run into issues with the `git-codereview` tool, please note
        that all error messages will assume that you have set up these aliases.

1. Change to a directory of your choosing and clone the repo.

    ```
    cd ~/code
    git clone https://code.googlesource.com/gocloud
    ```

    * If you have already checked out the source, make sure that the remote
    `git` `origin` is https://code.googlesource.com/gocloud:

        ```
        git remote -v
        # ...
        git remote set-url origin https://code.googlesource.com/gocloud
        ```

    * The project uses [Go Modules](https://blog.golang.org/using-go-modules)
    for dependency management See
    [`gopls`](https://github.com/golang/go/wiki/gopls) for making your editor
    work with modules.

1. Change to the project directory and add the github remote:

    ```
    cd ~/code/gocloud
    git remote add github https://github.com/googleapis/google-cloud-go
    ```

1. Make sure your `git` auth is configured correctly by visiting
https://code.googlesource.com, clicking "Generate Password" at the top-right,
and following the directions. Otherwise, `git codereview mail` in the next step
will fail.

# Which module to release?

The Go client libraries have several modules. Each module does not strictly
correspond to a single library - they correspond to trees of directories. If a
file needs to be released, you must release the closest ancestor module.

To see all modules:

```
$ cat `find . -name go.mod` | grep module
module cloud.google.com/go
module cloud.google.com/go/bigtable
module cloud.google.com/go/firestore
module cloud.google.com/go/bigquery
module cloud.google.com/go/storage
module cloud.google.com/go/datastore
module cloud.google.com/go/pubsub
module cloud.google.com/go/spanner
module cloud.google.com/go/logging
```

The `cloud.google.com/go` is the repository root module. Each other module is
a submodule.

So, if you need to release a change in `bigtable/bttest/inmem.go`, the closest
ancestor module is `cloud.google.com/go/bigtable` - so you should release a new
version of the `cloud.google.com/go/bigtable` submodule.

If you need to release a change in `asset/apiv1/asset_client.go`, the closest
ancestor module is `cloud.google.com/go` - so you should release a new version
of the `cloud.google.com/go` repository root module. Note: releasing
`cloud.google.com/go` has no impact on any of the submodules, and vice-versa.
They are released entirely independently.

# How to release `cloud.google.com/go`

1. Navigate to `~/code/gocloud/` and switch to master.
1. `git pull`
1. Run `git tag -l | grep -v beta | grep -v alpha` to see all existing releases.
   The current latest tag `$CV` is the largest tag. It should look something
   like `vX.Y.Z` (note: ignore all `LIB/vX.Y.Z` tags - these are tags for a
   specific library, not the module root). We'll call the current version `$CV`
   and the new version `$NV`.
1. On master, run `git log $CV...` to list all the changes since the last
   release. NOTE: You must manually visually parse out changes to submodules [1]
   (the `git log` is going to show you things in submodules, which are not going
   to be part of your release).
1. Edit `CHANGES.md` to include a summary of the changes.
1. `cd internal/version && go generate && cd -`
1. Mail the CL: `git add -A && git change <branch name> && git mail`
1. Wait for the CL to be submitted. Once it's submitted, and without submitting
   any other CLs in the meantime:
   a. Switch to master.
   b. `git pull`
   c. Tag the repo with the next version: `git tag $NV`.
   d. Push the tag to both remotes:
      `git push origin $NV`
      `git push github $NV`
1. Update [the releases page](https://github.com/googleapis/google-cloud-go/releases)
   with the new release, copying the contents of `CHANGES.md`.

# How to release a submodule

We have several submodules, including `cloud.google.com/go/logging`,
`cloud.google.com/go/datastore`, and so on.

To release a submodule:

(these instructions assume we're releasing `cloud.google.com/go/datastore` - adjust accordingly)

1. Navigate to `~/code/gocloud/` and switch to master.
1. `git pull`
1. Run `git tag -l | grep datastore | grep -v beta | grep -v alpha` to see all
   existing releases. The current latest tag `$CV` is the largest tag. It
   should look something like `datastore/vX.Y.Z`. We'll call the current version
   `$CV` and the new version `$NV`.
1. On master, run `git log $CV.. -- datastore/` to list all the changes to the
   submodule directory since the last release.
1. Edit `datastore/CHANGES.md` to include a summary of the changes.
1. `cd internal/version && go generate && cd -`
1. Mail the CL: `git add -A && git change <branch name> && git mail`
1. Wait for the CL to be submitted. Once it's submitted, and without submitting
   any other CLs in the meantime:
   a. Switch to master.
   b. `git pull`
   c. Tag the repo with the next version: `git tag $NV`.
   d. Push the tag to both remotes:
      `git push origin $NV`
      `git push github $NV`
1. Update [the releases page](https://github.com/googleapis/google-cloud-go/releases)
   with the new release, copying the contents of `datastore/CHANGES.md`.

# Appendix

1: This should get better as submodule tooling matures.
