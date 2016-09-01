<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Using godep to manage dependencies

This document is intended to show a way for managing `vendor/` tree dependencies
in Kubernetes. If you are not planning on managing `vendor` dependencies go here
[Godep dependency management](development.md#godep-dependency-management).

## Alternate GOPATH for installing and using godep

There are many ways to build and host Go binaries. Here is one way to get
utilities like `godep` installed:

Create a new GOPATH just for your go tools and install godep:

```sh
export GOPATH=$HOME/go-tools
mkdir -p $GOPATH
go get -u github.com/tools/godep
```

Add this $GOPATH/bin to your path. Typically you'd add this to your ~/.profile:

```sh
export GOPATH=$HOME/go-tools
export PATH=$PATH:$GOPATH/bin
```

## Using godep

Here's a quick walkthrough of one way to use godeps to add or update a
Kubernetes dependency into `vendor/`. For more details, please see the
instructions in [godep's documentation](https://github.com/tools/godep).

1) Devote a directory to this endeavor:

_Devoting a separate directory is not strictly required, but it is helpful to
separate dependency updates from other changes._

```sh
export KPATH=$HOME/code/kubernetes
mkdir -p $KPATH/src/k8s.io
cd $KPATH/src/k8s.io
git clone https://github.com/$YOUR_GITHUB_USERNAME/kubernetes.git # assumes your fork is 'kubernetes'
# Or copy your existing local repo here. IMPORTANT: making a symlink doesn't work.
```

2) Set up your GOPATH.

```sh
# This will *not* let your local builds see packages that exist elsewhere on your system.
export GOPATH=$KPATH
```

3) Populate your new GOPATH.

```sh
cd $KPATH/src/k8s.io/kubernetes
godep restore
```

4) Next, you can either add a new dependency or update an existing one.

To add a new dependency is simple (if a bit slow):

```sh
cd $KPATH/src/k8s.io/kubernetes
DEP=example.com/path/to/dependency
godep get $DEP/...
# Now change code in Kubernetes to use the dependency.
./hack/godep-save.sh
```

To update an existing dependency is a bit more complicated.  Godep has an
`update` command, but none of us can figure out how to actually make it work.
Instead, this procedure seems to work reliably:

```sh
cd $KPATH/src/k8s.io/kubernetes
DEP=example.com/path/to/dependency
# NB: For the next step, $DEP is assumed be the repo root.  If it is actually a
# subdir of the repo, use the repo root here.  This is required to keep godep
# from getting angry because `godep restore` left the tree in a "detached head"
# state.
rm -rf $KPATH/src/$DEP # repo root
godep get $DEP/...
# Change code in Kubernetes, if necessary.
rm -rf Godeps
rm -rf vendor
./hack/godep-save.sh
git co -- $(git st -s | grep "^ D" | awk '{print $2}' | grep ^Godeps)
```

_If `go get -u path/to/dependency` fails with compilation errors, instead try
`go get -d -u path/to/dependency` to fetch the dependencies without compiling
them. This is unusual, but has been observed._

After all of this is done, `git status` should show you what files have been
modified and added/removed.  Make sure to `git add` and `git rm` them.  It is
commonly advised to make one `git commit` which includes just the dependency
update and Godeps files, and another `git commit` that includes changes to
Kubernetes code to use the new/updated dependency.  These commits can go into a
single pull request.

5) Before sending your PR, it's a good idea to sanity check that your
Godeps.json file and the contents of `vendor/ `are ok by running `hack/verify-godeps.sh`

_If `hack/verify-godeps.sh` fails after a `godep update`, it is possible that a
transitive dependency was added or removed but not updated by godeps. It then
may be necessary to perform a `hack/godep-save.sh` to pick up the transitive
dependency changes._

It is sometimes expedient to manually fix the /Godeps/Godeps.json file to
minimize the changes. However without great care this can lead to failures
with `hack/verify-godeps.sh`. This must pass for every PR.

6) If you updated the Godeps, please also update `Godeps/LICENSES` by running
`hack/update-godep-licenses.sh`.






<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/godep.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
