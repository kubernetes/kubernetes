# Installing client-go

## For the casual user

If you want to write a simple script, don't care about a reproducible client
library install, don't mind getting head (which may be less stable than a
particular release), then simply:

```sh
$ go get k8s.io/client-go/...
```

This will install `k8s.io/client-go` in your `$GOPATH`. `k8s.io/client-go`
includes most of its own dependencies in its `k8s.io/client-go/vendor` path,
except for `k8s.io/apimachinery` and `glog`. `go get` will recursively download
these excluded repos to your `$GOPATH`, if they don't already exist. If
`k8s.io/apimachinery` preexisted in `$GOPATH`, you also need to:

```sh
$ go get -u k8s.io/apimachinery/...
```

because the head of client-go is only guaranteed to work with the head of
apimachinery.

We excluded `k8s.io/apimachinery` and `glog` from `k8s.io/client-go/vendor` to
prevent `go get` users from hitting issues like
[#19](https://github.com/kubernetes/client-go/issues/19) and
[#83](https://github.com/kubernetes/client-go/issues/83). If your project share
other dependencies with client-go, and you hit issues similar to #19 or #83,
then you'll need to look down at the next section.

Note: the official go policy is that libraries should not vendor their
dependencies. This is unworkable for us, since our dependencies change and HEAD
on every dependency has not necessarily been tested with client-go. In fact,
HEAD from all dependencies may not even compile with client-go!

## Dependency management for the serious (or reluctant) user

Reasons why you might need to use a dependency management system:
* You use a dependency that client-go also uses, and don't want two copies of
  the dependency compiled into your application. For some dependencies with
  singletons or global inits (e.g. `glog`) this wouldn't even compile...
* You want to lock in a particular version (so you don't have to change your
  code every time we change a public interface).
* You want your install to be reproducible. For example, for your CI system or
  for new team members.

There are three tools you could in theory use for this. Instructions
for each follows.

### Godep

[godep](https://github.com/tools/godep) is an older dependency management tool, which is
used by the main Kubernetes repo and `client-go` to manage dependencies.

Before proceeding with the below instructions, you should ensure that your
$GOPATH is empty except for containing your own package and its dependencies,
and you have a copy of godep somewhere in your $PATH.

To install `client-go` and place its dependencies in your `$GOPATH`:

```sh
go get k8s.io/client-go/...
cd $GOPATH/src/k8s.io/client-go
git checkout v9.0.0 # replace v9.0.0 with the required version
# cd 1.5 # only necessary with 1.5 and 1.4 clients.
godep restore ./...
```

At this point, `client-go`'s dependencies have been placed in your $GOPATH, but
if you were to build, `client-go` would still see its own copy of its
dependencies in its `vendor` directory. You have two options at this point.

If you would like to keep dependencies in your own project's vendor directory,
then you can continue like this:

```sh
cd $GOPATH/src/<my-pkg>
godep save ./...
```

Alternatively, if you want to build using the dependencies in your `$GOPATH`,
then `rm -rf vendor/` to remove `client-go`'s copy of its dependencies.

### Glide

[Glide](https://github.com/Masterminds/glide) is another popular dependency
management tool for Go. Glide will manage your /vendor directory, but unlike
godep, will not use or modify your $GOPATH (there's no equivalent of
`godep restore` or `godep save`).

Generally, it's best to avoid Glide's many subcommands, favoring modifying
Glide's manifest file (`glide.yaml`) directly, then running
`glide update --strip-vendor`. First create a `glide.yaml` file at the root of
your project:

```yaml
package: ( your project's import path ) # e.g. github.com/foo/bar
import:
- package: k8s.io/client-go
  version: v9.0.0 # replace v9.0.0 with the required version
```

Second, add a Go file that imports `client-go` somewhere in your project,
otherwise `client-go`'s dependencies will not be added to your project's
vendor/. Then run the following command in the same directory as `glide.yaml`:

```sh
glide update --strip-vendor
```

This can also be abbreviated as:

```sh
glide up -v
```

At this point, `k8s.io/client-go` should be added to your project's vendor/.
`client-go`'s dependencies should be flattened and be added to your project's
vendor/ as well.

Glide will detect the versions of dependencies `client-go` specified in
`client-go`'s Godep.json file, and automatically set the versions of these
imports in your /vendor directory. It will also record the detected version of
all dependencies in the `glide.lock` file.

Projects that require a different version of a dependency than `client-go`
requests can override the version manually in `glide.yaml`. For example:

```yaml
package: ( your project's import path ) # e.g. github.com/foo/bar
import:
- package: k8s.io/client-go
  version: v9.0.0 # replace v9.0.0 with the required version
# Use a newer version of go-spew even though client-go wants an old one.
- package: github.com/davecgh/go-spew
  version: v1.1.0
```

After modifying, run `glide up -v` again to re-populate your /vendor directory.

Optionally, Glide users can also use [`glide-vc`](https://github.com/sgotti/glide-vc)
after running `glide up -v` to remove unused files from /vendor.

### Dep (Not supported yet!)

[dep](https://github.com/golang/dep) is an up-and-coming dependency management
tool, which has the goal of being accepted as part of the standard go toolchain.
However, client-go does **NOT** work well with `dep` yet. To support `dep`, we
need to fix at least two issues:
1. publish native `Gopkg.toml` in client-go and other k8s.io repos, like `k8s.io/apimachinery`;
2. find a way to express transitive constraints (see https://github.com/golang/dep/issues/1124).

As a workaround, which may or may not be worthwhile, you can specify all
client-go dependencies manually as
[override](https://github.com/golang/dep/blob/master/docs/Gopkg.toml.md#override)
in Gopkg.toml with the versions listed in [Godeps.json](./Godeps/Godeps.json),
and manually update them when you upgrade client-go version.

We are actively working on the two issues blocking using `dep`. For the
meantime, we recommend using `glide` or `godeps`.
