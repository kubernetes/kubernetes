# Installing client-go

## For the casual user

If you want to write a simple script, don't care about a reproducible client
library install, don't mind getting HEAD (which may be less stable than a
particular release), then simply:

```sh
go get k8s.io/client-go@master
```

This will record a dependency on `k8s.io/client-go` in your go module.
You can now import and use the `k8s.io/client-go` APIs in your project.
The next time you `go build`, `go test`, or `go run` your project,
`k8s.io/client-go` and its dependencies will be downloaded (if needed),
and detailed dependency version info will be added to your `go.mod` file
(or you can also run `go mod tidy` to do this directly).

This assumes you are using go modules with go 1.11+.
If you get a message like `cannot use path@version syntax in GOPATH mode`,
you can choose to [opt into using go modules](#go-modules).
If you are using a version of go prior to 1.11, or do not wish to use 
go modules, you can download `k8s.io/client-go` to your `$GOPATH` instead:

```sh
go get -u k8s.io/client-go/...
go get -u k8s.io/apimachinery/...
cd $GOPATH/src/k8s.io/client-go
git checkout v11.0.0
cd $GOPATH/src/k8s.io/apimachinery
git checkout kubernetes-1.14.0
```

This downloads a version of `k8s.io/client-go` prior to v1.12.0,
which includes most of its dependencies in its `k8s.io/client-go/vendor` path
(except for `k8s.io/apimachinery` and `glog`).

We excluded `k8s.io/apimachinery` and `glog` from `k8s.io/client-go/vendor` to
prevent `go get` users from hitting issues like
[#19](https://github.com/kubernetes/client-go/issues/19) and
[#83](https://github.com/kubernetes/client-go/issues/83). If your project shares
other dependencies with client-go, and you hit issues similar to #19 or #83,
then you'll need to look down at the next section.

Note: the official go policy is that libraries should not vendor their
dependencies. This was unworkable for us, since our dependencies change and HEAD
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

### Go modules

Dependency management tools are built into go 1.11+ in the form of [go modules](https://github.com/golang/go/wiki/Modules).
These are used by the main Kubernetes repo (>= 1.15) and `client-go` (on master, and v12.0.0+) to manage dependencies.
When using `client-go` v12.0.0+ and go 1.11.4+, go modules are the recommended dependency management tool.

If you are using go 1.11 or 1.12 and are working with a project located within `$GOPATH`,
you must opt into using go modules:

```sh
export GO111MODULE=on
```

Ensure your project has a `go.mod` file defined at the root of your project.
If you do not already have one, `go mod init` will create one for you:

```sh
go mod init
```

Indicate which version of `client-go` your project requires.
For `client-go` on v12.0.0 (and later), this is a single step:

```sh
go get k8s.io/client-go@v12.0.0
```

For `client-go` prior to v12.0.0, you also need to indicate the required versions of `k8s.io/api` and `k8s.io/apimachinery`:

```sh
go get k8s.io/client-go@v11.0.0              # replace v11.0.0 with the required version (or use kubernetes-1.x.y tags if desired)
go get k8s.io/api@kubernetes-1.14.0          # replace kubernetes-1.14.0 with the required version
go get k8s.io/apimachinery@kubernetes-1.14.0 # replace kubernetes-1.14.0 with the required version
```

You can now import and use the `k8s.io/client-go` APIs in your project.
The next time you `go build`, `go test`, or `go run` your project,
`k8s.io/client-go` and its dependencies will be downloaded (if needed),
and detailed dependency version info will be added to your `go.mod` file
(or you can also run `go mod tidy` to do this directly).

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
  version: v12.0.0 # replace v12.0.0 with the required version
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
  version: v12.0.0 # replace v12.0.0 with the required version
# Use a newer version of go-spew even though client-go wants an old one.
- package: github.com/davecgh/go-spew
  version: v1.1.0
```

After modifying, run `glide up -v` again to re-populate your /vendor directory.

Optionally, Glide users can also use [`glide-vc`](https://github.com/sgotti/glide-vc)
after running `glide up -v` to remove unused files from /vendor.

### Godep

[godep](https://github.com/tools/godep) is an older dependency management tool, which was
used by the main Kubernetes repo (<= 1.14) and `client-go` (<= v11.0) to manage dependencies.

Before proceeding with the below instructions, you should ensure that your
$GOPATH is empty except for containing your own package and its dependencies,
and you have a copy of godep somewhere in your $PATH.

To install `client-go` and place its dependencies in your `$GOPATH`:

```sh
go get k8s.io/client-go/...
cd $GOPATH/src/k8s.io/client-go
git checkout v11.0.0 # v11.0.0 or older
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
