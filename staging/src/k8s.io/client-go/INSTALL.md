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
see the instructions for [enabling go modules](#enabling-go-modules).

## Dependency management for the serious (or reluctant) user

Reasons why you might need to use a dependency management system:
* You use a dependency that client-go also uses, and don't want two copies of
  the dependency compiled into your application. For some dependencies with
  singletons or global inits (e.g. `glog`) this wouldn't even compile...
* You want to lock in a particular version (so you don't have to change your
  code every time we change a public interface).
* You want your install to be reproducible. For example, for your CI system or
  for new team members.

### Enabling go modules

Dependency management tools are built into go 1.11+ in the form of [go modules](https://github.com/golang/go/wiki/Modules).
These are used by the main Kubernetes repo (>= `v1.15.0`) and `client-go` (>= `kubernetes-1.15.0`) to manage dependencies.
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

### Add client-go as a dependency

Indicate which version of `client-go` your project requires:

- If you are using Kubernetes versions >= `v1.17.0`, use a corresponding
`v0.x.y` tag. For example, `k8s.io/client-go@v0.17.0` corresponds to Kubernetes `v1.17.0`:

```sh
go get k8s.io/client-go@v0.17.0
```

You can also use a non-semver `kubernetes-1.x.y` tag to refer to a version
of `client-go` corresponding to a given Kubernetes release. Prior to Kubernetes
`v1.17.0` these were the only tags available for use with go modules.
For example, `kubernetes-1.16.3` corresponds to Kubernetes `v1.16.3`.
However, it is recommended to use semver-like `v0.x.y` tags over non-semver
`kubernetes-1.x.y` tags to have a seamless experience with go modules.

- If you are using Kubernetes versions < `v1.17.0` (replace `kubernetes-1.16.3` with the desired version):

```sh
go get k8s.io/client-go@kubernetes-1.16.3
```

You can now import and use the `k8s.io/client-go` APIs in your project.
The next time you `go build`, `go test`, or `go run` your project,
`k8s.io/client-go` and its dependencies will be downloaded (if needed),
and detailed dependency version info will be added to your `go.mod` file
(or you can also run `go mod tidy` to do this directly).
