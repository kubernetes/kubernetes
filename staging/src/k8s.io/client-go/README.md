# client-go

Go clients for talking to a [kubernetes](http://kubernetes.io/) cluster.

We recommend using the `v0.x.y` tags for Kubernetes releases >= `v1.17.0` and
`kubernetes-1.x.y` tags for Kubernetes releases < `v1.17.0`.

The fastest way to add this library to a project is to run `go get k8s.io/client-go@latest` with go1.16+.
See [INSTALL.md](/INSTALL.md) for detailed installation instructions and troubleshooting.

[![GoDocWidget]][GoDocReference]

[GoDocWidget]: https://godoc.org/k8s.io/client-go?status.svg
[GoDocReference]:https://godoc.org/k8s.io/client-go 

## Table of Contents

- [What's included](#whats-included)
- [Versioning](#versioning)
  - [Compatibility: your code <-> client-go](#compatibility-your-code---client-go)
  - [Compatibility: client-go <-> Kubernetes clusters](#compatibility-client-go---kubernetes-clusters)
  - [Compatibility matrix](#compatibility-matrix)
  - [Why do the 1.4 and 1.5 branch contain top-level folder named after the version?](#why-do-the-14-and-15-branch-contain-top-level-folder-named-after-the-version)
- [Kubernetes tags](#kubernetes-tags)
- [How to get it](#how-to-get-it)
- [How to use it](#how-to-use-it)
- [Dependency management](#dependency-management)
- [Contributing code](#contributing-code)

### What's included

* The `kubernetes` package contains the clientset to access Kubernetes API.
* The `discovery` package is used to discover APIs supported by a Kubernetes API server.
* The `dynamic` package contains a dynamic client that can perform generic operations on arbitrary Kubernetes API objects.
* The `plugin/pkg/client/auth` packages contain optional authentication plugins for obtaining credentials from external sources.
* The `transport` package is used to set up auth and start a connection.
* The `tools/cache` package is useful for writing controllers.

### Versioning

- For each `v1.x.y` Kubernetes release, the major version (first digit)
would remain `0`.

- Bugfixes will result in the patch version (third digit) changing. PRs that are
cherry-picked into an older Kubernetes release branch will result in an update
to the corresponding branch in `client-go`, with a corresponding new tag
changing the patch version.

#### Branches and tags.

We will create a new branch and tag for each increment in the minor version
number. We will create only a new tag for each increment in the patch
version number. See [semver](http://semver.org/) for definitions of major,
minor, and patch.

The HEAD of the master branch in client-go will track the HEAD of the master
branch in the main Kubernetes repo.

#### Compatibility: your code <-> client-go

The `v0.x.y` tags indicate that go APIs may change in incompatible ways in
different versions.

See [INSTALL.md](INSTALL.md) for guidelines on requiring a specific
version of client-go.

#### Compatibility: client-go <-> Kubernetes clusters

Since Kubernetes is backwards compatible with clients, older `client-go`
versions will work with many different Kubernetes cluster versions.

We will backport bugfixes--but not new features--into older versions of
`client-go`.


#### Compatibility matrix

|                               | Kubernetes 1.27 | Kubernetes 1.28 | Kubernetes 1.29 | Kubernetes 1.30 | Kubernetes 1.31 | Kubernetes 1.32 |
| ----------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| `kubernetes-1.27.0`/`v0.27.0` | ✓               | +-              | +-              | +-              | +-              | +-              |
| `kubernetes-1.28.0`/`v0.28.0` | +-              | ✓               | +-              | +-              | +-              | +-              |
| `kubernetes-1.29.0`/`v0.29.0` | +-              | +-              | ✓               | +-              | +-              | +-              |
| `kubernetes-1.30.0`/`v0.30.0` | +-              | +-              | +-              | ✓               | +-              | +-              |
| `kubernetes-1.31.0`/`v0.31.0` | +-              | +-              | +-              | +-              | ✓               | +-              |
| `kubernetes-1.32.0`/`v0.32.0` | +-              | +-              | +-              | +-              | +-              | ✓               |
| `HEAD`                        | +-              | +-              | +-              | +-              | +-              | +-              |

Key:

* `✓` Exactly the same features / API objects in both client-go and the Kubernetes
  version.
* `+` client-go has features or API objects that may not be present in the
  Kubernetes cluster, either due to that client-go has additional new API, or
  that the server has removed old API. However, everything they have in
  common (i.e., most APIs) will work. Please note that alpha APIs may vanish or
  change significantly in a single release.
* `-` The Kubernetes cluster has features the client-go library can't use,
  either due to the server has additional new API, or that client-go has
  removed old API. However, everything they share in common (i.e., most APIs)
  will work.

See the [CHANGELOG](./CHANGELOG.md) for a detailed description of changes
between client-go versions.

| Branch         | Canonical source code location      | Maintenance status |
| -------------- | ----------------------------------- | ------------------ |
| `release-1.23` | Kubernetes main repo, 1.23 branch   | =-                 |
| `release-1.24` | Kubernetes main repo, 1.24 branch   | =-                 |
| `release-1.25` | Kubernetes main repo, 1.25 branch   | =-                 |
| `release-1.26` | Kubernetes main repo, 1.26 branch   | =-                 |
| `release-1.27` | Kubernetes main repo, 1.27 branch   | =-                 |
| `release-1.28` | Kubernetes main repo, 1.28 branch   | =-                 |
| `release-1.29` | Kubernetes main repo, 1.29 branch   | ✓                  |
| `release-1.30` | Kubernetes main repo, 1.30 branch   | ✓                  |
| `release-1.31` | Kubernetes main repo, 1.31 branch   | ✓                  |
| `release-1.32` | Kubernetes main repo, 1.32 branch   | ✓                  |
| client-go HEAD | Kubernetes main repo, master branch | ✓                  |

Key:

* `✓` Changes in main Kubernetes repo are actively published to client-go by a bot
* `=` Maintenance is manual, only severe security bugs will be patched.
* `-` Deprecated; please upgrade.

#### Deprecation policy

We will maintain branches for at least six months after their first stable tag
is cut. (E.g., the clock for the release-2.0 branch started ticking when we
tagged v2.0.0, not when we made the first alpha.) This policy applies to
every version greater than or equal to 2.0.

#### Why do the 1.4 and 1.5 branch contain top-level folder named after the version?

For the initial release of client-go, we thought it would be easiest to keep
separate directories for each minor version. That soon proved to be a mistake.
We are keeping the top-level folders in the 1.4 and 1.5 branches so that
existing users won't be broken.

### Kubernetes tags

This repository is still a mirror of
[k8s.io/kubernetes/staging/src/client-go](https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/client-go),
the code development is still done in the staging area.

Since Kubernetes `v1.8.0`, when syncing the code from the staging area,
we also sync the Kubernetes version tags to client-go, prefixed with
`kubernetes-`. From Kubernetes `v1.17.0`, we also create matching semver
`v0.x.y` tags for each `v1.x.y` Kubernetes release.

For example, if you check out the `kubernetes-1.17.0` or the `v0.17.0` tag in
client-go, the code you get is exactly the same as if you check out the `v1.17.0`
tag in Kubernetes, and change directory to `staging/src/k8s.io/client-go`.

The purpose is to let users quickly find matching commits among published repos,
like [sample-apiserver](https://github.com/kubernetes/sample-apiserver),
[apiextension-apiserver](https://github.com/kubernetes/apiextensions-apiserver),
etc. The Kubernetes version tag does NOT claim any backwards compatibility
guarantees for client-go. Please check the [semantic versions](#versioning) if
you care about backwards compatibility.

### How to get it

To get the latest version, use go1.16+ and fetch using the `go get` command. For example:

```
go get k8s.io/client-go@latest
```

To get a specific version, use go1.11+ and fetch the desired version using the `go get` command. For example:

```
go get k8s.io/client-go@v0.20.4
```

See [INSTALL.md](/INSTALL.md) for detailed instructions and troubleshooting.

### How to use it

If your application runs in a Pod in the cluster, please refer to the
in-cluster [example](examples/in-cluster-client-configuration), otherwise please
refer to the out-of-cluster [example](examples/out-of-cluster-client-configuration).

### Dependency management

For details on how to correctly use a dependency management for installing client-go, please see [INSTALL.md](INSTALL.md).

### Contributing code
Please send pull requests against the client packages in the Kubernetes main [repository](https://github.com/kubernetes/kubernetes). Changes in the staging area will be published to this repository every day.
