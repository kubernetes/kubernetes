# etcd2

[![Go Report Card](https://goreportcard.com/badge/github.com/coreos/etcd)](https://goreportcard.com/report/github.com/coreos/etcd)
[![Build Status](https://travis-ci.org/coreos/etcd.svg?branch=master)](https://travis-ci.org/coreos/etcd)
[![Build Status](https://semaphoreci.com/api/v1/coreos/etcd/branches/master/shields_badge.svg)](https://semaphoreci.com/coreos/etcd)
[![Docker Repository on Quay.io](https://quay.io/repository/coreos/etcd-git/status "Docker Repository on Quay.io")](https://quay.io/repository/coreos/etcd-git)

**Note**: The `master` branch may be in an *unstable or even broken state* during development. Please use [releases][github-release] instead of the `master` branch in order to get stable binaries.

![etcd Logo](../../logos/etcd-horizontal-color.png)

etcd is a distributed, consistent key-value store for shared configuration and service discovery, with a focus on being:

* *Simple*: curl'able user-facing API (HTTP+JSON)
* *Secure*: optional SSL client cert authentication
* *Fast*: benchmarked 1000s of writes/s per instance
* *Reliable*: properly distributed using Raft

etcd is written in Go and uses the [Raft][raft] consensus algorithm to manage a highly-available replicated log.

etcd is used [in production by many companies](./production-users.md), and the development team stands behind it in critical deployment scenarios, where etcd is frequently teamed with applications such as [Kubernetes][k8s], [fleet][fleet], [locksmith][locksmith], [vulcand][vulcand], and many others.

See [etcdctl][etcdctl] for a simple command line client.
Or feel free to just use `curl`, as in the examples below.

[raft]: https://raft.github.io/
[k8s]: http://kubernetes.io/
[fleet]: https://github.com/coreos/fleet
[locksmith]: https://github.com/coreos/locksmith
[vulcand]: https://github.com/vulcand/vulcand
[etcdctl]: https://github.com/coreos/etcd/tree/master/etcdctl

## Getting Started

### Getting etcd

The easiest way to get etcd is to use one of the pre-built release binaries which are available for OSX, Linux, Windows, AppC (ACI), and Docker. Instructions for using these binaries are on the [GitHub releases page][github-release].

For those wanting to try the very latest version, you can build the latest version of etcd from the `master` branch.
You will first need [*Go*](https://golang.org/) installed on your machine (version 1.5+ is required).
All development occurs on `master`, including new features and bug fixes.
Bug fixes are first targeted at `master` and subsequently ported to release branches, as described in the [branch management][branch-management] guide.

[github-release]: https://github.com/coreos/etcd/releases/
[branch-management]: branch_management.md

### Running etcd

First start a single-member cluster of etcd:

```sh
./bin/etcd
```

This will bring up etcd listening on port 2379 for client communication and on port 2380 for server-to-server communication.

Next, let's set a single key, and then retrieve it:

```
curl -L http://127.0.0.1:2379/v2/keys/mykey -XPUT -d value="this is awesome"
curl -L http://127.0.0.1:2379/v2/keys/mykey
```

You have successfully started an etcd and written a key to the store.

### etcd TCP ports

The [official etcd ports][iana-ports] are 2379 for client requests, and 2380 for peer communication. To maintain compatibility, some etcd configuration and documentation continues to refer to the legacy ports 4001 and 7001, but all new etcd use and discussion should adopt the IANA-assigned ports. The legacy ports 4001 and 7001 will be fully deprecated, and support for their use removed, in future etcd releases.

[iana-ports]: http://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.txt

### Running local etcd cluster

First install [goreman](https://github.com/mattn/goreman), which manages Procfile-based applications.

Our [Procfile script](../../V2Procfile) will set up a local example cluster. You can start it with:

```sh
goreman start
```

This will bring up 3 etcd members `infra1`, `infra2` and `infra3` and etcd proxy `proxy`, which runs locally and composes a cluster.

You can write a key to the cluster and retrieve the value back from any member or proxy.

### Next Steps

Now it's time to dig into the full etcd API and other guides.

- Explore the full [API][api].
- Set up a [multi-machine cluster][clustering].
- Learn the [config format, env variables and flags][configuration].
- Find [language bindings and tools][libraries-and-tools].
- Use TLS to [secure an etcd cluster][security].
- [Tune etcd][tuning].
- [Upgrade from 0.4.9+ to 2.2.0][upgrade].

[api]: ./api.md
[clustering]: ./clustering.md
[configuration]: ./configuration.md
[libraries-and-tools]: ./libraries-and-tools.md
[security]: ./security.md
[tuning]: ./tuning.md
[upgrade]: ./04_to_2_snapshot_migration.md

## Contact

- Mailing list: [etcd-dev](https://groups.google.com/forum/?hl=en#!forum/etcd-dev)
- IRC: #[etcd](irc://irc.freenode.org:6667/#etcd) on freenode.org
- Planning/Roadmap: [milestones](https://github.com/coreos/etcd/milestones), [roadmap](../../ROADMAP.md)
- Bugs: [issues](https://github.com/coreos/etcd/issues)

## Contributing

See [CONTRIBUTING](../../CONTRIBUTING.md) for details on submitting patches and the contribution workflow.

## Reporting bugs

See [reporting bugs](reporting_bugs.md) for details about reporting any issue you may encounter.

## Known bugs

[GH518](https://github.com/coreos/etcd/issues/518) is a known bug. Issue is that:

```
curl http://127.0.0.1:2379/v2/keys/foo -XPUT -d value=bar
curl http://127.0.0.1:2379/v2/keys/foo -XPUT -d dir=true -d prevExist=true
```

If the previous node is a key and client tries to overwrite it with `dir=true`, it does not give warnings such as `Not a directory`. Instead, the key is set to empty value.

## Project Details

### Versioning

#### Service Versioning

etcd uses [semantic versioning](http://semver.org)
New minor versions may add additional features to the API.

You can get the version of etcd by issuing a request to /version:

```sh
curl -L http://127.0.0.1:2379/version
```

#### API Versioning

The `v2` API responses should not change after the 2.0.0 release but new features will be added over time.

#### 32-bit and other unsupported systems

etcd has known issues on 32-bit systems due to a bug in the Go runtime. See #[358][358] for more information.

To avoid inadvertently running a possibly unstable etcd server, `etcd` on unsupported architectures will print
a warning message and immediately exit if the environment variable `ETCD_UNSUPPORTED_ARCH` is not set to
the target architecture.

Currently only the amd64 architecture is officially supported by `etcd`.

[358]: https://github.com/coreos/etcd/issues/358

### License

etcd is under the Apache 2.0 license. See the [LICENSE](../../LICENSE) file for details.
