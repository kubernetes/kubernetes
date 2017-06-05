# etcd

[![Go Report Card](https://goreportcard.com/badge/github.com/coreos/etcd)](https://goreportcard.com/report/github.com/coreos/etcd)
[![Build Status](https://travis-ci.org/coreos/etcd.svg?branch=master)](https://travis-ci.org/coreos/etcd)
[![Build Status](https://semaphoreci.com/api/v1/coreos/etcd/branches/master/shields_badge.svg)](https://semaphoreci.com/coreos/etcd)
[![Docker Repository on Quay.io](https://quay.io/repository/coreos/etcd-git/status "Docker Repository on Quay.io")](https://quay.io/repository/coreos/etcd-git)

**Note**: The `master` branch may be in an *unstable or even broken state* during development. Please use [releases][github-release] instead of the `master` branch in order to get stable binaries.

*the etcd v2 [documentation](Documentation/v2/README.md) has moved*

![etcd Logo](logos/etcd-horizontal-color.png)

etcd is a distributed, consistent key-value store for shared configuration and service discovery, with a focus on being:

* *Simple*: well-defined, user-facing API (gRPC)
* *Secure*: automatic TLS with optional client cert authentication
* *Fast*: benchmarked 10,000 writes/sec
* *Reliable*: properly distributed using Raft

etcd is written in Go and uses the [Raft][raft] consensus algorithm to manage a highly-available replicated log.

etcd is used [in production by many companies](./Documentation/production-users.md), and the development team stands behind it in critical deployment scenarios, where etcd is frequently teamed with applications such as [Kubernetes][k8s], [fleet][fleet], [locksmith][locksmith], [vulcand][vulcand], [Doorman][doorman], and many others. Reliability is further ensured by rigorous [testing][etcd-tests].

See [etcdctl][etcdctl] for a simple command line client.

[raft]: https://raft.github.io/
[k8s]: http://kubernetes.io/
[doorman]: https://github.com/youtube/doorman
[fleet]: https://github.com/coreos/fleet
[locksmith]: https://github.com/coreos/locksmith
[vulcand]: https://github.com/vulcand/vulcand
[etcdctl]: https://github.com/coreos/etcd/tree/master/etcdctl
[etcd-tests]: http://dash.etcd.io

## Getting started

### Getting etcd

The easiest way to get etcd is to use one of the pre-built release binaries which are available for OSX, Linux, Windows, AppC (ACI), and Docker. Instructions for using these binaries are on the [GitHub releases page][github-release].

For those wanting to try the very latest version, you can build the latest version of etcd from the `master` branch.
You will first need [*Go*](https://golang.org/) installed on your machine (version 1.6+ is required).
All development occurs on `master`, including new features and bug fixes.
Bug fixes are first targeted at `master` and subsequently ported to release branches, as described in the [branch management][branch-management] guide.

[github-release]: https://github.com/coreos/etcd/releases/
[branch-management]: ./Documentation/branch_management.md

### Running etcd

First start a single-member cluster of etcd:

```sh
./bin/etcd
```

This will bring up etcd listening on port 2379 for client communication and on port 2380 for server-to-server communication.

Next, let's set a single key, and then retrieve it:

```
ETCDCTL_API=3 etcdctl put mykey "this is awesome"
ETCDCTL_API=3 etcdctl get mykey
```

That's it! etcd is now running and serving client requests. For more

- [Animated quick demo][demo-gif]
- [Interactive etcd playground][etcd-play]

[demo-gif]: ./Documentation/demo.md
[etcd-play]: http://play.etcd.io/

### etcd TCP ports

The [official etcd ports][iana-ports] are 2379 for client requests, and 2380 for peer communication. 

[iana-ports]: https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=etcd

### Running a local etcd cluster

First install [goreman](https://github.com/mattn/goreman), which manages Procfile-based applications.

Our [Procfile script](./Procfile) will set up a local example cluster. Start it with:

```sh
goreman start
```

This will bring up 3 etcd members `infra1`, `infra2` and `infra3` and etcd proxy `proxy`, which runs locally and composes a cluster.

Every cluster member and proxy accepts key value reads and key value writes.

### Next steps

Now it's time to dig into the full etcd API and other guides.

- Read the full [documentation][fulldoc].
- Explore the full gRPC [API][api].
- Set up a [multi-machine cluster][clustering].
- Learn the [config format, env variables and flags][configuration].
- Find [language bindings and tools][libraries-and-tools].
- Use TLS to [secure an etcd cluster][security].
- [Tune etcd][tuning].

[fulldoc]: ./Documentation/docs.md
[api]: ./Documentation/dev-guide/api_reference_v3.md
[clustering]: ./Documentation/op-guide/clustering.md
[configuration]: ./Documentation/op-guide/configuration.md
[libraries-and-tools]: ./Documentation/libraries-and-tools.md
[security]: ./Documentation/op-guide/security.md
[tuning]: ./Documentation/tuning.md

## Contact

- Mailing list: [etcd-dev](https://groups.google.com/forum/?hl=en#!forum/etcd-dev)
- IRC: #[etcd](irc://irc.freenode.org:6667/#etcd) on freenode.org
- Planning/Roadmap: [milestones](https://github.com/coreos/etcd/milestones), [roadmap](./ROADMAP.md)
- Bugs: [issues](https://github.com/coreos/etcd/issues)

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for details on submitting patches and the contribution workflow.

## Reporting bugs

See [reporting bugs](Documentation/reporting_bugs.md) for details about reporting any issue you may encounter.

### License

etcd is under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.

