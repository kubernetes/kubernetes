# rkt - App Container runtime

[![godoc](https://godoc.org/github.com/coreos/rkt?status.svg)](http://godoc.org/github.com/coreos/rkt)
[![Build Status (Travis)](https://travis-ci.org/coreos/rkt.svg?branch=master)](https://travis-ci.org/coreos/rkt)
[![Build Status (SemaphoreCI)](https://semaphoreci.com/api/v1/projects/28468e19-4fd0-483e-9c29-6c8368661333/395211/badge.svg)](https://semaphoreci.com/coreos/rkt)

![rkt Logo](logos/rkt-horizontal-color.png)

rkt (pronounced _"rock-it"_) is a CLI for running app containers on Linux. rkt is designed to be secure, composable, and standards-based.

Some of rkt's key features and goals include:

- _Security_: rkt is developed with a principle of "secure-by-default", and includes a number of important security features like support for [SELinux][selinux], [TPM measurement][tpm], and running app containers in [hardware-isolated VMs][lkvm].
- _Composability_: rkt is designed for first-class integration with init systems ([systemd][systemd], upstart) and cluster orchestration tools (fleet, [Kubernetes][kubernetes], [Nomad][nomad]), and supports [swappable execution engines][architecture].
- _Open standards and compatibility_: rkt implements the [appc specification][rkt-and-appc], supports the [Container Networking Interface specification][CNI], and can also run [Docker images][docker].

For more on the background and motivation behind rkt, read the original [launch announcement][blog-post].

[architecture]: Documentation/devel/architecture.md
[systemd]: Documentation/using-rkt-with-systemd.md
[kubernetes]: Documentation/using-rkt-with-kubernetes.md
[nomad]: Documentation/using-rkt-with-nomad.md
[docker]: Documentation/running-docker-images.md
[networking]: Documentation/networking.md
[lkvm]: Documentation/running-lkvm-stage1.md
[rkt-and-appc]: Documentation/app-container.md
[cni]: https://github.com/appc/cni
[selinux]: Documentation/selinux.md
[tpm]: Documentation/devel/tpm.md
[blog-post]: https://coreos.com/blog/rocket

## Project status

rkt's version 1.0 release marks the command line user interface and on-disk data structures as stable and reliable for external development. Any major changes to those primary areas will be clearly communicated, and a formal deprecation process conducted for any retired features. The (optional) API for pod inspection is not yet completely stabilized, but is quite usable, and an excellent area for testing and participation as it matures.

Check out the [roadmap](ROADMAP.md) for more details on the future of rkt.

## Trying out rkt

To get started quickly using rkt for the first time, start with the ["trying out rkt" document](Documentation/trying-out-rkt.md).
For an end-to-end example of building an application from scratch and running it with rkt, check out the [getting started guide](Documentation/getting-started-guide.md).

## Getting help with rkt

There are a number of different avenues for seeking help and communicating with the rkt community:
- For bugs and feature requests (including documentation!), file an [issue][new-issue]
- For general discussion about both using and developing rkt, join the [rkt-dev][rkt-dev] mailing list
- For real-time discussion, join us on IRC: #[rkt-dev][irc] on freenode.org
- For more details on rkt development plans, check out the GitHub [milestones][milestones]

Most discussion about rkt development happens on GitHub via issues and pull requests.
The rkt developers also host a semi-regular community sync meeting open to the public.
This sync usually features demos, updates on the roadmap, and time for anyone from the community to ask questions of the developers or share users stories with others.
For more details, including how to join and recordings of previous syncs, see the [sync doc on Google Docs][sync-doc].

[new-issue]: https://github.com/coreos/rkt/issues/new
[rkt-dev]: https://groups.google.com/forum/?hl=en#!forum/rkt-dev
[irc]: irc://irc.freenode.org:6667/#rkt-dev
[milestones]: https://github.com/coreos/rkt/milestones
[sync-doc]: https://docs.google.com/document/d/1NT_J5X2QErtKgd8Y3TFXNknWhJx_yOCMJnq3Iy2jPgE/edit#

## Contributing to rkt

rkt is an open source project and contributions are gladly welcomed!
See the [Hacking Guide](Documentation/hacking.md) for more information on how to build and work on rkt.
See [CONTRIBUTING](CONTRIBUTING.md) for details on submitting patches and the contribution workflow.

## Licensing

Unless otherwise noted, all code in the rkt repository is licensed under the [Apache 2.0 license](LICENSE).
Some portions of the codebase are derived from other projects under different licenses; the appropriate information can be found in the header of those source files, as applicable.

## Security disclosure

If you suspect you have found a security vulnerability in rkt, please *do not* file a GitHub issue, but instead email <security@coreos.com> with the full details, including steps to reproduce the issue.
CoreOS is currently the primary sponsor of rkt development, and all reports are thoroughly investigated by CoreOS engineers.
For more information, see the [CoreOS security disclosure page](https://coreos.com/security/disclosure/).

## Known issues

Due to a bug in the Linux kernel, using rkt's overlay support on top of an overlay filesystem requires Linux 4.3+ (see the [stacked overlay fix](https://github.com/torvalds/linux/commit/1c8a47d) and [#1537](https://github.com/coreos/rkt/issues/1537)).

Due to a bug in the Linux kernel, rkt will not work when /var/lib/rkt is on btrfs ([#2175](https://github.com/coreos/rkt/issues/2175)).

Linux 3.18+ is required to successfully garbage collect rkt pods when system services such as udevd are in a slave mount namespace (see [lazy umounts on unlinked files and directories](https://github.com/torvalds/linux/commit/8ed936b) and [#1922](https://github.com/coreos/rkt/issues/1922)).
