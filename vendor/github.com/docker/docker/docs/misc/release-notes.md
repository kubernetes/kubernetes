<!--[metadata]>
+++
draft=true
title = "Docker Engine"
description = "Release notes for Docker 1.x."
keywords = ["docker, documentation, about, technology, understanding,  release"]
[menu.main]
parent = "smn_release_notes"
+++
<![end-metadata]-->

# Release notes version 1.6.0
(2015-04-16)

You can view release notes for earlier version of Docker by selecting the
desired version from the drop-down list at the top right of this page. For the
formal release announcement, see [the Docker
blog](https://blog.docker.com/2015/04/docker-release-1-6/).



## Docker Engine 1.6.0 features

For a complete list of engine patches, fixes, and other improvements, see the
[merge PR on GitHub](https://github.com/docker/docker/pull/11635). You'll also
find [a changelog in the project
repository](https://github.com/docker/docker/blob/master/CHANGELOG.md).

## Docker Engine 1.6.0 features

For a complete list of engine patches, fixes, and other improvements, see the
[merge PR on GitHub](https://github.com/docker/docker/pull/11635). You'll also
find [a changelog in the project
repository](https://github.com/docker/docker/blob/master/CHANGELOG.md).


| Feature                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Container and Image Labels   | Labels allow you to attach user-defined metadata to containers and images that can be used by your tools. For additional information on using labels, see [Apply custom metadata](https://docs.docker.com/userguide/labels-custom-metadata/#add-labels-to-images-the-label-instruction) in the documentation.                                                                                                                                                    |
| Windows Client preview       | The Windows Client can be used just like the Mac OS X client is today with a remote host. Our testing infrastructure was scaled out to accommodate Windows Client testing on every PR to the Engine. See the Azure blog for [details on using this new client](http://azure.microsoft.com/blog/2015/04/16/docker-client-for-windows-is-now-available).                                                                                                           |
| Logging drivers              | The new logging driver follows the exec driver and storage driver concepts already available in Engine today. There is a new option `--log-driver` to `docker run` command. See the `run` reference for a [description on how to use this option](https://docs.docker.com/reference/run/#logging-drivers-log-driver).                                                                                                                                            |
| Image digests                | When you pull, build, or run images, you specify them in the form `namespace/repository:tag`, or even just `repository`. In this release, you are now able to pull, run, build and refer to images by a new content addressable identifier called a “digest” with the syntax `namespace/repo@digest`. See the the command line reference for [examples of using the digest](https://docs.docker.com/reference/commandline/cli/#listing-image-digests).           |
| Custom cgroups               | Containers are made from a combination of namespaces, capabilities, and cgroups. Docker already supports custom namespaces and capabilities. Additionally, in this release we’ve added support for custom cgroups. Using the `--cgroup-parent` flag, you can pass a specific `cgroup` to run a container in. See [the command line reference for more information](https://docs.docker.com/reference/commandline/cli/#create).                                   |
| Ulimits                      | You can now specify the default `ulimit` settings for all containers when configuring the daemon. For example:`docker -d --default-ulimit nproc=1024:2048` See [Default Ulimits](https://docs.docker.com/reference/commandline/cli/#default-ulimits) in this documentation.                                                                                                                                                                                   |
| Commit and import Dockerfile | You can now make changes to images on the fly without having to re-build the entire image. The feature `commit --change` and `import --change` allows you to apply standard changes to a new image. These are expressed in the Dockerfile syntax and used to modify the image. For details on how to use these, see the [commit](https://docs.docker.com/reference/commandline/cli/#commit) and [import](https://docs.docker.com/reference/commandline/cli/#import). |

### Known issues in Engine

This section lists significant known issues present in Docker as of release date.
For an exhaustive list of issues, see [the issues list on the project
repository](https://github.com/docker/docker/issues/).

* *Unexpected File Permissions in Containers*
An idiosyncrasy in AUFS prevented permissions from propagating predictably
between upper and lower layers. This caused issues with accessing private
keys, database instances, etc.  This issue was closed in this release:
[GitHub Issue 783](https://github.com/docker/docker/issues/783).


* *Docker Hub incompatible with Safari 8*
Docker Hub had multiple issues displaying on Safari 8, the default browser for
OS X 10.10 (Yosemite).  Most notably, changes in the way Safari handled cookies
means that the user was repeatedly logged out.
Recently, Safari fixed the bug that was causing all the issues. If you upgrade
to Safari 8.0.5 which was just released last week and see if that fixes your
issues. You might have to flush your cookies if it doesn't work right away.
For more information, see the [Docker forum
post](https://forums.docker.com/t/new-safari-in-yosemite-issue/300).

## Docker Registry 2.0 features

This release includes Registry 2.0. The Docker Registry is a central server for
pushing and pulling images. In this release, it was completely rewritten in Go
around a new set of distribution APIs

- **Webhook notifications**: You can now configure the Registry to send Webhooks
when images are pushed. Spin off a CI build, send a notification to IRC –
whatever you want! Included in the documentation is a detailed [notification
specification](https://docs.docker.com/registry/notifications/).

- **Native TLS support**: This release makes it easier to secure a registry with
TLS.  This documentation includes [expanded examples of secure
deployments](https://docs.docker.com/registry/deploying/).

- **New Distribution APIs**: This release includes an expanded set of new
distribution APIs. You can read the [detailed specification
here](https://docs.docker.com/registry/spec/api/).


## Docker Compose 1.2

For a complete list of compose patches, fixes, and other improvements, see the
[changelog in the project
repository](https://github.com/docker/compose/blob/master/CHANGES.md). The
project also makes a [set of release
notes](https://github.com/docker/compose/releases/tag/1.2.0) on the project.

- **extends**:  You can use `extends` to share configuration between services
with the keyword “extends”. With extends, you can refer to a service defined
elsewhere and include its configuration in a locally-defined service, while also
adding or overriding configuration as necessary. The documentation describes
[how to use extends in your
configuration](https://docs.docker.com/compose/extends/#extending-services-in-
compose).

- **Relative directory handling may cause breaking change**: Compose now treats
directories passed to build, filenames passed to `env_file` and volume host
paths passed to volumes as relative to the configuration file's directory.
Previously, they were treated as relative to the directory where you were
running `docker-compose`. In the majority of cases, the location of the
configuration file and where you ran `docker-compose` were the same directory.
Now, you can use the `-f|--file` argument to specify a configuration file in
another directory. 


## Docker Swarm 0.2

You'll find the [release for download on
GitHub](https://github.com/docker/swarm/releases/tag/v0.2.0) and [the
documentation here](https://docs.docker.com/swarm/).  This release includes the
following features:

- **Spread strategy**: A new strategy for scheduling containers on your cluster
which evenly spreads them over available nodes.
- **More Docker commands supported**: More progress has been made towards
supporting the complete Docker API, such as pulling and inspecting images.
- **Clustering drivers**: There are not any third-party drivers yet, but the
first steps have been made towards making a pluggable driver interface that will
make it possible to use Swarm with clustering systems such as Mesos.


## Docker Machine 0.2 Pre-release

You'll find the [release for download on
GitHub](https://github.com/docker/machine/releases) and [the documentation
here](https://docs.docker.com/machine/).  For a complete list of machine changes
see [the changelog in the project
repository](https://github.com/docker/machine/blob/master/CHANGES.md#020-2015-03
-22).

- **Cleaner driver interface**: It is now much easier to write drivers for providers.
- **More reliable and consistent provisioning**: Provisioning servers is now
handled centrally by Machine instead of letting each driver individually do it.
- **Regenerate TLS certificates**: A new command has been added to regenerate a
host’s TLS certificates for good security practice and for if a host’s IP
address changes. 

## Docker Hub Enterprise & Commercially Supported Docker Engine

See the [DHE and CS Docker Engine release notes](docker-hub-enterprise/release-notes.md).
