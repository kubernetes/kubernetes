# Dear Packager,

If you are looking to make Docker available on your favorite software
distribution, this document is for you. It summarizes the requirements for
building and running the Docker client and the Docker daemon.

## Getting Started

We want to help you package Docker successfully. Before doing any packaging, a
good first step is to introduce yourself on the [docker-dev mailing
list](https://groups.google.com/d/forum/docker-dev), explain what you're trying
to achieve, and tell us how we can help. Don't worry, we don't bite! There might
even be someone already working on packaging for the same distro!

You can also join the IRC channel - #docker and #docker-dev on Freenode are both
active and friendly.

We like to refer to Tianon ("@tianon" on GitHub and "tianon" on IRC) as our
"Packagers Relations", since he's always working to make sure our packagers have
a good, healthy upstream to work with (both in our communication and in our
build scripts). If you're having any kind of trouble, feel free to ping him
directly. He also likes to keep track of what distributions we have packagers
for, so feel free to reach out to him even just to say "Hi!"

## Package Name

If possible, your package should be called "docker". If that name is already
taken, a second choice is "docker-engine". Another possible choice is "docker.io".

## Official Build vs Distro Build

The Docker project maintains its own build and release toolchain. It is pretty
neat and entirely based on Docker (surprise!). This toolchain is the canonical
way to build Docker. We encourage you to give it a try, and if the circumstances
allow you to use it, we recommend that you do.

You might not be able to use the official build toolchain - usually because your
distribution has a toolchain and packaging policy of its own. We get it! Your
house, your rules. The rest of this document should give you the information you
need to package Docker your way, without denaturing it in the process.

## Build Dependencies

To build Docker, you will need the following:

* A recent version of Git and Mercurial
* Go version 1.6 or later
* A clean checkout of the source added to a valid [Go
  workspace](https://golang.org/doc/code.html#Workspaces) under the path
  *src/github.com/docker/docker* (unless you plan to use `AUTO_GOPATH`,
  explained in more detail below)

To build the Docker daemon, you will additionally need:

* An amd64/x86_64 machine running Linux
* SQLite version 3.7.9 or later
* libdevmapper version 1.02.68-cvs (2012-01-26) or later from lvm2 version
  2.02.89 or later
* btrfs-progs version 3.16.1 or later (unless using an older version is
  absolutely necessary, in which case 3.8 is the minimum)
* libseccomp version 2.2.1 or later (for build tag seccomp)

Be sure to also check out Docker's Dockerfile for the most up-to-date list of
these build-time dependencies.

### Go Dependencies

All Go dependencies are vendored under "./vendor". They are used by the official
build, so the source of truth for the current version of each dependency is
whatever is in "./vendor".

To use the vendored dependencies, simply make sure the path to "./vendor" is
included in `GOPATH` (or use `AUTO_GOPATH`, as explained below).

If you would rather (or must, due to distro policy) package these dependencies
yourself, take a look at "vendor.conf" for an easy-to-parse list of the
exact version for each.

NOTE: if you're not able to package the exact version (to the exact commit) of a
given dependency, please get in touch so we can remediate! Who knows what
discrepancies can be caused by even the slightest deviation. We promise to do
our best to make everybody happy.

## Stripping Binaries

Please, please, please do not strip any compiled binaries. This is really
important.

In our own testing, stripping the resulting binaries sometimes results in a
binary that appears to work, but more often causes random panics, segfaults, and
other issues. Even if the binary appears to work, please don't strip.

See the following quotes from Dave Cheney, which explain this position better
from the upstream Golang perspective.

### [go issue #5855, comment #3](https://code.google.com/p/go/issues/detail?id=5855#c3)

> Super super important: Do not strip go binaries or archives. It isn't tested,
> often breaks, and doesn't work.

### [launchpad golang issue #1200255, comment #8](https://bugs.launchpad.net/ubuntu/+source/golang/+bug/1200255/comments/8)

> To quote myself: "Please do not strip Go binaries, it is not supported, not
> tested, is often broken, and doesn't do what you want"
>
> To unpack that a bit
>
> * not supported, as in, we don't support it, and recommend against it when
>   asked
> * not tested, we don't test stripped binaries as part of the build CI process
> * is often broken, stripping a go binary will produce anywhere from no, to
>   subtle, to outright execution failure, see above

### [launchpad golang issue #1200255, comment #13](https://bugs.launchpad.net/ubuntu/+source/golang/+bug/1200255/comments/13)

> To clarify my previous statements.
>
> * I do not disagree with the debian policy, it is there for a good reason
> * Having said that, it stripping Go binaries doesn't work, and nobody is
>   looking at making it work, so there is that.
>
> Thanks for patching the build formula.

## Building Docker

Please use our build script ("./hack/make.sh") for all your compilation of
Docker. If there's something you need that it isn't doing, or something it could
be doing to make your life as a packager easier, please get in touch with Tianon
and help us rectify the situation. Chances are good that other packagers have
probably run into the same problems and a fix might already be in the works, but
none of us will know for sure unless you harass Tianon about it. :)

All the commands listed within this section should be run with the Docker source
checkout as the current working directory.

### `AUTO_GOPATH`

If you'd rather not be bothered with the hassles that setting up `GOPATH`
appropriately can be, and prefer to just get a "build that works", you should
add something similar to this to whatever script or process you're using to
build Docker:

```bash
export AUTO_GOPATH=1
```

This will cause the build scripts to set up a reasonable `GOPATH` that
automatically and properly includes both docker/docker from the local
directory, and the local "./vendor" directory as necessary.

### `DOCKER_BUILDTAGS`

If you're building a binary that may need to be used on platforms that include
AppArmor, you will need to set `DOCKER_BUILDTAGS` as follows:
```bash
export DOCKER_BUILDTAGS='apparmor'
```

If you're building a binary that may need to be used on platforms that include
SELinux, you will need to use the `selinux` build tag:
```bash
export DOCKER_BUILDTAGS='selinux'
```

If you're building a binary that may need to be used on platforms that include
seccomp, you will need to use the `seccomp` build tag:
```bash
export DOCKER_BUILDTAGS='seccomp'
```

There are build tags for disabling graphdrivers as well. By default, support
for all graphdrivers are built in.

To disable btrfs:
```bash
export DOCKER_BUILDTAGS='exclude_graphdriver_btrfs'
```

To disable devicemapper:
```bash
export DOCKER_BUILDTAGS='exclude_graphdriver_devicemapper'
```

To disable aufs:
```bash
export DOCKER_BUILDTAGS='exclude_graphdriver_aufs'
```

NOTE: if you need to set more than one build tag, space separate them:
```bash
export DOCKER_BUILDTAGS='apparmor selinux exclude_graphdriver_aufs'
```

### Static Daemon

If it is feasible within the constraints of your distribution, you should
seriously consider packaging Docker as a single static binary. A good comparison
is Busybox, which is often packaged statically as a feature to enable mass
portability. Because of the unique way Docker operates, being similarly static
is a "feature".

To build a static Docker daemon binary, run the following command (first
ensuring that all the necessary libraries are available in static form for
linking - see the "Build Dependencies" section above, and the relevant lines
within Docker's own Dockerfile that set up our official build environment):

```bash
./hack/make.sh binary
```

This will create a static binary under
"./bundles/$VERSION/binary/docker-$VERSION", where "$VERSION" is the contents of
the file "./VERSION". This binary is usually installed somewhere like
"/usr/bin/docker".

### Dynamic Daemon / Client-only Binary

If you are only interested in a Docker client binary, you can build using:

```bash
./hack/make.sh binary-client
```

If you need to (due to distro policy, distro library availability, or for other
reasons) create a dynamically compiled daemon binary, or if you are only
interested in creating a client binary for Docker, use something similar to the
following:

```bash
./hack/make.sh dynbinary-client
```

This will create "./bundles/$VERSION/dynbinary-client/docker-$VERSION", which for
client-only builds is the important file to grab and install as appropriate.

## System Dependencies

### Runtime Dependencies

To function properly, the Docker daemon needs the following software to be
installed and available at runtime:

* iptables version 1.4 or later
* procps (or similar provider of a "ps" executable)
* e2fsprogs version 1.4.12 or later (in use: mkfs.ext4, tune2fs)
* xfsprogs (in use: mkfs.xfs)
* XZ Utils version 4.9 or later
* a [properly
  mounted](https://github.com/tianon/cgroupfs-mount/blob/master/cgroupfs-mount)
  cgroupfs hierarchy (having a single, all-encompassing "cgroup" mount point
  [is](https://github.com/docker/docker/issues/2683)
  [not](https://github.com/docker/docker/issues/3485)
  [sufficient](https://github.com/docker/docker/issues/4568))

Additionally, the Docker client needs the following software to be installed and
available at runtime:

* Git version 1.7 or later

### Kernel Requirements

The Docker daemon has very specific kernel requirements. Most pre-packaged
kernels already include the necessary options enabled. If you are building your
own kernel, you will either need to discover the options necessary via trial and
error, or check out the [Gentoo
ebuild](https://github.com/tianon/docker-overlay/blob/master/app-emulation/docker/docker-9999.ebuild),
in which a list is maintained (and if there are any issues or discrepancies in
that list, please contact Tianon so they can be rectified).

Note that in client mode, there are no specific kernel requirements, and that
the client will even run on alternative platforms such as Mac OS X / Darwin.

### Optional Dependencies

Some of Docker's features are activated by using optional command-line flags or
by having support for them in the kernel or userspace. A few examples include:

* AUFS graph driver (requires AUFS patches/support enabled in the kernel, and at
  least the "auplink" utility from aufs-tools)
* BTRFS graph driver (requires BTRFS support enabled in the kernel)
* ZFS graph driver (requires userspace zfs-utils and a corresponding kernel module)
* Libseccomp to allow running seccomp profiles with containers

## Daemon Init Script

Docker expects to run as a daemon at machine startup. Your package will need to
include a script for your distro's process supervisor of choice. Be sure to
check out the "contrib/init" folder in case a suitable init script already
exists (and if one does not, contact Tianon about whether it might be
appropriate for your distro's init script to live there too!).

In general, Docker should be run as root, similar to the following:

```bash
dockerd
```

Generally, a `DOCKER_OPTS` variable of some kind is available for adding more
flags (such as changing the graph driver to use BTRFS, switching the location of
"/var/lib/docker", etc).

## Communicate

As a final note, please do feel free to reach out to Tianon at any time for
pretty much anything. He really does love hearing from our packagers and wants
to make sure we're not being a "hostile upstream". As should be a given, we
appreciate the work our packagers do to make sure we have broad distribution!
