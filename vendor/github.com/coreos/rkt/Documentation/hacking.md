# Hacking Guide

## Overview

This guide contains instructions for those looking to hack on rkt.
For more information on the rkt internals, see the [`devel`](devel/) documentation.

## Building rkt

The easiest way to build rkt is by using the coreos.com/rkt/builder ACI image. See instructions for how to use it in the README at [github.com/coreos/rkt-builder][rkt-builder].

Alternatively, you should be able build rkt on any modern Linux system with [Go](https://golang.org/) (1.5+) installed.
For the most part the codebase is self-contained (e.g. all dependencies are vendored), but assembly of the stage1 requires some other tools to be installed on the system.
Please see [the list of the build-time dependencies][build-time-dependencies].
Once the dependencies have been satisfied you can build rkt with a default configuration by running the following commands:

```
git clone https://github.com/coreos/rkt.git
cd rkt
./autogen.sh && ./configure && make
```

Build verbosity can be controlled with the V variable.
Set V to 0 to have a silent build.
Set V to either 1 or 2 to get short messages about what is being done (level 2 prints more of them).
Set V to 3 to get raw output.
Instead of a number, English words can be used: `quiet` or `silent` for level 0, `info` for level 1, `all` for level 2 and `raw` for level 3.
For example, `make V=raw` is equivalent to `make V=3`.

To be able to run rkt, please see [the list of the run-time dependencies][run-time-dependencies].

### Building rkt with Docker

Alternatively, you can build rkt in a Docker container with the following command.
Replace `$SRC` with the absolute path to your rkt source code:

```
# docker run -v $SRC:/opt/rkt debian:sid /bin/bash -c "cd /opt/rkt && ./scripts/install-deps-debian-sid.sh && ./autogen.sh && ./configure && make"
```

### Building systemd in stage1 from source

By default, rkt gets systemd from a CoreOS image to generate stage1.
It's also possible to build systemd from source.
To do this, use the following `configure` parameters after running `./autogen.sh`:

- `--with-stage1-flavors`
- `--with-stage1-default-flavor` (optional)
- `--with-stage1-systemd-version`
- `--with-stage1-systemd-revision` (optional)
- `--with-stage1-systemd-src`

For more details, see [configure script parameters documentation][build-configure].
Example:

```
./autogen.sh && ./configure --with-stage1-flavors=src --with-stage1-systemd-version=v231 --with-stage1-systemd-revision=master --with-stage1-systemd-src=$HOME/src/systemd && make
```

### Building stage1 with kvm as execution engine

The stage1 kvm image is based on CoreOS, but with additional components for running containers on top of a hypervisor.

To build this stage1 image, pass `kvm` to `--with-stage1-flavors` parameter in `./configure`

This will generate a stage1 with an embedded kernel and kvmtool, which launches each pod in a separate virtual machine.

Additional build dependencies for the stage1 kvm follow.
If building with docker, these must be added to the `apt-get install` command.

* wget
* xz-utils
* patch
* bc
* libssl-dev

### Alternative stage1 paths

rkt is designed and intended to be modular, using a [staged architecture][architecture].

`rkt run` determines the stage1 image it should use via its `--stage1-{url,path,name,hash,from-dir}` flags.
If this flag is not given to rkt, the stage1 image will default to the settings taken from the configuration.
If those are unset, rkt will fall back to the settings configured when rkt was built from source.
It usually means that rkt will look for a file called `stage1-<default flavor>.aci` that is in the same directory as the rkt binary itself.

However, a default value can be set for this parameter at build time by setting the option `--with-stage1-default-location` when invoking `./configure`.
It can be set with the `paths` kind of configuration.
For more details, see [configure script parameters documentation][build-configure] and [configuration documentation][configuration].

rkt expects stage1 images to be signed except in the following cases:

* it is the default stage1 image and it's in the same directory as the rkt binary
* `--stage1-{name,hash}` is used and the image is already in the store
* `--stage1-{url,path,from-dir}` is used and the image is in the default directory configured at build time

### Updating the coreos flavor stage1

Follow the instructions on [Update coreos flavor stage1][update-coreos-stage1].

## Managing dependencies

rkt uses [`glide`][glide] and [`glide-vc`][glide-vc] to manage third-party dependencies.
The build process is crafted to make this transparent to most users (i.e. if you're just building rkt from source, or modifying any of the codebase without changing dependencies, you should have no need to interact with glide).
But occasionally the need arises to either a) add a new dependency or b) update/remove an existing dependency.

We might want to vendor an application for several reasons:

- it will be used at build-time (like actool to build stage1 images)
- it will be a part of a stage1 image (like CNI plugins for networking)
- it will be used in functional tests (like ACE validator)

### Update glide/glide-vc

Ensure you have the **latest version** of `glide` and `glide-vc` available in your `PATH`.

#### Add a new dependency

Use the glide tool to add a new dependency. In order to add a dependency to a package i.e. `github.com/fizz/buzz` for version `1.2.3`, execute:
```
$ glide get github.com/fizz/buzz#v1.2.3
$ ./scripts/glide-update.sh
```

Note that although glide does support [versions and ranges][glide-versioning] currently it is preferred to pin to concrete versions as described above.

*Note*: Do *not* use `go get` and `glide update` to add new dependencies. It will cause both `glide.lock` and `glide.yaml` files to diverge.

#### Update existing dependencies

Once in a while new versions of dependencies are available. Entries in the `glide.yaml` file specify the target version. To update a dependency, edit the appropriate entry and specify the updated target version.

*Note*: Changing specific entries in `glide.yaml` does not imply that only those will be updated. Glide will pull potential updates for all dependencies.

To update a vendored dependency to a newer version, first update its target version directly in `glade.yaml`. The glide update script will then take care of pulling all dependencies and refreshing any updated ones, according to version constraints specified in the YAML manifest.

*Note*: Glide will pull all dependencies from all referenced repos potentially causing a lot of network traffic.

Once done editing glide.yaml, execute the glide update script:
```
$ ./scripts/glide-update.sh
```

#### Resolving transitive dependency conflicts

Glide currently has no deterministic mechanism to resolve transitive dependency conflicts. A transitive dependency conflict exists if package `A` depends on `B`, and a package `C` also depends on `B`, but on a different version.

To resolve this conflict on package `C` specify the version directly in the `glide.yaml` file as described above.

#### Removing an existing dependency

Execute:
```
$ glide rm github.com/fizz/buzz
$ ./scripts/glide-update.sh
```

## Errors & Output

rkt attempts to offer consistent and structured error output. To achieve this, we use a couple helper types which we'll describe below.

### Wrapping errors

rkt uses the errwrap package to structure errors. This allows us to manage how we output errors. You can wrap errors by doing the following.

```
err := funcReturningSomeError()
errwrap.Wrap(errors.New("My new error"), err)
```

### Logging errors

For writing to output rkt uses its own log package which is essentially a wrapper around the golang log package. This is used to write to both `stderr` and `stdout`. By doing this, rkt can easily change the way it formats its output.

A few new methods are added to control the output of the wrapped errors. For example, the following outputs an error to `stderr`.

```
log := rktlog.Logger(os.Stderr, "rkt", debug)

log.PrintE("a message to accompany the error", err)
```

There are similar functions named `FatalE`, `PanicE`. All the other methods from golang's log package are available.

### Writing to stdout

In order to write to `stdout`, we also use the rkt log package. If not already set up in your package, one can be created as follows.

```
stdout := rktlog.Logger(os.Stdout, "", false)
```

Here, the prefix is an empty string and debug is set to `false`.

## Finishing Up

At this point, you should be good to submit a PR.
As well as a simple sanity check that the code actually builds and tests pass, here are some things to look out for:
- `git status Godeps/` should show only a minimal and relevant change (i.e. only the dependencies you actually intended to touch).
- `git diff Godeps/` should be free of any changes to import paths within the vendored dependencies
- `git diff` should show _all_ third-party import paths prefixed with `Godeps/_workspace`

If something looks awry, restart, pray to your preferred deity, and try again.


[architecture]: devel/architecture.md
[build-configure]: build-configure.md
[build-time-dependencies]: dependencies.md#build-time-dependencies
[configuration]: configuration.md
[glide]: https://glide.sh
[glide-vc]: https://github.com/sgotti/glide-vc
[glide-versioning]: https://glide.readthedocs.io/en/latest/versions/
[go]: https://golang.org/
[rkt-builder]: https://github.com/coreos/rkt-builder
[run-time-dependencies]: dependencies.md#run-time-dependencies
[update-coreos-stage1]: devel/update-coreos-stage1.md
