# Hacking Guide

## Overview

This guide contains instructions for those looking to hack on rkt.
For more information on the rkt internals, see the [`devel`](devel/) documentation.

## Building rkt

You should be able build rkt on any modern Linux system.
For the most part the codebase is self-contained (e.g. all dependencies are vendored), but assembly of the stage1 requires some other tools to be installed on the system.
Please see [the list of the build-time dependencies](dependencies.md#build-time-dependencies).
Once the dependencies have been satisfied you can build rkt by running the following commands:

```
git clone https://github.com/coreos/rkt.git
cd rkt
./autogen.sh && ./configure && make
```

Build verbosity can be controlled with the V variable.
Set V to 0 to have a silent build.
Set V to either 1 or 2 to get short messages about what is being done (level 2 prints more of them).
Set V to 3 to get raw output.
Instead of a number, english words can be used.
`quiet` or `silent` for level 0, `info` for level 1, `all` for level 2 and `raw` for level 3. Example:

`make V=raw`

To be able to run rkt, please see [the list of the run-time dependencies](dependencies.md#run-time-dependencies).

### With Docker

Alternatively, you can build rkt in a Docker container with the following command.
Replace $SRC with the absolute path to your rkt source code:

```
# docker run -v $SRC:/opt/rkt -i -t golang:1.5 /bin/bash -c "apt-get update && apt-get install -y coreutils cpio squashfs-tools realpath autoconf file xz-utils patch bc && cd /opt/rkt && go get github.com/appc/spec/... && ./autogen.sh && ./configure && make"
```

### Building systemd in stage1 from the sources

By default, rkt gets systemd from a CoreOS image to generate stage1.
But it's also possible to build systemd from the sources.
To do this, use the following configure parameters after running `./autogen.sh`:

- `--with-stage1-flavors`
- `--with-stage1-default-flavor` (optional)
- `--with-stage1-systemd-version`
- `--with-stage1-systemd-src`

For more details, see [configure script parameters documentation](build-configure.md).
Example:

```
./autogen.sh && ./configure --with-stage1-flavors=src --with-stage1-systemd-version=master --with-stage1-systemd-src=$HOME/src/systemd && make
```

### Building stage1 with kvm as execution engine

The stage1 kvm image is based on CoreOS, but with additional components for running containers on top of a hypervisor.

To build, pass `kvm` to `--with-stage1-flavors` parameter in `./configure`

This will generate stage1 with embedded kernel and kvmtool to start pod in virtual machine.

Additional build dependencies for the stage1 kvm follow.
If building with docker, these must be added to the `apt-get install` command.

* wget
* xz-utils
* patch
* bc

### Alternative stage1 paths

rkt is designed and intended to be modular, using a [staged architecture](devel/architecture.md).

`rkt run` determines the stage1 image it should use via its `--stage1-{url,path,name,hash,from-dir}` flags.
If this flag is not given to rkt, the stage1 image will default to the settings taken from the configuration.
If those are unset, rkt will fall back to the settings configured when rkt was built from source.
It usually means that rkt will look for a file called `stage1-<default flavor>.aci` that is in the same directory as the rkt binary itself.

However, a default value can be set for this parameter at build time by setting the option `--with-stage1-default-location` when invoking `./configure`.
It can be set with the `paths` kind of configuration.
For more details, see [configure script parameters documentation](build-configure.md) and [configuration documentation](configuration.md).

## Managing Dependencies

rkt uses [`godep`](https://github.com/tools/godep) to manage third-party dependencies.
The build process is crafted to make this transparent to most users (i.e. if you're just building rkt from source, or modifying any of the codebase without changing dependencies, you should have no need to interact with godep).
But occasionally the need arises to either a) add a new dependency or b) update/remove an existing dependency.
There are two types of dependencies:

- libraries - Go code imported by some Go source file in the repository
- applications - Go code that we need to build to get some executable binary; this means that we want to "import" a "main" package.

We might want to vendor an application for several reasons:

- it will be used at build-time (like actool to build stage1 images)
- it will be a part of a stage1 image (like CNI plugins for networking)
- it will be used in functional tests (like ACE validator)

At this point, the ramblings below from an experienced Godep victim^Wenthusiast might prove of use...

### Update godep

Step zero is generally to ensure you have the **latest version** of `godep` available in your `PATH`.

### Having the right directory layout (i.e. `GOPATH`)

To work with `godep`, you'll need to have the repository (i.e. `github.com/coreos/rkt`) checked out in a valid `GOPATH`.
If you use the [standard Go workflow](https://golang.org/doc/code.html#Organization), with every package in its proper place in a workspace, this should be no problem.
As an example, if one was obtaining the repository for the first time, one would do the following:

```
$ export GOPATH=/tmp/foo               # or any directory you please
$ go get -d github.com/coreos/rkt/...  # or 'git clone https://github.com/coreos/rkt $GOPATH/src/github.com/coreos/rkt'
$ cd $GOPATH/src/github.com/coreos/rkt
```

If, however, you instead prefer to manage your source code in directories like `~/src/rkt`, there's a problem: `godep` doesn't like symbolic links (which is what the rkt build process uses by default to create a self-contained GOPATH).
Hence, you'll need to work around this with bind mounts, with something like the following:

```
$ export GOPATH=/tmp/foo        # or any directory you please
$ mkdir -p $GOPATH/src/github.com/coreos/rkt
# mount --bind ~/src/rkt $GOPATH/src/github.com/coreos/rkt
$ cd $GOPATH/src/github.com/coreos/rkt
```

One benefit of this approach over the single-workspace workflow is that checking out different versions of dependencies in the `GOPATH` (as we are about to do) is guaranteed to not affect any other packages in the `GOPATH`.
(Using [gvm](https://github.com/moovweb/gvm) or other such tomfoolery to manage `GOPATH`s is an exercise left for the reader.)

### Restoring the current state of dependencies

Now that we have a functional `GOPATH`, use `godep` to restore the full set of vendored dependencies to their correct versions.
(What this command does is essentially just loop over the set of dependencies codified in `Godeps/Godeps.json`, using `go get` to retrieve and then `git checkout` (or equivalent) to set each to their correct revision.)

```
$ godep restore # might take a while if it's the first time...
```

At this stage, your path forks, depending on what exactly you want to do: add, update or remove a dependency, or add, update or remove an application.
But in _all six cases_, the procedure finishes with the [same save command](#saving-the-set-of-dependencies).

#### Add a new dependency

In this case you'll first need to retrieve the dependency you're working with into `GOPATH`.
As a simple example, assuming we're adding `github.com/fizz/buzz/tazz`:

```
$ go get -d github.com/fizz/buzz
```

##### If it is a library

Add the new dependency into `godep`'s purview by simply importing the standard package name in one of your sources:

```
$ vim $GOPATH/src/github.com/coreos/rkt/some/file.go
...
import "github.com/fizz/buzz/tazz"
...
```

Now, GOTO [saving](#saving-the-set-of-dependencies)

##### If it is an application

Add the new application to the `vendoredApps` file in the root of the repository:

```
$ vim vendoredApps
...
github.com/fizz/buzz/tazz
...
```

Now, GOTO [saving](#saving-the-set-of-dependencies)

#### Update an existing dependency

The steps here are the same for both libraries and applications.
In this case, assuming we're updating `github.com/foo/bar`:

```
$ cd $GOPATH/src/github.com/foo/bar
$ git pull   # or 'go get -d -u github.com/foo/bar/...'
$ git checkout $DESIRED_REVISION
$ cd $GOPATH/src/github.com/coreos/rkt
$ godep update github.com/foo/bar/...
```

Now, GOTO [saving](#saving-the-set-of-dependencies)

#### Removing an existing dependency

This is the simplest case of all.

##### If it is a library

Simply remove all references to a dependency from the source files.

Now, GOTO [saving](#saving-the-set-of-dependencies)

##### If it is an application

Simply remove the relevant line from the `vendoredApps` file.

Now, GOTO [saving](#saving-the-set-of-dependencies)

### Saving the set of dependencies

Finally, here we are, the magic command, the holy grail, the ultimate conclusion of all `godep` operations.
Provided you have followed the preceding instructions, regardless of whether you are adding/removing/modifying dependencies, this innocuous script will cast the necessary spells to solve all of your dependency worries:

```
$ ./scripts/godep-save
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
