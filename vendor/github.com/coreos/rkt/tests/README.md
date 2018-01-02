# rkt functional tests

This directory contains a set of functional tests for rkt.
The tests use [gexpect](https://github.com/coreos/gexpect) to spawn various `rkt run` commands and look for expected output.

## Semaphore Continuous Integration System

The tests run on the [Semaphore CI system](https://semaphoreci.com/) through the [`rktbot`](https://semaphoreci.com/rktbot) user, which is part of the [`coreos`](https://semaphoreci.com/coreos/) org on Semaphore.
This user is authorized against the corresponding [`rktbot`](https://github.com/rktbot) GitHub account.
The credentials for `rktbot` are currently managed by CoreOS.

The tests are executed on Semaphore at each Pull Request (PR).
Each GitHub PR page should have a link to the [test results on Semaphore](https://semaphoreci.com/coreos/rkt).

Developers can disable the tests by adding `[skip ci]` in the last commit message of the PR.

### Build settings

Select the "Other" language.
We don't use "Go" language setting, because rkt is not a typical go project (building it with a go get won't get you too far).
Also, the "Go" setting is creating a proper GOPATH directory structure with some symlinks on top, which rkt does not need at all and some go tools we use do not like the symlinks in GOPATH at all.

The tests will run on two VMs.
The "Setup" and "Post thread" sections will be executed on both VMs.
The "Thread 1" and "Thread 2" will be executed in parallel in separate VMs.

#### Setup

```
sudo groupadd rkt
sudo gpasswd -a runner rkt
./tests/install-deps.sh
```

#### Thread 1

```
./tests/build-and-run-tests.sh -f none -c
./tests/build-and-run-tests.sh -f kvm -c
```

#### Thread 2

```
./tests/build-and-run-tests.sh -f coreos -c
./tests/build-and-run-tests.sh -f host -c
```

#### Post thread

```
git clean -ffdx
```

#### Other possible commands

The LKVM stage1 or other versions of systemd are not currently tested.
It would be possible to add more tests with the following commands:

```
./tests/build-and-run-tests.sh -f src -s v227 -c
./tests/build-and-run-tests.sh -f src -s master -c
./tests/build-and-run-tests.sh -f src -s v229 -c
```

#### build-and-run-tests.sh parameters description

The build script has the following parameters:
- `-c` - Run cleanup. Cleanup has two phases: *after build* and *after tests*. In the *after build* phase, this script removes artifacts from external dependencies (like kernel sources in the `kvm` flavor). In the  *after tests* phase, it removes `rkt` build artifacts and (if the build is running on CI or if the `-x` flag is used) it unmounts the remaining `rkt` mountpoints, removes unused `rkt` NICs and flushes the current state of IPAM IP reservation.
- `-d` - Run build based on current state of local rkt repository instead of commited changes.
- `-f` - Select flavor for rkt. You can choose only one from the following list: "`coreos`, `host`, `kvm`, `none`, `src`".
- `-j` - Build without running unit and functional tests. Artifacts are available after build.
- `-s` - Systemd version. You can choose `master` or a tag from the [systemd GitHub repository](https://github.com/systemd/systemd).
- `-u` - Show usage message and exit.
- `-x` - Force after-test cleanup on a non-CI system. **WARNING: This flag can affect your system. Use with caution.**

### Platform

Select `Ubuntu 14.04 LTS v1503 (beta with Docker support)`.
The platform with *Docker support* means the tests will run in a VM.

## Manually running the tests

The tests can be run manually. There is a rule to run unit, functional and all tests.

### Unit tests

The unit tests can be run with `make unit-check` after you [built](../Documentation/hacking.md#building-rkt) the project.

### Functional tests

The functional tests require to pass `--enable-functional-tests` to the configure script, then, after building the project, you can run the tests.

```
./autogen.sh
./configure --enable-functional-tests
make -j4
make functional-check
```

For more details about the `--enable-functional-tests` parameter, see [configure script parameters documentation](../Documentation/build-configure.md#--enable-functional-tests).

### All tests

To run all tests, see [functional tests](./README.md#functional-tests) to configure and build it with functional tests enabled. Instead of `make functional-check` you have to call `make check` to run all tests.

### Passing additional parameters

You can use a `GO_TEST_FUNC_ARGS` variable to pass additional parameters to `go test`.
This is mostly useful for running only the selected functional tests.
The variable is ignored in unit tests.

```
make check GO_TEST_FUNC_ARGS='-run NameOfTheTest'
make functional-check GO_TEST_FUNC_ARGS='-run NameOfTheTest'
```

Run `go help testflag` to get more informations about possible flags accepted by `go test`.

## Running the benchmark

Running the benchmark is similar to running the other tests, we just need to pass additional
parameters to `go test`:

```
make check GO_TEST_FUNC_ARGS='-bench=. -run=Benchmark'
make functional-check GO_TEST_FUNC_ARGS='-bench=. -run=Benchmark'
```
