# rkt functional tests

This directory contains a set of functional tests for rkt.
The tests use [gexpect](https://github.com/coreos/gexpect) to spawn various `rkt run` commands and look for expected output.

## Semaphore

The tests run on the [Semaphore](https://semaphoreci.com/) CI system through the [`rktbot`](https://semaphoreci.com/rktbot) user, which is part of the [`coreos`](https://semaphoreci.com/coreos/) org on Semaphore.
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
./tests/run-build.sh none
./tests/run-build.sh src v229
```

#### Thread 2

```
./tests/run-build.sh coreos
./tests/run-build.sh host
```

#### Post thread

```
git clean -ffdx
```

#### Other possible commands

The LKVM stage1 or other versions of systemd are not currently tested.
It would be possible to add more tests with the following commands:

```
./tests/run-build.sh src v227
./tests/run-build.sh src master
./tests/run-build.sh kvm
```

### Platform

Select `Ubuntu 14.04 LTS v1503 (beta with Docker support)`.
The platform with *Docker support* means the tests will run in a VM.

## Manually running the functional tests

Make sure to pass `--enable-functional-tests` to the configure script, then, after building the project, you can run the tests.

```
./configure --enable-functional-tests
make -j4
make check
```

For more details about the `--enable-functional-tests` parameter, see [configure script parameters documentation](build-configure.md).
The snippet above will run both unit and functional tests.
If you want to run only functional tests, use `make functional-check`.
There is also a counterpart target for running unit tests only - it is named `unit-check`.

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
