# LibNetwork Integration Tests

Integration tests provide end-to-end testing of LibNetwork and Drivers.

While unit tests verify the code is working as expected by relying on mocks and
artificially created fixtures, integration tests actually use real docker
engines and communicate to it through the CLI.

Note that integration tests do **not** replace unit tests and Docker is used as a good use-case.

As a rule of thumb, code should be tested thoroughly with unit tests.
Integration tests on the other hand are meant to test a specific feature end to end.

Integration tests are written in *bash* using the
[bats](https://github.com/sstephenson/bats) framework.

## Pre-Requisites

1. Bats (https://github.com/sstephenson/bats#installing-bats-from-source)
2. Docker Machine (https://github.com/docker/machine)
3. Virtualbox (as a Docker machine driver)

## Running integration tests

* Start by [installing] (https://github.com/sstephenson/bats#installing-bats-from-source) *bats* on your system.
* If not done already, [install](https://docs.docker.com/machine/) *docker-machine* into /usr/bin
* Make sure Virtualbox is installed as well, which will be used by docker-machine as a driver to launch VMs

In order to run all integration tests, pass *bats* the test path:
```
$ bats test/integration/daemon-configs.bats
```


