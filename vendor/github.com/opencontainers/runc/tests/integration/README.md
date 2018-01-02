# runc Integration Tests

Integration tests provide end-to-end testing of runc.

Note that integration tests do **not** replace unit tests.

As a rule of thumb, code should be tested thoroughly with unit tests.
Integration tests on the other hand are meant to test a specific feature end
to end.

Integration tests are written in *bash* using the
[bats](https://github.com/sstephenson/bats) framework.

## Running integration tests

The easiest way to run integration tests is with Docker:
```
$ make integration
```
Alternatively, you can run integration tests directly on your host through make:
```
$ sudo make localintegration
```
Or you can just run them directly using bats
```
$ sudo bats tests/integration
```
To run a single test bucket:
```
$ make integration TESTFLAGS="/checkpoint.bats"
```


To run them on your host, you will need to setup a development environment plus
[bats](https://github.com/sstephenson/bats#installing-bats-from-source)
For example:
```
$ cd ~/go/src/github.com
$ git clone https://github.com/sstephenson/bats.git
$ cd bats
$ ./install.sh /usr/local
```

> **Note**: There are known issues running the integration tests using
> **devicemapper** as a storage driver, make sure that your docker daemon
> is using **aufs** if you want to successfully run the integration tests.

## Writing integration tests

[helper functions]
(https://github.com/opencontainers/runc/blob/master/test/integration/helpers.bash)
are provided in order to facilitate writing tests.

```sh
#!/usr/bin/env bats

# This will load the helpers.
load helpers

# setup is called at the beginning of every test.
function setup() {
  # see functions teardown_hello and setup_hello in helpers.bash, used to
  # create a pristine environment for running your tests
  teardown_hello
  setup_hello
}

# teardown is called at the end of every test.
function teardown() {
  teardown_hello
}

@test "this is a simple test" {
  runc run containerid
  # "The runc macro" automatically populates $status, $output and $lines.
  # Please refer to bats documentation to find out more.
  [ "$status" -eq 0 ]

  # check expected output
  [[ "${output}" == *"Hello"* ]]
}

```
