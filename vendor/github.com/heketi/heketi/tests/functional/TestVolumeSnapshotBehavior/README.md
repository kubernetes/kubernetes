# TestVolumeSnapshotBehavior
Test that Heketi manages volumes with snapshots correctly

## Requirements

* Vagrant
* Andible
* Hypervisor: Only Libvirt/KVM

For simplicity you can use the ansible script in https://github.com/heketi/setup-vagrant-libvirt
to setup your system for functional tests.

## Setup

Type:

```
$ ./up
```

## Running the tests

* Go to the top of the source tree build and run a new Heketi server:

```
$ rm heketi.db
$ make
$ ./heketi --config=tests/functional/TestVolumeSnapshotBehavior/config/heketi.json 2>&1 | tee log

```

* Then start running the tests

```
$ cd tests/functional/large/tests
$ go test -timeout=1h -tags ftlarge
```

Output will be shows by the logs on the heketi server.
