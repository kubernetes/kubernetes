# Small Functional Test
This functional test can be used on a system with at least 8GB of RAM.

## Requirements

* Vagrant
* Ansible
* Hypervisor: VirtualBox or Libvirt/KVM

## Setup 

* Go to `tests/functional/TestSmokeTest/vagrant`
Type:
```
$ ./up.sh --provider=PROVIDER
```
where PROVIDER is virtualbox or libvirt.

## Running the Tests

* Go to the top of the source tree build and run a new Heketi server:

```
$ rm heketi.db
$ make
$ ./heketi --config=tests/functional/TestSmokeTest/config/heketi.json | tee log

```

* Once it is ready, then start running the tests in another terminal

```
$ cd tests/functional/TestSmokeTest/tests
$ go test -tags ftsmall
```

Output will be shown by the logs on the heketi server.
