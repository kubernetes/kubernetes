# vcsim - A vCenter and ESXi API based simulator

This package implements a vSphere Web Services (SOAP) SDK endpoint intended for testing consumers of the API.
While the mock framework is written in the Go language, it can be used by any language that can talk to the vSphere
API.

## Installation

```sh
% export GOPATH=$HOME/gopath
% go get -u github.com/vmware/govmomi/vcsim
% $GOPATH/bin/vcsim -h
```

## Usage

The **vcsim** program by default creates a *vCenter* model with a datacenter, hosts, cluster, resource pools, networks
and a datastore.  The naming is similar to that of the original *vcsim* mode that was included with vCenter.  The number
of resources can be increased or decreased using the various resource type flags.  Resources can also be created and
removed using the API.

Example using the default settings:

```
% export GOVC_URL=https://user:pass@127.0.0.1:8989
% $GOPATH/bin/vcsim
% govc find
/
/DC0
/DC0/vm
/DC0/vm/DC0_H0_VM0
/DC0/vm/DC0_H0_VM1
/DC0/vm/DC0_C0_RP0_VM0
/DC0/vm/DC0_C0_RP0_VM1
/DC0/host
/DC0/host/DC0_H0
/DC0/host/DC0_H0/DC0_H0
/DC0/host/DC0_H0/Resources
/DC0/host/DC0_C0
/DC0/host/DC0_C0/DC0_C0_H0
/DC0/host/DC0_C0/DC0_C0_H1
/DC0/host/DC0_C0/DC0_C0_H2
/DC0/host/DC0_C0/Resources
/DC0/datastore
/DC0/datastore/LocalDS_0
/DC0/network
/DC0/network/VM Network
/DC0/network/DVS0
/DC0/network/DC0_DVPG0
```

Example using ESX mode:

```
% $GOPATH/vcsim -esx
% govc find
/
/ha-datacenter
/ha-datacenter/vm
/ha-datacenter/vm/ha-host_VM0
/ha-datacenter/vm/ha-host_VM1
/ha-datacenter/host
/ha-datacenter/host/localhost.localdomain
/ha-datacenter/host/localhost.localdomain/localhost.localdomain
/ha-datacenter/host/localhost.localdomain/Resources
/ha-datacenter/datastore
/ha-datacenter/datastore/LocalDS_0
/ha-datacenter/network
/ha-datacenter/network/VM Network

```

## Supported methods

The simulator supports a subset of API methods.  However, the generated [govmomi](https://github.com/vmware/govmomi)
code includes all types and methods defined in the vmodl, which can be used to implement any method documented in the
[VMware vSphere API Reference](http://pubs.vmware.com/vsphere-6-5/index.jsp#com.vmware.wssdk.apiref.doc/right-pane.html).

To see the list of supported methods:

```
curl -sk https://user:pass@127.0.0.1:8989/about
```

## Listen address

The default vcsim listen address is `127.0.0.1:8989`.  Use the `-httptest.serve` flag to listen on another address:


``` shell
vcsim -l 10.118.69.224:8989 # specific address

vcsim -l :8989 # any address
```

When given a port value of '0', an unused port will be chosen.  You can then source the GOVC_URL from another
process, for example:

```sh
govc_sim_env=$TMPDIR/vcsim-$(uuidgen)

mkfifo $govc_sim_env

vcsim -l 127.0.0.1:0 -E $govc_sim_env &

eval "$(cat $govc_sim_env)"

# ... run tests ...

kill $GOVC_SIM_PID
rm -f $govc_sim_env
```

Tests written in Go can also use the [simulator package](https://godoc.org/github.com/vmware/govmomi/simulator)
directly, rather than the vcsim binary.

## Introducing delays
Sometimes, especially when debugging software, it can be useful to introduce delays to simulate network latency or a poorly performing vCenter. There are three command line options for dealing with delays.

```-delay <ms>``` Adds a constant delay (experessed in milliseconds) to every call

```-method-delay <method:milliseconds,method:milliseconds...>``` Adds a specified delay to individual methods. If both ```-method-delay``` and ```-delay``` are specified, they are added together

```delay-jitter``` Specifies a jitter, i.e. a random value added to or subtracted from the delay. It is specified as a <i>Coefficient of Variation</i>, which is the same as the standard deviation divided by the mean. A reasonable starting value is 0.5, as it gives a nice variation without extreme outliers.

## Projects using vcsim

* [VMware VIC Engine](https://github.com/vmware/vic)

* [Kubernetes](https://github.com/kubernetes/kubernetes/tree/master/pkg/cloudprovider/providers/vsphere)

* [Ansible](https://github.com/ansible/vcenter-test-container)

* [Telegraf](https://github.com/influxdata/telegraf/tree/master/plugins/inputs/vsphere)

## Related projects

[LocalStack](https://github.com/localstack/localstack/blob/master/README.md#why-localstack)

## Blog posts on vcsim

* [Beginning vCenter Server simulation with Govcsim](https://opensourceforu.com/2017/10/vcenter-server-simulation-govcsim/) By Abhijeet Kasurde - October 3, 2017

* [govcsim – Neat incubation project (vCenter Server & ESXi API based simulator)](https://www.virtuallyghetto.com/2017/04/govcsim-neat-incubation-project-vcenter-server-esxi-api-based-simulator.html) by [William Lam](https://twitter.com/lamw) - April 21, 2017
