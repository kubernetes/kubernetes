# Kubernetes Cluster Boostrap Made Easy

## Usage

### `kubeadm init`

It's usually enough to run `kubeadm init`, but in some case you might like to override the
default behaviour. The flags used for said purpose are described below.

- `--token=<token>`

By default, a token is generated, but if you are to automate cluster deployment, you will want to
set the token ahead of time. Read the docs for more information on the token format.

- `--api-advertise-addresses=<ips>` (multiple values are allowed by having multiple flag declarations or multiple values separated by comma)
- `--api-external-dns-names=<domain>` (multiple values are allowed by having multiple flag declarations or multiple values separated by comma)

By default, `kubeadm` will auto detect IP addresses and use that to generate API server certificates.
If you would like to access the API via any external IPs and/or hostnames, which it might not be able
to detect, you can use `--api-advertise-addresses` and `--api-external-dns-names` to add multiple
different IP addresses and hostnames (DNS).

- `--service-cidr=<cidr>` (default: "100.64.0.0/12")

By default, `kubeadm` sets `100.64.0.0/12` as the subnet for services. This means when a service is created, its cluster IP, if not manually specified, 
will be automatically assigned from the services subnet. If you would like to set a different one, use `--service-cidr`.

- `--service-dns-domain=<domain>` (default: "cluster.local")

By default, `kubeadm` sets `cluster.local` as the cluster DNS domain. If you would like to set a different one, use `--service-dns-domain`.

- `--schedule-workload=<bool>` (default: "false")

By default, `kubeadm` sets the master node kubelet as non-schedulable for workloads. This means the master node won't run your pods. If you want to change that, 
use `--schedule-workload=true`.

- `--cloud-provider=<cloud provider>`

By default, `kubeadm` doesn't perform auto-detection of the current cloud provider. If you want to specify it, use `--cloud-provider`. Possible values are
the ones supported by controller-manager, namely `"aws"`, `"azure"`, `"cloudstack"`, `"gce"`, `"mesos"`, `"openstack"`, `"ovirt"`, `"rackspace"`, `"vsphere"`.

***TODO(phase1+)***

- `--api-bind-address=<ip>`
- `--api-bind-port=<port>`

***TODO(phase2)***

- `--api-bind-loopback-unsecure=<bool>`

- `--prefer-private-network=<bool>`
- `--prefer-public-network=<bool>`

### `kubeadm join`

`kubeadm join` has one mandatory flag, the token used to secure cluster bootstrap, and one mandatory argument, the master IP address.
Here's an example on how to use it:

`kubeadm join --token=the_secret_token 192.168.1.1`

- `--token=<token>`

By default, when `kubeadm init` runs, a token is generated and revealed in the output. That's the token you should use here.

# User Experience Considerations

> ***TODO*** _Move this into the design document

a) `kube-apiserver` will listen on `0.0.0.0:443` and `127.0.0.1:8080`, which is not configurable right now and make things a bit easier for the MVP
b) there is `kube-discovery`, which will listen on `0.0.0.0:9898`


from the point of view of `kubeadm init`, we need to have
a) a primary IP address as will be seen by the nodes and needs to go into the cert and `kube-discovery` configuration secret
b) some other names and addresses API server may be known by (e.g. external DNS and/or LB and/or NAT)

from that perspective we don’t can assume default ports for now, but for all the address we really only care about two ports (i.e.  443 and 9898)

we should make ports configurable and expose some way of making API server bind to a specific address/interface

but I think for the MVP we need solve the issue with hardcode IPs and DNS names in the certs

so it sounds rather simple enough to introduce  `--api-advertise-addr=<ip>` and `--api-external-dns-name=<domain>`, and allowing multiple of those sounds also simple enough

from the `kubeadm join` perspective, it cares about the two ports we mentioned, and we can make those configurable too

but for now it’s easier to pass just the IP address

plust it’s also require, so passing it without a named flag sounds convenient, and it’s something users are familiar with

that’s what Consul does, what what Weave does, and now Docker SwarmKit does the same thing also (edited)

flags will differ, as there are some Kubernetes-specifics aspects to what join does, but basic join semantics will remain _familiar_

if we do marry `kube-discovery` with the API, we might do `kubeadm join host:port`, as we’d end-up with a single port to care about

but we haven’t yet
