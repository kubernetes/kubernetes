# Kubernetes Cluster Boostrap Made Easy

## Usage

### `kubeadm init`

It's usually enough to run `kubeadm init`, but in some case you might like to override the
default behaviour.

- `--token=<str>`

By default, a token is generated, but if you are to automate cluster deployment, you want to
set the token ahead of time. Read the docs for more information on the token format.

- `--api-advertise-addr=<ip>` (multiple values allowed)
- `--api-external-dns-name=<domain>` (multiple values allowed)

By default, `kubeadm` will auto detect IP address and use that to generate API server certificates.
If you would like to access the API via any external IPs and/or DNS, which it might not be able
to detect, you can use `--api-advertise-addr` and `--api-external-dns-name` to add multiple
different IP addresses and DNS names.

- `--service-cidr=<cidr>` (default: "100.64/12")
- `--service-dns-domain=<domain>` (default: "cluster.local")

- `--use-hyperkube=<bool>` (default: "false")

***TODO(phase1+)***

- `--api-bind-addr=<ip>`
- `--api-bind-port=<port>`

***TODO(phase2)***

- `--api-bind-loopback-unsecure=<bool>`

***TODO(pahse2)***

- `--prefer-private-network=<bool>`
- `--prefer-public-network=<bool>`

### `kubeadm join`

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
