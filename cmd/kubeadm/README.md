# Kubernetes Cluster Bootstrap Made Easy

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

- `--schedule-pods-here=<bool>` (default: "false")

  By default, `kubeadm` sets the master node kubelet as non-schedulable for workloads. This means the master node won't run your pods. If you want to change that,
  use `--schedule-pods-here=true`.

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
