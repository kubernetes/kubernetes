# DNS in Kubernetes

Kubernetes offers a DNS cluster addon, which most of the supported environments
enable by default. The source code is in [cmd/kube-dns][kube-dns].

The [Kubernetes DNS Admin Guide][dns-admin] provides further details on this plugin.

[kube-dns]: https://github.com/kubernetes/kubernetes/tree/master/cmd/kube-dns
[dns-admin]: http://kubernetes.io/docs/admin/dns/

## Making Changes

The container containing the kube-dns binary needs to be built for every
architecture and pushed to the registry manually whenever the kube-dns binary
has code changes. Every significant change to the functionality should result
in a bump of the TAG in the Makefile.

Any significant changes to the YAML template for `kube-dns` should result a bump
of the version number for the `kube-dns` replication controller and well as the
`version` label. This will permit a rolling update of `kube-dns`.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build-tools/kube-dns/README.md?pixel)]()
