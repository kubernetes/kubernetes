## coredns

Users can enable CoreDNS instead of kube-dns for service discovery.
`coredns` is schedules DNS Pods and Service on the cluster, other pods in cluster
can use the DNS Service’s IP to resolve DNS names.

* [Administrators guide](https://coredns.io/2018/01/29/deploying-kubernetes-with-coredns-using-kubeadm/)
* [Code repository](https://github.com/coredns/coredns)


### Base Template files

These are the authoritative base templates.
Run 'make' to generate the Salt and Sed yaml templates from these.

```
coredns.yaml.base
```

### Generated Salt files

```
coredns.yaml.in
```

### Generated Sed files

```
coredns.yaml.sed
```
