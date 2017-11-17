
Generate all PKI assets necessary to establish the control plane

### Synopsis


Generate all PKI assets necessary to establish the control plane

```
kubeadm alpha phase certs all
```

### Options

```
      --apiserver-advertise-address string      The IP address the API Server will advertise it is listening on. Specify '0.0.0.0' to use the address of the default network interface.
      --apiserver-cert-extra-sans stringSlice   Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.
      --cert-dir string                         The path where to save and store the certificates. (default "/etc/kubernetes/pki")
      --config string                           Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --service-cidr string                     Use alternative range of IP address for service VIPs. (default "10.96.0.0/12")
      --service-dns-domain string               Use alternative domain for services, e.g. "myorg.internal". (default "cluster.local")
```

