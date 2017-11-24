
Generates API server serving certificate and key

### Synopsis


Generates the API server serving certificate and key and saves them into apiserver.crt and apiserver.key files. 

The certificate includes default subject alternative names and additional sans eventually provided by the user; default sans are: <node-name>, <apiserver-advertise-address>, kubernetes, kubernetes.default, kubernetes.default.svc, kubernetes.default.svc. <service-dns-domain>, <internalAPIServerVirtualIP>(that is the .10 address in <service-cidr>address space). 

If both files already exist, kubeadm skips the generation step and existing files will be used. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase certs apiserver
```

### Options

```
      --apiserver-advertise-address string      The IP address the API server is accessible on, to use for the API server serving cert
      --apiserver-cert-extra-sans stringSlice   Optional extra altnames to use for the API server serving cert. Can be both IP addresses and dns names
      --cert-dir string                         The path where to save the certificates (default "/etc/kubernetes/pki")
      --config string                           Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --service-cidr string                     Alternative range of IP address for service VIPs, from which derives the internal API server VIP that will be added to the API Server serving cert (default "10.96.0.0/12")
      --service-dns-domain string               Alternative domain for services, to use for the API server serving cert (default "cluster.local")
```

