
Generates front proxy CA certificate and key for a Kubernetes cluster

### Synopsis


Generates the front proxy CA certificate and key and saves them into front-proxy-ca.crt and front-proxy-ca.key files. 

If both files already exist, kubeadm skips the generation step and existing files will be used. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase certs front-proxy-ca
```

### Options

```
      --cert-dir string   The path where to save the certificates (default "/etc/kubernetes/pki")
      --config string     Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
```

