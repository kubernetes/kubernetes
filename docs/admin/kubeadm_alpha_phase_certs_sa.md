
Generates a private key for signing service account tokens along with its public key

### Synopsis


Generates the private key for signing service account tokens along with its public key, and saves them into sa.key and sa.pub files. 

If both files already exist, kubeadm skips the generation step and existing files will be used. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase certs sa
```

### Options

```
      --cert-dir string   The path where to save the certificates (default "/etc/kubernetes/pki")
      --config string     Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
```

