
Generates client certificate for the API server to connect to the kubelets securely

### Synopsis


Generates the client certificate for the API server to connect to the kubelet securely and the respective key, and saves them into apiserver-kubelet-client.crt and apiserver-kubelet-client.key files. 

If both files already exist, kubeadm skips the generation step and existing files will be used. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase certs apiserver-kubelet-client
```

### Options

```
      --cert-dir string   The path where to save the certificates (default "/etc/kubernetes/pki")
      --config string     Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
```

