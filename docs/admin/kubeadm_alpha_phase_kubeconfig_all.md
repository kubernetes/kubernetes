
Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file

### Synopsis


Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase kubeconfig all
```

### Examples

```
  # Generates all kubeconfig files, functionally equivalent to what generated
  # by kubeadm init.
  kubeadm alpha phase kubeconfig all
  
  # Generates all kubeconfig files using options read from a configuration file.
  kubeadm alpha phase kubeconfig all --config masterconfiguration.yaml
```

### Options

```
      --apiserver-advertise-address string   The IP address the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --cert-dir string                      The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubeconfig-dir string                The port where to save the kubeconfig file (default "/etc/kubernetes")
      --node-name string                     The node name that should be used for the kubelet client certificate
```

