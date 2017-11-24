
Generates a kubeconfig file for the controller manager to use

### Synopsis


Generates the kubeconfig file for the controller manager to use and saves it to /etc/kubernetes/controller-manager.conf file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase kubeconfig controller-manager
```

### Options

```
      --apiserver-advertise-address string   The IP address the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --cert-dir string                      The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubeconfig-dir string                The port where to save the kubeconfig file (default "/etc/kubernetes")
```

