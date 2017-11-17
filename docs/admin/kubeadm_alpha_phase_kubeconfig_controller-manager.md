
Generate a kubeconfig file for the Controller Manager to use.

### Synopsis


Generate a kubeconfig file for the Controller Manager to use.

```
kubeadm alpha phase kubeconfig controller-manager
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API Server is accessible on.
      --apiserver-bind-port int32            The port the API Server is accessible on. (default 6443)
      --cert-dir string                      The path where certificates are stored. (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --kubeconfig-dir string                The path where to save and store the kubeconfig file. (default "/etc/kubernetes")
```

