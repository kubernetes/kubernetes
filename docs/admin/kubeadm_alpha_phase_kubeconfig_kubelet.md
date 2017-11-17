
Generate a kubeconfig file for the Kubelet to use. Please note that this should *only* be used for bootstrapping purposes! After your control plane is up, you should request all kubelet credentials from the CSR API.

### Synopsis


Generate a kubeconfig file for the Kubelet to use. Please note that this should *only* be used for bootstrapping purposes! After your control plane is up, you should request all kubelet credentials from the CSR API.

```
kubeadm alpha phase kubeconfig kubelet
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API Server is accessible on.
      --apiserver-bind-port int32            The port the API Server is accessible on. (default 6443)
      --cert-dir string                      The path where certificates are stored. (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --kubeconfig-dir string                The path where to save and store the kubeconfig file. (default "/etc/kubernetes")
      --node-name string                     The node name that the kubelet client cert should use.
```

