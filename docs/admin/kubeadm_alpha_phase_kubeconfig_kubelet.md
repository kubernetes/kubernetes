
Generates a kubeconfig file for the kubelet to use. Please note that this should be used *only* for bootstrapping purposes.

### Synopsis


Generates the kubeconfig file for the kubelet to use and saves it to /etc/kubernetes/kubelet.conf file. 

Please note that this should only be used for bootstrapping purposes. After your control plane is up, you should request all kubelet credentials from the CSR API. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase kubeconfig kubelet
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

