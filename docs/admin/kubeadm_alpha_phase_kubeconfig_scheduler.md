
Generates a kubeconfig file for the scheduler to use

### Synopsis


Generates the kubeconfig file for the scheduler to use and saves it to /etc/kubernetes/scheduler.conf file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase kubeconfig scheduler
```

### Options

```
      --apiserver-advertise-address string   The IP address the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --cert-dir string                      The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubeconfig-dir string                The port where to save the kubeconfig file (default "/etc/kubernetes")
```

