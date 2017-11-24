
Outputs a kubeconfig file for an additional user

### Synopsis


Outputs a kubeconfig file for an additional user. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase kubeconfig user
```

### Examples

```
  # Outputs a kubeconfig file for an additional user named foo
  kubeadm alpha phase kubeconfig user --client-name=foo
```

### Options

```
      --apiserver-advertise-address string   The IP address the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --cert-dir string                      The path where certificates are stored (default "/etc/kubernetes/pki")
      --client-name string                   The name of user. It will be used as the CN if client certificates are created
      --kubeconfig-dir string                The port where to save the kubeconfig file (default "/etc/kubernetes")
      --token string                         The token that should be used as the authentication mechanism for this kubeconfig (instead of client certificates)
```

