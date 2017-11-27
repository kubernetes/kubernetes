
Generates the API server static Pod manifest.

### Synopsis


Generates the static Pod manifest file for the API server and saves it into /etc/kubernetes/manifests/kube-apiserver.yaml file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase controlplane apiserver
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --cert-dir string                      The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string                        Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --feature-gates string                 A set of key=value pairs that describe feature gates for various features. Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --kubernetes-version string            Choose a specific Kubernetes version for the control plane (default "stable-1.8")
      --service-cidr string                  The range of IP address used for service VIPs (default "10.96.0.0/12")
```

