
Generate all static pod manifest files necessary to establish the control plane.

### Synopsis


Generate all static pod manifest files necessary to establish the control plane.

```
kubeadm alpha phase controlplane all
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API Server is accessible on.
      --apiserver-bind-port int32            The port the API Server is accessible on. (default 6443)
      --cert-dir string                      The path where certificates are stored. (default "/etc/kubernetes/pki")
      --config string                        Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --feature-gates string                 A set of key=value pairs that describe feature gates for various features. Options are:
CoreDNS=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
SupportIPVSProxyMode=true|false (ALPHA - default=false)
      --kubernetes-version string            Choose a specific Kubernetes version for the control plane. (default "stable-1.8")
      --pod-network-cidr string              The range of IP addresses used for the pod network.
      --service-cidr string                  The range of IP address used for service VIPs. (default "10.96.0.0/12")
```

