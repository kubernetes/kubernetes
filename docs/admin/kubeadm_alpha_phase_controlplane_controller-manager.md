
Generates the controller-manager static Pod manifest.

### Synopsis


Generates the static Pod manifest file for the controller-manager and saves it into /etc/kubernetes/manifests/kube-controller-manager.yaml file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase controlplane controller-manager
```

### Options

```
      --cert-dir string             The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string               Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubernetes-version string   Choose a specific Kubernetes version for the control plane (default "stable-1.8")
      --pod-network-cidr string     The range of IP addresses used for the Pod network
```

