
Generates the scheduler static Pod manifest.

### Synopsis


Generates the static Pod manifest file for the scheduler and saves it into /etc/kubernetes/manifests/kube-scheduler.yaml file. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase controlplane scheduler
```

### Options

```
      --cert-dir string             The path where certificates are stored (default "/etc/kubernetes/pki")
      --config string               Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubernetes-version string   Choose a specific Kubernetes version for the control plane (default "stable-1.8")
```

