
Installs the kube-proxy addon to a Kubernetes cluster

### Synopsis


Installs the kube-proxy addon components via the API server. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase addon kube-proxy
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --config string                        Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --image-repository string              Choose a container registry to pull control plane images from (default "gcr.io/google_containers")
      --kubeconfig string                    The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
      --kubernetes-version string            Choose a specific Kubernetes version for the control plane (default "stable-1.8")
      --pod-network-cidr string              The range of IP addresses used for the Pod network
```

