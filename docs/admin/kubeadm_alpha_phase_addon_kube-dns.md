
Install the kube-dns addon to a Kubernetes cluster.

### Synopsis


Install the kube-dns addon to a Kubernetes cluster.

```
kubeadm alpha phase addon kube-dns
```

### Options

```
      --config string               Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --image-repository string     Choose a container registry to pull control plane images from. (default "gcr.io/google_containers")
      --kubeconfig string           The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
      --kubernetes-version string   Choose a specific Kubernetes version for the control plane. (default "stable-1.8")
      --service-cidr string         Use alternative range of IP address for service VIPs (default "10.96.0.0/12")
      --service-dns-domain string   Use alternative domain for services, e.g. "myorg.internal. (default "cluster.local")
```

