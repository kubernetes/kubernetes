
Installs the kube-dns addon to a Kubernetes cluster

### Synopsis


Installs the kube-dns addon components via the API server.
Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase addon kube-dns
```

### Options

```
      --config string               Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --feature-gates string        A set of key=value pairs that describe feature gates for various features.Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --image-repository string     Choose a container registry to pull control plane images from (default "gcr.io/google_containers")
      --kubeconfig string           The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
      --kubernetes-version string   Choose a specific Kubernetes version for the control plane (default "stable-1.8")
      --service-cidr string         The range of IP address used for service VIPs (default "10.96.0.0/12")
      --service-dns-domain string   Alternative domain for services (default "cluster.local")
```

