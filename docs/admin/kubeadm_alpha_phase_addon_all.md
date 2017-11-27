
Installs all addons to a Kubernetes cluster

### Synopsis


Installs the kube-dns and the kube-proxys addons components via the API server.
Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase addon all
```

### Examples

```
  # Installs the kube-dns and the kube-proxys addons components via the API server,
  # functionally equivalent to what installed by kubeadm init.
  
  kubeadm alpha phase selfhosting from-staticpods
```

### Options

```
      --apiserver-advertise-address string   The IP address or DNS name the API server is accessible on
      --apiserver-bind-port int32            The port the API server is accessible on (default 6443)
      --config string                        Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --feature-gates string                 A set of key=value pairs that describe feature gates for various features.Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --image-repository string              Choose a container registry to pull control plane images from (default "gcr.io/google_containers")
      --kubeconfig string                    The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
      --kubernetes-version string            Choose a specific Kubernetes version for the control plane (default "stable-1.8")
      --pod-network-cidr string              The range of IP addresses used for the Pod network
      --service-cidr string                  The range of IP address used for service VIPs (default "10.96.0.0/12")
      --service-dns-domain string            Alternative domain for services (default "cluster.local")
```

