
Create the in-cluster configuration file for the first time from using flags.

### Synopsis



Using this command, you can upload configuration to the ConfigMap in the cluster using the same flags you gave to 'kubeadm init'.
If you initialized your cluster using a v1.7.x or lower kubeadm client and set certain flags, you need to run this command with the
same flags before upgrading to v1.8 using 'kubeadm upgrade'.

The configuration is located in the "kube-system" namespace in the "kubeadm-config" ConfigMap.


```
kubeadm config upload from-flags
```

### Options

```
      --apiserver-advertise-address string      The IP address the API Server will advertise it's listening on. Specify '0.0.0.0' to use the address of the default network interface.
      --apiserver-bind-port int32               Port for the API Server to bind to. (default 6443)
      --apiserver-cert-extra-sans stringSlice   Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.
      --cert-dir string                         The path where to save and store the certificates. (default "/etc/kubernetes/pki")
      --feature-gates string                    A set of key=value pairs that describe feature gates for various features. Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --kubernetes-version string               Choose a specific Kubernetes version for the control plane. (default "stable-1.8")
      --node-name string                        Specify the node name.
      --pod-network-cidr string                 Specify range of IP addresses for the pod network. If set, the control plane will automatically allocate CIDRs for every node.
      --service-cidr string                     Use alternative range of IP address for service VIPs. (default "10.96.0.0/12")
      --service-dns-domain string               Use alternative domain for services, e.g. "myorg.internal". (default "cluster.local")
      --token string                            The token to use for establishing bidirectional trust between nodes and masters.
      --token-ttl duration                      The duration before the bootstrap token is automatically deleted. If set to '0', the token will never expire. (default 24h0m0s)
```

### Options inherited from parent commands

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster. (default "/etc/kubernetes/admin.conf")
```

