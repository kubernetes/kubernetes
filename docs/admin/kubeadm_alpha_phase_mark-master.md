
Mark a node as master

### Synopsis


Applies a label that specifies that a node is a master and a taint that forces workloads to be deployed accordingly. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase mark-master
```

### Examples

```
  # Applies master label and taint to the current node, functionally equivalent to what executed by kubeadm init.
  kubeadm alpha phase mark-master
  
  # Applies master label and taint to a specific node
  kubeadm alpha phase mark-master --node-name myNode
```

### Options

```
      --config string       Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
      --node-name string    The node name to which label and taints should apply
```

