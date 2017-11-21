
Uploads the cluster-info ConfigMap from the given kubeconfig file

### Synopsis


Uploads the "cluster-info" ConfigMap in the "kube-public" namespace, populating it with cluster information extracted from the given kubeconfig file. The ConfigMap is used for the node bootstrap process in its initial phases, before the client trusts the API server. 

See online documentation about Authenticating with Bootstrap Tokens for more details. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase bootstrap-token cluster-info
```

### Options inherited from parent commands

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
```

