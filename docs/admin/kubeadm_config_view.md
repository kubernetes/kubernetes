
View the kubeadm configuration stored inside the cluster.

### Synopsis



Using this command, you can view the ConfigMap in the cluster where the configuration for kubeadm is located.

The configuration is located in the "kube-system" namespace in the "kubeadm-config" ConfigMap.


```
kubeadm config view
```

### Options inherited from parent commands

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster. (default "/etc/kubernetes/admin.conf")
```

