
Manage configuration for a kubeadm cluster persisted in a ConfigMap in the cluster.

### Synopsis



There is a ConfigMap in the kube-system namespace called "kubeadm-config" that kubeadm uses to store internal configuration about the
cluster. kubeadm CLI v1.8.0+ automatically creates this ConfigMap with the config used with 'kubeadm init', but if you
initialized your cluster using kubeadm v1.7.x or lower, you must use the 'config upload' command to create this
ConfigMap. This is required so that 'kubeadm upgrade' can configure your upgraded cluster correctly.


```
kubeadm config
```

### Options

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster. (default "/etc/kubernetes/admin.conf")
```

