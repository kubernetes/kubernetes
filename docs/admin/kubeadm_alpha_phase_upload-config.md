
Upload the currently used configuration for kubeadm to a ConfigMap in the cluster for future use in reconfiguration and upgrades of the cluster.

### Synopsis


Upload the currently used configuration for kubeadm to a ConfigMap in the cluster for future use in reconfiguration and upgrades of the cluster.

```
kubeadm alpha phase upload-config
```

### Options

```
      --config string       Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
```

