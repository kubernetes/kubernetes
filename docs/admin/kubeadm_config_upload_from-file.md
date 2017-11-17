
Upload a configuration file to the in-cluster ConfigMap for kubeadm configuration.

### Synopsis



Using this command, you can upload configuration to the ConfigMap in the cluster using the same config file you gave to 'kubeadm init'.
If you initialized your cluster using a v1.7.x or lower kubeadm client and used the --config option, you need to run this command with the
same config file before upgrading to v1.8 using 'kubeadm upgrade'.

The configuration is located in the "kube-system" namespace in the "kubeadm-config" ConfigMap.


```
kubeadm config upload from-file
```

### Options

```
      --config string   Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental.
```

### Options inherited from parent commands

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster. (default "/etc/kubernetes/admin.conf")
```

