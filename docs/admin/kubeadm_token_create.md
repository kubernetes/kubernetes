
Create bootstrap tokens on the server.

### Synopsis



This command will create a bootstrap token for you.
You can specify the usages for this token, the "time to live" and an optional human friendly description.

The [token] is the actual token to write.
This should be a securely generated random token of the form "[a-z0-9]{6}.[a-z0-9]{16}".
If no [token] is given, kubeadm will generate a random token instead.


```
kubeadm token create [token]
```

### Options

```
      --description string   A human friendly description of how this token is used.
      --groups stringSlice   Extra groups that this token will authenticate as when used for authentication. Must match "system:bootstrappers:[a-z0-9:-]{0,255}[a-z0-9]". (default [system:bootstrappers:kubeadm:default-node-token])
      --print-join-command   Instead of printing only the token, print the full 'kubeadm join' flag needed to join the cluster using the token.
      --ttl duration         The duration before the token is automatically deleted (e.g. 1s, 2m, 3h). If set to '0', the token will never expire. (default 24h0m0s)
      --usages stringSlice   Describes the ways in which this token can be used. You can pass --usages multiple times or provide a comma separated list of options. Valid options: [signing,authentication]. (default [signing,authentication])
```

### Options inherited from parent commands

```
      --dry-run             Whether to enable dry-run mode or not
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
```

