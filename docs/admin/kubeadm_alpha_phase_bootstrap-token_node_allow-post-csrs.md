
Configures RBAC to allow node bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials

### Synopsis


Configures RBAC rules to allow node bootstrap tokens to post a certificate signing request, thus enabling nodes joining the cluster to request long term certificate credentials. 

See online documentation about TLS bootstrapping for more details. 

Alpha Disclaimer: this command is currently alpha.

```
kubeadm alpha phase bootstrap-token node allow-post-csrs
```

### Options inherited from parent commands

```
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
```

