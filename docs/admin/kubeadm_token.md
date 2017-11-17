
Manage bootstrap tokens.

### Synopsis



This command manages bootstrap tokens. It is optional and needed only for advanced use cases.

In short, bootstrap tokens are used for establishing bidirectional trust between a client and a server.
A bootstrap token can be used when a client (for example a node that is about to join the cluster) needs
to trust the server it is talking to. Then a bootstrap token with the "signing" usage can be used.
bootstrap tokens can also function as a way to allow short-lived authentication to the API Server
(the token serves as a way for the API Server to trust the client), for example for doing the TLS Bootstrap.

What is a bootstrap token more exactly?
 - It is a Secret in the kube-system namespace of type "bootstrap.kubernetes.io/token".
 - A bootstrap token must be of the form "[a-z0-9]{6}.[a-z0-9]{16}". The former part is the public token ID,
   while the latter is the Token Secret and it must be kept private at all circumstances!
 - The name of the Secret must be named "bootstrap-token-(token-id)".

You can read more about bootstrap tokens here:
  https://kubernetes.io/docs/admin/bootstrap-tokens/


```
kubeadm token
```

### Options

```
      --dry-run             Whether to enable dry-run mode or not
      --kubeconfig string   The KubeConfig file to use when talking to the cluster (default "/etc/kubernetes/admin.conf")
```

