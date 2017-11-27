
Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.

### Synopsis


Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.

```
kubeadm reset
```

### Options

```
      --cert-dir string                       The path to the directory where the certificates are stored. If specified, clean this directory. (default "/etc/kubernetes/pki")
      --cri-socket string                     The path to the CRI socket to use with crictl when cleaning up containers. (default "/var/run/dockershim.sock")
      --ignore-preflight-errors stringSlice   A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.
```

