
Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.

### Synopsis


Run this to revert any changes made to this host by 'kubeadm init' or 'kubeadm join'.

```
kubeadm reset
```

### Options

```
      --cert-dir string         The path to the directory where the certificates are stored. If specified, clean this directory. (default "/etc/kubernetes/pki")
      --cri-socket string       The path to the CRI socket to use with crictl when cleaning up containers. (default "/var/run/dockershim.sock")
      --skip-preflight-checks   Skip preflight checks which normally run before modifying the system.
```

