
Check which versions are available to upgrade to and validate whether your current cluster is upgradeable.

### Synopsis


Check which versions are available to upgrade to and validate whether your current cluster is upgradeable.

```
kubeadm upgrade plan
```

### Options inherited from parent commands

```
      --allow-experimental-upgrades           Show unstable versions of Kubernetes as an upgrade alternative and allow upgrading to an alpha/beta/release candidate versions of Kubernetes.
      --allow-release-candidate-upgrades      Show release candidate versions of Kubernetes as an upgrade alternative and allow upgrading to a release candidate versions of Kubernetes.
      --config string                         Path to kubeadm config file. WARNING: Usage of a configuration file is experimental!
      --feature-gates string                  A set of key=value pairs that describe feature gates for various features.Options are:
CoreDNS=true|false (ALPHA - default=false)
DynamicKubeletConfig=true|false (ALPHA - default=false)
HighAvailability=true|false (ALPHA - default=false)
SelfHosting=true|false (BETA - default=false)
StoreCertsInSecrets=true|false (ALPHA - default=false)
      --ignore-preflight-errors stringSlice   A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.
      --kubeconfig string                     The KubeConfig file to use when talking to the cluster. (default "/etc/kubernetes/admin.conf")
      --print-config                          Specifies whether the configuration file that will be used in the upgrade should be printed or not.
```

