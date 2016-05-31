# kube2sky
==============

A bridge between Kubernetes and SkyDNS.  This will watch the kubernetes API for
changes in Services and then publish those changes to SkyDNS through etcd.

For now, this is expected to be run in a pod alongside the etcd and SkyDNS
containers.

## Namespaces

Kubernetes namespaces become another level of the DNS hierarchy.  See the
description of `--domain` below.

## API Access

In most functioning Kubernetes installations, the DNS pod runs with a
ServiceAccount (usually the per-Namespace default) which grants it access to
the apiserver.  The client library should find and use the ServiceAccount
tokens automatically.  If your installation does not use ServiceAccounts, see
the flags below for other options.

See
https://github.com/kubernetes/kubernetes/issues/10265#issuecomment-117267504
for help in diagnosing ServiceAccount problems.

## Flags

`--domain`: Set the domain under which all DNS names will be hosted.  For
example, if this is set to `cluster.local`, then a service named "nifty" in the
"default" namespace would be exposed through DNS as
"nifty.default.svc.cluster.local".

`--v`: Set logging level

`--etcd-mutation-timeout`: For how long the application will keep retrying etcd
mutation (insertion or removal of a dns entry) before giving up and crashing.

`--etcd-server`: The etcd server that is being used by skydns.

`--kube-master-url`: URL of kubernetes master. Required if service account
tokens are not available and `--kubecfg-file` is not set.

`--kubecfg-file`: Path to kubecfg file that contains the master URL and tokens
to authenticate with the master.

`--log-dir`: If non empty, write log files in this directory

`--logtostderr`: Logs to stderr instead of files

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/kube2sky/README.md?pixel)]()
