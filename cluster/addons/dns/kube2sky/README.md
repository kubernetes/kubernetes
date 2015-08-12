# kube2sky
==============

A bridge between Kubernetes and SkyDNS.  This will watch the kubernetes API for
changes in Services and then publish those changes to SkyDNS through etcd.

For now, this is expected to be run in a pod alongside the etcd and SkyDNS
containers.

## Namespaces

Kubernetes namespaces become another level of the DNS hierarchy.  See the
description of `-domain` below.

## Flags

`-domain`: Set the domain under which all DNS names will be hosted.  For
example, if this is set to `kubernetes.io`, then a service named "nifty" in the
"default" namespace would be exposed through DNS as
"nifty.default.kubernetes.io".

`-verbose`: Log additional information.

`-etcd_mutation_timeout`: For how long the application will keep retrying etcd 
mutation (insertion or removal of a dns entry) before giving up and crashing.

`--etcd-server`: The etcd server that is being used by skydns.

`--kube_master_url`: URL of kubernetes master. Required if `--kubecfg_file` is not set.

`--kubecfg_file`: Path to kubecfg file that contains the master URL and tokens to authenticate with the master.

## Domain Label

It is possible to specify a label in the metadata of a Service definition with
the name "domain" to control the name of the A record created by kube2dns. Consider
the following Service definition:

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: example-service
  namespace: web
  labels:
    domain: example.com
spec:
  ports:
    - port: 80
      targetPort: 80
  selector:
    name: example-web-server
```

Normally, kube2dns would create an A record with the following name:

```
example-service.web.svc.skydns.local.
```

Because the "domain" label has been specified, the following name is used instead:

```
example.com.web.svc.skydns.local.
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/kube2sky/README.md?pixel)]()
