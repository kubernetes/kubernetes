# kube-dns
kube-dns schedules DNS Pods and Service on the cluster, other pods in cluster can
use the DNS Service’s IP to resolve DNS names.

More details on http://kubernetes.io/docs/admin/dns/.

## Manually scale kube-dns Deployment
kube-dns creates only one DNS Pod by default. If
[dns-horizontal-autoscaler](../dns-horizontal-autoscaler/)
is not enabled, you may need to manually scale kube-dns Deployment.

Please use below `kubectl scale` command to scale:
```
kubectl --namespace=kube-system scale deployment kube-dns --replicas=<NUM_YOU_WANT>
```

Do not use `kubectl edit` to modify kube-dns Deployment object if it is controlled by
[Addon Manager](../addon-manager/). Otherwise the modifications will be clobbered,
in addition the replicas count for kube-dns Deployment will be reset to 1. See
[Cluster add-ons README](../README.md) and [#36411](https://github.com/kubernetes/kubernetes/issues/36411)
for reference.

## kube-dns Deployment and Service templates

This directory contains the base UNDERSCORE templates that can be used
to generate the skydns-rc.yaml.in and skydns.rc.yaml.in needed in Salt format.

Due to a varied preference in templating language choices, the transform
Makefile in this directory should be enhanced to generate all required
formats from the base underscore templates.

**NOTE WELL**: Developers, when you add a parameter you should also
update the various scripts that supply values for your new parameter.
Here is one way you might find those scripts:
```
cd kubernetes
find [a-zA-Z0-9]* -type f -exec grep skydns-rc.yaml \{\} \; -print -exec echo \;
```

### Base Template files

These are the authoritative base templates.
Run 'make' to generate the Salt and Sed yaml templates from these.

skydns-rc.yaml.base
skydns-svc.yaml.base

### Generated Salt files

skydns-rc.yaml.in
skydns-svc.yaml.in

### Generated Sed files

skydns-rc.yaml.sed
skydns-svc.yaml.sed

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/README.md?pixel)]()
