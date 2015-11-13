<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/docker-multinode/deployDNS.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Deploy DNS

### Get the template file

First of all, download the template dns rc and svc file from

[skydns-rc template](skydns-rc.yaml.in)

[skydns-svc template](skydns-svc.yaml.in)

### Set env

Then you need to set `DNS_REPLICAS` , `DNS_DOMAIN` , `DNS_SERVER_IP` , `KUBE_SERVER` ENV.

```
$ export DNS_REPLICAS=1

$ export DNS_DOMAIN=cluster.local # specify in startup parameter `--cluster-domain` for containerized kubelet 

$ export DNS_SERVER_IP=10.0.0.10  # specify in startup parameter `--cluster-dns` for containerized kubelet 

$ export KUBE_SERVER=10.10.103.250 # your master server ip, you may change it
```

### Replace the corresponding value in the template.

```
$ sed -e "s/{{ pillar\['dns_replicas'\] }}/${DNS_REPLICAS}/g;s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g;s/{kube_server_url}/${KUBE_SERVER}/g;" skydns-rc.yaml.in > ./skydns-rc.yaml

$ sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" skydns-svc.yaml.in > ./skydns-svc.yaml
```

### Use `kubectl` to create skydns rc and service


```
$ kubectl -s "$KUBE_SERVER:8080" --namespace=kube-system create -f ./skydns-rc.yaml

$ kubectl -s "$KUBE_SERVER:8080" --namespace=kube-system create -f ./skydns-svc.yaml
```

### Test if DNS works

Follow [this link](../../../cluster/addons/dns/#how-do-i-test-if-it-is-working) to check it out.





<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/deployDNS.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
