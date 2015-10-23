---
layout: docwithnav
title: "</strong>"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Step 1. Setting up iSCSI target and iSCSI initiator

**Setup A.** On Fedora 21 nodes

If you use Fedora 21 on Kubernetes node, then first install iSCSI initiator on the node:

    # yum -y install iscsi-initiator-utils

then edit */etc/iscsi/iscsid.conf* to match your iSCSI target configuration.

I mostly followed these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi) to setup iSCSI target. and these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi&f=2) to setup iSCSI initiator.

**Setup B.** On Unbuntu 12.04 and Debian 7 nodes on Google Compute Engine (GCE)

GCE does not provide preconfigured Fedora 21 image, so I set up the iSCSI target on a preconfigured Ubuntu 12.04 image, mostly following these [instructions](http://www.server-world.info/en/note?os=Ubuntu_12.04&p=iscsi). My Kubernetes cluster on GCE was running Debian 7 images, so I followed these [instructions](http://www.server-world.info/en/note?os=Debian_7.0&p=iscsi&f=2) to set up the iSCSI initiator.

## Step 2. Creating the pod with iSCSI persistent storage

Once you have installed iSCSI initiator and new Kubernetes, you can create a pod based on the example *iscsi.json*. In the pod JSON, you need to provide *targetPortal* (the iSCSI target's **IP** address and *port* if not the default port 3260), target's *iqn*, *lun*, and the type of the filesystem that has been created on the lun, and *readOnly* boolean. No initiator information is required.

**Note:** If you have followed the instructions in the links above you
may have partitioned the device, the iSCSI volume plugin does not
currently support partitions so format the device as one partition.
Make sure you have the correct device name then run the following as
root to format it:

{% highlight console %}
{% raw %}
mkfs.ext4 /dev/<name of device>
{% endraw %}
{% endhighlight %}

Once your pod is created, run it on the Kubernetes master:

{% highlight console %}
{% raw %}
kubectl create -f ./your_new_pod.json
{% endraw %}
{% endhighlight %}

Here is my command and output:

{% highlight console %}
{% raw %}
# kubectl create -f examples/iscsi/iscsi.json
# kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
iscsipd   2/2       RUNNING   0           2m
{% endraw %}
{% endhighlight %}

On the Kubernetes node, verify the mount output

{% highlight console %}
{% raw %}
# mount |grep kub
/dev/sdb on /var/lib/kubelet/plugins/kubernetes.io/iscsi/10.0.2.15:3260-iqn.2001-04.com.example:storage.kube.sys1.xyz-lun-0 type ext4 (ro,relatime,data=ordered)
/dev/sdb on /var/lib/kubelet/pods/f527ca5b-6d87-11e5-aa7e-080027ff6387/volumes/kubernetes.io~iscsi/iscsipd-ro type ext4 (ro,relatime,data=ordered)
/dev/sdc on /var/lib/kubelet/plugins/kubernetes.io/iscsi/10.0.2.15:3260-iqn.2001-04.com.example:storage.kube.sys1.xyz-lun-1 type ext4 (rw,relatime,data=ordered)
/dev/sdc on /var/lib/kubelet/pods/f527ca5b-6d87-11e5-aa7e-080027ff6387/volumes/kubernetes.io~iscsi/iscsipd-rw type ext4 (rw,relatime,data=ordered)
{% endraw %}
{% endhighlight %}

If you ssh to that machine, you can run `docker ps` to see the actual pod.

{% highlight console %}
{% raw %}
# docker ps
CONTAINER ID        IMAGE                                  COMMAND             CREATED             STATUS              PORTS               NAMES
f855336407f4        kubernetes/pause                       "/pause"            6 minutes ago       Up 6 minutes                            k8s_iscsipd-ro.d130ec3e_iscsipd_default_f527ca5b-6d87-11e5-aa7e-080027ff6387_5409a4cb
3b8a772515d2        kubernetes/pause                       "/pause"            6 minutes ago       Up 6 minutes                            k8s_iscsipd-rw.ed58ec4e_iscsipd_default_f527ca5b-6d87-11e5-aa7e-080027ff6387_d25592c5
{% endraw %}
{% endhighlight %}

Run *docker inspect* and verify the container mounted the host directory into the their */mnt/iscsipd* directory.

{% highlight console  %}
{% raw %}
# docker inspect --format '{{ range .Mounts }}{{ if eq .Destination "/mnt/iscsipd" }}{{ .Source }}{{ end }}{{ end }}' f855336407f4
/var/lib/kubelet/pods/f527ca5b-6d87-11e5-aa7e-080027ff6387/volumes/kubernetes.io~iscsi/iscsipd-ro

# docker inspect --format '{{ range .Mounts }}{{ if eq .Destination "/mnt/iscsipd" }}{{ .Source }}{{ end }}{{ end }}' 3b8a772515d2
/var/lib/kubelet/pods/f527ca5b-6d87-11e5-aa7e-080027ff6387/volumes/kubernetes.io~iscsi/iscsipd-rw
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/iscsi/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

