---
layout: docwithnav
title: "Configuring Kubernetes on Fedora via Ansible"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Configuring Kubernetes on [Fedora](http://fedoraproject.org) via [Ansible](http://www.ansible.com/home)
-------------------------------------------------------------------------------------------------------

Configuring Kubernetes on Fedora via Ansible offers a simple way to quickly create a clustered environment with little effort.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Architecture of the cluster](#architecture-of-the-cluster)
- [Setting up ansible access to your nodes](#setting-up-ansible-access-to-your-nodes)
- [Setting up the cluster](#setting-up-the-cluster)
- [Testing and using your new cluster](#testing-and-using-your-new-cluster)

## Prerequisites

1. Host able to run ansible and able to clone the following repo: [kubernetes](https://github.com/GoogleCloudPlatform/kubernetes.git)
2. A Fedora 21+ host to act as cluster master
3. As many Fedora 21+ hosts as you would like, that act as cluster nodes

The hosts can be virtual or bare metal. Ansible will take care of the rest of the configuration for you - configuring networking, installing packages, handling the firewall, etc. This example will use one master and two nodes.

## Architecture of the cluster

A Kubernetes cluster requires etcd, a master, and n nodes, so we will create a cluster with three hosts, for example:

{% highlight console %}
{% raw %}
    master,etcd = kube-master.example.com
    node1 = kube-node-01.example.com
    node2 = kube-node-02.example.com
{% endraw %}
{% endhighlight %}

**Make sure your local machine has**

 - ansible (must be 1.9.0+)
 - git
 - python-netaddr

If not

{% highlight sh %}
{% raw %}
yum install -y ansible git python-netaddr
{% endraw %}
{% endhighlight %}

**Now clone down the Kubernetes repository**

{% highlight sh %}
{% raw %}
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
cd kubernetes/contrib/ansible
{% endraw %}
{% endhighlight %}

**Tell ansible about each machine and its role in your cluster**

Get the IP addresses from the master and nodes.  Add those to the `~/kubernetes/contrib/ansible/inventory` file on the host running Ansible.

{% highlight console %}
{% raw %}
[masters]
kube-master.example.com

[etcd]
kube-master.example.com

[nodes]
kube-node-01.example.com
kube-node-02.example.com
{% endraw %}
{% endhighlight %}

## Setting up ansible access to your nodes

If you already are running on a machine which has passwordless ssh access to the kube-master and kube-node-{01,02} nodes, and 'sudo' privileges, simply set the value of `ansible_ssh_user` in `~/kubernetes/contrib/ansible/group_vars/all.yaml` to the username which you use to ssh to the nodes (i.e. `fedora`), and proceed to the next step...

*Otherwise* setup ssh on the machines like so (you will need to know the root password to all machines in the cluster).

edit: ~/kubernetes/contrib/ansible/group_vars/all.yml

{% highlight yaml %}
{% raw %}
ansible_ssh_user: root
{% endraw %}
{% endhighlight %}

**Configuring ssh access to the cluster**

If you already have ssh access to every machine using ssh public keys you may skip to [setting up the cluster](#setting-up-the-cluster)

Make sure your local machine (root) has an ssh key pair if not

{% highlight sh %}
{% raw %}
ssh-keygen
{% endraw %}
{% endhighlight %}

Copy the ssh public key to **all** nodes in the cluster

{% highlight sh %}
{% raw %}
for node in kube-master.example.com kube-node-01.example.com kube-node-02.example.com; do
  ssh-copy-id ${node}
done
{% endraw %}
{% endhighlight %}

## Setting up the cluster

Although the default value of variables in `~/kubernetes/contrib/ansible/group_vars/all.yml` should be good enough, if not, change them as needed.

edit: ~/kubernetes/contrib/ansible/group_vars/all.yml

**Configure the IP addresses used for services**

Each Kubernetes service gets its own IP address.  These are not real IPs.  You need only select a range of IPs which are not in use elsewhere in your environment.

{% highlight yaml %}
{% raw %}
kube_service_addresses: 10.254.0.0/16
{% endraw %}
{% endhighlight %}

**Managing flannel**

Modify `flannel_subnet`, `flannel_prefix` and `flannel_host_prefix` only if defaults are not appropriate for your cluster.


**Managing add on services in your cluster**

Set `cluster_logging` to false or true (default) to disable or enable logging with elasticsearch.

{% highlight yaml %}
{% raw %}
cluster_logging: true
{% endraw %}
{% endhighlight %}

Turn `cluster_monitoring` to true (default) or false to enable or disable cluster monitoring with heapster and influxdb.

{% highlight yaml %}
{% raw %}
cluster_monitoring: true
{% endraw %}
{% endhighlight %}

Turn `dns_setup` to true (recommended) or false to enable or disable whole DNS configuration.

{% highlight yaml %}
{% raw %}
dns_setup: true
{% endraw %}
{% endhighlight %}

**Tell ansible to get to work!**

This will finally setup your whole Kubernetes cluster for you.

{% highlight sh %}
{% raw %}
cd ~/kubernetes/contrib/ansible/

./setup.sh
{% endraw %}
{% endhighlight %}

## Testing and using your new cluster

That's all there is to it.  It's really that easy.  At this point you should have a functioning Kubernetes cluster.

**Show kubernets nodes**

Run the following on the kube-master:

{% highlight sh %}
{% raw %}
kubectl get nodes
{% endraw %}
{% endhighlight %}

**Show services running on masters and nodes**

{% highlight sh %}
{% raw %}
systemctl | grep -i kube
{% endraw %}
{% endhighlight %}

**Show firewall rules on the masters and nodes**

{% highlight sh %}
{% raw %}
iptables -nvL
{% endraw %}
{% endhighlight %}

**Create /tmp/apache.json on the master with the following contents and deploy pod**

{% highlight json %}
{% raw %}
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "fedoraapache",
    "labels": {
      "name": "fedoraapache"
    }
  },
  "spec": {
    "containers": [
      {
        "name": "fedoraapache",
        "image": "fedora/apache",
        "ports": [
          {
            "hostPort": 80,
            "containerPort": 80
          }
        ]
      }
    ]
  }
}
{% endraw %}
{% endhighlight %}

{% highlight sh %}
{% raw %}
kubectl create -f /tmp/apache.json
{% endraw %}
{% endhighlight %}

**Check where the pod was created**

{% highlight sh %}
{% raw %}
kubectl get pods
{% endraw %}
{% endhighlight %}

**Check Docker status on nodes**

{% highlight sh %}
{% raw %}
docker ps
docker images
{% endraw %}
{% endhighlight %}

**After the pod is 'Running' Check web server access on the node**

{% highlight sh %}
{% raw %}
curl http://localhost
{% endraw %}
{% endhighlight %}

That's it !


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/fedora/fedora_ansible_config.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

