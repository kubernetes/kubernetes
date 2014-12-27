#Configuring kubernetes on [Fedora](http://fedoraproject.org) via [Ansible](http://www.ansible.com/home).

Configuring kubernetes on Fedora via Ansible offers a simple way to quickly create a clustered environment with little effort.

Requirements:

1. Host able to run ansible and able to clone the following repo: [kubernetes-ansible](https://github.com/eparis/kubernetes-ansible)
2. A Fedora 20+ or RHEL7 host to act as cluster master
3. As many Fedora 20+ or RHEL7 hosts as you would like, that act as cluster minions

The hosts can be virtual or bare metal.  The only requirement to make the ansible network setup work is that all of the machines are connected via the same layer 2 network.

Ansible will take care of the rest of the configuration for you - configuring networking, installing packages, handling the firewall, etc... This example will use one master and two minions.

## Configuring cluster information:

Hosts:
```
fed1 (master) = 192.168.121.205
fed2 (minion) = 192.168.121.84
fed3 (minion) = 192.168.121.116
```

**Make sure your local machine has ansible installed**

```
yum install -y ansible
```

**Clone the kubernetes-ansible repo on the host running Ansible.**

```
git clone https://github.com/eparis/kubernetes-ansible.git
cd kubernetes-ansible
```

**Tell ansible about each machine and its role in your cluster.**

Get the IP addresses from the master and minions.  Add those to the inventory file at the root of the repo on the host running Ansible.  Ignore the kube_ip_addr= option for a moment.

```
[masters]
192.168.121.205
    
[etcd]
192.168.121.205

[minions]
192.168.121.84  kube_ip_addr=[ignored]
192.168.121.116 kube_ip_addr=[ignored]
```

**Tell ansible which user has ssh access (and sudo access to root)**

edit: group_vars/all.yml

```
ansible_ssh_user: root
```

## Configuring ssh access to the cluster

If you already have ssh access to every machine using ssh public keys you may skip to [configuring the network](#configuring-the-network)

**Create a password file.**

The password file should contain the root password for every machine in the cluster.  It will be used in order to lay down your ssh public key. Make sure your machines sshd-config allows password logins from root.

```
echo "password" > ~/rootpassword
```

**Agree to accept each machine's ssh public key**

```
ansible-playbook -i inventory ping.yml # This will look like it fails, that's ok
```

**Push your ssh public key to every machine**

```
ansible-playbook -i inventory keys.yml
```

## Configuring the network

If you already have configured your network and docker will use it correctly, skip to [setting up the cluster](#setting-up-the-cluster)

The ansible scripts are quite hacky configuring the network, see the README

**Configure the ip addresses which should be used to run pods on each machine**

The IP address pool used to assign addresses to pods for each minion is the kube_ip_addr= option.  Choose a /24 to use for each minion and add that to you inventory file.

```
[minions]
192.168.121.84  kube_ip_addr=10.0.1.0
192.168.121.116 kube_ip_addr=10.0.2.0
```

**Run the network setup playbook**

```
ansible-playbook -i inventory hack-network.yml
```

## Setting up the cluster

**Configure the IP addresses used for services**

Each kubernetes service gets its own IP address.  These are not real IPs.  You need only select a range of IPs which are not in use elsewhere in your environment.  This must be done even if you do not use the network setup provided by the ansible scripts.

edit: group_vars/all.yml

```
kube_service_addresses: 10.254.0.0/16
```

**Tell ansible to get to work!**

```
ansible-playbook -i inventory setup.yml
```

## Testing and using your new cluster

That's all there is to it.  It's really that easy.  At this point you should have a functioning kubernetes cluster.  


**Show services running on masters and minions.**

```
systemctl | grep -i kube
```

**Show firewall rules on the masters and minions.**

```
iptables -nvL
```

**Create the following apache.json file and deploy pod to minion.**

```
cat ~/apache.json
{
  "id": "fedoraapache",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "fedoraapache",
      "containers": [{
        "name": "fedoraapache",
        "image": "fedora/apache",
        "ports": [{
          "containerPort": 80,
          "hostPort": 80
        }]
      }]
    }
  },
  "labels": {
    "name": "fedoraapache"
  }
}

/usr/bin/kubectl create -f apache.json
```

**Check where the pod was created**

```
/usr/bin/kubectl get pod fedoraapache
```

**Check Docker status on minion.**

```
docker ps
docker images
```

**After the pod is 'Running' Check web server access on the minion**

```
curl http://localhost
```
