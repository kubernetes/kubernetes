#Configuring kubernetes on [Fedora](http://fedoraproject.org) via [Ansible](http://www.ansible.com/home).

Configuring kubernetes on Fedora via Ansible offers a simple way to quickly create a clustered environment with little effort.

Requirements:

1. Host running Ansible with the following repo cloned: [kubernetes-ansible](https://github.com/eparis/kubernetes-ansible)
2. A Fedora 20 or greater host to act as cluster master
3. As many Fedora 20 hosts as you would like, that act as cluster minions

The hosts can be virtual or bare metal.  It's your choice. Ansible will take care of the rest of the configuration for you - configuring networking, installing packages, handling the firewall, etc... This example will use one master and two minions.

**System Information:**

Hosts:
```
fed1 (master) = 192.168.121.205
fed2 (minion) = 192.168.121.84
fed2 (minion)= 192.168.121.116
```

Versions:

```
Fedora release 20 (Heisenbug)

etcd-0.4.6-3.fc20.x86_64
kubernetes-0.2-0.4.gitcc7999c.fc20.x86_64
```


Now, let's get started with the configuration.

* Show Ansible on the host running Ansible.

```
rpm -ql ansible | grep bin
cat /etc/fedora-release
```

* Clone the kubernetes-ansible repo on the host running Ansible.

```
git clone https://github.com/eparis/kubernetes-ansible.git
cd kubernetes-ansible

```

* Get IP addresses from master and minion, add to inventory file at the root of the repo on the host running Ansible.

```
[masters]
192.168.121.205
    
[etcd]
192.168.121.205

[minions]
192.168.121.84  kube_ip_addr=10.0.1.1
192.168.121.116 kube_ip_addr=10.0.2.1
```

* Explore the playbooks and the Ansible files.

```
tree roles/
cat keys.yml
cat setup.yml
```

* Create a password file.  Hopefully you don't use the password below.

```
echo "password" > ~/rootpassword
```

* Set root password on all atomic hosts to match the password in the _rootpassword_ file.  Ansible will use the ansible_ssh_pass method to parse the file and gain access all the hosts.

* Ping the hosts.

```
ansible-playbook -i inventory ping.yml # This will look like it fails, that's ok
```

* Configure the SSH keys.

```
ansible-playbook -i inventory keys.yml
```

* Run the playbook

```
ansible-playbook -i inventory setup.yml
```

That's all there is to it.  It's really that easy.  At this point you should have a functioning kubernetes cluster.  


* Show services running on masters and minions.

```
systemctl | grep -i kube
```

* Show firewall rules on the masters and minions.

```
iptables -nvL
```

* Create the following apache.json file and deploy pod to minion.

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

/bin/kubecfg -c apache.json create pods
```

* Check Docker status on minion.

```
docker ps
docker images
```

* Check web server access on a minion.

```
curl http://localhost
```

