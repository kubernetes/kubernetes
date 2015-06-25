#Configuring kubernetes on [Fedora](http://fedoraproject.org) via [Ansible](http://www.ansible.com/home).

Configuring kubernetes on Fedora via Ansible offers a simple way to quickly create a clustered environment with little effort.

Requirements:

1. Host able to run ansible and able to clone the following repo: [kubernetes-ansible](https://github.com/eparis/kubernetes-ansible)
2. A Fedora 20+ or RHEL7 host to act as cluster master
3. As many Fedora 20+ or RHEL7 hosts as you would like, that act as cluster minions

The hosts can be virtual or bare metal.  The only requirement to make the ansible network setup work is that all of the machines are connected via the same layer 2 network.

Ansible will take care of the rest of the configuration for you - configuring networking, installing packages, handling the firewall, etc... This example will use one master and two minions.

## Architecture of the cluster

A Kubernetes cluster reqiures etcd, a master, and n minions, so we will create a cluster with three hosts, for example:

```
    fed1 (master,etcd) = 192.168.121.205
    fed2 (minion) = 192.168.121.84
    fed3 (minion) = 192.168.121.116
```

**Make sure your local machine** 

 - has ansible
 - has git

**then we just clone down the kubernetes-ansible repository** 

```
   yum install -y ansible git
   git clone https://github.com/eparis/kubernetes-ansible.git
   cd kubernetes-ansible
```

**Tell ansible about each machine and its role in your cluster.**

Get the IP addresses from the master and minions.  Add those to the `inventory` file (at the root of the repo) on the host running Ansible.  

We will set the kube_ip_addr to '10.254.0.[1-3]', for now.  The reason we do this is explained later...  It might work for you as a default.

```
[masters]
192.168.121.205
    
[etcd]
192.168.121.205

[minions]
192.168.121.84  kube_ip_addr=[10.254.0.1]
192.168.121.116 kube_ip_addr=[10.254.0.2]
```

**Setup ansible access to your nodes**

If you already are running on a machine which has passwordless ssh access to the fed[1-3] nodes, and 'sudo' privileges, simply set the value of `ansible_ssh_user` in `group_vars/all.yaml` to the username which you use to ssh to the nodes (i.e. `fedora`), and proceed to the next step...

*Otherwise* setup ssh on the machines like so (you will need to know the root password to all machines in the cluster).

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

After this is completed, ansible is now enabled to ssh into any of the machines you're configuring.

```
ansible-playbook -i inventory ping.yml # This will look like it fails, that's ok
```

**Push your ssh public key to every machine**

Again, you can skip this step if your ansible machine has ssh access to the nodes you are going to use in the kubernetes cluster.
```
ansible-playbook -i inventory keys.yml
```

## Configuring the internal kubernetes network

If you already have configured your network and docker will use it correctly, skip to [setting up the cluster](#setting-up-the-cluster)

The ansible scripts are quite hacky configuring the network, you can see the [README](https://github.com/eparis/kubernetes-ansible) for details, or you can simply enter in variants of the 'kube_service_addresses' (in the all.yaml file) as `kube_ip_addr` entries in the minions field, as shown in the next section.

**Configure the ip addresses which should be used to run pods on each machine**

The IP address pool used to assign addresses to pods for each minion is the `kube_ip_addr`= option.  Choose a /24 to use for each minion and add that to you inventory file.

For this example, as shown earlier, we can do something like this...

```
[minions]
192.168.121.84  kube_ip_addr=10.254.0.1
192.168.121.116 kube_ip_addr=10.254.0.2
```

**Run the network setup playbook**

There are two ways to do this: via flannel, or using NetworkManager. 

Flannel is a cleaner mechanism to use, and is the recommended choice.

- If you are using flannel, you should check the kubernetes-ansible repository above. 

Currently, you essentially have to (1) update group_vars/all.yml, and then (2) run
```
ansible-playbook -i inventory flannel.yml
```

- On the other hand, if using the NetworkManager based setup (i.e. you do not  want to use flannel).

On EACH node, make sure NetworkManager is installed, and the service "NetworkManager" is running, then you can run 
the network manager playbook...

```
ansible-playbook -i inventory ./old-network-config/hack-network.yml
```

## Setting up the cluster

**Configure the IP addresses used for services**

Each kubernetes service gets its own IP address.  These are not real IPs.  You need only select a range of IPs which are not in use elsewhere in your environment.  This must be done even if you do not use the network setup provided by the ansible scripts.

edit: group_vars/all.yml

```
kube_service_addresses: 10.254.0.0/16
```

**Tell ansible to get to work!**

This will finally setup your whole kubernetes cluster for you.

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
cat << EOF > apache.json
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
EOF 

/usr/bin/kubectl create -f apache.json

**Testing your new kube cluster**

```

**Check where the pod was created**

```
kubectl get pods
```

Important : Note that the IP of the pods IP fields are on the network which you created in the kube_ip_addr file.

In this example, that was the 10.254 network.

If you see 172 in the IP fields, networking was not setup correctly, and you may want to re run or dive deeper into the way networking is being setup by looking at the details of the networking scripts used above.

**Check Docker status on minion.**

```
docker ps
docker images
```

**After the pod is 'Running' Check web server access on the minion**

```
curl http://localhost
```

That's it !


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/fedora/fedora_ansible_config.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/getting-started-guides/fedora/fedora_ansible_config.md?pixel)]()
