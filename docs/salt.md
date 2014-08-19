# Using Salt to configure Kubernetes

The Kubernetes cluster can be configured using Salt.

The Salt scripts are shared across multiple hosting providers, so it's important to understand some background information prior to making a modification to ensure your changes do not break hosting Kubernetes across multiple environments.  Depending on where you host your Kubernetes cluster, you may be using different operating systems and different networking configurations.  As a result, it's important to understand some background information before making Salt changes in order to minimize introducing failures for other hosting providers.

## Salt cluster setup

The **salt-master** service runs on the kubernetes-master node.

The **salt-minion** service runs on the kubernetes-master node and each kubernetes-minion node in the cluster.

Each salt-minion service is configured to interact with the **salt-master** service hosted on the kubernetes-master via the **master.conf** file.  

```
[root@kubernetes-master] $ cat /etc/salt/minion.d/master.conf
master: kubernetes-master
```
The salt-master is contacted by each salt-minion and depending upon the machine information presented, the salt-master will provision the machine as either a kubernetes-master or kubernetes-minion with all the required capabilities needed to run Kubernetes.

If you are running the Vagrant based environment, the **salt-api** service is running on the kubernetes-master.  It is configured to enable the vagrant user to introspect the salt cluster in order to find out about machines in the Vagrant environment via a REST API.  

## Salt security

Security is not enabled on the salt-master, and the salt-master is configured to auto-accept incoming requests from minions.  It is not recommended to use this security configuration in production environments.

```
[root@kubernetes-master] $ cat /etc/salt/master.d/auto-accept.conf
open_mode: True
auto_accept: True
```
## Salt minion configuration

Each minion in the salt cluster has an associated configuration that instructs the salt-master how to provision the required resources on the machine.

An example file is presented below using the Vagrant based environment.

```
[root@kubernetes-master] $ cat /etc/salt/minion.d/grains.conf
grains:
  master_ip: $MASTER_IP
  etcd_servers: $MASTER_IP
  cloud_provider: vagrant
  roles:
    - kubernetes-master
```

Each hosting environment has a slightly different grains.conf file that is used to build conditional logic where required in the Salt files.

The following enumerates the set of defined key/value pairs that are supported today.  If you add new ones, please make sure to update this list.

Key | Value
------------- | -------------
cbr-cidr | (Optional) The minion IP address range used for the docker container bridge.
cloud | (Optional) Which IaaS platform is used to host kubernetes, *gce*, *azure*
cloud_provider | (Optional) The cloud_provider used by apiserver: *gce*, *azure*, *vagrant*
etcd_servers | (Required) Comma-delimited list of IP addresses the apiserver and kubelet use to reach etcd
hostnamef | (Optional) The full host name of the machine, i.e. hostname -f
master_ip | (Optional) The IP address that the apiserver will bind against
roles | (Required) 1. **kubernetes-master** means this machine is the master in the kubernetes cluster.  2. **kubernetes-pool** means this machine is a kubernetes-minion.  Depending on the role, the Salt scripts will provision different resources on the machine.

These keys may be leveraged by the Salt sls files to branch behavior.

In addition, a cluster may be running a Debian based operating system or Red Hat based operating system (Centos, Fedora, RHEL, etc.).  As a result, its important to sometimes distinguish behavior based on operating system using if branches like the following.

```
{% if grains['os_family'] == 'RedHat' %}
// something specific to a RedHat environment (Centos, Fedora, RHEL) where you may use yum, systemd, etc.
{% else %}
// something specific to Debian environment (apt-get, initd)
{% endif %}
```

## Best Practices

1.  When configuring default arguments for processes, its best to avoid the use of EnvironmentFiles (Systemd in Red Hat environments) or init.d files (Debian distributions) to hold default values that should be common across operating system environments.  This helps keep our Salt template files easy to understand for editors that may not be familiar with the particulars of each distribution.

## Future enhancements (Networking)

Per pod IP configuration is provider specific, so when making networking changes, its important to sand-box these as all providers may not use the same mechanisms (iptables, openvswitch, etc.)

We should define a grains.conf key that captures more specifically what network configuration environment is being used to avoid future confusion across providers.