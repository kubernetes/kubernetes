# Kubernetes Ansible

This playbook helps you to set up a Kubernetes cluster on machines where you
can't or don't want to use the salt scripts and cluster up/down tools. They
can be real hardware, VMs, things in a public cloud, etc.

## Before starting

* Record the IP address/hostname of which machine you want to be your master (only support a single master)
* Record the IP address/hostname of the machine you want to be your etcd server (often same as master, only one)
* Record the IP addresses/hostname of the machines you want to be your nodes. (the master can also be a node)
* Make sure your ansible running machine has ansible 1.9 and python-netaddr installed.

### Configure the inventory file

Stick the system information gathered above into the 'inventory' file.

### Configure your cluster

You will want to look though all of the options in `group_vars/all.yml` and
set the variables to reflect your needs. The options should be described there
in full detail.

### Set up the actual kubernetes cluster

Now run the setup:

`$ ./setup.sh`

In generel this will work on very recent Fedora, rawhide or F21.  Future work to
support RHEL7, CentOS, and possible other distros should be forthcoming.

### You can just set up certain parts instead of doing it all

Only etcd:

`$ ./setup.sh --tags=etcd`

Only the kubernetes master:

`$ ./setup.sh --tags=masters`

Only the kubernetes nodes:

`$ ./setup.sh --tags=nodes`

### You may overwrite the inventory file by doing

`INVENTORY=myinventory ./setup.sh`

Only flannel:

    $ ./setup.sh --tags=flannel

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/ansible/README.md?pixel)]()
