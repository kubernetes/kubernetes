# Kubernetes Ansible

This playbook and set of roles set up a Kubernetes cluster onto machines. They
can be real hardware, VMs, things in a public cloud, etc. Anything that you can connect to via SSH.

## Before starting

* Record the IP address/hostname of which machine you want to be your master (only support a single master)
* Record the IP address/hostname of the machine you want to be your etcd server (often same as master, only one)
* Record the IP addresses/hostname of the machines you want to be your nodes. (the master can also be a node)
* Make sure your ansible running machine has ansible 1.9 and python-netaddr installed.

## Setup

### Configure inventory

Add the system information gathered above into the 'inventory' file, or create a new inventory file for the cluster.

### Configure Cluster options

Look though all of the options in `group_vars/all.yml` and
set the variables to reflect your needs. The options are described there
in full detail.

## Running the playbook

After going through the setup, run the setup script provided:

`$ ./setup.sh`

You may override the inventory file by doing:

`INVENTORY=myinventory ./setup.sh`


In general this will work on very recent Fedora, rawhide or F21.  Future work to
support RHEL7, CentOS, and possible other distros should be forthcoming.

### Targeted runs

You can just setup certain parts instead of doing it all.

#### etcd

`$ ./setup.sh --tags=etcd`

#### Kubernetes master

`$ ./setup.sh --tags=masters`

#### kubernetes nodes

`$ ./setup.sh --tags=nodes`

### flannel

`$ ./setup.sh --tags=flannel`

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/ansible/README.md?pixel)]()
