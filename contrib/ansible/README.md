# Kubernetes Ansible

This playbook helps you to set up a Kubernetes cluster on machines where you
can't or don't want to use the salt scripts and cluster up/down tools. They
can be real hardware, VMs, things in a public cloud, etc.

## Usage

* Record the IP address of which machine you want to be your master
* Record the IP address of the machine you want to be your etcd server (often same as master)
* Record the IP addresses of the machines you want to be your nodes. (master can be a node)

Stick the system information into the 'inventory' file.

### Configure your cluster

You will want to look though all of the options in `group_vars/all.yml` and
set the variables to reflect your needs. The options should be described there
in full detail.

### Set up the actual kubernetes cluster

Now run the setup:

    $ ansible-playbook -i inventory cluster.yml

In generel this will work on very recent Fedora, rawhide or F21.  Future work to
support RHEL7, CentOS, and possible other distros should be forthcoming.

### You can just set up certain parts instead of doing it all

Only the kubernetes daemons:

    $ ansible-playbook -i inventory kubernetes-services.yml

Only etcd:

    $ ansible-playbook -i inventory etcd.yml

Only flannel:

    $ ansible-playbook -i inventory flannel.yml


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/ansible/README.md?pixel)]()
