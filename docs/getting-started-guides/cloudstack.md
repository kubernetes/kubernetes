## Deploying LMKTFY on [CloudStack](http://cloudstack.apache.org)

CloudStack is a software to build public and private clouds based on hardware virtualization principles (traditional IaaS). To deploy LMKTFY on CloudStack there are several possibilities depending on the Cloud being used and what images are made available. [Exoscale](http://exoscale.ch) for instance makes a [CoreOS](http://coreos.com) template available, therefore instructions to deploy LMKTFY on coreOS can be used. CloudStack also has a vagrant plugin available, hence Vagrant could be used to deploy LMKTFY either using the existing shell provisioner or using new Salt based recipes.

[CoreOS](http://coreos.com) templates for CloudStack are built [nightly](http://stable.release.core-os.net/amd64-usr/current/). CloudStack operators need to [register](http://docs.cloudstack.apache.org/projects/cloudstack-administration/en/latest/templates.html) this template in their cloud before proceeding with these LMKTFY deployment instructions.

There are currently two deployment techniques.

* [LMKTFY on Exoscale](https://github.com/runseb/lmktfy-exoscale).
   This uses [libcloud](http://libcloud.apache.org) to launch CoreOS instances and pass the appropriate cloud-config setup using userdata. Several manual steps are required. This is obsoleted by the Ansible playbook detailed below.

* [Ansible playbook](https://github.com/runseb/ansible-lmktfy).
  This is completely automated, a single playbook deploys LMKTFY based on the coreOS [instructions](https://github.com/GoogleCloudPlatform/lmktfy/blob/master/docs/getting-started-guides/coreos/coreos_multinode_cluster.md).

#Ansible playbook

This [Ansible](http://ansibleworks.com) playbook deploys LMKTFY on a CloudStack based Cloud using CoreOS images. The playbook, creates an ssh key pair, creates a security group and associated rules and finally starts coreOS instances configured via cloud-init.

Prerequisites
-------------

    $ sudo apt-get install -y python-pip
    $ sudo pip install ansible
    $ sudo pip install cs

[_cs_](http://github.com/exoscale/cs) is a python module for the CloudStack API.

Set your CloudStack endpoint, API keys and HTTP method used.

You can define them as environment variables: `CLOUDSTACK_ENDPOINT`, `CLOUDSTACK_KEY`, `CLOUDSTACK_SECRET` and `CLOUDSTACK_METHOD`.

Or create a `~/.cloudstack.ini` file:

    [cloudstack]
    endpoint = <your cloudstack api endpoint>
    key = <your api access key> 
    secret = <your api secret key> 
    method = post

We need to use the http POST method to pass the _large_ userdata to the coreOS instances.

Clone the playbook
------------------

    $ git clone --recursive https://github.com/runseb/ansible-lmktfy.git
    $ cd ansible-lmktfy

The [ansible-cloudstack](https://github.com/resmo/ansible-cloudstack) module is setup in this repository as a submodule, hence the `--recursive`.

Create a LMKTFY cluster
---------------------------

You simply need to run the playbook.

    $ ansible-playbook lmktfy.yml

Some variables can be edited in the `lmktfy.yml` file.

    vars:
      ssh_key: lmktfy
      lmktfy_num_nodes: 2
      lmktfy_security_group_name: lmktfy
      lmktfy_node_prefix: lmktfy2
      lmktfy_template: Linux CoreOS alpha 435 64-bit 10GB Disk
      lmktfy_instance_type: Tiny

This will start a LMKTFY master node and a number of compute nodes (by default 2).
The `instance_type` and `template` by default are specific to [exoscale](http://exoscale.ch), edit them to specify your CloudStack cloud specific template and instance type (i.e service offering).

Check the tasks and templates in `roles/lmktfy` if you want to modify anything.

Once the playbook as finished, it will print out the IP of the LMKTFY master:

    TASK: [lmktfy | debug msg='lmktfy master IP is {{ lmktfy_master.default_ip }}'] ******** 

SSH to it using the key that was created and using the _core_ user and you can list the machines in your cluster:

    $ ssh -i ~/.ssh/id_rsa_lmktfy core@<maste IP>
    $ fleetctl list-machines
    MACHINE		IP		       METADATA
    a017c422...	<node #1 IP>   role=node
    ad13bf84...	<master IP>	   role=master
    e9af8293...	<node #2 IP>   role=node







