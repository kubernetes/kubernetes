<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/cloudstack.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started on [CloudStack](http://cloudstack.apache.org)
------------------------------------------------------------

**Table of Contents**

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Clone the playbook](#clone-the-playbook)
- [Create a Kubernetes cluster](#create-a-kubernetes-cluster)

### Introduction

CloudStack is a software to build public and private clouds based on hardware virtualization principles (traditional IaaS). To deploy Kubernetes on CloudStack there are several possibilities depending on the Cloud being used and what images are made available. [Exoscale](http://exoscale.ch) for instance makes a [CoreOS](http://coreos.com) template available, therefore instructions to deploy Kubernetes on coreOS can be used. CloudStack also has a vagrant plugin available, hence Vagrant could be used to deploy Kubernetes either using the existing shell provisioner or using new Salt based recipes.

[CoreOS](http://coreos.com) templates for CloudStack are built [nightly](http://stable.release.core-os.net/amd64-usr/current/). CloudStack operators need to [register](http://docs.cloudstack.apache.org/projects/cloudstack-administration/en/latest/templates.html) this template in their cloud before proceeding with these Kubernetes deployment instructions.

This guide uses an [Ansible playbook](https://github.com/runseb/ansible-kubernetes).
This is a completely automated, a single playbook deploys Kubernetes based on the coreOS [instructions](coreos/coreos_multinode_cluster.md).


This [Ansible](http://ansibleworks.com) playbook deploys Kubernetes on a CloudStack based Cloud using CoreOS images. The playbook, creates an ssh key pair, creates a security group and associated rules and finally starts coreOS instances configured via cloud-init.

### Prerequisites

    $ sudo apt-get install -y python-pip
    $ sudo pip install ansible
    $ sudo pip install cs

[_cs_](https://github.com/exoscale/cs) is a python module for the CloudStack API.

Set your CloudStack endpoint, API keys and HTTP method used.

You can define them as environment variables: `CLOUDSTACK_ENDPOINT`, `CLOUDSTACK_KEY`, `CLOUDSTACK_SECRET` and `CLOUDSTACK_METHOD`.

Or create a `~/.cloudstack.ini` file:

    [cloudstack]
    endpoint = <your cloudstack api endpoint>
    key = <your api access key>
    secret = <your api secret key>
    method = post

We need to use the http POST method to pass the _large_ userdata to the coreOS instances.

### Clone the playbook

    $ git clone --recursive https://github.com/runseb/ansible-kubernetes.git
    $ cd ansible-kubernetes

The [ansible-cloudstack](https://github.com/resmo/ansible-cloudstack) module is setup in this repository as a submodule, hence the `--recursive`.

### Create a Kubernetes cluster

You simply need to run the playbook.

    $ ansible-playbook k8s.yml

Some variables can be edited in the `k8s.yml` file.

    vars:
      ssh_key: k8s
      k8s_num_nodes: 2
      k8s_security_group_name: k8s
      k8s_node_prefix: k8s2
      k8s_template: Linux CoreOS alpha 435 64-bit 10GB Disk
      k8s_instance_type: Tiny

This will start a Kubernetes master node and a number of compute nodes (by default 2).
The `instance_type` and `template` by default are specific to [exoscale](http://exoscale.ch), edit them to specify your CloudStack cloud specific template and instance type (i.e service offering).

Check the tasks and templates in `roles/k8s` if you want to modify anything.

Once the playbook as finished, it will print out the IP of the Kubernetes master:

    TASK: [k8s | debug msg='k8s master IP is {{ k8s_master.default_ip }}'] ********

SSH to it using the key that was created and using the _core_ user and you can list the machines in your cluster:

    $ ssh -i ~/.ssh/id_rsa_k8s core@<maste IP>
    $ fleetctl list-machines
    MACHINE		IP		       METADATA
    a017c422...	<node #1 IP>   role=node
    ad13bf84...	<master IP>	   role=master
    e9af8293...	<node #2 IP>   role=node


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/cloudstack.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
