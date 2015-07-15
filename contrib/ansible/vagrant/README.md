## Vagrant deployer for Kubernetes Ansible

This deployer sets up a vagrant cluster and installs kubernetes with flannel on it.

The URI's in the Vagrantfile may need to be changed depending on the exact version of openstack which you have.

## Before you start !

If running the openstack provider, then of course, you need to modify the key credentials and so on to match your particular openstack credentials.

At the time of this writing (july 2 2015) no other providers are supported, but this recipe is pretty easy to port to virtualbox, kvm, and so on if you want.

## USAGE

To use, first modify the Vagrantfile to reflect the machines you want.

This is easy: You just change the number of nodes.

Then, update the kubernetes ansible data structure to include more nodes if you want them.

## Provider

Now make sure to install openstack provider for vagrant.

`vagrant plugin install vagrant-openstack-provider`

NOTE This is a more up-to-date provider than the similar  `vagrant-openstack-plugin`.

# Now, vagrant up!

Now lets run it.  Again, make sure you look at your openstack dashboard to see the URLs and security groups and tokens that you want.  In general, you want an open security group (i.e. for port 8080 and so on) and you want an SSH key that is named that you can use to ssh into all machines, and make sure you set those in the Vagrantfile correctly.  ESPECIALLY also make sure you set your tenant-name is right.

`VAGRANT_LOG=info vagrant up --provision-with=shell ; vagrant provision provision-with=ansible` 

This will run a first pass provisioning, which sets up the raw machines, followed by a second pass,

which sets up kuberentes, etcd, and so on.


