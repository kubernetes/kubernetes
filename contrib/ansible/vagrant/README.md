## Vagrant deployer for Kubernetes Ansible

This deployer sets up a vagrant cluster and installs kubernetes with flannel on it.

## Before you start !

You will need a functioning vagrant provider. Currently supported are openstack.

## USAGE

In general all that should be needed it to run

```
vagrant up
```

If you export an env variable such as
```
export NUM_MINIONS=4
```

The system will create that number of nodes. Default is 2.

## Provider Specific Information
Vagrant tries to be intelligent and pick the first provider supported by your installation. If you want to specify a provider you can do so by running vagrant like so:
```
vagrant up --provider=openstack
```

### OpenStack
Make sure to install the openstack provider for vagrant.
```
vagrant plugin install vagrant-openstack-provider --plugin-version ">= 0.6.1"
```
NOTE This is a more up-to-date provider than the similar  `vagrant-openstack-plugin`.

Also note that current (required) versions of `vagrant-openstack-provider` are not compatible with ruby 2.2.
https://github.com/ggiamarchi/vagrant-openstack-provider/pull/237
So make sure you get at least version 0.6.1.

To use the vagrant openstack provider you will need
- Copy `openstack_config.yml.example` to `openstack_config.yml`
- Edit `openstack_config.yml` to include your relevant details.

For vagrant (1.7.2) does not seem to ever want to pick openstack as the provider. So you will need to tell it to use openstack explicitly.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/ansible/vagrant/README.md?pixel)]()
