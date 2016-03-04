<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# CoreOS Multinode Cluster

Use the [master.yaml](cloud-configs/master.yaml) and [node.yaml](cloud-configs/node.yaml) cloud-configs to provision a multi-node Kubernetes cluster.

> **Attention**: This requires at least CoreOS version **[695.0.0][coreos695]**, which includes `etcd2`.

[coreos695]: https://coreos.com/releases/#695.0.0

## Overview

* Provision the master node
* Capture the master node private IP address
* Edit node.yaml
* Provision one or more worker nodes

### AWS

*Attention:* Replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/).

#### Provision the Master

```sh
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

```sh
aws ec2 run-instances \
--image-id <ami_image_id> \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://master.yaml
```

#### Capture the private IP address

```sh
aws ec2 describe-instances --instance-id <master-instance-id>
```

#### Edit node.yaml

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the private IP address of the master node.

#### Provision worker nodes

```sh
aws ec2 run-instances \
--count 1 \
--image-id <ami_image_id> \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://node.yaml
```

### Google Compute Engine (GCE)

*Attention:* Replace `<gce_image_id>` below for a [suitable version of CoreOS image for Google Compute Engine](https://coreos.com/docs/running-coreos/cloud-providers/google-compute-engine/).

#### Provision the Master

```sh
gcloud compute instances create master \
--image-project coreos-cloud \
--image <gce_image_id> \
--boot-disk-size 200GB \
--machine-type n1-standard-1 \
--zone us-central1-a \
--metadata-from-file user-data=master.yaml
```

#### Capture the private IP address

```sh
gcloud compute instances list
```

#### Edit node.yaml

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the private IP address of the master node.

#### Provision worker nodes

```sh
gcloud compute instances create node1 \
--image-project coreos-cloud \
--image <gce_image_id> \
--boot-disk-size 200GB \
--machine-type n1-standard-1 \
--zone us-central1-a \
--metadata-from-file user-data=node.yaml
```

#### Establish network connectivity

Next, setup an ssh tunnel to the master so you can run kubectl from your local host.
In one terminal, run `gcloud compute ssh master --ssh-flag="-L 8080:127.0.0.1:8080"` and in a second
run `gcloud compute ssh master --ssh-flag="-R 8080:127.0.0.1:8080"`.

### OpenStack

These instructions are for running on the command line.  Most of this you can also do through the Horizon dashboard.
These instructions were tested on the Ice House release on a Metacloud distribution of OpenStack but should be similar if not the same across other versions/distributions of OpenStack.

#### Make sure you can connect with OpenStack

Make sure the environment variables are set for OpenStack such as:

```sh
OS_TENANT_ID
OS_PASSWORD
OS_AUTH_URL
OS_USERNAME
OS_TENANT_NAME
```

Test this works with something like:

```
nova list
```

#### Get a Suitable CoreOS Image

You'll need a [suitable version of CoreOS image for OpenStack](https://coreos.com/os/docs/latest/booting-on-openstack.html)
Once you download that, upload it to glance.  An example is shown below:

```sh
glance image-create --name CoreOS723 \
--container-format bare --disk-format qcow2 \
--file coreos_production_openstack_image.img \
--is-public True
```

#### Create security group

```sh
nova secgroup-create kubernetes "Kubernetes Security Group"
nova secgroup-add-rule kubernetes tcp 22 22   0.0.0.0/0
nova secgroup-add-rule kubernetes tcp 80 80   0.0.0.0/0
```

#### Provision the Master

```sh
nova boot \
--image <image_name> \
--key-name <my_key> \
--flavor <flavor id> \
--security-group kubernetes \
--user-data files/master.yaml \
kube-master
```

```<image_name>``` is the CoreOS image name.  In our example we can use the image we created in the previous step and put in 'CoreOS723'

```<my_key>``` is the keypair name that you already generated to access the instance.

```<flavor_id>``` is the flavor ID you use to size the instance.  Run ```nova flavor-list``` to get the IDs.  3 on the system this was tested with gives the m1.large size.

The important part is to ensure you have the files/master.yml as this is what will do all the post boot configuration. This path is relevant so we are assuming in this example that you are running the nova command in a directory where there is a subdirectory called files that has the master.yml file in it.  Absolute paths also work.

Next, assign it a public IP address:

``` 
nova floating-ip-list
```

Get an IP address that's free and run:

```
nova floating-ip-associate kube-master <ip address>
```

where ```<ip address>``` is the IP address that was available from the ```nova floating-ip-list``` command.

#### Provision Worker Nodes

Edit ```node.yaml``` and replace all instances of ```<master-private-ip>``` with the private IP address of the master node.  You can get this by running ```nova show kube-master``` assuming you named your instance kube master.  This is not the floating IP address you just assigned it.

```sh
nova boot \
--image <image_name> \
--key-name <my_key> \
--flavor <flavor id> \
--security-group kubernetes \
--user-data files/node.yaml \
minion01
```

This is basically the same as the master nodes but with the node.yaml post-boot script instead of the master.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/coreos/coreos_multinode_cluster.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
