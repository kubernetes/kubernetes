# CoreOS Multinode Cluster

Use the [master.yaml](cloud-configs/master.yaml) and [node.yaml](cloud-configs/node.yaml) cloud-configs to provision a multi-node Kubernetes cluster.

> **Attention**: This requires at least CoreOS version **653.0.0**.

## Overview

* Provision the master node
* Capture the master node private IP address
* Edit node.yaml
* Provision one or more worker nodes 

### AWS

*Attention:* Replace ```<ami_image_id>``` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/).

#### Provision the Master

```
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

```
aws ec2 run-instances \
--image-id <ami_image_id> \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://master.yaml
```

#### Capture the private IP address

```
aws ec2 describe-instances --instance-id <master-instance-id>
```

#### Edit node.yaml

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the private IP address of the master node.

#### Provision worker nodes

```
aws ec2 run-instances \
--count 1 \
--image-id <ami_image_id> \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://node.yaml
```

### GCE

*Attention:* Replace ```<gce_image_id>``` below for a [suitable version of CoreOS image for GCE](https://coreos.com/docs/running-coreos/cloud-providers/google-compute-engine/).

#### Provision the Master

```
gcloud compute instances create master \
--image-project coreos-cloud \
--image <gce_image_id> \
--boot-disk-size 200GB \
--machine-type n1-standard-1 \
--zone us-central1-a \
--metadata-from-file user-data=master.yaml
```

#### Capture the private IP address

```
gcloud compute instances list
```

#### Edit node.yaml

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the private IP address of the master node.

#### Provision worker nodes

```
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

### VMware Fusion

#### Create the master config-drive

```
mkdir -p /tmp/new-drive/openstack/latest/
cp master.yaml /tmp/new-drive/openstack/latest/user_data
hdiutil makehybrid -iso -joliet -joliet-volume-name "config-2" -joliet -o master.iso /tmp/new-drive
```

#### Provision the Master

Boot the [vmware image](https://coreos.com/docs/running-coreos/platforms/vmware) using `master.iso` as a config drive.

#### Capture the master private IP address

#### Edit node.yaml

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the private IP address of the master node.

#### Create the node config-drive

```
mkdir -p /tmp/new-drive/openstack/latest/
cp node.yaml /tmp/new-drive/openstack/latest/user_data
hdiutil makehybrid -iso -joliet -joliet-volume-name "config-2" -joliet -o node.iso /tmp/new-drive
```

#### Provision worker nodes

Boot one or more the [vmware image](https://coreos.com/docs/running-coreos/platforms/vmware) using `node.iso` as a config drive.
