# CoreOS - Single Node Kubernetes Cluster

Use the [standalone.yaml](cloud-configs/standalone.yaml) cloud-config to provision a single node Kubernetes cluster.

> **Attention**: This requires at least CoreOS version **653.0.0**.

### CoreOS image versions

### AWS

```
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

*Attention:* Replace ```<ami_image_id>``` bellow for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/).

```
aws ec2 run-instances \
--image-id <ami_image_id> \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://standalone.yaml
```

### GCE

*Attention:* Replace ```<gce_image_id>``` bellow for a [suitable version of CoreOS image for GCE](https://coreos.com/docs/running-coreos/cloud-providers/google-compute-engine/).

```
gcloud compute instances create standalone \
--image-project coreos-cloud \
--image <gce_image_id> \
--boot-disk-size 200GB \
--machine-type n1-standard-1 \
--zone us-central1-a \
--metadata-from-file user-data=standalone.yaml 
```

Next, setup an ssh tunnel to the instance so you can run kubectl from your local host.
In one terminal, run `gcloud compute ssh standalone --ssh-flag="-L 8080:127.0.0.1:8080"` and in a second
run `gcloud compute ssh standalone --ssh-flag="-R 8080:127.0.0.1:8080"`.


### VMware Fusion

Create a [config-drive](https://coreos.com/docs/cluster-management/setup/cloudinit-config-drive) ISO.

```
mkdir -p /tmp/new-drive/openstack/latest/
cp standalone.yaml /tmp/new-drive/openstack/latest/user_data
hdiutil makehybrid -iso -joliet -joliet-volume-name "config-2" -joliet -o standalone.iso /tmp/new-drive
```

Boot the [vmware image](https://coreos.com/docs/running-coreos/platforms/vmware) using the `standalone.iso` as a config drive.
