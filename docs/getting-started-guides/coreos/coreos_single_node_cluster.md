# CoreOS - Single Node Kubernetes Cluster

Use the [standalone.yaml](cloud-configs/standalone.yaml) cloud-config to provision a single node Kubernetes cluster.

### AWS

```
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

```
aws ec2 run-instances \
--image-id ami-c484f5ac \
--key-name <keypair> \
--region us-west-2 \
--security-groups kubernetes \
--instance-type m3.medium \
--user-data file://standalone.yaml
```

### GCE

```
gcloud compute instances create standalone \
--image-project coreos-cloud \
--image coreos-alpha-557-0-0-v20150109 \
--boot-disk-size 200GB \
--machine-type n1-standard-1 \
--zone us-central1-a \
--metadata-from-file user-data=standalone.yaml 
```

### VMware Fusion

Create a [config-drive](https://coreos.com/docs/cluster-management/setup/cloudinit-config-drive) ISO.

```
mkdir -p /tmp/new-drive/openstack/latest/
cp standalone.yaml /tmp/new-drive/openstack/latest/user_data
hdiutil makehybrid -iso -joliet -joliet-volume-name "config-2" -joliet -o standalone.iso /tmp/new-drive
```

Boot the [vmware image](https://coreos.com/docs/running-coreos/platforms/vmware) using the `standalone.iso` as a config drive.
