## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

### Cluster turnup

#### Download LMKTFY
##### a) Preferred Option: Install from [0.10.0 release](https://github.com/GoogleCloudPlatform/lmktfy/releases/tag/v0.10.0)
1. ```wget https://github.com/GoogleCloudPlatform/lmktfy/releases/download/v0.10.0/lmktfy.tar.gz```
2. ```tar -xzf lmktfy.tar.gz; cd lmktfy```
3. ```export PATH=$PATH:$PWD/platforms/<os>/<platform>```

##### b) Alternate Option: Install from source at head
1. ```git clone https://github.com/GoogleCloudPlatform/lmktfy.git```
2. ```cd lmktfy; make release```
3. ```export PATH=$PATH:$PWD/_output/local/bin/<os>/<platform>```

#### Turn up the cluster
```
export LMKTFYRNETES_PROVIDER=aws
cluster/lmktfy-up.sh
```

The script above relies on AWS S3 to deploy the software to instances running in EC2.

NOTE: The script will provision a new VPC and a 5 node lmktfy cluster in us-west-2 (Oregon). It'll also try to create or
reuse a keypair called "lmktfy", and IAM profiles called "lmktfy-master" and "lmktfy-minion".  If these
already exist, make sure you want them to be used here.

Once the cluster is up, it will print the ip address of your cluster, this process takes about 5 to 10 minutes.

```
export LMKTFYRNETES_MASTER=https://<ip-address>
```

Also setup your path to point to the released binaries:
```
export PATH=$PATH:$PWD:/cluster
```

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/lmktfy#where-to-go-next)

### Tearing down the cluster
```
cd lmktfy
cluster/lmktfy-down.sh
```

### Cloud Formation [optional]
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.
