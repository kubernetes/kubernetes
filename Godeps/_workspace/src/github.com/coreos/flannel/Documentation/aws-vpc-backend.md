# Amazon VPC Backend for Flannel

When running within an Amazon VPC, we recommend using the aws-vpc backend which, instead of using encapsulation, manipulates IP routes to achieve maximum performance. Because of this, a separate flannel interface is not created.

In order to run flannel on AWS we need to first create an [Amazon VPC](http://aws.amazon.com/vpc/).
Amazon VPC enables us to launch EC2 instances into a virtual network, which we can configure via its route table.

From the VPC dashboard start out by running the "VPC Wizard":

- Select "VPC with a Single Public Subnet"
- Configure the network and the subnet address ranges 

<br/>
<div class="row">
  <div class="col-lg-10 col-lg-offset-1 col-md-10 col-md-offset-1 col-sm-12 col-xs-12 co-m-screenshot">
    <a href="{{site.baseurl}}/assets/images/media/aws-vpc.png">
      <img src="{{site.baseurl}}/assets/images/media/aws-vpc.png" alt="New Amazon VPC" />
    </a>
  </div>
</div>
<div class="caption">Creating a new Amazon VPC</div>

Now that we have set up our VPC and subnet, let’s create an Identity and Access Management ([IAM](http://aws.amazon.com/iam/)) role to grant the required permissions to our EC2 instances. 

From the console, select Services -> Administration & Security -> IAM. 

We first need to create a [policy](http://docs.aws.amazon.com/IAM/latest/UserGuide/policies_overview.html) that we will later assign to an IAM role.
Under "Create Policy" select the "Create Your Own Policy" option.
The following permissions are required as shown below in the sample policy document.

- ec2:CreateRoute
- ec2:DeleteRoute
- ec2:ReplaceRoute
- ec2:DescribeRouteTables
- ec2:DescribeInstances

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
          "Effect": "Allow",
          "Action": [
              "ec2:CreateRoute",
              "ec2:DeleteRoute",
              "ec2:ReplaceRoute"
          ],
          "Resource": [
              "*"
          ]
    },
    {
          "Effect": "Allow",
          "Action": [
              "ec2:DescribeRouteTables",
              "ec2:DescribeInstances"
          ],
          "Resource": "*"
    }
  ]
}
```

Note that although the first three permissions can be tied to the route table resource of our subnet, the ec2:Describe\* permissions can not be limited to a particular resource.
For simplicity, we leave the "Resource" as wildcard in both. 

With the policy added, let's attach it to a new IAM role by clicking the "Create New Role" button and setting the following options:

- Role Name: `demo-role`
- Role Type: "Amazon EC2"
- Attach the policy we created earlier

We are now all set to launch an EC2 instance. 
In the launch wizard, choose the `CoreOS-stable-681.2.0` image and under "Configure Instance Details" perform the following steps:

- Change the "Network" to the VPC we just created
- Enable "Auto-assign Public IP"
- Select IAM `demo-role`

<br/>
<div class="row">
  <div class="col-lg-10 col-lg-offset-1 col-md-10 col-md-offset-1 col-sm-12 col-xs-12 co-m-screenshot">
    <a href="{{site.baseurl}}/assets/images/media/aws-instance-configuration.png" class="co-m-screenshot">
      <img src="{{site.baseurl}}/assets/images/media/aws-instance-configuration.png" alt="AWS EC2 Instance Details" />
    </a>
  </div>
</div>
<div class="caption">Configuring AWS EC2 instance details</div>

Under the "Configure Security Group" tab add the rules to allow etcd traffic (tcp/2379), SSH and ICMP.

Go ahead and launch the instance! 

Since our instance will be sending and receiving traffic for IPs other than the one assigned by our subnet, we need to disable source/destination checks. 

<br/>
<div class="row">
  <div class="col-lg-10 col-lg-offset-1 col-md-10 col-md-offset-1 col-sm-12 col-xs-12 co-m-screenshot">
    <a href="{{site.baseurl}}/assets/images/media/aws-src-dst-check.png" class="co-m-screenshot">
      <img src="{{site.baseurl}}/assets/images/media/aws-src-dst-check.png" alt="Disable AWS Source/Dest Check" />
    </a>
  </div>
</div>
<div class="caption">Disable AWS Source/Dest Check</div>

All that’s left now is to start etcd, publish the network configuration and run the flannel daemon. 
First, SSH into `demo-instance-1`:

- Start etcd:

```
$ etcd2 -advertise-client-urls http://$INTERNAL_IP:2379 -listen-client-urls http://0.0.0.0:2379
```
- Publish configuration in etcd (ensure that the network range does not overlap with the one configured for the VPC)

```
$ etcdctl set /coreos.com/network/config '{"Network":"10.20.0.0/16", "Backend": {"Type": "aws-vpc"}}'
```
- Fetch the latest release using wget from [here](https://github.com/coreos/flannel/releases/download/v0.5.0/flannel-0.5.0-linux-amd64.tar.gz)
- Run flannel daemon:

```
sudo ./flanneld --etcd-endpoints=http://127.0.0.1:2379
```

Next, create and connect to a clone of `demo-instance-1`.
Run flannel with the `--etcd-endpoints` flag set to the *internal* IP of the instance running etcd.

Confirm that the subnet route table has entries for the lease acquired by each of the subnets.

<br/>
<div class="row">
  <div class="col-lg-10 col-lg-offset-1 col-md-10 col-md-offset-1 col-sm-12 col-xs-12 co-m-screenshot">
    <a href="{{site.baseurl}}/assets/images/media/aws-routes.png" class="co-m-screenshot">
      <img src="{{site.baseurl}}/assets/images/media/aws-routes.png" alt="AWS Routes" />
    </a>
  </div>
</div>
<div class="caption">AWS Routes</div>

### Limitations

Keep in mind that the Amazon VPC [limits](http://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_Appendix_Limits.html) the number of entries per route table to 50. If you require more routes, request a quota increase or simply switch to the VXLAN backend.
