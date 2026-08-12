# Amazon EC2 Construct Library



The `aws-cdk-lib/aws-ec2` package contains primitives for setting up networking and
instances.

```ts nofixture
import * as ec2 from 'aws-cdk-lib/aws-ec2';
```

## VPC

Most projects need a Virtual Private Cloud to provide security by means of
network partitioning. This is achieved by creating an instance of
`Vpc`:

```ts
const vpc = new ec2.Vpc(this, 'VPC');
```

All default constructs require EC2 instances to be launched inside a VPC, so
you should generally start by defining a VPC whenever you need to launch
instances for your project.

### Subnet Types

A VPC consists of one or more subnets that instances can be placed into. CDK
distinguishes three different subnet types:

* **Public (`SubnetType.PUBLIC`)** - public subnets connect directly to the Internet using an
  Internet Gateway. If you want your instances to have a public IP address
  and be directly reachable from the Internet, you must place them in a
  public subnet.
* **Private with Internet Access (`SubnetType.PRIVATE_WITH_EGRESS`)** - instances in private subnets are not directly routable from the
  Internet, and you must provide a way to connect out to the Internet.
  By default, a NAT gateway is created in every public subnet for maximum availability. Be
  aware that you will be charged for NAT gateways.
  Alternatively you can set `natGateways:0` and provide your own egress configuration (i.e through Transit Gateway)
* **Isolated (`SubnetType.PRIVATE_ISOLATED`)** - isolated subnets do not route from or to the Internet, and
  as such do not require NAT gateways. They can only connect to or be
  connected to from other instances in the same VPC. A default VPC configuration
  will not include isolated subnets,

A default VPC configuration will create public and **private** subnets. However, if
`natGateways:0` **and** `subnetConfiguration` is undefined, default VPC configuration
will create public and **isolated** subnets. See [*Advanced Subnet Configuration*](#advanced-subnet-configuration)
below for information on how to change the default subnet configuration.

Constructs using the VPC will "launch instances" (or more accurately, create
Elastic Network Interfaces) into one or more of the subnets. They all accept
a property called `subnetSelection` (sometimes called `vpcSubnets`) to allow
you to select in what subnet to place the ENIs, usually defaulting to
*private* subnets if the property is omitted.

If you would like to save on the cost of NAT gateways, you can use
*isolated* subnets instead of *private* subnets (as described in Advanced
*Subnet Configuration*). If you need private instances to have
internet connectivity, another option is to reduce the number of NAT gateways
created by setting the `natGateways` property to a lower value (the default
is one NAT gateway per availability zone). Be aware that this may have
availability implications for your application.

[Read more about
subnets](https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_Subnets.html).

### Control over availability zones

By default, a VPC will spread over at most 3 Availability Zones available to
it. To change the number of Availability Zones that the VPC will spread over,
specify the `maxAzs` property when defining it.

The number of Availability Zones that are available depends on the *region*
and *account* of the Stack containing the VPC. If the [region and account are
specified](https://docs.aws.amazon.com/cdk/latest/guide/environments.html) on
the Stack, the CLI will [look up the existing Availability
Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#using-regions-availability-zones-describe)
and get an accurate count. The result of this operation will be written to a file
called `cdk.context.json`. You must commit this file to source control so
that the lookup values are available in non-privileged environments such
as CI build steps, and to ensure your template builds are repeatable.


If region and account are not specified, the stack
could be deployed anywhere and it will have to make a safe choice, limiting
itself to 2 Availability Zones.

Therefore, to get the VPC to spread over 3 or more availability zones, you
must specify the environment where the stack will be deployed.

You can gain full control over the availability zones selection strategy by overriding the Stack's [`get availabilityZones()`](https://github.com/aws/aws-cdk/blob/main/packages/@aws-cdk/core/lib/stack.ts) method:

```text
// This example is only available in TypeScript

class MyStack extends Stack {

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // ...
  }

  get availabilityZones(): string[] {
    return ['us-west-2a', 'us-west-2b'];
  }

}
```

Note that overriding the `get availabilityZones()` method will override the default behavior for all constructs defined within the Stack.

### Choosing subnets for resources

When creating resources that create Elastic Network Interfaces (such as
databases or instances), there is an option to choose which subnets to place
them in. For example, a VPC endpoint by default is placed into a subnet in
every availability zone, but you can override which subnets to use. The property
is typically called one of `subnets`, `vpcSubnets` or `subnetSelection`.

The example below will place the endpoint into two AZs (`us-east-1a` and `us-east-1c`),
in Isolated subnets:

```ts
declare const vpc: ec2.Vpc;

new ec2.InterfaceVpcEndpoint(this, 'VPC Endpoint', {
  vpc,
  service: new ec2.InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
  subnets: {
    subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
    availabilityZones: ['us-east-1a', 'us-east-1c']
  }
});
```

You can also specify specific subnet objects for granular control:

```ts
declare const vpc: ec2.Vpc;
declare const subnet1: ec2.Subnet;
declare const subnet2: ec2.Subnet;

new ec2.InterfaceVpcEndpoint(this, 'VPC Endpoint', {
  vpc,
  service: new ec2.InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
  subnets: {
    subnets: [subnet1, subnet2]
  }
});
```

Which subnets are selected is evaluated as follows:

* `subnets`: if specific subnet objects are supplied, these are selected, and no other
  logic is used.
* `subnetType`/`subnetGroupName`: otherwise, a set of subnets is selected by
  supplying either type or name:
  * `subnetType` will select all subnets of the given type.
  * `subnetGroupName` should be used to distinguish between multiple groups of subnets of
    the same type (for example, you may want to separate your application instances and your
    RDS instances into two distinct groups of Isolated subnets).
  * If neither are given, the first available subnet group of a given type that
    exists in the VPC will be used, in this order: Private, then Isolated, then Public.
    In short: by default ENIs will preferentially be placed in subnets not connected to
    the Internet.
* `availabilityZones`/`onePerAz`: finally, some availability-zone based filtering may be done.
  This filtering by availability zones will only be possible if the VPC has been created or
  looked up in a non-environment agnostic stack (so account and region have been set and
  availability zones have been looked up).
  * `availabilityZones`: only the specific subnets from the selected subnet groups that are
    in the given availability zones will be returned.
  * `onePerAz`: per availability zone, a maximum of one subnet will be returned (Useful for resource
    types that do not allow creating two ENIs in the same availability zone).
* `subnetFilters`: additional filtering on subnets using any number of user-provided filters which
  extend `SubnetFilter`.  The following methods on the `SubnetFilter` class can be used to create
  a filter:
  * `byIds`: chooses subnets from a list of ids
  * `availabilityZones`: chooses subnets in the provided list of availability zones
  * `onePerAz`: chooses at most one subnet per availability zone
  * `containsIpAddresses`: chooses a subnet which contains *any* of the listed ip addresses
  * `byCidrMask`: chooses subnets that have the provided CIDR netmask
  * `byCidrRanges`: chooses subnets which are inside any of the specified CIDR ranges

### Using NAT instances

By default, the `Vpc` construct will create NAT *gateways* for you, which
are managed by AWS. If you would prefer to use your own managed NAT
*instances* instead, specify a different value for the `natGatewayProvider`
property, as follows:

The construct will automatically selects the latest version of Amazon Linux 2023.
If you prefer to use a custom AMI, use `machineImage:
MachineImage.genericLinux({ ... })` and configure the right AMI ID for the
regions you want to deploy to.

> **Warning**
> The NAT instances created using this method will be **unmonitored**.
> They are not part of an Auto Scaling Group,
> and if they become unavailable or are terminated for any reason,
> will not be restarted or replaced.

By default, the NAT instances will route all traffic. To control what traffic
gets routed, pass a custom value for `defaultAllowedTraffic` and access the
`NatInstanceProvider.connections` member after having passed the NAT provider to
the VPC:

```ts
declare const instanceType: ec2.InstanceType;

const provider = ec2.NatProvider.instanceV2({
  instanceType,
  defaultAllowedTraffic: ec2.NatTrafficDirection.OUTBOUND_ONLY,
});
new ec2.Vpc(this, 'TheVPC', {
  natGatewayProvider: provider,
});
provider.connections.allowFrom(ec2.Peer.ipv4('1.2.3.4/8'), ec2.Port.HTTP);
```

You can also customize the characteristics of your NAT instances, including their security group,
as well as their initialization scripts:

```ts
declare const bucket: s3.Bucket;

const userData = ec2.UserData.forLinux();
userData.addCommands(
  ...ec2.NatInstanceProviderV2.DEFAULT_USER_DATA_COMMANDS,
  'echo "hello world!" > hello.txt',
  `aws s3 cp hello.txt s3://${bucket.bucketName}`,
);

const provider = ec2.NatProvider.instanceV2({
  instanceType: new ec2.InstanceType('t3.small'),
  creditSpecification: ec2.CpuCredits.UNLIMITED,
  defaultAllowedTraffic: ec2.NatTrafficDirection.NONE,
});

const vpc = new ec2.Vpc(this, 'TheVPC', {
  natGatewayProvider: provider,
  natGateways: 2,
});

const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });
    securityGroup.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443));
for (const gateway of provider.gatewayInstances) {
  bucket.grants.write(gateway);
  gateway.addSecurityGroup(securityGroup);
}
```

[using NAT instances](test/integ.nat-instances.lit.ts) [Deprecated]

The V1 `NatProvider.instance` construct will use the AWS official NAT instance AMI, which has already
reached EOL on Dec 31, 2023. For more information, see the following blog post:
[Amazon Linux AMI end of life](https://aws.amazon.com/blogs/aws/update-on-amazon-linux-ami-end-of-life/).

```ts
declare const instanceType: ec2.InstanceType;

const provider = ec2.NatProvider.instance({
  instanceType,
  defaultAllowedTraffic: ec2.NatTrafficDirection.OUTBOUND_ONLY,
});
new ec2.Vpc(this, 'TheVPC', {
  natGatewayProvider: provider,
});
provider.connections.allowFrom(ec2.Peer.ipv4('1.2.3.4/8'), ec2.Port.HTTP);
```

### Associate Public IP Address to NAT Instance

You can choose to associate public IP address to a NAT instance V2 by specifying `associatePublicIpAddress`
like the following:

```ts
const natGatewayProvider = ec2.NatProvider.instanceV2({
  instanceType: new ec2.InstanceType('t3.small'),
  associatePublicIpAddress: true,
});
```

In certain scenarios where the public subnet has set `mapPublicIpOnLaunch` to `false`, NAT instances does not
get public IP addresses assigned which would result in non-working NAT instance as NAT instance requires a public
IP address to enable outbound internet connectivity. Users can specify `associatePublicIpAddress` to `true` to
solve this problem.

### Ip Address Management

The VPC spans a supernet IP range, which contains the non-overlapping IPs of its contained subnets. Possible sources for this IP range are:

* You specify an IP range directly by specifying a CIDR
* You allocate an IP range of a given size automatically from AWS IPAM

By default the Vpc will allocate the `10.0.0.0/16` address range which will be exhaustively spread across all subnets in the subnet configuration. This behavior can be changed by passing an object that implements `IIpAddresses` to the `ipAddress` property of a Vpc. See the subsequent sections for the options.

Be aware that if you don't explicitly reserve subnet groups in `subnetConfiguration`, the address space will be fully allocated! If you predict you may need to add more subnet groups later, add them early on and set `reserved: true` (see the "Advanced Subnet Configuration" section for more information).

#### Specifying a CIDR directly

Use `IpAddresses.cidr` to define a Cidr range for your Vpc directly in code:

```ts
import { IpAddresses } from 'aws-cdk-lib/aws-ec2';

new ec2.Vpc(this, 'TheVPC', {
  ipAddresses: IpAddresses.cidr('10.0.1.0/20')
});
```

Space will be allocated to subnets in the following order:

* First, spaces is allocated for all subnets groups that explicitly have a `cidrMask` set as part of their configuration (including reserved subnets).
* Afterwards, any remaining space is divided evenly between the rest of the subnets (if any).

The argument to `IpAddresses.cidr` may not be a token, and concrete Cidr values are generated in the synthesized CloudFormation template.

#### Allocating an IP range from AWS IPAM

Amazon VPC IP Address Manager (IPAM) manages a large IP space, from which chunks can be allocated for use in the Vpc. For information on Amazon VPC IP Address Manager please see the [official documentation](https://docs.aws.amazon.com/vpc/latest/ipam/what-it-is-ipam.html). An example of allocating from AWS IPAM looks like this:

```ts
import { IpAddresses } from 'aws-cdk-lib/aws-ec2';

declare const pool: ec2.CfnIPAMPool;

new ec2.Vpc(this, 'TheVPC', {
  ipAddresses: IpAddresses.awsIpamAllocation({
    ipv4IpamPoolId: pool.ref,
    ipv4NetmaskLength: 18,
    defaultSubnetIpv4NetmaskLength: 24
  })
});
```

`IpAddresses.awsIpamAllocation` requires the following:

* `ipv4IpamPoolId`, the id of an IPAM Pool from which the VPC range should be allocated.
* `ipv4NetmaskLength`, the size of the IP range that will be requested from the Pool at deploy time.
* `defaultSubnetIpv4NetmaskLength`, the size of subnets in groups that don't have `cidrMask` set.

With this method of IP address management, no attempt is made to guess at subnet group sizes or to exhaustively allocate the IP range. All subnet groups must have an explicit `cidrMask` set as part of their subnet configuration, or `defaultSubnetIpv4NetmaskLength` must be set for a default size. If not, synthesis will fail and you must provide one or the other.

### Dual Stack configuration

To allocate both IPv4 and IPv6 addresses in your VPC, you can configure your VPC to have a dual stack protocol.

```ts
new ec2.Vpc(this, 'DualStackVpc', {
  ipProtocol: ec2.IpProtocol.DUAL_STACK,
})
```

By default, a dual stack VPC will create an Amazon provided IPv6 /56 CIDR block associated to the VPC. It will then assign /64 portions of the block to each subnet. For each subnet, auto-assigning an IPv6 address will be enabled, and auto-asigning a public IPv4 address will be disabled. An egress only internet gateway will be created for `PRIVATE_WITH_EGRESS` subnets, and IPv6 routes will be added for IGWs and EIGWs.

Disabling the auto-assigning of a public IPv4 address by default can avoid the cost of public IPv4 addresses starting 2/1/2024. For use cases that need an IPv4 address, the `mapPublicIpOnLaunch` property in `subnetConfiguration` can be set to auto-assign the IPv4 address. Note that private IPv4 address allocation will not be changed.

See [Advanced Subnet Configuration](#advanced-subnet-configuration) for all IPv6 specific properties.

### Reserving availability zones

There are situations where the IP space for availability zones will
need to be reserved. This is useful in situations where availability
zones would need to be added after the vpc is originally deployed,
without causing IP renumbering for availability zones subnets. The IP
space for reserving `n` availability zones can be done by setting the
`reservedAzs` to `n` in vpc props, as shown below:

```ts
const vpc = new ec2.Vpc(this, 'TheVPC', {
  cidr: '10.0.0.0/21',
  maxAzs: 3,
  reservedAzs: 1,
});
```

In the example above, the subnets for reserved availability zones is not
actually provisioned but its IP space is still reserved. If, in the future,
new availability zones needs to be provisioned, then we would decrement
the value of `reservedAzs` and increment the `maxAzs` or `availabilityZones`
accordingly. This action would not cause the IP address of subnets to get
renumbered, but rather the IP space that was previously reserved will be
used for the new availability zones subnets.

### Advanced Subnet Configuration

If the default VPC configuration (public and private subnets spanning the
size of the VPC) don't suffice for you, you can configure what subnets to
create by specifying the `subnetConfiguration` property. It allows you
to configure the number and size of all subnets. Specifying an advanced
subnet configuration could look like this:

```ts
const vpc = new ec2.Vpc(this, 'TheVPC', {
  // 'IpAddresses' configures the IP range and size of the entire VPC.
  // The IP space will be divided based on configuration for the subnets.
  ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/21'),

  // 'maxAzs' configures the maximum number of availability zones to use.
  // If you want to specify the exact availability zones you want the VPC
  // to use, use `availabilityZones` instead.
  maxAzs: 3,

  // 'subnetConfiguration' specifies the "subnet groups" to create.
  // Every subnet group will have a subnet for each AZ, so this
  // configuration will create `3 groups × 3 AZs = 9` subnets.
  subnetConfiguration: [
    {
      // 'subnetType' controls Internet access, as described above.
      subnetType: ec2.SubnetType.PUBLIC,

      // 'name' is used to name this particular subnet group. You will have to
      // use the name for subnet selection if you have more than one subnet
      // group of the same type.
      name: 'Ingress',

      // 'cidrMask' specifies the IP addresses in the range of individual
      // subnets in the group. Each of the subnets in this group will contain
      // `2^(32 address bits - 24 subnet bits) - 2 reserved addresses = 254`
      // usable IP addresses.
      //
      // If 'cidrMask' is left out the available address space is evenly
      // divided across the remaining subnet groups.
      cidrMask: 24,
    },
    {
      cidrMask: 24,
      name: 'Application',
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    },
    {
      cidrMask: 28,
      name: 'Database',
      subnetType: ec2.SubnetType.PRIVATE_ISOLATED,

      // 'reserved' can be used to reserve IP address space. No resources will
      // be created for this subnet, but the IP range will be kept available for
      // future creation of this subnet, or even for future subdivision.
      reserved: true
    }
  ],
});
```

The example above is one possible configuration, but the user can use the
constructs above to implement many other network configurations.

The `Vpc` from the above configuration in a Region with three
availability zones will be the following:

Subnet Name       |Type      |IP Block      |AZ|Features
------------------|----------|--------------|--|--------
IngressSubnet1    |`PUBLIC`  |`10.0.0.0/24` |#1|NAT Gateway
IngressSubnet2    |`PUBLIC`  |`10.0.1.0/24` |#2|NAT Gateway
IngressSubnet3    |`PUBLIC`  |`10.0.2.0/24` |#3|NAT Gateway
ApplicationSubnet1|`PRIVATE` |`10.0.3.0/24` |#1|Route to NAT in IngressSubnet1
ApplicationSubnet2|`PRIVATE` |`10.0.4.0/24` |#2|Route to NAT in IngressSubnet2
ApplicationSubnet3|`PRIVATE` |`10.0.5.0/24` |#3|Route to NAT in IngressSubnet3
DatabaseSubnet1   |`ISOLATED`|`10.0.6.0/28` |#1|Only routes within the VPC
DatabaseSubnet2   |`ISOLATED`|`10.0.6.16/28`|#2|Only routes within the VPC
DatabaseSubnet3   |`ISOLATED`|`10.0.6.32/28`|#3|Only routes within the VPC

#### Dual Stack Configurations

Here is a break down of IPv4 and IPv6 specific `subnetConfiguration` properties in a dual stack VPC:

```ts
const vpc = new ec2.Vpc(this, 'TheVPC', {
  ipProtocol: ec2.IpProtocol.DUAL_STACK,

  subnetConfiguration: [
    {
      // general properties
      name: 'Public',
      subnetType: ec2.SubnetType.PUBLIC,
      reserved: false,

      // IPv4 specific properties
      mapPublicIpOnLaunch: true,
      cidrMask: 24,

      // new IPv6 specific property
      ipv6AssignAddressOnCreation: true,
    },
  ],
});
```

The property `mapPublicIpOnLaunch` controls if a public IPv4 address will be assigned. This defaults to `false` for dual stack VPCs to avoid inadvertant costs of having the public address. However, a public IP must be enabled (or otherwise configured with BYOIP or IPAM) in order for services that rely on the address to function.

The `ipv6AssignAddressOnCreation` property controls the same behavior for the IPv6 address. It defaults to true.

Using IPv6 specific properties in an IPv4 only VPC will result in errors.

### Accessing the Internet Gateway

If you need access to the internet gateway, you can get its ID like so:

```ts
declare const vpc: ec2.Vpc;

const igwId = vpc.internetGatewayId;
```

For a VPC with only `ISOLATED` subnets, this value will be undefined.

This is only supported for VPCs created in the stack - currently you're
unable to get the ID for imported VPCs. To do that you'd have to specifically
look up the Internet Gateway by name, which would require knowing the name
beforehand.

This can be useful for configuring routing using a combination of gateways:
for more information see [Routing](#routing) below.

### Disabling the creation of the default internet gateway

If you need to control the creation of the internet gateway explicitly,
you can disable the creation of the default one using the `createInternetGateway`
property:

```ts
const vpc = new ec2.Vpc(this, "VPC", {
  createInternetGateway: false,
  subnetConfiguration: [{
      subnetType: ec2.SubnetType.PUBLIC,
      name: 'Public',
    }]
});
```

#### Routing

It's possible to add routes to any subnets using the `addRoute()` method. If for
example you want an isolated subnet to have a static route via the default
Internet Gateway created for the public subnet - perhaps for routing a VPN
connection - you can do so like this:

```ts
const vpc = new ec2.Vpc(this, "VPC", {
  subnetConfiguration: [{
      subnetType: ec2.SubnetType.PUBLIC,
      name: 'Public',
    },{
      subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      name: 'Isolated',
    }]
});

(vpc.isolatedSubnets[0] as ec2.Subnet).addRoute("StaticRoute", {
    routerId: vpc.internetGatewayId!,
    routerType: ec2.RouterType.GATEWAY,
    destinationCidrBlock: "8.8.8.8/32",
})
```

*Note that we cast to `Subnet` here because the list of subnets only returns an
`ISubnet`.*

### Reserving subnet IP space

There are situations where the IP space for a subnet or number of subnets
will need to be reserved. This is useful in situations where subnets would
need to be added after the vpc is originally deployed, without causing IP
renumbering for existing subnets. The IP space for a subnet may be reserved
by setting the `reserved` subnetConfiguration property to true, as shown
below:

```ts
const vpc = new ec2.Vpc(this, 'TheVPC', {
  natGateways: 1,
  subnetConfiguration: [
    {
      cidrMask: 26,
      name: 'Public',
      subnetType: ec2.SubnetType.PUBLIC,
    },
    {
      cidrMask: 26,
      name: 'Application1',
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    },
    {
      cidrMask: 26,
      name: 'Application2',
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      reserved: true,   // <---- This subnet group is reserved
    },
    {
      cidrMask: 27,
      name: 'Database',
      subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
    }
  ],
});
```

In the example above, the subnet for Application2 is not actually provisioned
but its IP space is still reserved. If in the future this subnet needs to be
provisioned, then the `reserved: true` property should be removed. Reserving
parts of the IP space prevents the other subnets from getting renumbered.

### Sharing VPCs between stacks

If you are creating multiple `Stack`s inside the same CDK application, you
can reuse a VPC defined in one Stack in another by simply passing the VPC
instance around:

[sharing VPCs between stacks](test/integ.share-vpcs.lit.ts)

> Note: If you encounter an error like "Delete canceled. Cannot delete export ..."
> when using a cross-stack reference to a VPC, it's likely due to CloudFormation
> export/import constraints. In such cases, it's safer to use Vpc.fromLookup()
> in the consuming stack instead of directly referencing the VPC object, more details
> is provided in [Importing an existing VPC](#importing-an-existing-vpc). This
> avoids creating CloudFormation exports and gives more flexibility, especially
> when stacks need to be deleted or updated independently.

### Importing an existing VPC

If your VPC is created outside your CDK app, you can use `Vpc.fromLookup()`.
The CDK CLI will search for the specified VPC in the stack's region and
account, and import the subnet configuration. Looking up can be done by VPC
ID, but more flexibly by searching for a specific tag on the VPC.

Subnet types will be determined from the `aws-cdk:subnet-type` tag on the
subnet if it exists, or the presence of a route to an Internet Gateway
otherwise. Subnet names will be determined from the `aws-cdk:subnet-name` tag
on the subnet if it exists, or will mirror the subnet type otherwise (i.e.
a public subnet will have the name `"Public"`).

The result of the `Vpc.fromLookup()` operation will be written to a file
called `cdk.context.json`. You must commit this file to source control so
that the lookup values are available in non-privileged environments such
as CI build steps, and to ensure your template builds are repeatable.

Here's how `Vpc.fromLookup()` can be used:

[importing existing VPCs](test/integ.import-default-vpc.lit.ts)

`Vpc.fromLookup` is the recommended way to import VPCs. If for whatever
reason you do not want to use the context mechanism to look up a VPC at
synthesis time, you can also use `Vpc.fromVpcAttributes`. This has the
following limitations:

* Every subnet group in the VPC must have a subnet in each availability zone
  (for example, each AZ must have both a public and private subnet). Asymmetric
  VPCs are not supported.
* All VpcId, SubnetId, RouteTableId, ... parameters must either be known at
  synthesis time, or they must come from deploy-time list parameters whose
  deploy-time lengths are known at synthesis time.

Using `Vpc.fromVpcAttributes()` looks like this:

```ts
const vpc = ec2.Vpc.fromVpcAttributes(this, 'VPC', {
  vpcId: 'vpc-1234',
  availabilityZones: ['us-east-1a', 'us-east-1b'],

  // Either pass literals for all IDs
  publicSubnetIds: ['s-12345', 's-67890'],

  // OR: import a list of known length
  privateSubnetIds: Fn.importListValue('PrivateSubnetIds', 2),

  // OR: split an imported string to a list of known length
  isolatedSubnetIds: Fn.split(',', ssm.StringParameter.valueForStringParameter(this, `MyParameter`), 2),
});
```

For each subnet group the import function accepts optional parameters for subnet
names, route table ids and IPv4 CIDR blocks. When supplied, the length of these
lists are required to match the length of the list of subnet ids, allowing the
lists to be zipped together to form `ISubnet` instances.

Public subnet group example (for private or isolated subnet groups, use the properties with the respective prefix):

```ts
const vpc = ec2.Vpc.fromVpcAttributes(this, 'VPC', {
  vpcId: 'vpc-1234',
  availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
  publicSubnetIds: ['s-12345', 's-34567', 's-56789'],
  publicSubnetNames: ['Subnet A', 'Subnet B', 'Subnet C'],
  publicSubnetRouteTableIds: ['rt-12345', 'rt-34567', 'rt-56789'],
  publicSubnetIpv4CidrBlocks: ['10.0.0.0/24', '10.0.1.0/24', '10.0.2.0/24'],
});
```

The above example will create an `IVpc` instance with three public subnets:

| Subnet id | Availability zone | Subnet name | Route table id | IPv4 CIDR   |
| --------- | ----------------- | ----------- | -------------- | ----------- |
| s-12345   | us-east-1a        | Subnet A    | rt-12345       | 10.0.0.0/24 |
| s-34567   | us-east-1b        | Subnet B    | rt-34567       | 10.0.1.0/24 |
| s-56789   | us-east-1c        | Subnet B    | rt-56789       | 10.0.2.0/24 |

### Restricting access to the VPC default security group

AWS Security best practices recommend that the [VPC default security group should
not allow inbound and outbound
traffic](https://docs.aws.amazon.com/securityhub/latest/userguide/ec2-controls.html#ec2-2).
When the `@aws-cdk/aws-ec2:restrictDefaultSecurityGroup` feature flag is set to
`true` (default for new projects) this will be enabled by default. If you do not
have this feature flag set you can either set the feature flag _or_ you can set
the `restrictDefaultSecurityGroup` property to `true`.

```ts
new ec2.Vpc(this, 'VPC', {
  restrictDefaultSecurityGroup: true,
});
```

If you set this property to `true` and then later remove it or set it to `false`
the default ingress/egress will be restored on the default security group.

## Allowing Connections

In AWS, all network traffic in and out of **Elastic Network Interfaces** (ENIs)
is controlled by **Security Groups**. You can think of Security Groups as a
firewall with a set of rules. By default, Security Groups allow no incoming
(ingress) traffic and all outgoing (egress) traffic. You can add ingress rules
to them to allow incoming traffic streams. To exert fine-grained control over
egress traffic, set `allowAllOutbound: false` on the `SecurityGroup`, after
which you can add egress traffic rules.

You can manipulate Security Groups directly:

```ts fixture=with-vpc
const mySecurityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
  description: 'Allow ssh access to ec2 instances',
  allowAllOutbound: true   // Can be set to false
});
mySecurityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22), 'allow ssh access from the world');
```

All constructs that create ENIs on your behalf (typically constructs that create
EC2 instances or other VPC-connected resources) will all have security groups
automatically assigned. Those constructs have an attribute called
**connections**, which is an object that makes it convenient to update the
security groups. If you want to allow connections between two constructs that
have security groups, you have to add an **Egress** rule to one Security Group,
and an **Ingress** rule to the other. The connections object will automatically
take care of this for you:

```ts
declare const loadBalancer: elbv2.ApplicationLoadBalancer;
declare const appFleet: autoscaling.AutoScalingGroup;
declare const dbFleet: autoscaling.AutoScalingGroup;

// Allow connections from anywhere
loadBalancer.connections.allowFromAnyIpv4(ec2.Port.HTTPS, 'Allow inbound HTTPS');

// The same, but an explicit IP address
loadBalancer.connections.allowFrom(ec2.Peer.ipv4('1.2.3.4/32'), ec2.Port.HTTPS, 'Allow inbound HTTPS');

// Allow connection between AutoScalingGroups
appFleet.connections.allowTo(dbFleet, ec2.Port.HTTPS, 'App can call database');
```

### Connection Peers

There are various classes that implement the connection peer part:

```ts
declare const appFleet: autoscaling.AutoScalingGroup;
declare const dbFleet: autoscaling.AutoScalingGroup;

// Simple connection peers
let peer = ec2.Peer.ipv4('10.0.0.0/16');
peer = ec2.Peer.anyIpv4();
peer = ec2.Peer.ipv6('::0/0');
peer = ec2.Peer.anyIpv6();
appFleet.connections.allowTo(peer, ec2.Port.HTTPS, 'Allow outbound HTTPS');
```

Any object that has a security group can itself be used as a connection peer:

```ts
declare const fleet1: autoscaling.AutoScalingGroup;
declare const fleet2: autoscaling.AutoScalingGroup;
declare const appFleet: autoscaling.AutoScalingGroup;

// These automatically create appropriate ingress and egress rules in both security groups
fleet1.connections.allowTo(fleet2, ec2.Port.HTTP, 'Allow between fleets');

appFleet.connections.allowFromAnyIpv4(ec2.Port.HTTP, 'Allow from load balancer');
```

A managed prefix list is also a connection peer:

``` ts
declare const appFleet: autoscaling.AutoScalingGroup;

const prefixList = new ec2.PrefixList(this, 'PrefixList', { maxEntries: 10 });
appFleet.connections.allowFrom(prefixList, ec2.Port.HTTPS);
```

#### Rule Configuration Interfaces

The `IPeer` interface provides type-safe methods for generating security group rule configurations.
The `toIngressRuleConfig()` and `toEgressRuleConfig()` methods return strongly-typed interfaces
instead of `any`, enabling better IDE autocompletion and compile-time type checking:

- `IngressRuleConfig`: Configuration for ingress rules with properties like `cidrIp`, `cidrIpv6`,
  `sourcePrefixListId`, `sourceSecurityGroupId`, and `sourceSecurityGroupOwnerId`
- `EgressRuleConfig`: Configuration for egress rules with properties like `cidrIp`, `cidrIpv6`,
  `destinationPrefixListId`, and `destinationSecurityGroupId`

### Port Ranges

The connections that are allowed are specified by port ranges. A number of classes provide
the connection specifier:

```ts
ec2.Port.tcp(80)
ec2.Port.HTTPS
ec2.Port.tcpRange(60000, 65535)
ec2.Port.allTcp()
ec2.Port.allIcmp()
ec2.Port.allIcmpV6()
ec2.Port.allTraffic()
```

> NOTE: Not all protocols have corresponding helper methods. In the absence of a helper method,
> you can instantiate `Port` yourself with your own settings. You are also welcome to contribute
> new helper methods.

### Default Ports

Some Constructs have default ports associated with them. For example, the
listener of a load balancer does (it's the public port), or instances of an
RDS database (it's the port the database is accepting connections on).

If the object you're calling the peering method on has a default port associated with it, you can call
`allowDefaultPortFrom()` and omit the port specifier. If the argument has an associated default port, call
`allowDefaultPortTo()`.

For example:

```ts
declare const listener: elbv2.ApplicationListener;
declare const appFleet: autoscaling.AutoScalingGroup;
declare const rdsDatabase: rds.DatabaseCluster;

// Port implicit in listener
listener.connections.allowDefaultPortFromAnyIpv4('Allow public');

// Port implicit in peer
appFleet.connections.allowDefaultPortTo(rdsDatabase, 'Fleet can access database');
```

### Security group rules

By default, security group wills be added inline to the security group in the output cloud formation
template, if applicable.  This includes any static rules by ip address and port range.  This
optimization helps to minimize the size of the template.

In some environments this is not desirable, for example if your security group access is controlled
via tags. You can disable inline rules per security group or globally via the context key
`@aws-cdk/aws-ec2.securityGroupDisableInlineRules`.

```ts fixture=with-vpc
const mySecurityGroupWithoutInlineRules = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
  description: 'Allow ssh access to ec2 instances',
  allowAllOutbound: true,
  disableInlineRules: true
});
//This will add the rule as an external cloud formation construct
mySecurityGroupWithoutInlineRules.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.SSH, 'allow ssh access from the world');
```

### Importing an existing security group

If you know the ID and the configuration of the security group to import, you can use `SecurityGroup.fromSecurityGroupId`:

```ts
const sg = ec2.SecurityGroup.fromSecurityGroupId(this, 'SecurityGroupImport', 'sg-1234', {
  allowAllOutbound: true,
});
```

Alternatively, use lookup methods to import security groups if you do not know the ID or the configuration details. Method `SecurityGroup.fromLookupByName` looks up a security group if the security group ID is unknown.

```ts fixture=with-vpc
const sg = ec2.SecurityGroup.fromLookupByName(this, 'SecurityGroupLookup', 'security-group-name', vpc);
```

If the security group ID is known and configuration details are unknown, use method `SecurityGroup.fromLookupById` instead. This method will lookup property `allowAllOutbound` from the current configuration of the security group.

```ts
const sg = ec2.SecurityGroup.fromLookupById(this, 'SecurityGroupLookup', 'sg-1234');
```

The result of `SecurityGroup.fromLookupByName` and `SecurityGroup.fromLookupById` operations will be
written to a file called `cdk.context.json`.
You must commit this file to source control so that the lookup values are available in non-privileged
environments such as CI build steps, and to ensure your template builds are repeatable.

### Cross Stack Connections

If you are attempting to add a connection from a peer in one stack to a peer in a different stack, sometimes it is necessary to ensure that you are making the connection in
a specific stack in order to avoid a cyclic reference. If there are no other dependencies between stacks then it will not matter in which stack you make
the connection, but if there are existing dependencies (i.e. stack1 already depends on stack2), then it is important to make the connection in the dependent stack (i.e. stack1).

Whenever you make a `connections` function call, the ingress and egress security group rules will be added to the stack that the calling object exists in.
So if you are doing something like `peer1.connections.allowFrom(peer2)`, then the security group rules (both ingress and egress) will be created in `peer1`'s Stack.

As an example, if we wanted to allow a connection from a security group in one stack (egress) to a security group in a different stack (ingress),
we would make the connection like:

**If Stack1 depends on Stack2**

```ts fixture=with-vpc
// Stack 1
declare const stack1: Stack;
declare const stack2: Stack;

const sg1 = new ec2.SecurityGroup(stack1, 'SG1', {
  allowAllOutbound: false, // if this is `true` then no egress rule will be created
  vpc,
});

// Stack 2
const sg2 = new ec2.SecurityGroup(stack2, 'SG2', {
  allowAllOutbound: false, // if this is `true` then no egress rule will be created
  vpc,
});


// `connections.allowTo` on `sg1` since we want the
// rules to be created in Stack1
sg1.connections.allowTo(sg2, ec2.Port.tcp(3333));
```

In this case both the Ingress Rule for `sg2` and the Egress Rule for `sg1` will both be created
in `Stack 1` which avoids the cyclic reference.


**If Stack2 depends on Stack1**

```ts fixture=with-vpc
// Stack 1
declare const stack1: Stack;
declare const stack2: Stack;

const sg1 = new ec2.SecurityGroup(stack1, 'SG1', {
  allowAllOutbound: false, // if this is `true` then no egress rule will be created
  vpc,
});

// Stack 2
const sg2 = new ec2.SecurityGroup(stack2, 'SG2', {
  allowAllOutbound: false, // if this is `true` then no egress rule will be created
  vpc,
});


// `connections.allowFrom` on `sg2` since we want the
// rules to be created in Stack2
sg2.connections.allowFrom(sg1, ec2.Port.tcp(3333));
```

In this case both the Ingress Rule for `sg2` and the Egress Rule for `sg1` will both be created
in `Stack 2` which avoids the cyclic reference.

## Machine Images (AMIs)

AMIs control the OS that gets launched when you start your EC2 instance. The EC2
library contains constructs to select the AMI you want to use.

Depending on the type of AMI, you select it a different way. Here are some
examples of images you might want to use:

[example of creating images](test/example.images.lit.ts)

> NOTE: The AMIs selected by `MachineImage.lookup()` will be cached in
> `cdk.context.json`, so that your AutoScalingGroup instances aren't replaced while
> you are making unrelated changes to your CDK app.
>
> To query for the latest AMI again, remove the relevant cache entry from
> `cdk.context.json`, or use the `cdk context` command. For more information, see
> [Runtime Context](https://docs.aws.amazon.com/cdk/latest/guide/context.html) in the CDK
> developer guide.
>
> To customize the cache key, use the `additionalCacheKey` parameter.
> This allows you to have multiple lookups with the same parameters
> cache their values separately. This can be useful if you want to
> scope the context variable to a construct (ie, using `additionalCacheKey: this.node.path`),
> so that if the value in the cache needs to be updated, it does not need to be updated
> for all constructs at the same time.
>
> `MachineImage.genericLinux()`, `MachineImage.genericWindows()` will use `CfnMapping` in
> an agnostic stack.

## Special VPC configurations

### VPN connections to a VPC

Create your VPC with VPN connections by specifying the `vpnConnections` props (keys are construct `id`s):

```ts
import { SecretValue } from 'aws-cdk-lib/core';

const vpc = new ec2.Vpc(this, 'MyVpc', {
  vpnConnections: {
    dynamic: { // Dynamic routing (BGP)
      ip: '1.2.3.4',
      tunnelOptions: [
        {
          preSharedKeySecret: SecretValue.unsafePlainText('secretkey1234'),
        },
        {
          preSharedKeySecret: SecretValue.unsafePlainText('secretkey5678'),
        },
      ],
    },
    static: { // Static routing
      ip: '4.5.6.7',
      staticRoutes: [
        '192.168.10.0/24',
        '192.168.20.0/24'
      ]
    }
  }
});
```

To create a VPC that can accept VPN connections, set `vpnGateway` to `true`:

```ts
const vpc = new ec2.Vpc(this, 'MyVpc', {
  vpnGateway: true
});
```

VPN connections can then be added:

```ts fixture=with-vpc
vpc.addVpnConnection('Dynamic', {
  ip: '1.2.3.4'
});
```

By default, routes will be propagated on the route tables associated with the private subnets. If no
private subnets exist, isolated subnets are used. If no isolated subnets exist, public subnets are
used. Use the `Vpc` property `vpnRoutePropagation` to customize this behavior.

VPN connections expose [metrics (cloudwatch.Metric)](https://github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-cloudwatch/README.md) across all tunnels in the account/region and per connection:

```ts fixture=with-vpc
// Across all tunnels in the account/region
const allDataOut = ec2.VpnConnection.metricAllTunnelDataOut();

// For a specific vpn connection
const vpnConnection = vpc.addVpnConnection('Dynamic', {
  ip: '1.2.3.4'
});
const state = vpnConnection.metricTunnelState();
```

### VPC endpoints

A VPC endpoint enables you to privately connect your VPC to supported AWS services and VPC endpoint services powered by PrivateLink without requiring an internet gateway, NAT device, VPN connection, or AWS Direct Connect connection. Instances in your VPC do not require public IP addresses to communicate with resources in the service. Traffic between your VPC and the other service does not leave the Amazon network.

Endpoints are virtual devices. They are horizontally scaled, redundant, and highly available VPC components that allow communication between instances in your VPC and services without imposing availability risks or bandwidth constraints on your network traffic.

[example of setting up VPC endpoints](test/integ.vpc-endpoint.lit.ts)

By default, CDK will place a VPC endpoint in one subnet per AZ. If you wish to override the AZs CDK places the VPC endpoint in,
use the `subnets` parameter as follows:

```ts
declare const vpc: ec2.Vpc;

new ec2.InterfaceVpcEndpoint(this, 'VPC Endpoint', {
  vpc,
  service: new ec2.InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
  // Choose which availability zones to place the VPC endpoint in, based on
  // available AZs
  subnets: {
    availabilityZones: ['us-east-1a', 'us-east-1c']
  }
});
```

Per the [AWS documentation](https://aws.amazon.com/premiumsupport/knowledge-center/interface-endpoint-availability-zone/), not all
VPC endpoint services are available in all AZs. If you specify the parameter `lookupSupportedAzs`, CDK attempts to discover which
AZs an endpoint service is available in, and will ensure the VPC endpoint is not placed in a subnet that doesn't match those AZs.
These AZs will be stored in cdk.context.json.

```ts
declare const vpc: ec2.Vpc;

new ec2.InterfaceVpcEndpoint(this, 'VPC Endpoint', {
  vpc,
  service: new ec2.InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
  // Choose which availability zones to place the VPC endpoint in, based on
  // available AZs
  lookupSupportedAzs: true
});
```

Pre-defined AWS services are defined in the [InterfaceVpcEndpointAwsService](lib/vpc-endpoint.ts) class, and can be used to
create VPC endpoints without having to configure name, ports, etc. For example, a Keyspaces endpoint can be created for
use in your VPC:

``` ts
declare const vpc: ec2.Vpc;

new ec2.InterfaceVpcEndpoint(this, 'VPC Endpoint', {
  vpc,
  service: ec2.InterfaceVpcEndpointAwsService.KEYSPACES,
});
```

For cross-region VPC endpoints, specify the `serviceRegion` parameter:

```ts
declare const vpc: ec2.Vpc;

new ec2.InterfaceVpcEndpoint(this, 'CrossRegionEndpoint', {
  vpc,
  service: new ec2.InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-123456', 443),
  serviceRegion: 'us-east-1', // Same region as the service endpoint above
});
```

#### Security groups for interface VPC endpoints

By default, interface VPC endpoints create a new security group and all traffic to the endpoint from within the VPC will be automatically allowed.

Use the `connections` object to allow other traffic to flow to the endpoint:

```ts
declare const myEndpoint: ec2.InterfaceVpcEndpoint;

myEndpoint.connections.allowDefaultPortFromAnyIpv4();
```

Alternatively, existing security groups can be used by specifying the `securityGroups` prop.

#### IPv6 and Dualstack support

As IPv4 addresses are running out, many AWS services are adding support for IPv6 or Dualstack (IPv4 and IPv6 support) for their VPC Endpoints.

IPv6 and Dualstack address types can be configured for both interface and gateway endpoints using `ipAddressType` and `dnsRecordIpType`:

```ts fixture=with-vpc
// Interface endpoint with IPv6
vpc.addInterfaceEndpoint('ExampleEndpoint', {
  service: ec2.InterfaceVpcEndpointAwsService.ECR,
  ipAddressType: ec2.VpcEndpointIpAddressType.IPV6,
  dnsRecordIpType: ec2.VpcEndpointDnsRecordIpType.IPV6,
});

// Gateway endpoint with dualstack
vpc.addGatewayEndpoint('S3DualstackEndpoint', {
  service: ec2.GatewayVpcEndpointAwsService.S3,
  ipAddressType: ec2.VpcEndpointIpAddressType.DUALSTACK,
  dnsRecordIpType: ec2.VpcEndpointDnsRecordIpType.DUALSTACK,
});
```

The possible values for `ipAddressType` are:
* `IPV4` This option is supported only if all selected subnets have IPv4 address ranges and the endpoint service accepts IPv4 requests.
* `IPV6` This option is supported only if all selected subnets are IPv6 only subnets and the endpoint service accepts IPv6 requests.
* `DUALSTACK` Assign both IPv4 and IPv6 addresses to the endpoint network interfaces. This option is supported only if all selected subnets have both IPv4 and IPv6 address ranges and the endpoint service accepts both IPv4 and IPv6 requests.
  The possible values for `dnsRecordIpType` are:
* `IPV4` Create A records for the private, Regional, and zonal DNS names. `ipAddressType` MUST be `IPV4` or `DUALSTACK`
* `IPV6` Create AAAA records for the private, Regional, and zonal DNS names. `ipAddressType` MUST be `IPV6` or `DUALSTACK`
* `DUALSTACK` Create A and AAAA records for the private, Regional, and zonal DNS names. `ipAddressType` MUST be `DUALSTACK`
* `SERVICE_DEFINED` Create A records for the private, Regional, and zonal DNS names and AAAA records for the Regional and zonal DNS names. `ipAddressType` MUST be `DUALSTACK`
  We can only configure dnsRecordIpType when ipAddressType is specified and private DNS must be enabled to use any DNS related features. To avoid complications, it is recommended to always set `privateDnsEnabled` to true (defaults to true) and set the `ipAddressType` and `dnsRecordIpType` explicitly when needing specific IP type behavior. Furthermore, check that the VPC being used supports the IP address type that is being configued.
  More documentation on compatibility and specifications can be found [here](https://docs.aws.amazon.com/vpc/latest/privatelink/create-endpoint-service.html#connect-to-endpoint-service)

### VPC endpoint services

A VPC endpoint service enables you to expose a Network Load Balancer(s) as a provider service to consumers, who connect to your service over a VPC endpoint. You can restrict access to your service via allowed principals (anything that extends ArnPrincipal), and require that new connections be manually accepted. You can also enable Contributor Insight rules.

```ts
declare const networkLoadBalancer1: elbv2.NetworkLoadBalancer;
declare const networkLoadBalancer2: elbv2.NetworkLoadBalancer;

new ec2.VpcEndpointService(this, 'EndpointService', {
  vpcEndpointServiceLoadBalancers: [networkLoadBalancer1, networkLoadBalancer2],
  acceptanceRequired: true,
  allowedPrincipals: [new iam.ArnPrincipal('arn:aws:iam::123456789012:root')],
  contributorInsights: true
});
```

You can also include a service principal in the `allowedPrincipals` property by specifying it as a parameter to the  `ArnPrincipal` constructor.
The resulting VPC endpoint will have an allowlisted principal of type `Service`, instead of `Arn` for that item in the list.
```ts
declare const networkLoadBalancer: elbv2.NetworkLoadBalancer;

new ec2.VpcEndpointService(this, 'EndpointService', {
  vpcEndpointServiceLoadBalancers: [networkLoadBalancer],
  allowedPrincipals: [new iam.ArnPrincipal('ec2.amazonaws.com')],
});
```

You can specify which IP address types (IPv4, IPv6, or both) are supported for your VPC endpoint service:

```ts
declare const networkLoadBalancer: elbv2.NetworkLoadBalancer;

new ec2.VpcEndpointService(this, 'EndpointService', {
  vpcEndpointServiceLoadBalancers: [networkLoadBalancer],
  // Support both IPv4 and IPv6 connections to the endpoint service
  supportedIpAddressTypes: [
    ec2.IpAddressType.IPV4,
    ec2.IpAddressType.IPV6,
  ],
});
```

You can restrict access to your endpoint service to specific AWS regions:

```ts
declare const networkLoadBalancer: elbv2.NetworkLoadBalancer;

new ec2.VpcEndpointService(this, 'EndpointService', {
  vpcEndpointServiceLoadBalancers: [networkLoadBalancer],
  // Allow service consumers from these regions only
  allowedRegions: ['us-east-1', 'eu-west-1'],
});
```

Endpoint services support private DNS, which makes it easier for clients to connect to your service by automatically setting up DNS in their VPC.
You can enable private DNS on an endpoint service like so:

```ts
import { PublicHostedZone, VpcEndpointServiceDomainName } from 'aws-cdk-lib/aws-route53';
declare const zone: PublicHostedZone;
declare const vpces: ec2.VpcEndpointService;

new VpcEndpointServiceDomainName(this, 'EndpointDomain', {
  endpointService: vpces,
  domainName: 'my-stuff.aws-cdk.dev',
  publicHostedZone: zone,
});
```

Note: The domain name must be owned (registered through Route53) by the account the endpoint service is in, or delegated to the account.
The VpcEndpointServiceDomainName will handle the AWS side of domain verification, the process for which can be found
[here](https://docs.aws.amazon.com/vpc/latest/userguide/endpoint-services-dns-validation.html)

### Client VPN endpoint

AWS Client VPN is a managed client-based VPN service that enables you to securely access your AWS
resources and resources in your on-premises network. With Client VPN, you can access your resources
from any location using an OpenVPN-based VPN client.

Use the `addClientVpnEndpoint()` method to add a client VPN endpoint to a VPC:

```ts fixture=client-vpn
vpc.addClientVpnEndpoint('Endpoint', {
  cidr: '10.100.0.0/16',
  serverCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/server-certificate-id',
  // Mutual authentication
  clientCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/client-certificate-id',
  // User-based authentication
  userBasedAuthentication: ec2.ClientVpnUserBasedAuthentication.federated(samlProvider),
});
```

The endpoint must use at least one [authentication method](https://docs.aws.amazon.com/vpn/latest/clientvpn-admin/client-authentication.html):

* Mutual authentication with a client certificate
* User-based authentication (directory or federated)

If user-based authentication is used, the [self-service portal URL](https://docs.aws.amazon.com/vpn/latest/clientvpn-user/self-service-portal.html)
is made available via a CloudFormation output.

By default, a new security group is created, and logging is enabled. Moreover, a rule to
authorize all users to the VPC CIDR is created.

To customize authorization rules, set the `authorizeAllUsersToVpcCidr` prop to `false`
and use `addAuthorizationRule()`:

```ts fixture=client-vpn
const endpoint = vpc.addClientVpnEndpoint('Endpoint', {
  cidr: '10.100.0.0/16',
  serverCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/server-certificate-id',
  userBasedAuthentication: ec2.ClientVpnUserBasedAuthentication.federated(samlProvider),
  authorizeAllUsersToVpcCidr: false,
});

endpoint.addAuthorizationRule('Rule', {
  cidr: '10.0.10.0/32',
  groupId: 'group-id',
});
```

Use `addRoute()` to configure network routes:

```ts fixture=client-vpn
const endpoint = vpc.addClientVpnEndpoint('Endpoint', {
  cidr: '10.100.0.0/16',
  serverCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/server-certificate-id',
  userBasedAuthentication: ec2.ClientVpnUserBasedAuthentication.federated(samlProvider),
});

// Client-to-client access
endpoint.addRoute('Route', {
  cidr: '10.100.0.0/16',
  target: ec2.ClientVpnRouteTarget.local(),
});
```

Use the `connections` object of the endpoint to allow traffic to other security groups.

To enable [client route enforcement](https://docs.aws.amazon.com/vpn/latest/clientvpn-admin/cvpn-working-cre.html), configure the `clientRouteEnforcementOptions.enforced` prop to `true`:

```ts fixture=client-vpn
const endpoint = vpc.addClientVpnEndpoint('Endpoint', {
  cidr: '10.100.0.0/16',
  serverCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/server-certificate-id',
  clientCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/client-certificate-id',
  clientRouteEnforcementOptions: {
    enforced: true,
  },
});
```

To control whether clients are automatically disconnected when the maximum session duration is reached, use the `disconnectOnSessionTimeout` prop.
By default (`true`), clients are disconnected and must manually reconnect.
Set to `false` to allow automatic reconnection attempts:

```ts fixture=client-vpn
const endpoint = vpc.addClientVpnEndpoint('Endpoint', {
  cidr: '10.100.0.0/16',
  serverCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/server-certificate-id',
  clientCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/client-certificate-id',
  disconnectOnSessionTimeout: false, // Allow automatic reconnection attempts
});
```

Detail information about maximum VPN session duration timeout can be found in the [AWS documentation](https://docs.aws.amazon.com/vpn/latest/clientvpn-admin/cvpn-working-max-duration.html).

## Instances

You can use the `Instance` class to start up a single EC2 instance. For production setups, we recommend
you use an `AutoScalingGroup` from the `aws-autoscaling` module instead, as AutoScalingGroups will take
care of restarting your instance if it ever fails.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;

// Amazon Linux 2
new ec2.Instance(this, 'Instance2', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
});

// Amazon Linux 2 with kernel 5.x
new ec2.Instance(this, 'Instance3', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2({
    kernel: ec2.AmazonLinux2Kernel.KERNEL_5_10,
  }),
});

// Amazon Linux 2023
new ec2.Instance(this, 'Instance4', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
});

// Graviton 3 Processor
new ec2.Instance(this, 'Instance5', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2023({
    cpuType: ec2.AmazonLinuxCpuType.ARM_64,
  }),
});
```

### Latest Amazon Linux Images

Rather than specifying a specific AMI ID to use, it is possible to specify a SSM
Parameter that contains the AMI ID. AWS publishes a set of [public parameters](https://docs.aws.amazon.com/systems-manager/latest/userguide/parameter-store-public-parameters-ami.html)
that contain the latest Amazon Linux AMIs. To make it easier to query a
particular image parameter, the CDK provides a couple of constructs `AmazonLinux2ImageSsmParameter`,
`AmazonLinux2022ImageSsmParameter`, & `AmazonLinux2023SsmParameter`. For example
to use the latest `al2023` image:

```ts
declare const vpc: ec2.Vpc;

new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
});
```

> **Warning**
> Since this retrieves the value from an SSM parameter at deployment time, the
> value will be resolved each time the stack is deployed. This means that if
> the parameter contains a different value on your next deployment, the instance
> will be replaced.

It is also possible to perform the lookup once at synthesis time and then cache
the value in CDK context. This way the value will not change on future
deployments unless you manually refresh the context.

```ts
declare const vpc: ec2.Vpc;

new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2023({
    cachedInContext: true, // default is false
  }),
});

// or
new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  // context cache is turned on by default
  machineImage: new ec2.AmazonLinux2023ImageSsmParameter(),
});

// or
new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2023({
    cachedInContext: true,
    // creates a distinct context variable for this image, instead of resolving to the same
    // value anywhere this lookup is done in your app
    additionalCacheKey: this.node.path,
  }),
});

// or
new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  // context cache is turned on by default
  machineImage: new ec2.AmazonLinux2023ImageSsmParameter({
    // creates a distinct context variable for this image, instead of resolving to the same
    // value anywhere this lookup is done in your app
    additionalCacheKey: this.node.path,
  }),
});

```

#### Kernel Versions

Each Amazon Linux AMI uses a specific kernel version. Most Amazon Linux
generations come with an AMI using the "default" kernel and then 1 or more
AMIs using a specific kernel version, which may or may not be different from the
default kernel version.

For example, Amazon Linux 2 has two different AMIs available from the SSM
parameters.

- `/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-ebs`
  - This is the "default" kernel which uses `kernel-4.14`
- `/aws/service/ami-amazon-linux-latest/amzn2-ami-kernel-5.10-hvm-x86_64-ebs`

If a new Amazon Linux generation AMI is published with a new kernel version,
then a new SSM parameter will be created with the new version
(e.g. `/aws/service/ami-amazon-linux-latest/amzn2-ami-kernel-5.15-hvm-x86_64-ebs`),
but the "default" AMI may or may not be updated.

If you would like to make sure you always have the latest kernel version, then
either specify the specific latest kernel version or opt-in to using the CDK
latest kernel version.

```ts
declare const vpc: ec2.Vpc;

new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  // context cache is turned on by default
  machineImage: new ec2.AmazonLinux2023ImageSsmParameter({
    kernel: ec2.AmazonLinux2023Kernel.KERNEL_6_1,
  }),
});
```
_CDK managed latest_

```ts
declare const vpc: ec2.Vpc;

new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  // context cache is turned on by default
  machineImage: new ec2.AmazonLinux2023ImageSsmParameter({
    kernel: ec2.AmazonLinux2023Kernel.CDK_LATEST,
  }),
});

// or

new ec2.Instance(this, 'LatestAl2023', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.C7G, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2023(), // always uses latest kernel version
});
```

When using the CDK managed latest version, when a new kernel version is made
available the `LATEST` will be updated to point to the new kernel version. You
then would be required to update the newest CDK version for it to take effect.

### Configuring Instances using CloudFormation Init (cfn-init)

CloudFormation Init allows you to configure your instances by writing files to them, installing software
packages, starting services and running arbitrary commands. By default, if any of the instance setup
commands throw an error; the deployment will fail and roll back to the previously known good state.
The following documentation also applies to `AutoScalingGroup`s.

For the full set of capabilities of this system, see the documentation for
[`AWS::CloudFormation::Init`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-init.html).
Here is an example of applying some configuration to an instance:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,

  // Showing the most complex setup, if you have simpler requirements
  // you can use `CloudFormationInit.fromElements()`.
  init: ec2.CloudFormationInit.fromConfigSets({
    configSets: {
      // Applies the configs below in this order
      default: ['yumPreinstall', 'config'],
    },
    configs: {
      yumPreinstall: new ec2.InitConfig([
        // Install an Amazon Linux package using yum
        ec2.InitPackage.yum('git'),
      ]),
      config: new ec2.InitConfig([
        // Create a JSON file from tokens (can also create other files)
        ec2.InitFile.fromObject('/etc/stack.json', {
          stackId: Stack.of(this).stackId,
          stackName: Stack.of(this).stackName,
          region: Stack.of(this).region,
        }),

        // Create a group and user
        ec2.InitGroup.fromName('my-group'),
        ec2.InitUser.fromName('my-user'),

        // Install an RPM from the internet
        ec2.InitPackage.rpm('http://mirrors.ukfast.co.uk/sites/dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/r/rubygem-git-1.5.0-2.el8.noarch.rpm'),
      ]),
    },
  }),
  initOptions: {
    // Optional, which configsets to activate (['default'] by default)
    configSets: ['default'],

    // Optional, how long the installation is expected to take (5 minutes by default)
    timeout: Duration.minutes(30),

    // Optional, whether to include the --url argument when running cfn-init and cfn-signal commands (false by default)
    includeUrl: true,

    // Optional, whether to include the --role argument when running cfn-init and cfn-signal commands (false by default)
    includeRole: true,
  },
});
```

`InitCommand` can not be used to start long-running processes. At deploy time,
`cfn-init` will always wait for the process to exit before continuing, causing
the CloudFormation deployment to fail because the signal hasn't been received
within the expected timeout.

Instead, you should install a service configuration file onto your machine `InitFile`,
and then use `InitService` to start it.

If your Linux OS is using SystemD (like Amazon Linux 2 or higher), the CDK has
helpers to create a long-running service using CFN Init. You can create a
SystemD-compatible config file using `InitService.systemdConfigFile()`, and
start it immediately. The following examples shows how to start a trivial Python
3 web server:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),

  init: ec2.CloudFormationInit.fromElements(
    // Create a simple config file that runs a Python web server
    ec2.InitService.systemdConfigFile('simpleserver', {
      command: '/usr/bin/python3 -m http.server 8080',
      cwd: '/var/www/html',
    }),
    // Start the server using SystemD
    ec2.InitService.enable('simpleserver', {
      serviceManager: ec2.ServiceManager.SYSTEMD,
    }),
    // Drop an example file to show the web server working
    ec2.InitFile.fromString('/var/www/html/index.html', 'Hello! It\'s working!'),
  ),
});
```

You can have services restarted after the init process has made changes to the system.
To do that, instantiate an `InitServiceRestartHandle` and pass it to the config elements
that need to trigger the restart and the service itself. For example, the following
config writes a config file for nginx, extracts an archive to the root directory, and then
restarts nginx so that it picks up the new config and files:

```ts
declare const myBucket: s3.Bucket;

const handle = new ec2.InitServiceRestartHandle();

ec2.CloudFormationInit.fromElements(
  ec2.InitFile.fromString('/etc/nginx/nginx.conf', '...', { serviceRestartHandles: [handle] }),
  ec2.InitSource.fromS3Object('/var/www/html', myBucket, 'html.zip', { serviceRestartHandles: [handle] }),
  ec2.InitService.enable('nginx', {
    serviceRestartHandle: handle,
  })
);
```

You can use the `environmentVariables` or `environmentFiles` parameters to specify environment variables
for your services:

```ts
new ec2.InitConfig([
  ec2.InitFile.fromString('/myvars.env', 'VAR_FROM_FILE="VAR_FROM_FILE"'),
  ec2.InitService.systemdConfigFile('myapp', {
    command: '/usr/bin/python3 -m http.server 8080',
    cwd: '/var/www/html',
    environmentVariables: {
      MY_VAR: 'MY_VAR',
    },
    environmentFiles: ['/myvars.env'],
  }),
])
```

### Bastion Hosts

A bastion host functions as an instance used to access servers and resources in a VPC without open up the complete VPC on a network level.
You can use bastion hosts using a standard SSH connection targeting port 22 on the host. As an alternative, you can connect the SSH connection
feature of AWS Systems Manager Session Manager, which does not need an opened security group. (https://aws.amazon.com/about-aws/whats-new/2019/07/session-manager-launches-tunneling-support-for-ssh-and-scp/)

A default bastion host for use via SSM can be configured like:

```ts fixture=with-vpc
const host = new ec2.BastionHostLinux(this, 'BastionHost', { vpc });
```

If you want to connect from the internet using SSH, you need to place the host into a public subnet. You can then configure allowed source hosts.

```ts fixture=with-vpc
const host = new ec2.BastionHostLinux(this, 'BastionHost', {
  vpc,
  subnetSelection: { subnetType: ec2.SubnetType.PUBLIC },
});
host.allowSshAccessFrom(ec2.Peer.ipv4('1.2.3.4/32'));
```

As there are no SSH public keys deployed on this machine, you need to use [EC2 Instance Connect](https://aws.amazon.com/de/blogs/compute/new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances/)
with the command `aws ec2-instance-connect send-ssh-public-key` to provide your SSH public key.

EBS volume for the bastion host can be encrypted like:

```ts fixture=with-vpc
const host = new ec2.BastionHostLinux(this, 'BastionHost', {
  vpc,
  blockDevices: [{
    deviceName: '/dev/sdh',
    volume: ec2.BlockDeviceVolume.ebs(10, {
      encrypted: true,
    }),
  }],
});
```

It's recommended to set the `@aws-cdk/aws-ec2:bastionHostUseAmazonLinux2023ByDefault`
[feature flag](https://docs.aws.amazon.com/cdk/v2/guide/featureflags.html) to `true` to use Amazon Linux 2023 as the
bastion host AMI. Without this flag set, the bastion host will default to Amazon Linux 2, which will be unsupported in
June 2025.

```json
{
  "context": {
    "@aws-cdk/aws-ec2:bastionHostUseAmazonLinux2023ByDefault": true
  }
}
```

### Placement Group

Specify `placementGroup` to enable the placement group support:

```ts fixture=with-vpc
declare const instanceType: ec2.InstanceType;

const pg = new ec2.PlacementGroup(this, 'test-pg', {
  strategy: ec2.PlacementGroupStrategy.SPREAD,
});

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  placementGroup: pg,
});
```

### Block Devices

To add EBS block device mappings, specify the `blockDevices` property. The following example sets the EBS-backed
root device (`/dev/sda1`) size to 50 GiB, and adds another EBS-backed device mapped to `/dev/sdm` that is 100 GiB in
size:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,

  // ...

  blockDevices: [
    {
      deviceName: '/dev/sda1',
      volume: ec2.BlockDeviceVolume.ebs(50),
    },
    {
      deviceName: '/dev/sdm',
      volume: ec2.BlockDeviceVolume.ebs(100),
    },
  ],
});

```

It is also possible to encrypt the block devices. In this example we will create an customer managed key encrypted EBS-backed root device:

```ts
import { Key } from 'aws-cdk-lib/aws-kms';

declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

const kmsKey = new Key(this, 'KmsKey')

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,

  // ...

  blockDevices: [
    {
      deviceName: '/dev/sda1',
      volume: ec2.BlockDeviceVolume.ebs(50, {
        encrypted: true,
        kmsKey: kmsKey,
      }),
    },
  ],
});

```

To specify the throughput value for `gp3` volumes, use the `throughput` property:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,

  // ...

  blockDevices: [
    {
      deviceName: '/dev/sda1',
      volume: ec2.BlockDeviceVolume.ebs(100, {
        volumeType: ec2.EbsDeviceVolumeType.GP3,
        throughput: 250,
      }),
    },
  ],
});

```

#### EBS Optimized Instances

An Amazon EBS–optimized instance uses an optimized configuration stack and provides additional, dedicated capacity for Amazon EBS I/O. This optimization provides the best performance for your EBS volumes by minimizing contention between Amazon EBS I/O and other traffic from your instance.

Depending on the instance type, this features is enabled by default while others require explicit activation. Please refer to the [documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) for details.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,
  ebsOptimized: true,
  blockDevices: [{
    deviceName: '/dev/xvda',
    volume: ec2.BlockDeviceVolume.ebs(8),
  }],
});
```

### Volumes

Whereas a `BlockDeviceVolume` is an EBS volume that is created and destroyed as part of the creation and destruction of a specific instance. A `Volume` is for when you want an EBS volume separate from any particular instance. A `Volume` is an EBS block device that can be attached to, or detached from, any instance at any time. Some types of `Volume`s can also be attached to multiple instances at the same time to allow you to have shared storage between those instances.

A notable restriction is that a Volume can only be attached to instances in the same availability zone as the Volume itself.

The following demonstrates how to create a 500 GiB encrypted Volume in the `us-west-2a` availability zone, and give a role the ability to attach that Volume to a specific instance:

```ts
declare const instance: ec2.Instance;
declare const role: iam.Role;

const volume = new ec2.Volume(this, 'Volume', {
  availabilityZone: 'us-west-2a',
  size: Size.gibibytes(500),
  encrypted: true,
});

volume.grantAttachVolume(role, [instance]);
```

#### Instances Attaching Volumes to Themselves

If you need to grant an instance the ability to attach/detach an EBS volume to/from itself, then using `grantAttachVolume` and `grantDetachVolume` as outlined above
will lead to an unresolvable circular reference between the instance role and the instance. In this case, use `grantAttachVolumeByResourceTag` and `grantDetachVolumeByResourceTag` as follows:

```ts
declare const instance: ec2.Instance;
declare const volume: ec2.Volume;

const attachGrant = volume.grantAttachVolumeByResourceTag(instance.grantPrincipal, [instance]);
const detachGrant = volume.grantDetachVolumeByResourceTag(instance.grantPrincipal, [instance]);
```

#### Attaching Volumes

The Amazon EC2 documentation for
[Linux Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AmazonEBS.html) and
[Windows Instances](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/ebs-volumes.html) contains information on how
to attach and detach your Volumes to/from instances, and how to format them for use.

The following is a sample skeleton of EC2 UserData that can be used to attach a Volume to the Linux instance that it is running on:

```ts
declare const instance: ec2.Instance;
declare const volume: ec2.Volume;

volume.grantAttachVolumeByResourceTag(instance.grantPrincipal, [instance]);
const targetDevice = '/dev/xvdz';
instance.userData.addCommands(
  // Retrieve token for accessing EC2 instance metadata (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)
  `TOKEN=$(curl -SsfX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")`,
  // Retrieve the instance Id of the current EC2 instance
  `INSTANCE_ID=$(curl -SsfH "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)`,
  // Attach the volume to /dev/xvdz
  `aws --region ${Stack.of(this).region} ec2 attach-volume --volume-id ${volume.volumeId} --instance-id $INSTANCE_ID --device ${targetDevice}`,
  // Wait until the volume has attached
  `while ! test -e ${targetDevice}; do sleep 1; done`
  // The volume will now be mounted. You may have to add additional code to format the volume if it has not been prepared.
);
```

#### Tagging Volumes

You can configure [tag propagation on volume creation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ec2-instance.html#cfn-ec2-instance-propagatetagstovolumeoncreation).

```ts
  declare const vpc: ec2.Vpc;
  declare const instanceType: ec2.InstanceType;
  declare const machineImage: ec2.IMachineImage;

  new ec2.Instance(this, 'Instance', {
    vpc,
    machineImage,
    instanceType,
    propagateTagsToVolumeOnCreation: true,
  });
```

#### Throughput on GP3 Volumes

You can specify the `throughput` of a GP3 volume from 125 (default) to 2000.

```ts
new ec2.Volume(this, 'Volume', {
  availabilityZone: 'us-east-1a',
  size: Size.gibibytes(125),
  volumeType: ec2.EbsDeviceVolumeType.GP3,
  throughput: 125,
});
```

#### Volume initialization rate

When creating an EBS volume from a snapshot, you can specify the [volume initialization rate](https://docs.aws.amazon.com/ebs/latest/userguide/initalize-volume.html#volume-initialization-rate) at which the snapshot blocks are downloaded from Amazon S3 to the volume.
Specifying a volume initialization rate ensures that the volume is initialized at a predictable and consistent rate after creation.

```ts
new ec2.Volume(this, 'Volume', {
  availabilityZone: 'us-east-1a',
  size: Size.gibibytes(500),
  snapshotId: 'snap-1234567890abcdef0',
  volumeInitializationRate: Size.mebibytes(250),
});
```

The `volumeInitializationRate` must be:

* Between 100 and 300 MiB/s
* Only specified when creating a volume from a snapshot

### Configuring Instance Metadata Service (IMDS)

#### Comprehensive Metadata Options

You can configure [EC2 Instance Metadata Service](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html) options using individual properties. This provides comprehensive control over all metadata service settings:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

// Example 1: Enforce IMDSv2 with comprehensive options
new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,
  httpEndpoint: true,
  httpProtocolIpv6: false,
  httpPutResponseHopLimit: 2,
  httpTokens: ec2.HttpTokens.REQUIRED,
  instanceMetadataTags: true,
});

// Example 2: Enforce IMDSv2 with minimal configuration
new ec2.Instance(this, 'SecureInstance', {
  vpc,
  instanceType,
  machineImage,
  httpTokens: ec2.HttpTokens.REQUIRED,
});
```

#### Simple IMDSv2 Enforcement

For simple IMDSv2 enforcement without additional configuration, you can use the `requireImdsv2` property:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage,

  // Simple IMDSv2 enforcement
  requireImdsv2: true,
});
```

#### Applying to Multiple Instances

You can also use the either the `InstanceRequireImdsv2Aspect` for EC2 instances or the `LaunchTemplateRequireImdsv2Aspect` for EC2 launch templates
to apply the operation to multiple instances or launch templates, respectively.

The following example demonstrates how to use the `InstanceRequireImdsv2Aspect` to require IMDSv2 for all EC2 instances in a stack:

```ts
const aspect = new ec2.InstanceRequireImdsv2Aspect();
Aspects.of(this).add(aspect);
```

### Associating a Public IP Address with an Instance

All subnets have an attribute that determines whether instances launched into that subnet are assigned a public IPv4 address. This attribute is set to true by default for default public subnets. Thus, an EC2 instance launched into a default public subnet will be assigned a public IPv4 address. Nondefault public subnets have this attribute set to false by default and any EC2 instance launched into a nondefault public subnet will not be assigned a public IPv4 address automatically. To automatically assign a public IPv4 address to an instance launched into a nondefault public subnet, you can set the `associatePublicIpAddress` property on the `Instance` construct to true. Alternatively, to not automatically assign a public IPv4 address to an instance launched into a default public subnet, you can set `associatePublicIpAddress` to false. Including this property, removing this property, or updating the value of this property on an existing instance will result in replacement of the instance.

```ts
const vpc = new ec2.Vpc(this, 'VPC', {
  cidr: '10.0.0.0/16',
  natGateways: 0,
  maxAzs: 3,
  subnetConfiguration: [
    {
      name: 'public-subnet-1',
      subnetType: ec2.SubnetType.PUBLIC,
      cidrMask: 24,
    },
  ],
});

const instance = new ec2.Instance(this, 'Instance', {
  vpc,
  vpcSubnets: { subnetGroupName: 'public-subnet-1' },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.NANO),
  machineImage: new ec2.AmazonLinuxImage({ generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2 }),
  detailedMonitoring: true,
  associatePublicIpAddress: true,
});
```

### Specifying a key pair

To allow SSH access to an EC2 instance by default, a Key Pair must be specified. Key pairs can
be provided with the `keyPair` property to instances and launch templates. You can create a
key pair for an instance like this:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;

const keyPair = new ec2.KeyPair(this, 'KeyPair', {
  type: ec2.KeyPairType.ED25519,
  format: ec2.KeyPairFormat.PEM,
});
const instance = new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  // Use the custom key pair
  keyPair,
});
```

When a new EC2 Key Pair is created (without imported material), the private key material is
automatically stored in Systems Manager Parameter Store. This can be retrieved from the key pair
construct:

```ts
const keyPair = new ec2.KeyPair(this, 'KeyPair');
const privateKey = keyPair.privateKey;
```

If you already have an SSH key that you wish to use in EC2, that can be provided when constructing the
`KeyPair`. If public key material is provided, the key pair is considered "imported" and there
will not be any data automatically stored in Systems Manager Parameter Store and the `type` property
cannot be specified for the key pair.

```ts
const keyPair = new ec2.KeyPair(this, 'KeyPair', {
  publicKeyMaterial: "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB7jpNzG+YG0s+xIGWbxrxIZiiozHOEuzIJacvASP0mq",
})
```

#### Using an existing EC2 Key Pair

If you already have an EC2 Key Pair created outside of the CDK, you can import that key to
your CDK stack.

You can import it purely by name:

```ts
const keyPair = ec2.KeyPair.fromKeyPairName(this, 'KeyPair', 'the-keypair-name');
```

Or by specifying additional attributes:

```ts
const keyPair = ec2.KeyPair.fromKeyPairAttributes(this, 'KeyPair', {
  keyPairName: 'the-keypair-name',
  type: ec2.KeyPairType.RSA,
})
```

### Using IPv6 IPs

Instances can be given IPv6 IPs by launching them into a subnet of a dual stack VPC.

```ts
const vpc = new ec2.Vpc(this, 'Ip6VpcDualStack', {
  ipProtocol: ec2.IpProtocol.DUAL_STACK,
  subnetConfiguration: [
    {
      name: 'Public',
      subnetType: ec2.SubnetType.PUBLIC,
      mapPublicIpOnLaunch: true,
    },
    {
      name: 'Private',
      subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
    },
  ],
});

const instance = new ec2.Instance(this, 'MyInstance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  vpc: vpc,
  vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
  allowAllIpv6Outbound: true,

  // ...
});

instance.connections.allowFrom(ec2.Peer.anyIpv6(), ec2.Port.allIcmpV6(), 'allow ICMPv6');
```

Note to set `mapPublicIpOnLaunch` to true in the `subnetConfiguration`.

Additionally, IPv6 support varies by instance type. Most instance types have IPv6 support with exception of m1-m3, c1, g2, and t1.micro. A full list can be found here: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-eni.html#AvailableIpPerENI.

#### Specifying the IPv6 Address

If you want to specify [the number of IPv6 addresses](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/MultipleIP.html#assign-multiple-ipv6) to assign to the instance, you can use the `ipv6AddresseCount` property:

```ts
// dual stack VPC
declare const vpc: ec2.Vpc;

const instance = new ec2.Instance(this, 'MyInstance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  vpc: vpc,
  vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
  // Assign 2 IPv6 addresses to the instance
  ipv6AddressCount: 2,
});
```

### Credit configuration modes for burstable instances

You can set the [credit configuration mode](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-credits-baseline-concepts.html) for burstable instances (T2, T3, T3a and T4g instance types):

```ts
declare const vpc: ec2.Vpc;

const instance = new ec2.Instance(this, 'Instance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  vpc: vpc,
  creditSpecification: ec2.CpuCredits.STANDARD,
});
```

It is also possible to set the credit configuration mode for NAT instances.

```ts
const natInstanceProvider = ec2.NatProvider.instance({
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.LARGE),
  machineImage: new ec2.AmazonLinuxImage(),
  creditSpecification: ec2.CpuCredits.UNLIMITED,
});
new ec2.Vpc(this, 'VPC', {
  natGatewayProvider: natInstanceProvider,
});
```

**Note**: `CpuCredits.UNLIMITED` mode is not supported for T3 instances that are launched on a Dedicated Host.

### Shutdown behavior

You can specify the behavior of the instance when you initiate shutdown from the instance (using the operating system command for system shutdown).

```ts
declare const vpc: ec2.Vpc;

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.NANO),
  machineImage: new ec2.AmazonLinuxImage({ generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2 }),
  instanceInitiatedShutdownBehavior: ec2.InstanceInitiatedShutdownBehavior.TERMINATE, // default is STOP
});
```

### Enabling Nitro Enclaves

You can enable [AWS Nitro Enclaves](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave.html) for
your EC2 instances by setting the `enclaveEnabled` property to `true`. Nitro Enclaves is a feature of
AWS Nitro System that enables creating isolated and highly constrained CPU environments known as enclaves.

```ts
declare const vpc: ec2.Vpc;

const instance = new ec2.Instance(this, 'Instance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
  machineImage: new ec2.AmazonLinuxImage(),
  vpc: vpc,
  enclaveEnabled: true,
});
```

> NOTE: You must use an instance type and operating system that support Nitro Enclaves.
> For more information, see [Requirements](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave.html#nitro-enclave-reqs).

### Enabling Termination Protection

You can enable [Termination Protection](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingDisableAPITermination.html) for
your EC2 instances by setting the `disableApiTermination` property to `true`. Termination Protection controls whether the instance can be terminated using the AWS Management Console, AWS Command Line Interface (AWS CLI), or API.

```ts
declare const vpc: ec2.Vpc;

const instance = new ec2.Instance(this, 'Instance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
  machineImage: new ec2.AmazonLinuxImage(),
  vpc: vpc,
  disableApiTermination: true,
});
```

### Enabling Instance Hibernation

You can enable [Instance Hibernation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Hibernate.html) for
your EC2 instances by setting the `hibernationEnabled` property to `true`. Instance Hibernation saves the
instance's in-memory (RAM) state when an instance is stopped, and restores that state when the instance is started.

```ts
declare const vpc: ec2.Vpc;

const instance = new ec2.Instance(this, 'Instance', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
  machineImage: new ec2.AmazonLinuxImage(),
  vpc: vpc,
  hibernationEnabled: true,
  blockDevices: [{
    deviceName: '/dev/xvda',
    volume: ec2.BlockDeviceVolume.ebs(30, {
      volumeType: ec2.EbsDeviceVolumeType.GP3,
      encrypted: true,
      deleteOnTermination: true,
    }),
  }],
});
```

> NOTE: You must use an instance and a volume that meet the requirements for hibernation.
> For more information, see [Prerequisites for Amazon EC2 instance hibernation](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave.html#nitro-enclave-reqs).


## VPC Flow Logs

VPC Flow Logs is a feature that enables you to capture information about the IP traffic going to and from network interfaces in your VPC. Flow log data can be published to Amazon CloudWatch Logs and Amazon S3. After you've created a flow log, you can retrieve and view its data in the chosen destination. (<https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html>).

By default, a flow log will be created with CloudWatch Logs as the destination.

You can create a flow log like this:

```ts
declare const vpc: ec2.Vpc;

new ec2.FlowLog(this, 'FlowLog', {
  resourceType: ec2.FlowLogResourceType.fromVpc(vpc)
})
```

Or you can add a Flow Log to a VPC by using the addFlowLog method like this:

```ts
const vpc = new ec2.Vpc(this, 'Vpc');

vpc.addFlowLog('FlowLog');
```

You can also add multiple flow logs with different destinations.

```ts
const vpc = new ec2.Vpc(this, 'Vpc');

vpc.addFlowLog('FlowLogS3', {
  destination: ec2.FlowLogDestination.toS3()
});

// Only reject traffic and interval every minute.
vpc.addFlowLog('FlowLogCloudWatch', {
  trafficType: ec2.FlowLogTrafficType.REJECT,
  maxAggregationInterval: ec2.FlowLogMaxAggregationInterval.ONE_MINUTE,
});
```

To create a Transit Gateway flow log, you can use the `fromTransitGatewayId` method:

```ts
declare const tgw: ec2.CfnTransitGateway;

new ec2.FlowLog(this, 'TransitGatewayFlowLog', {
  resourceType: ec2.FlowLogResourceType.fromTransitGatewayId(tgw.ref)
})
```

To create a Transit Gateway Attachment flow log, you can use the `fromTransitGatewayAttachmentId` method:

```ts
declare const tgwAttachment: ec2.CfnTransitGatewayAttachment;

new ec2.FlowLog(this, 'TransitGatewayAttachmentFlowLog', {
  resourceType: ec2.FlowLogResourceType.fromTransitGatewayAttachmentId(tgwAttachment.ref)
})
```

For flow logs targeting TransitGateway and TransitGatewayAttachment, specifying the `trafficType` is not possible.

### Custom Formatting

You can also custom format flow logs.

```ts
const vpc = new ec2.Vpc(this, 'Vpc');

vpc.addFlowLog('FlowLog', {
  logFormat: [
    ec2.LogFormat.DST_PORT,
    ec2.LogFormat.SRC_PORT,
  ],
});

// If you just want to add a field to the default field
vpc.addFlowLog('FlowLog', {
  logFormat: [
    ec2.LogFormat.VERSION,
    ec2.LogFormat.ALL_DEFAULT_FIELDS,
  ],
});

// If AWS CDK does not support the new fields
vpc.addFlowLog('FlowLog', {
  logFormat: [
    ec2.LogFormat.SRC_PORT,
    ec2.LogFormat.custom('${new-field}'),
  ],
});
```


By default, the CDK will create the necessary resources for the destination. For the CloudWatch Logs destination
it will create a CloudWatch Logs Log Group as well as the IAM role with the necessary permissions to publish to
the log group. In the case of an S3 destination, it will create the S3 bucket.

If you want to customize any of the destination resources you can provide your own as part of the `destination`.

*CloudWatch Logs*

```ts
declare const vpc: ec2.Vpc;

const logGroup = new logs.LogGroup(this, 'MyCustomLogGroup');

const role = new iam.Role(this, 'MyCustomRole', {
  assumedBy: new iam.ServicePrincipal('vpc-flow-logs.amazonaws.com')
});

new ec2.FlowLog(this, 'FlowLog', {
  resourceType: ec2.FlowLogResourceType.fromVpc(vpc),
  destination: ec2.FlowLogDestination.toCloudWatchLogs(logGroup, role)
});
```

*S3*

```ts
declare const vpc: ec2.Vpc;

const bucket = new s3.Bucket(this, 'MyCustomBucket');

new ec2.FlowLog(this, 'FlowLog', {
  resourceType: ec2.FlowLogResourceType.fromVpc(vpc),
  destination: ec2.FlowLogDestination.toS3(bucket)
});

new ec2.FlowLog(this, 'FlowLogWithKeyPrefix', {
  resourceType: ec2.FlowLogResourceType.fromVpc(vpc),
  destination: ec2.FlowLogDestination.toS3(bucket, 'prefix/')
});
```

*Amazon Data Firehose*

```ts
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose';

declare const vpc: ec2.Vpc;
declare const deliveryStream: firehose.IDeliveryStream;

vpc.addFlowLog('FlowLogsFirehose', {
  destination: ec2.FlowLogDestination.toFirehose(deliveryStream),
});
```

When the S3 destination is configured, AWS will automatically create an S3 bucket policy
that allows the service to write logs to the bucket. This makes it impossible to later update
that bucket policy. To have CDK create the bucket policy so that future updates can be made,
the `@aws-cdk/aws-s3:createDefaultLoggingPolicy` [feature flag](https://docs.aws.amazon.com/cdk/v2/guide/featureflags.html) can be used. This can be set
in the `cdk.json` file.

```json
{
  "context": {
    "@aws-cdk/aws-s3:createDefaultLoggingPolicy": true
  }
}
```

## User Data

User data enables you to run a script when your instances start up.  In order to configure these scripts you can add commands directly to the script
 or you can use the UserData's convenience functions to aid in the creation of your script.

A user data could be configured to run a script found in an asset through the following:

```ts
import { Asset } from 'aws-cdk-lib/aws-s3-assets';

declare const instance: ec2.Instance;

const asset = new Asset(this, 'Asset', {
  path: './configure.sh'
});

const localPath = instance.userData.addS3DownloadCommand({
  bucket:asset.bucket,
  bucketKey:asset.s3ObjectKey,
  region: 'us-east-1', // Optional
});
instance.userData.addExecuteFileCommand({
  filePath:localPath,
  arguments: '--verbose -y'
});
asset.grantRead(instance.role);
```

### Persisting user data

By default, EC2 UserData is run once on only the first time that an instance is started. It is possible to make the
user data script run on every start of the instance.

When creating a Windows UserData you can use the `persist` option to set whether or not to add
`<persist>true</persist>` [to the user data script](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/ec2-windows-user-data.html#user-data-scripts). it can be used as follows:

```ts
const windowsUserData = ec2.UserData.forWindows({ persist: true });
```

For a Linux instance, this can be accomplished by using a Multipart user data to configure cloud-config as detailed
in: https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/

### Multipart user data

In addition, to above the `MultipartUserData` can be used to change instance startup behavior. Multipart user data are composed
from separate parts forming archive. The most common parts are scripts executed during instance set-up. However, there are other
kinds, too.

The advantage of multipart archive is in flexibility when it's needed to add additional parts or to use specialized parts to
fine tune instance startup. Some services (like AWS Batch) support only `MultipartUserData`.

The parts can be executed at different moment of instance start-up and can serve a different purpose. This is controlled by `contentType` property.
For common scripts, `text/x-shellscript; charset="utf-8"` can be used as content type.

In order to create archive the `MultipartUserData` has to be instantiated. Than, user can add parts to multipart archive using `addPart`. The `MultipartBody` contains methods supporting creation of body parts.

If the very custom part is required, it can be created using `MultipartUserData.fromRawBody`, in this case full control over content type,
transfer encoding, and body properties is given to the user.

Below is an example for creating multipart user data with single body part responsible for installing `awscli` and configuring maximum size
of storage used by Docker containers:

```ts
const bootHookConf = ec2.UserData.forLinux();
bootHookConf.addCommands('cloud-init-per once docker_options echo \'OPTIONS="${OPTIONS} --storage-opt dm.basesize=40G"\' >> /etc/sysconfig/docker');

const setupCommands = ec2.UserData.forLinux();
setupCommands.addCommands('sudo yum install awscli && echo Packages installed らと > /var/tmp/setup');

const multipartUserData = new ec2.MultipartUserData();
// The docker has to be configured at early stage, so content type is overridden to boothook
multipartUserData.addPart(ec2.MultipartBody.fromUserData(bootHookConf, 'text/cloud-boothook; charset="us-ascii"'));
// Execute the rest of setup
multipartUserData.addPart(ec2.MultipartBody.fromUserData(setupCommands));

new ec2.LaunchTemplate(this, '', {
  userData: multipartUserData,
  blockDevices: [
    // Block device configuration rest
  ]
});
```

For more information see
[Specifying Multiple User Data Blocks Using a MIME Multi Part Archive](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/bootstrap_container_instance.html#multi-part_user_data)

#### Using add*Command on MultipartUserData

To use the `add*Command` methods, that are inherited from the `UserData` interface, on `MultipartUserData` you must add a part
to the `MultipartUserData` and designate it as the receiver for these methods. This is accomplished by using the `addUserDataPart()`
method on `MultipartUserData` with the `makeDefault` argument set to `true`:

```ts
const multipartUserData = new ec2.MultipartUserData();
const commandsUserData = ec2.UserData.forLinux();
multipartUserData.addUserDataPart(commandsUserData, ec2.MultipartBody.SHELL_SCRIPT, true);

// Adding commands to the multipartUserData adds them to commandsUserData, and vice-versa.
multipartUserData.addCommands('touch /root/multi.txt');
commandsUserData.addCommands('touch /root/userdata.txt');
```

When used on an EC2 instance, the above `multipartUserData` will create both `multi.txt` and `userdata.txt` in `/root`.

## Importing existing subnet

To import an existing Subnet, call `Subnet.fromSubnetAttributes()` or
`Subnet.fromSubnetId()`. Only if you supply the subnet's Availability Zone
and Route Table Ids when calling `Subnet.fromSubnetAttributes()` will you be
able to use the CDK features that use these values (such as selecting one
subnet per AZ).

Importing an existing subnet looks like this:

```ts
// Supply all properties
const subnet1 = ec2.Subnet.fromSubnetAttributes(this, 'SubnetFromAttributes', {
  subnetId: 's-1234',
  availabilityZone: 'pub-az-4465',
  routeTableId: 'rt-145'
});

// Supply only subnet id
const subnet2 = ec2.Subnet.fromSubnetId(this, 'SubnetFromId', 's-1234');
```

## Launch Templates

A Launch Template is a standardized template that contains the configuration information to launch an instance.
They can be used when launching instances on their own, through Amazon EC2 Auto Scaling, EC2 Fleet, and Spot Fleet.
Launch templates enable you to store launch parameters so that you do not have to specify them every time you launch
an instance. For information on Launch Templates please see the
[official documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-templates.html).

The following demonstrates how to create a launch template with an Amazon Machine Image, security group, and an instance profile.

```ts
declare const vpc: ec2.Vpc;

const role = new iam.Role(this, 'Role', {
  assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
});
const instanceProfile = new iam.InstanceProfile(this, 'InstanceProfile', {
  role,
});

const template = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  launchTemplateName: 'MyTemplateV1',
  versionDescription: 'This is my v1 template',
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  securityGroup: new ec2.SecurityGroup(this, 'LaunchTemplateSG', {
    vpc: vpc,
  }),
  instanceProfile,
});
```

And the following demonstrates how to enable metadata options support.

```ts
new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  httpEndpoint: true,
  httpProtocolIpv6: true,
  httpPutResponseHopLimit: 1,
  httpTokens: ec2.LaunchTemplateHttpTokens.REQUIRED,
  instanceMetadataTags: true,
});
```

And the following demonstrates how to add one or more security groups to launch template.

```ts
declare const vpc: ec2.Vpc;

const sg1 = new ec2.SecurityGroup(this, 'sg1', {
  vpc: vpc,
});
const sg2 = new ec2.SecurityGroup(this, 'sg2', {
  vpc: vpc,
});

const launchTemplate = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  securityGroup: sg1,
});

launchTemplate.addSecurityGroup(sg2);
```

To use [AWS Systems Manager parameters instead of AMI IDs](https://docs.aws.amazon.com/autoscaling/ec2/userguide/using-systems-manager-parameters.html) in launch templates and resolve the AMI IDs at instance launch time:

```ts
const launchTemplate = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  machineImage: ec2.MachineImage.resolveSsmParameterAtLaunch('parameterName'),
});
```

To specify the EBS Provisioned Rate for Volume Initialization for a snapshot-backed EBS volume in a launch template, use the `volumeInitializationRate` property. The snapshot can be specified explicitly with `BlockDeviceVolume.ebsFromSnapshot(...)` or inherited from the AMI block device mapping when overriding an existing AMI device such as the root volume.

```ts
const launchTemplate = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  blockDevices: [
    {
      deviceName: 'deviceName',
      volume: ec2.BlockDeviceVolume.ebsFromSnapshot('snap-1234567890abcdef0', {
        volumeSize: 150,
        volumeInitializationRate: Size.mebibytes(200),
      }),
    },
  ],
});

```

### Placement Group

Specify `placementGroup` to enable the placement group support:

```ts fixture=with-vpc
declare const instanceType: ec2.InstanceType;

const pg = new ec2.PlacementGroup(this, 'test-pg', {
  strategy: ec2.PlacementGroupStrategy.SPREAD,
});

new ec2.LaunchTemplate(this, 'LaunchTemplate', {
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  placementGroup: pg,
});
```

Please note this feature does not support Launch Configurations.

## Detailed Monitoring

The following demonstrates how to enable [Detailed Monitoring](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-cloudwatch-new.html) for an EC2 instance. Keep in mind that Detailed Monitoring results in [additional charges](https://aws.amazon.com/cloudwatch/pricing/).

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;

new ec2.Instance(this, 'Instance1', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  detailedMonitoring: true,
});
```

## Connecting to your instances using SSM Session Manager

SSM Session Manager makes it possible to connect to your instances from the
AWS Console, without preparing SSH keys.

To do so, you need to:

* Use an image with [SSM agent](https://docs.aws.amazon.com/systems-manager/latest/userguide/ssm-agent.html) installed
  and configured. [Many images come with SSM Agent
  preinstalled](https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html), otherwise you
  may need to manually put instructions to [install SSM
  Agent](https://docs.aws.amazon.com/systems-manager/latest/userguide/sysman-manual-agent-install.html) into your
  instance's UserData or use EC2 Init).
* Create the instance with `ssmSessionPermissions: true`.

If these conditions are met, you can connect to the instance from the EC2 Console. Example:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;

new ec2.Instance(this, 'Instance1', {
  vpc,
  instanceType,

  // Amazon Linux 2023 comes with SSM Agent by default
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),

  // Turn on SSM
  ssmSessionPermissions: true,
});
```

## Managed Prefix Lists

Create and manage customer-managed prefix lists. If you don't specify anything in this construct, it will manage IPv4 addresses.

You can also create an empty Prefix List with only the maximum number of entries specified, as shown in the following code. If nothing is specified, maxEntries=1.

```ts
new ec2.PrefixList(this, 'EmptyPrefixList', {
  maxEntries: 100,
});
```

`maxEntries` can also be omitted as follows. In this case `maxEntries: 2`, will be set.

```ts
new ec2.PrefixList(this, 'PrefixList', {
  entries: [
    { cidr: '10.0.0.1/32' },
    { cidr: '10.0.0.2/32', description: 'sample1' },
  ],
});
```

To import AWS-managed prefix list, you can use `PrefixList.fromLookup()`.

``` ts
ec2.PrefixList.fromLookup(this, 'PrefixListFromName', {
  prefixListName: 'com.amazonaws.global.cloudfront.origin-facing',
});
```

For more information see [Work with customer-managed prefix lists](https://docs.aws.amazon.com/vpc/latest/userguide/working-with-managed-prefix-lists.html).

### IAM instance profile

Use `instanceProfile` to apply specific IAM Instance Profile. Cannot be used with role

```ts
declare const instanceType: ec2.InstanceType;
declare const vpc: ec2.Vpc;

const role = new iam.Role(this, 'Role', {
  assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
});
const instanceProfile = new iam.InstanceProfile(this, 'InstanceProfile', {
  role,
});

new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType,
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  instanceProfile,
});
```
