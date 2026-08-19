# Amazon Elastic Load Balancing V2 Construct Library

The `aws-cdk-lib/aws-elasticloadbalancingv2` package provides constructs for
configuring application and network load balancers.

For more information, see the AWS documentation for
[Application Load Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
and [Network Load Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/introduction.html).

## Defining an Application Load Balancer

You define an application load balancer by creating an instance of
`ApplicationLoadBalancer`, adding a Listener to the load balancer
and adding Targets to the Listener:

```ts
import { AutoScalingGroup } from 'aws-cdk-lib/aws-autoscaling';
declare const asg: AutoScalingGroup;
declare const vpc: ec2.Vpc;

// Create the load balancer in a VPC. 'internetFacing' is 'false'
// by default, which creates an internal load balancer.
const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true
});

// Add a listener and open up the load balancer's security group
// to the world.
const listener = lb.addListener('Listener', {
  port: 80,

  // 'open: true' is the default, you can leave it out if you want. Set it
  // to 'false' and use `listener.connections` if you want to be selective
  // about who can access the load balancer.
  open: true,
});

// Create an AutoScaling group and add it as a load balancing
// target to the listener.
listener.addTargets('ApplicationFleet', {
  port: 8080,
  targets: [asg]
});
```

The security groups of the load balancer and the target are automatically
updated to allow the network traffic.

> NOTE: If the `@aws-cdk/aws-elasticloadbalancingV2:albDualstackWithoutPublicIpv4SecurityGroupRulesDefault` feature flag is set (the default for new projects), and `addListener()` is called with `open: true`,
the load balancer's security group will automatically include both IPv4 and IPv6 ingress rules when using `IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4`.
>
> For existing projects that only have IPv4 rules, you can opt-in to IPv6 ingress rules
by enabling the feature flag in your cdk.json file. Note that enabling this feature flag
will modify existing security group rules.

One (or more) security groups can be associated with the load balancer;
if a security group isn't provided, one will be automatically created.

```ts
declare const vpc: ec2.Vpc;

const securityGroup1 = new ec2.SecurityGroup(this, 'SecurityGroup1', { vpc });
const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,
  securityGroup: securityGroup1, // Optional - will be automatically created otherwise
});

const securityGroup2 = new ec2.SecurityGroup(this, 'SecurityGroup2', { vpc });
lb.addSecurityGroup(securityGroup2);
```

### Conditions

It's possible to route traffic to targets based on conditions in the incoming
HTTP request. For example, the following will route requests to the indicated
AutoScalingGroup only if the requested host in the request is either for
`example.com/ok` or `example.com/path`:

```ts
declare const listener: elbv2.ApplicationListener;
declare const asg: autoscaling.AutoScalingGroup;

listener.addTargets('Example.Com Fleet', {
  priority: 10,
  conditions: [
    elbv2.ListenerCondition.hostHeaders(['example.com']),
    elbv2.ListenerCondition.pathPatterns(['/ok', '/path']),
  ],
  port: 8080,
  targets: [asg]
});
```

A target with a condition contains either `pathPatterns` or `hostHeader`, or
both. If both are specified, both conditions must be met for the requests to
be routed to the given target. `priority` is a required field when you add
targets with conditions. The lowest number wins.

Every listener must have at least one target without conditions, which is
where all requests that didn't match any of the conditions will be sent.

### Convenience methods and more complex Actions

Routing traffic from a Load Balancer to a Target involves the following steps:

* Create a Target Group, register the Target into the Target Group
* Add an Action to the Listener which forwards traffic to the Target Group.

A new listener can be added to the Load Balancer by calling `addListener()`.
Listeners that have been added to the load balancer can be listed using the
`listeners` property.  Note that the `listeners` property will throw an Error
for imported or looked up Load Balancers.

Various methods on the `Listener` take care of this work for you to a greater
or lesser extent:

* `addTargets()` performs both steps: automatically creates a Target Group and the
  required Action.
* `addTargetGroups()` gives you more control: you create the Target Group (or
  Target Groups) yourself and the method creates Action that routes traffic to
  the Target Groups.
* `addAction()` gives you full control: you supply the Action and wire it up
  to the Target Groups yourself (or access one of the other ELB routing features).

Using `addAction()` gives you access to some of the features of an Elastic Load
Balancer that the other two convenience methods don't:

* **Routing stickiness**: use `ListenerAction.forward()` and supply a
  `stickinessDuration` to make sure requests are routed to the same target group
  for a given duration.
* **Weighted Target Groups**: use `ListenerAction.weightedForward()`
  to give different weights to different target groups.
* **Fixed Responses**: use `ListenerAction.fixedResponse()` to serve
  a static response (ALB only).
* **Redirects**: use `ListenerAction.redirect()` to serve an HTTP
  redirect response (ALB only).
* **Authentication**: use `ListenerAction.authenticateOidc()` to
  perform OpenID authentication before serving a request, or
  `ListenerAction.authenticateJwt()` to verify JSON Web Tokens (JWT)
  for secure service-to-service communications (see the
  `aws-cdk-lib/aws-elasticloadbalancingv2-actions` package for direct authentication
  integration with Cognito) (ALB only).

Here's an example of serving a fixed response at the `/ok` URL:

```ts
declare const listener: elbv2.ApplicationListener;

listener.addAction('Fixed', {
  priority: 10,
  conditions: [
    elbv2.ListenerCondition.pathPatterns(['/ok']),
  ],
  action: elbv2.ListenerAction.fixedResponse(200, {
    contentType: 'text/plain',
    messageBody: 'OK',
  })
});
```

Here's an example of using OIDC authentication before forwarding to a TargetGroup:

```ts
declare const listener: elbv2.ApplicationListener;
declare const myTargetGroup: elbv2.ApplicationTargetGroup;

listener.addAction('DefaultAction', {
  action: elbv2.ListenerAction.authenticateOidc({
    authorizationEndpoint: 'https://example.com/openid',
    // Other OIDC properties here
    clientId: '...',
    clientSecret: SecretValue.secretsManager('...'),
    issuer: '...',
    tokenEndpoint: '...',
    userInfoEndpoint: '...',

    // Next
    next: elbv2.ListenerAction.forward([myTargetGroup]),
  }),
});
```

Here's an example of using JWT authentication for service-to-service communication:

**Note:** JWT authentication requires an HTTPS listener. If you attempt to use it with an HTTP listener, a validation error will be thrown.

```ts
declare const lb: elbv2.ApplicationLoadBalancer;
declare const certificate: elbv2.IListenerCertificate;
declare const myTargetGroup: elbv2.ApplicationTargetGroup;

// JWT authentication requires HTTPS
const listener = lb.addListener('Listener', {
  protocol: elbv2.ApplicationProtocol.HTTPS,
  port: 443,
  certificates: [certificate],
  defaultAction: elbv2.ListenerAction.authenticateJwt({
    issuer: 'https://issuer.example.com',
    jwksEndpoint: 'https://issuer.example.com/.well-known/jwks.json',
    next: elbv2.ListenerAction.forward([myTargetGroup]),
  }),
});
```

If you just want to redirect all incoming traffic on one port to another port, you can use the following code:

```ts
declare const lb: elbv2.ApplicationLoadBalancer;

lb.addRedirect({
  sourceProtocol: elbv2.ApplicationProtocol.HTTPS,
  sourcePort: 8443,
  targetProtocol: elbv2.ApplicationProtocol.HTTP,
  targetPort: 8080,
});
```

If you do not provide any options for this method, it redirects HTTP port 80 to HTTPS port 443.

By default all ingress traffic will be allowed on the source port. If you want to be more selective with your
ingress rules then set `open: false` and use the listener's `connections` object to selectively grant access to the listener.

**Note**: The `path` parameter must start with a `/`.

### Application Load Balancer attributes

You can modify attributes of Application Load Balancers:

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,

  // Whether HTTP/2 is enabled
  http2Enabled: false,

  // The idle timeout value, in seconds
  idleTimeout: Duration.seconds(1000),

  // Whether HTTP headers with header fields that are not valid
  // are removed by the load balancer (true), or routed to targets
  dropInvalidHeaderFields: true,

  // How the load balancer handles requests that might
  // pose a security risk to your application
  desyncMitigationMode: elbv2.DesyncMitigationMode.DEFENSIVE,

  // The type of IP addresses to use.
  ipAddressType: elbv2.IpAddressType.IPV4,

  // The duration of client keep-alive connections
  clientKeepAlive: Duration.seconds(500),

  // Whether cross-zone load balancing is enabled.
  crossZoneEnabled: true,

  // Whether the load balancer blocks traffic through the Internet Gateway (IGW).
  denyAllIgwTraffic: false,

  // Whether to preserve host header in the request to the target
  preserveHostHeader: true,

  // Whether to add the TLS information header to the request
  xAmznTlsVersionAndCipherSuiteHeaders: true,

  // Whether the X-Forwarded-For header should preserve the source port
  preserveXffClientPort: true,

  // The processing mode for X-Forwarded-For headers
  xffHeaderProcessingMode: elbv2.XffHeaderProcessingMode.APPEND,

  // Whether to allow a load balancer to route requests to targets if it is unable to forward the request to AWS WAF.
  wafFailOpen: true,
});
```

For more information, see [Load balancer attributes](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancers.html#load-balancer-attributes)

### Setting up Access Log Bucket on Application Load Balancer

The only server-side encryption option that's supported is Amazon S3-managed keys (SSE-S3). For more information
Documentation: <https://docs.aws.amazon.com/elasticloadbalancing/latest/application/enable-access-logging.html>

```ts

declare const vpc: ec2.Vpc;

const bucket = new s3.Bucket(this, 'ALBAccessLogsBucket',{
  encryption: s3.BucketEncryption.S3_MANAGED,
  });

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', { vpc });
lb.logAccessLogs(bucket);

```

### Setting up Connection Log Bucket on Application Load Balancer

Like access log bucket, the only server-side encryption option that's supported is Amazon S3-managed keys (SSE-S3). For more information
Documentation: <https://docs.aws.amazon.com/elasticloadbalancing/latest/application/enable-connection-logging.html>

```ts
declare const vpc: ec2.Vpc;

const bucket = new s3.Bucket(this, 'ALBConnectionLogsBucket',{
  encryption: s3.BucketEncryption.S3_MANAGED,
});

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', { vpc });
lb.logConnectionLogs(bucket);
```

### Dualstack Application Load Balancer

You can create a dualstack Network Load Balancer using the `ipAddressType` property:

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  ipAddressType: elbv2.IpAddressType.DUAL_STACK,
});
```

By setting `DUAL_STACK_WITHOUT_PUBLIC_IPV4`, you can provision load balancers without public IPv4s:

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,
  ipAddressType: elbv2.IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4,
});
```

### Defining a reserved Application Load Balancer Capacity Unit (LCU)

You can define a [reserved LCU for your Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/capacity-unit-reservation.html).
To reserve an LCU, you must specify a `minimumCapacityUnit`.

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  // Valid value is between 100 and 1500.
  minimumCapacityUnit: 100,
});
```

## Defining a Network Load Balancer

Network Load Balancers are defined in a similar way to Application Load
Balancers:

```ts
declare const vpc: ec2.Vpc;
declare const asg: autoscaling.AutoScalingGroup;

// Create the load balancer in a VPC. 'internetFacing' is 'false'
// by default, which creates an internal load balancer.
const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,
});

// Add a listener on a particular port.
const listener = lb.addListener('Listener', {
  port: 443,
});

// Add targets on a particular port.
listener.addTargets('AppFleet', {
  port: 443,
  targets: [asg]
});
```

### Security Groups for Network Load Balancer

By default, Network Load Balancers (NLB) have a security group associated with them.
This is controlled by the feature flag `@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault`.
When this flag is enabled (the default for new projects), a security group will be automatically created and attached to the NLB unless you explicitly provide your own security groups via the `securityGroups` property.

If you wish to create an NLB without any security groups, you can set the `disableSecurityGroups` property to `true`. When this property is set, no security group will be associated with the NLB, regardless of the feature flag.

```ts
declare const vpc: ec2.IVpc;

const nlb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  // To disable security groups for this NLB
  disableSecurityGroups: true,
});
```

If you want to use your own security groups, provide them via the `securityGroups` property:

```ts
declare const vpc: ec2.IVpc;
declare const sg1: ec2.ISecurityGroup;
declare const sg2: ec2.ISecurityGroup;

const nlb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  // Provide your own security groups
  securityGroups: [sg1],
});
// Add another security group to the NLB
nlb.addSecurityGroup(sg2);
```

### Enforce security group inbound rules on PrivateLink traffic for a Network Load Balancer

You can indicate whether to evaluate inbound security group rules for traffic
sent to a Network Load Balancer through AWS PrivateLink.
The evaluation is enabled by default.

```ts
declare const vpc: ec2.Vpc;

const nlb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  enforceSecurityGroupInboundRulesOnPrivateLinkTraffic: true,
});
```

One thing to keep in mind is that network load balancers do not have security
groups, and no automatic security group configuration is done for you. You will
have to configure the security groups of the target yourself to allow traffic by
clients and/or load balancer instances, depending on your target types.  See
[Target Groups for your Network Load
Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/load-balancer-target-groups.html)
and [Register targets with your Target
Group](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/target-group-register-targets.html)
for more information.

### Dualstack Network Load Balancer

You can create a dualstack Network Load Balancer using the `ipAddressType` property:

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  ipAddressType: elbv2.IpAddressType.DUAL_STACK,
});
```

You can configure whether to use an IPv6 prefix from each subnet for source NAT by setting `enablePrefixForIpv6SourceNat` to `true`.
This must be enabled if you want to create a dualstack Network Load Balancer with a listener that uses UDP protocol.

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  ipAddressType: elbv2.IpAddressType.DUAL_STACK,
  enablePrefixForIpv6SourceNat: true,
});

const listener = lb.addListener('Listener', {
  port: 1229,
  protocol: elbv2.Protocol.UDP,
});
```

### Configuring a subnet information for a Network Load Balancer

You can specify the subnets for a Network Load Balancer easily by setting the `vpcSubnets` property.

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  vpcSubnets: {
    subnetType: ec2.SubnetType.PRIVATE,
  },
});
```

If you want to configure detailed information about the subnets, you can use the `subnetMappings` property as follows:

```ts
declare const vpc: ec2.IVpc;
declare const dualstackVpc: ec2.IVpc;
declare const subnet: ec2.ISubnet;
declare const dualstackSubnet: ec2.ISubnet;
declare const cfnEip: ec2.CfnEIP;

// Internet facing Network Load Balancer with an Elastic IPv4 address
new elbv2.NetworkLoadBalancer(this, 'InternetFacingLb', {
  vpc,
  internetFacing: true,
  subnetMappings: [
    {
      subnet,
      // The allocation ID of the Elastic IP address
      allocationId: cfnEip.attrAllocationId,
    },
  ],
});

// Internal Network Load Balancer with a private IPv4 address
new elbv2.NetworkLoadBalancer(this, 'InternalLb', {
  vpc,
  internetFacing: false,
  subnetMappings: [
    {
      subnet,
      // The private IPv4 address from the subnet
      // The address must be in the subnet's CIDR range and
      // can not be configured for a internet facing Network Load Balancer.
      privateIpv4Address: '10.0.12.29',
    },
  ],
});

// Dualstack Network Load Balancer with an IPv6 address and prefix for source NAT
new elbv2.NetworkLoadBalancer(this, 'DualstackLb', {
  vpc: dualstackVpc,
  // Configure the dualstack Network Load Balancer
  ipAddressType: elbv2.IpAddressType.DUAL_STACK,
  enablePrefixForIpv6SourceNat: true,
  subnetMappings: [
    {
      subnet: dualstackSubnet,
      // The IPv6 address from the subnet
      // `ipAddresstype` must be `DUAL_STACK` or `DUAL_STACK_WITHOUT_PUBLIC_IPV4` to set the IPv6 address.
      ipv6Address: '2001:db8:1234:1a00::10',
      // The IPv6 prefix to use for source NAT
      // `enablePrefixForIpv6SourceNat` must be `true` to set `sourceNatIpv6Prefix`.
      sourceNatIpv6Prefix: elbv2.SourceNatIpv6Prefix.autoAssigned(),
    },
  ],
});
```

### Network Load Balancer attributes

You can modify attributes of Network Load Balancers:

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  // Whether deletion protection is enabled.
  deletionProtection: true,

  // Whether cross-zone load balancing is enabled.
  crossZoneEnabled: true,

  // Whether the load balancer blocks traffic through the Internet Gateway (IGW).
  denyAllIgwTraffic: false,

  // Indicates how traffic is distributed among the load balancer Availability Zones.
  clientRoutingPolicy: elbv2.ClientRoutingPolicy.AVAILABILITY_ZONE_AFFINITY,

  // Indicates whether zonal shift is enabled.
  zonalShift: true,
});
```

### Network Load Balancer Listener attributes

You can modify attributes of Network Load Balancer Listener:

```ts
declare const lb: elbv2.NetworkLoadBalancer;
declare const group: elbv2.NetworkTargetGroup;

const listener = lb.addListener('Listener', {
  port: 80,
  defaultAction: elbv2.NetworkListenerAction.forward([group]),

  // The tcp idle timeout value. The valid range is 60-6000 seconds. The default is 350 seconds.
  tcpIdleTimeout: Duration.seconds(100),
});
```

### Network Load Balancer and EC2 IConnectable interface

Network Load Balancer implements EC2 `IConnectable` and exposes `connections` property. EC2 Connections allows manage the allowed network connections for constructs with Security Groups. This class makes it easy to allow network connections to and from security groups, and between security groups individually. One thing to keep in mind is that network load balancers do not have security groups, and no automatic security group configuration is done for you. You will have to configure the security groups of the target yourself to allow traffic by clients and/or load balancer instances, depending on your target types.

```ts
declare const vpc: ec2.Vpc;
declare const sg1: ec2.ISecurityGroup;
declare const sg2: ec2.ISecurityGroup;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,
  securityGroups: [sg1],
});
lb.addSecurityGroup(sg2);
lb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));
```

### Defining a reserved Network Load Balancer Capacity Unit (LCU)

You can define a [reserved LCU for your Network Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/capacity-unit-reservation.html).

When requesting a LCU reservation, convert your capacity needs from Mbps to LCUs using the conversion rate of 1 LCU to 2.2 Mbps.

To reserve an LCU, you must specify a `minimumCapacityUnit`.

```ts
declare const vpc: ec2.Vpc;

const lb = new elbv2.NetworkLoadBalancer(this, 'LB', {
  vpc,
  minimumCapacityUnit: 5500,
});
```

**Note**: The `minimumCapacityUnit` value is evenly distributed across all active Availability Zones (AZs) for the network load balancer. The distributed value per AZ must be between 2,750 and 45,000 units.

## Targets and Target Groups

Application and Network Load Balancers organize load balancing targets in Target
Groups. If you add your balancing targets (such as AutoScalingGroups, ECS
services or individual instances) to your listener directly, the appropriate
`TargetGroup` will be automatically created for you.

If you need more control over the Target Groups created, create an instance of
`ApplicationTargetGroup` or `NetworkTargetGroup`, add the members you desire,
and add it to the listener by calling `addTargetGroups` instead of `addTargets`.

`addTargets()` will always return the Target Group it just created for you:

```ts
declare const listener: elbv2.NetworkListener;
declare const asg1: autoscaling.AutoScalingGroup;
declare const asg2: autoscaling.AutoScalingGroup;

const group = listener.addTargets('AppFleet', {
  port: 443,
  targets: [asg1],
});

group.addTarget(asg2);
```

### Sticky sessions for your Application Load Balancer

By default, an Application Load Balancer routes each request independently to a registered target based on the chosen load-balancing algorithm. However, you can use the sticky session feature (also known as session affinity) to enable the load balancer to bind a user's session to a specific target. This ensures that all requests from the user during the session are sent to the same target. This feature is useful for servers that maintain state information in order to provide a continuous experience to clients. To use sticky sessions, the client must support cookies.

Application Load Balancers support both duration-based cookies (`lb_cookie`) and application-based cookies (`app_cookie`). The key to managing sticky sessions is determining how long your load balancer should consistently route the user's request to the same target. Sticky sessions are enabled at the target group level. You can use a combination of duration-based stickiness, application-based stickiness, and no stickiness across all of your target groups.

```ts
declare const vpc: ec2.Vpc;

// Target group with duration-based stickiness with load-balancer generated cookie
const tg1 = new elbv2.ApplicationTargetGroup(this, 'TG1', {
  targetType: elbv2.TargetType.INSTANCE,
  port: 80,
  stickinessCookieDuration: Duration.minutes(5),
  vpc,
});

// Target group with application-based stickiness
const tg2 = new elbv2.ApplicationTargetGroup(this, 'TG2', {
  targetType: elbv2.TargetType.INSTANCE,
  port: 80,
  stickinessCookieDuration: Duration.minutes(5),
  stickinessCookieName: 'MyDeliciousCookie',
  vpc,
});
```

### Slow start mode for your Application Load Balancer

By default, a target starts to receive its full share of requests as soon as it is registered with a target group and passes an initial health check. Using slow start mode gives targets time to warm up before the load balancer sends them a full share of requests.

After you enable slow start for a target group, its targets enter slow start mode when they are considered healthy by the target group. A target in slow start mode exits slow start mode when the configured slow start duration period elapses or the target becomes unhealthy. The load balancer linearly increases the number of requests that it can send to a target in slow start mode. After a healthy target exits slow start mode, the load balancer can send it a full share of requests.

The allowed range is 30-900 seconds (15 minutes). The default is 0 seconds (disabled).

```ts
declare const vpc: ec2.Vpc;

// Target group with slow start mode enabled
const tg = new elbv2.ApplicationTargetGroup(this, 'TG', {
  targetType: elbv2.TargetType.INSTANCE,
  slowStart: Duration.seconds(60),
  port: 80,
  vpc,
});
```

For more information see: <https://docs.aws.amazon.com/elasticloadbalancing/latest/application/sticky-sessions.html#application-based-stickiness>

### Setting the target group protocol version

By default, Application Load Balancers send requests to targets using HTTP/1.1. You can use the [protocol version](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#target-group-protocol-version) to send requests to targets using HTTP/2 or gRPC.

```ts
declare const vpc: ec2.Vpc;

const tg = new elbv2.ApplicationTargetGroup(this, 'TG', {
  targetType: elbv2.TargetType.IP,
  port: 50051,
  protocol: elbv2.ApplicationProtocol.HTTP,
  protocolVersion: elbv2.ApplicationProtocolVersion.GRPC,
  healthCheck: {
    enabled: true,
    healthyGrpcCodes: '0-99',
  },
  vpc,
});
```

### Weighted random routing algorithms and automatic target weights for your Application Load Balancer

You can use the `weighted_random` routing algorithms by setting the `loadBalancingAlgorithmType` property.

When using this algorithm, Automatic Target Weights (ATW) anomaly mitigation can be used by setting `enableAnomalyMitigation` to `true`.

Also you can't use this algorithm with slow start mode.

For more information, see [Routing algorithms](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#modify-routing-algorithm) and [Automatic Target Weights (ATW)](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#automatic-target-weights).

```ts
declare const vpc: ec2.Vpc;

const tg = new elbv2.ApplicationTargetGroup(this, 'TargetGroup', {
  vpc,
  loadBalancingAlgorithmType: elbv2.TargetGroupLoadBalancingAlgorithmType.WEIGHTED_RANDOM,
  enableAnomalyMitigation: true,
});
```

### Target Group level cross-zone load balancing setting for Application Load Balancers and Network Load Balancers

You can set cross-zone load balancing setting at the target group level by setting `crossZone` property.

If not specified, it will use the load balancer's configuration.

For more information, see [How Elastic Load Balancing works](https://docs.aws.amazon.com/elasticloadbalancing/latest/userguide/how-elastic-load-balancing-works.html).

```ts
declare const vpc: ec2.Vpc;

const targetGroup = new elbv2.ApplicationTargetGroup(this, 'TargetGroup', {
  vpc,
  port: 80,
  targetType: elbv2.TargetType.INSTANCE,

  // Whether cross zone load balancing is enabled.
  crossZoneEnabled: true,
});
```

### IP Address Type for Target Groups

You can set the IP address type for the target group by setting the `ipAddressType` property for both Application and Network target groups.

If you set the `ipAddressType` property to `IPV6`, the VPC for the target group must have an associated IPv6 CIDR block.

For more information, see IP address type for [Network Load Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/load-balancer-target-groups.html#target-group-ip-address-type) and [Application Load Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#target-group-ip-address-type).

```ts
declare const vpc: ec2.Vpc;

const ipv4ApplicationTargetGroup = new elbv2.ApplicationTargetGroup(this, 'IPv4ApplicationTargetGroup', {
  vpc,
  port: 80,
  targetType: elbv2.TargetType.INSTANCE,
  ipAddressType: elbv2.TargetGroupIpAddressType.IPV4,
});

const ipv6ApplicationTargetGroup = new elbv2.ApplicationTargetGroup(this, 'Ipv6ApplicationTargetGroup', {
  vpc,
  port: 80,
  targetType: elbv2.TargetType.INSTANCE,
  ipAddressType: elbv2.TargetGroupIpAddressType.IPV6,
});

const ipv4NetworkTargetGroup = new elbv2.NetworkTargetGroup(this, 'IPv4NetworkTargetGroup', {
  vpc,
  port: 80,
  targetType: elbv2.TargetType.INSTANCE,
  ipAddressType: elbv2.TargetGroupIpAddressType.IPV4,
});

const ipv6NetworkTargetGroup = new elbv2.NetworkTargetGroup(this, 'Ipv6NetworkTargetGroup', {
  vpc,
  port: 80,
  targetType: elbv2.TargetType.INSTANCE,
  ipAddressType: elbv2.TargetGroupIpAddressType.IPV6,
});
```

### Target Group level health setting for Application Load Balancers and Network Load Balancers

You can set target group health setting at target group level by setting `targetGroupHealth` property.

For more information, see [How Elastic Load Balancing works](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#target-group-attributes).

```ts
declare const vpc: ec2.Vpc;

const targetGroup = new elbv2.ApplicationTargetGroup(this, 'TargetGroup', {
  vpc,
  port: 80,
  targetGroupHealth: {
    dnsMinimumHealthyTargetCount: 3,
    dnsMinimumHealthyTargetPercentage: 70,
    routingMinimumHealthyTargetCount: 2,
    routingMinimumHealthyTargetPercentage: 50,
  },
});
```

## Using Lambda Targets

To use a Lambda Function as a target, use the integration class in the
`aws-cdk-lib/aws-elasticloadbalancingv2-targets` package:

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as targets from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets';

declare const lambdaFunction: lambda.Function;
declare const lb: elbv2.ApplicationLoadBalancer;

const listener = lb.addListener('Listener', { port: 80 });
listener.addTargets('Targets', {
  targets: [new targets.LambdaTarget(lambdaFunction)],

  // For Lambda Targets, you need to explicitly enable health checks if you
  // want them.
  healthCheck: {
    enabled: true,
  }
});
```

Only a single Lambda function can be added to a single listener rule.

### Multi-Value Headers with Lambda Targets

When using a Lambda function as a target, you can enable [multi-value headers](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/lambda-functions.html#multi-value-headers) to allow the load balancer to send headers with multiple values:

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as targets from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets';

declare const vpc: ec2.Vpc;
declare const lambdaFunction: lambda.Function;

// Create a target group with multi-value headers enabled
const targetGroup = new elbv2.ApplicationTargetGroup(this, 'LambdaTargetGroup', {
  vpc,
  targets: [new targets.LambdaTarget(lambdaFunction)],

  // Enable multi-value headers
  targetType: elbv2.TargetType.LAMBDA,
  multiValueHeadersEnabled: true,
});
```

When multi-value headers are enabled, the request and response headers exchanged between the load balancer and the Lambda function include headers with multiple values. If this option is disabled (the default) and the request contains a duplicate header field name, the load balancer uses the last value sent by the client.

## Using Application Load Balancer Targets

To use a single application load balancer as a target for the network load balancer, use the integration class in the
`aws-cdk-lib/aws-elasticloadbalancingv2-targets` package:

```ts
import * as targets from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as patterns from 'aws-cdk-lib/aws-ecs-patterns';

declare const vpc: ec2.Vpc;

const task = new ecs.FargateTaskDefinition(this, 'Task', { cpu: 256, memoryLimitMiB: 512 });
task.addContainer('nginx', {
  image: ecs.ContainerImage.fromRegistry('public.ecr.aws/nginx/nginx:latest'),
  portMappings: [{ containerPort: 80 }],
});

const svc = new patterns.ApplicationLoadBalancedFargateService(this, 'Service', {
  vpc,
  taskDefinition: task,
  publicLoadBalancer: false,
});

const nlb = new elbv2.NetworkLoadBalancer(this, 'Nlb', {
  vpc,
  crossZoneEnabled: true,
  internetFacing: true,
});

const listener = nlb.addListener('listener', { port: 80 });

listener.addTargets('Targets', {
  targets: [new targets.AlbListenerTarget(svc.listener)],
  port: 80,
});

new CfnOutput(this, 'NlbEndpoint', { value: `http://${nlb.loadBalancerDnsName}`})
```

Only the network load balancer is allowed to add the application load balancer as the target.

## Configuring Health Checks

Health checks are configured upon creation of a target group:

```ts
declare const listener: elbv2.ApplicationListener;
declare const asg: autoscaling.AutoScalingGroup;

listener.addTargets('AppFleet', {
  port: 8080,
  targets: [asg],
  healthCheck: {
    path: '/ping',
    interval: Duration.minutes(1),
  }
});
```

The health check can also be configured after creation by calling
`configureHealthCheck()` on the created object.

No attempts are made to configure security groups for the port you're
configuring a health check for, but if the health check is on the same port
you're routing traffic to, the security group already allows the traffic.
If not, you will have to configure the security groups appropriately:

```ts
declare const lb: elbv2.ApplicationLoadBalancer;
declare const listener: elbv2.ApplicationListener;
declare const asg: autoscaling.AutoScalingGroup;

listener.addTargets('AppFleet', {
  port: 8080,
  targets: [asg],
  healthCheck: {
    port: '8088',
  }
});

asg.connections.allowFrom(lb, ec2.Port.tcp(8088));
```

## Using a Load Balancer from a different Stack

If you want to put your Load Balancer and the Targets it is load balancing to in
different stacks, you may not be able to use the convenience methods
`loadBalancer.addListener()` and `listener.addTargets()`.

The reason is that these methods will create resources in the same Stack as the
object they're called on, which may lead to cyclic references between stacks.
Instead, you will have to create an `ApplicationListener` in the target stack,
or an empty `TargetGroup` in the load balancer stack that you attach your
service to.

For an example of the alternatives while load balancing to an ECS service, see the
[ecs/cross-stack-load-balancer
example](https://github.com/aws-samples/aws-cdk-examples/tree/master/typescript/ecs/cross-stack-load-balancer/).

## Protocol for Load Balancer Targets

Constructs that want to be a load balancer target should implement
`IApplicationLoadBalancerTarget` and/or `INetworkLoadBalancerTarget`, and
provide an implementation for the function `attachToXxxTargetGroup()`, which can
call functions on the load balancer and should return metadata about the
load balancing target:

```ts
class MyTarget implements elbv2.IApplicationLoadBalancerTarget {
  public attachToApplicationTargetGroup(targetGroup: elbv2.ApplicationTargetGroup): elbv2.LoadBalancerTargetProps {
    // If we need to add security group rules
    // targetGroup.registerConnectable(...);
    return {
      targetType: elbv2.TargetType.IP,
      targetJson: { id: '1.2.3.4', port: 8080 },
    };
  }
}
```

`targetType` should be one of `Instance` or `Ip`. If the target can be
directly added to the target group, `targetJson` should contain the `id` of
the target (either instance ID or IP address depending on the type) and
optionally a `port` or `availabilityZone` override.

Application load balancer targets can call `registerConnectable()` on the
target group to register themselves for addition to the load balancer's security
group rules.

If your load balancer target requires that the TargetGroup has been
associated with a LoadBalancer before registration can happen (such as is the
case for ECS Services for example), take a resource dependency on
`targetGroup.loadBalancerAttached` as follows:

```ts
declare const resource: Resource;
declare const targetGroup: elbv2.ApplicationTargetGroup;

// Make sure that the listener has been created, and so the TargetGroup
// has been associated with the LoadBalancer, before 'resource' is created.

Node.of(resource).addDependency(targetGroup.loadBalancerAttached);
```

## Looking up Load Balancers and Listeners

You may look up load balancers and load balancer listeners by using one of the
following lookup methods:

* `ApplicationLoadBalancer.fromLookup(options)` - Look up an application load
  balancer.
* `ApplicationListener.fromLookup(options)` - Look up an application load
  balancer listener.
* `NetworkLoadBalancer.fromLookup(options)` - Look up a network load balancer.
* `NetworkListener.fromLookup(options)` - Look up a network load balancer
  listener.

### Load Balancer lookup options

You may look up a load balancer by ARN or by associated tags. When you look a
load balancer up by ARN, that load balancer will be returned unless CDK detects
that the load balancer is of the wrong type. When you look up a load balancer by
tags, CDK will return the load balancer matching all specified tags. If more
than one load balancer matches, CDK will throw an error requesting that you
provide more specific criteria.

**Look up a Application Load Balancer by ARN**

```ts
const loadBalancer = elbv2.ApplicationLoadBalancer.fromLookup(this, 'ALB', {
  loadBalancerArn: 'arn:aws:elasticloadbalancing:us-east-2:123456789012:loadbalancer/app/my-load-balancer/1234567890123456',
});
```

**Look up an Application Load Balancer by tags**

```ts
const loadBalancer = elbv2.ApplicationLoadBalancer.fromLookup(this, 'ALB', {
  loadBalancerTags: {
    // Finds a load balancer matching all tags.
    some: 'tag',
    someother: 'tag',
  },
});
```

## Load Balancer Listener lookup options

You may look up a load balancer listener by the following criteria:

* Associated load balancer ARN
* Associated load balancer tags
* Listener ARN
* Listener port
* Listener protocol

The lookup method will return the matching listener. If more than one listener
matches, CDK will throw an error requesting that you specify additional
criteria.

**Look up a Listener by associated Load Balancer, Port, and Protocol**

```ts
const listener = elbv2.ApplicationListener.fromLookup(this, 'ALBListener', {
  loadBalancerArn: 'arn:aws:elasticloadbalancing:us-east-2:123456789012:loadbalancer/app/my-load-balancer/1234567890123456',
  listenerProtocol: elbv2.ApplicationProtocol.HTTPS,
  listenerPort: 443,
});
```

**Look up a Listener by associated Load Balancer Tag, Port, and Protocol**

```ts
const listener = elbv2.ApplicationListener.fromLookup(this, 'ALBListener', {
  loadBalancerTags: {
    Cluster: 'MyClusterName',
  },
  listenerProtocol: elbv2.ApplicationProtocol.HTTPS,
  listenerPort: 443,
});
```

**Look up a Network Listener by associated Load Balancer Tag, Port, and Protocol**

```ts
const listener = elbv2.NetworkListener.fromLookup(this, 'ALBListener', {
  loadBalancerTags: {
    Cluster: 'MyClusterName',
  },
  listenerProtocol: elbv2.Protocol.TCP,
  listenerPort: 12345,
});
```

## Metrics

You may create metrics for Load Balancers and Target Groups through the `metrics` attribute:

**Load Balancer:**

```ts
declare const alb: elbv2.IApplicationLoadBalancer;

const albMetrics: elbv2.IApplicationLoadBalancerMetrics = alb.metrics;
const metricConnectionCount: cloudwatch.Metric = albMetrics.activeConnectionCount();
```

**Target Group:**

```ts
declare const targetGroup: elbv2.IApplicationTargetGroup;

const targetGroupMetrics: elbv2.IApplicationTargetGroupMetrics = targetGroup.metrics;
const metricHealthyHostCount: cloudwatch.Metric = targetGroupMetrics.healthyHostCount();
```

Metrics are also available to imported resources:

```ts
declare const stack: Stack;

const targetGroup = elbv2.ApplicationTargetGroup.fromTargetGroupAttributes(this, 'MyTargetGroup', {
  targetGroupArn: Fn.importValue('TargetGroupArn'),
  loadBalancerArns: Fn.importValue('LoadBalancerArn'),
});

const targetGroupMetrics: elbv2.IApplicationTargetGroupMetrics = targetGroup.metrics;
```

Notice that TargetGroups must be imported by supplying the Load Balancer too, otherwise accessing the `metrics` will
throw an error:

```ts
declare const stack: Stack;
const targetGroup = elbv2.ApplicationTargetGroup.fromTargetGroupAttributes(this, 'MyTargetGroup', {
  targetGroupArn: Fn.importValue('TargetGroupArn'),
});

const targetGroupMetrics: elbv2.IApplicationTargetGroupMetrics = targetGroup.metrics; // throws an Error()
```

## logicalIds on ExternalApplicationListener.addTargetGroups() and .addAction()

By default, the `addTargetGroups()` method does not follow the standard behavior
of adding a `Rule` suffix to the logicalId of the `ListenerRule` it creates.
If you are deploying new `ListenerRule`s using `addTargetGroups()` the recommendation
is to set the `removeRuleSuffixFromLogicalId: false` property.
If you have `ListenerRule`s deployed using the legacy behavior of `addTargetGroups()`,
which you need to switch over to being managed by the `addAction()` method,
then you will need to enable the `removeRuleSuffixFromLogicalId: true` property in the `addAction()` method.

`ListenerRule`s have a unique `priority` for a given `Listener`.
Because the `priority` must be unique, CloudFormation will always fail when creating a new `ListenerRule` to replace the existing one, unless you change the `priority` as well as the logicalId.

## Configuring Mutual authentication with TLS in Application Load Balancer

You can configure Mutual authentication with TLS (mTLS) for Application Load Balancer.

To set mTLS, you must create an instance of `TrustStore` and set it to `ApplicationListener`.

For more information, see [Mutual authentication with TLS in Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/mutual-authentication.html)

```ts
import * as acm from 'aws-cdk-lib/aws-certificatemanager';

declare const certificate: acm.Certificate;
declare const lb: elbv2.ApplicationLoadBalancer;
declare const bucket: s3.Bucket;

const trustStore = new elbv2.TrustStore(this, 'Store', {
  bucket,
  key: 'rootCA_cert.pem',
});

lb.addListener('Listener', {
  port: 443,
  protocol: elbv2.ApplicationProtocol.HTTPS,
  certificates: [certificate],
  // mTLS settings
  mutualAuthentication: {
    advertiseTrustStoreCaNames: true,
    ignoreClientCertificateExpiry: false,
    mutualAuthenticationMode: elbv2.MutualAuthenticationMode.VERIFY,
    trustStore,
  },
  defaultAction: elbv2.ListenerAction.fixedResponse(200,
    { contentType: 'text/plain', messageBody: 'Success mTLS' }),
});
```

Optionally, you can create a certificate revocation list for a trust store by creating an instance of `TrustStoreRevocation`.

```ts
declare const trustStore: elbv2.TrustStore;
declare const bucket: s3.Bucket;

new elbv2.TrustStoreRevocation(this, 'Revocation', {
  trustStore,
  revocationContents: [
    {
      revocationType: elbv2.RevocationType.CRL,
      bucket,
      key: 'crl.pem',
    },
  ],
});
```
