import { Match, Template } from '../../../assertions';
import * as ec2 from '../../../aws-ec2';
import { IpAddressType } from '../../../aws-opensearchservice/lib/domain';
import * as route53 from '../../../aws-route53';
import * as s3 from '../../../aws-s3';
import * as cdk from '../../../core';
import * as elbv2 from '../../lib';

describe('tests', () => {
  describe('subnet mappings', () => {
    test('set subnet', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC', { maxAzs: 2 });

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        subnetMappings: [
          { subnet: vpc.publicSubnets[0] },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        SubnetMappings: [
          {
            SubnetId: { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
          },
        ],
      });
    });

    test('set allocation id', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC', { maxAzs: 2 });
      const elasticIp = new ec2.CfnEIP(stack, 'EIP');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        internetFacing: true,
        subnetMappings: [
          {
            subnet: vpc.publicSubnets[0],
            allocationId: elasticIp.attrAllocationId,
          },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        SubnetMappings: [
          {
            SubnetId: { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
            AllocationId: { 'Fn::GetAtt': ['EIP', 'AllocationId'] },
          },
        ],
      });
    });

    test('set private ipv4 address', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC', {
        maxAzs: 2,
        ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      });

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        internetFacing: false,
        subnetMappings: [
          {
            subnet: vpc.publicSubnets[0],
            privateIpv4Address: '10.0.12.29',
          },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        SubnetMappings: [
          {
            SubnetId: { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
            PrivateIPv4Address: '10.0.12.29',
          },
        ],
      });
    });

    test('set ipv6 address', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC', {
        maxAzs: 2,
        ipProtocol: ec2.IpProtocol.DUAL_STACK,
      });

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        ipAddressType: IpAddressType.DUAL_STACK,
        enablePrefixForIpv6SourceNat: true,
        subnetMappings: [
          {
            subnet: vpc.publicSubnets[0],
            ipv6Address: 'fd00:1234:5678:abcd::1',
            sourceNatIpv6Prefix: elbv2.SourceNatIpv6Prefix.autoAssigned(),
          },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        SubnetMappings: [
          {
            SubnetId: { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
            IPv6Address: 'fd00:1234:5678:abcd::1',
            SourceNatIpv6Prefix: 'auto_assigned',
          },
        ],
      });
    });

    test('throw error for configuring both vpcSubnets and subnetMappings', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // THEN
      expect(() => {
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
          subnetMappings: [
            { subnet: vpc.publicSubnets[0] },
          ],
        });
      }).toThrow('You can specify either `vpcSubnets` or `subnetMappings`, not both.');
    });

    test('throw error for configuring private ipv4 address for internet-facing load balancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // THEN
      expect(() => {
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          subnetMappings: [
            {
              subnet: vpc.publicSubnets[0],
              privateIpv4Address: '10.0.12.29',
            },
          ],
        });
      }).toThrow('Cannot specify `privateIpv4Address` for a internet facing load balancer.');
    });

    test('throw error for configuring allocation id for internal load balancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const elasticIp = new ec2.CfnEIP(stack, 'EIP');

      // THEN
      expect(() => {
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: false,
          subnetMappings: [
            {
              subnet: vpc.publicSubnets[0],
              allocationId: elasticIp.attrAllocationId,
            },
          ],
        });
      }).toThrow('Cannot specify `allocationId` for a internal load balancer.');
    });

    test('throw error for configuring source nat ipv6 prefix for a load balancer that does not have `enablePrefixForIpv6SourceNat` enabled', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC', {
        maxAzs: 2,
        ipProtocol: ec2.IpProtocol.DUAL_STACK,
      });

      // THEN
      expect(() => {
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          subnetMappings: [
            {
              subnet: vpc.publicSubnets[0],
              sourceNatIpv6Prefix: elbv2.SourceNatIpv6Prefix.autoAssigned(),
            },
          ],
        });
      }).toThrow('Cannot specify `sourceNatIpv6Prefix` for a load balancer that does not have `enablePrefixForIpv6SourceNat` enabled.');
    });
  });

  test('specify minimum capacity unit', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      minimumCapacityUnit: 5500,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      MinimumLoadBalancerCapacity: {
        CapacityUnits: 5500,
      },
    });
  });

  test.each([-1, 2750, 5499, 10000.1, 90001])('throw error for invalid range minimum capacity unit', (minimumCapacityUnit) => {
    // GIVEN
    const stack = new cdk.Stack();
    // two AZs
    const vpc = new ec2.Vpc(stack, 'VPC');
    const capacityUnitPerAz = minimumCapacityUnit / vpc.availabilityZones.length;

    // THEN
    expect(() => {
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        minimumCapacityUnit,
      });
    }).toThrow(`'minimumCapacityUnit' must be a positive value between 2750 and 45000 per AZ for Network Load Balancer, got ${capacityUnitPerAz} LCU per AZ.`);
  });

  test('Trivial construction: internet facing', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internet-facing',
      Subnets: [
        { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
        { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
      ],
      Type: 'network',
    });
  });

  test('Trivial construction: internal', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', { vpc });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'StackPrivateSubnet1Subnet47AC2BC7' },
        { Ref: 'StackPrivateSubnet2SubnetA2F8EDD8' },
      ],
      Type: 'network',
    });
  });

  test('VpcEndpointService with Domain Name imported from public hosted zone', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const nlb = new elbv2.NetworkLoadBalancer(stack, 'Nlb', { vpc });
    const endpointService = new ec2.VpcEndpointService(stack, 'EndpointService', { vpcEndpointServiceLoadBalancers: [nlb] });

    // WHEN
    const importedPHZ = route53.PublicHostedZone.fromPublicHostedZoneAttributes(stack, 'MyPHZ', {
      hostedZoneId: 'sampleid',
      zoneName: 'MyZone',
    });
    new route53.VpcEndpointServiceDomainName(stack, 'EndpointServiceDomainName', {
      endpointService,
      domainName: 'MyDomain',
      publicHostedZone: importedPHZ,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Route53::RecordSet', {
      HostedZoneId: 'sampleid',
    });
  });

  test('Attributes', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      crossZoneEnabled: true,
      clientRoutingPolicy: elbv2.ClientRoutingPolicy.PARTIAL_AVAILABILITY_ZONE_AFFINITY,
      zonalShift: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: Match.arrayWith([
        {
          Key: 'load_balancing.cross_zone.enabled',
          Value: 'true',
        },
        {
          Key: 'dns_record.client_routing_policy',
          Value: 'partial_availability_zone_affinity',
        },
        {
          Key: 'zonal_shift.config.enabled',
          Value: 'true',
        },
      ]),
    });
  });

  test('Access logging', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, undefined, { env: { region: 'us-east-1' } });
    const vpc = new ec2.Vpc(stack, 'Stack');
    const bucket = new s3.Bucket(stack, 'AccessLoggingBucket');
    const lb = new elbv2.NetworkLoadBalancer(stack, 'LB', { vpc });

    // WHEN
    lb.logAccessLogs(bucket);

    // THEN

    // verify that the LB attributes reference the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: Match.arrayWith([
        {
          Key: 'access_logs.s3.enabled',
          Value: 'true',
        },
        {
          Key: 'access_logs.s3.bucket',
          Value: { Ref: 'AccessLoggingBucketA6D88F29' },
        },
        {
          Key: 'access_logs.s3.prefix',
          Value: '',
        },
      ]),
    });

    // verify the bucket policy allows the NLB to put objects in the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 's3:PutObject',
            Effect: 'Allow',
            Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
            Resource: {
              'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'] }, '/AWSLogs/',
                { Ref: 'AWS::AccountId' }, '/*']],
            },
          },
          {
            Action: 's3:PutObject',
            Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'] }, '/AWSLogs/',
                { Ref: 'AWS::AccountId' }, '/*']],
            },
          },
          {
            Action: 's3:GetBucketAcl',
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'],
            },
          },
        ],
      },
    });

    // verify the NLB depends on the bucket policy
    Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      DependsOn: ['AccessLoggingBucketPolicy700D7CC6'],
    });
  });

  test('access logging with prefix', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, undefined, { env: { region: 'us-east-1' } });
    const vpc = new ec2.Vpc(stack, 'Stack');
    const bucket = new s3.Bucket(stack, 'AccessLoggingBucket');
    const lb = new elbv2.NetworkLoadBalancer(stack, 'LB', { vpc });

    // WHEN
    lb.logAccessLogs(bucket, 'prefix-of-access-logs');

    // THEN
    // verify that the LB attributes reference the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: Match.arrayWith([
        {
          Key: 'access_logs.s3.enabled',
          Value: 'true',
        },
        {
          Key: 'access_logs.s3.bucket',
          Value: { Ref: 'AccessLoggingBucketA6D88F29' },
        },
        {
          Key: 'access_logs.s3.prefix',
          Value: 'prefix-of-access-logs',
        },
      ]),
    });

    // verify the bucket policy allows the NLB to put objects in the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 's3:PutObject',
            Effect: 'Allow',
            Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
            Resource: {
              'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'] }, '/prefix-of-access-logs/AWSLogs/',
                { Ref: 'AWS::AccountId' }, '/*']],
            },
          },
          {
            Action: 's3:PutObject',
            Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'] }, '/prefix-of-access-logs/AWSLogs/',
                { Ref: 'AWS::AccountId' }, '/*']],
            },
          },
          {
            Action: 's3:GetBucketAcl',
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::GetAtt': ['AccessLoggingBucketA6D88F29', 'Arn'],
            },
          },
        ],
      },
    });
  });

  test('Access logging on imported bucket', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, undefined, { env: { region: 'us-east-1' } });
    const vpc = new ec2.Vpc(stack, 'Stack');
    const bucket = s3.Bucket.fromBucketName(stack, 'ImportedAccessLoggingBucket', 'imported-bucket');
    // Imported buckets have `autoCreatePolicy` disabled by default
    bucket.policy = new s3.BucketPolicy(stack, 'ImportedAccessLoggingBucketPolicy', {
      bucket,
    });
    const lb = new elbv2.NetworkLoadBalancer(stack, 'LB', { vpc });

    // WHEN
    lb.logAccessLogs(bucket);

    // THEN

    // verify that the LB attributes reference the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: Match.arrayWith([
        {
          Key: 'access_logs.s3.enabled',
          Value: 'true',
        },
        {
          Key: 'access_logs.s3.bucket',
          Value: 'imported-bucket',
        },
        {
          Key: 'access_logs.s3.prefix',
          Value: '',
        },
      ]),
    });

    // verify the bucket policy allows the NLB to put objects in the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 's3:PutObject',
            Effect: 'Allow',
            Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':s3:::imported-bucket/AWSLogs/',
                  { Ref: 'AWS::AccountId' },
                  '/*',
                ],
              ],
            },
          },
          {
            Action: 's3:PutObject',
            Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':s3:::imported-bucket/AWSLogs/',
                  { Ref: 'AWS::AccountId' },
                  '/*',
                ],
              ],
            },
          },
          {
            Action: 's3:GetBucketAcl',
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':s3:::imported-bucket',
                ],
              ],
            },
          },
        ],
      },
    });

    // verify the NLB depends on the bucket policy
    Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      DependsOn: ['ImportedAccessLoggingBucketPolicy97AE3371'],
    });
  });

  test('loadBalancerName', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'myLoadBalancer',
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Name: 'myLoadBalancer',
    });
  });

  test('can set EnforceSecurityGroupInboundRulesOnPrivateLinkTraffic on', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'myLoadBalancer',
      enforceSecurityGroupInboundRulesOnPrivateLinkTraffic: true,
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Name: 'myLoadBalancer',
      EnforceSecurityGroupInboundRulesOnPrivateLinkTraffic: 'on',
    });
  });

  test('can set EnforceSecurityGroupInboundRulesOnPrivateLinkTraffic off', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'myLoadBalancer',
      enforceSecurityGroupInboundRulesOnPrivateLinkTraffic: false,
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Name: 'myLoadBalancer',
      EnforceSecurityGroupInboundRulesOnPrivateLinkTraffic: 'off',
    });
  });

  test('loadBalancerName unallowed: more than 32 characters', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'a'.repeat(33),
      vpc,
    });

    // THEN
    expect(() => {
      app.synth();
    }).toThrow('Load balancer name: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" can have a maximum of 32 characters.');
  });

  test('loadBalancerName unallowed: starts with "internal-"', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'internal-myLoadBalancer',
      vpc,
    });

    // THEN
    expect(() => {
      app.synth();
    }).toThrow('Load balancer name: "internal-myLoadBalancer" must not begin with "internal-".');
  });

  test('loadBalancerName unallowed: starts with hyphen', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: '-myLoadBalancer',
      vpc,
    });

    // THEN
    expect(() => {
      app.synth();
    }).toThrow('Load balancer name: "-myLoadBalancer" must not begin or end with a hyphen.');
  });

  test('loadBalancerName unallowed: ends with hyphen', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'myLoadBalancer-',
      vpc,
    });

    // THEN
    expect(() => {
      app.synth();
    }).toThrow('Load balancer name: "myLoadBalancer-" must not begin or end with a hyphen.');
  });

  test('loadBalancerName unallowed: unallowed characters', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'NLB', {
      loadBalancerName: 'my load balancer',
      vpc,
    });

    // THEN
    expect(() => {
      app.synth();
    }).toThrow('Load balancer name: "my load balancer" must contain only alphanumeric characters or hyphens.');
  });

  test.each([
    [false, undefined],
    [true, undefined],
    [false, elbv2.IpAddressType.IPV4],
    [true, elbv2.IpAddressType.IPV4],
  ])('throw error for denyAllIgwTraffic set to %s for Ipv4 (default) addressing.', (denyAllIgwTraffic, ipAddressType) => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // THEN
    expect(() => {
      new elbv2.NetworkLoadBalancer(stack, 'NLB', {
        vpc,
        denyAllIgwTraffic: denyAllIgwTraffic,
        ipAddressType: ipAddressType,
      });
    }).toThrow(`'denyAllIgwTraffic' may only be set on load balancers with ${elbv2.IpAddressType.DUAL_STACK} addressing.`);
  });

  test('imported network load balancer with no vpc specified throws error when calling addTargets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const nlbArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const nlb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
      loadBalancerArn: nlbArn,
    });
    // WHEN
    const listener = nlb.addListener('Listener', { port: 80 });
    expect(() => listener.addTargets('targetgroup', { port: 8080 })).toThrow();
  });

  test('imported network load balancer with vpc does not throw error when calling addTargets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const nlbArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const nlb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
      loadBalancerArn: nlbArn,
      vpc,
    });
    // WHEN
    const listener = nlb.addListener('Listener', { port: 80 });
    expect(() => listener.addTargets('targetgroup', { port: 8080 })).not.toThrow();
  });

  test('imported load balancer knows its region', () => {
    const stack = new cdk.Stack();

    // WHEN
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const alb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
      loadBalancerArn: albArn,
    });

    // THEN
    expect(alb.env.region).toEqual('us-west-2');
  });

  test('imported load balancer can have metrics', () => {
    const stack = new cdk.Stack();

    // WHEN
    const arn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/network/my-load-balancer/50dc6c495c0c9188';
    const nlb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
      loadBalancerArn: arn,
    });

    const metric = nlb.metrics.custom('MetricName');

    // THEN
    expect(metric.namespace).toEqual('AWS/NetworkELB');
    expect(stack.resolve(metric.dimensions)).toEqual({
      LoadBalancer: 'network/my-load-balancer/50dc6c495c0c9188',
    });
  });

  test('Trivial construction: internal with Isolated subnets only', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC', {
      subnetConfiguration: [{
        cidrMask: 20,
        name: 'Isolated',
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      }],
    });

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'VPCIsolatedSubnet1SubnetEBD00FC6' },
        { Ref: 'VPCIsolatedSubnet2Subnet4B1C8CAA' },
      ],
      Type: 'network',
    });
  });
  test('Internal with Public, Private, and Isolated subnets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC', {
      subnetConfiguration: [{
        cidrMask: 24,
        name: 'Public',
        subnetType: ec2.SubnetType.PUBLIC,
      }, {
        cidrMask: 24,
        name: 'Private',
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      }, {
        cidrMask: 28,
        name: 'Isolated',
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      }],
    });

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'VPCPrivateSubnet1Subnet8BCA10E0' },
        { Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A' },
      ],
      Type: 'network',
    });
  });
  test('Internet-facing with Public, Private, and Isolated subnets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC', {
      subnetConfiguration: [{
        cidrMask: 24,
        name: 'Public',
        subnetType: ec2.SubnetType.PUBLIC,
      }, {
        cidrMask: 24,
        name: 'Private',
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      }, {
        cidrMask: 28,
        name: 'Isolated',
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      }],
    });

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internet-facing',
      Subnets: [
        { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
        { Ref: 'VPCPublicSubnet2Subnet74179F39' },
      ],
      Type: 'network',
    });
  });
  test('Internal load balancer supplying public subnets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: false,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'VPCPublicSubnet1SubnetB4246D30' },
        { Ref: 'VPCPublicSubnet2Subnet74179F39' },
      ],
      Type: 'network',
    });
  });
  test('Internal load balancer supplying isolated subnets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC', {
      subnetConfiguration: [{
        cidrMask: 24,
        name: 'Public',
        subnetType: ec2.SubnetType.PUBLIC,
      }, {
        cidrMask: 24,
        name: 'Private',
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      }, {
        cidrMask: 28,
        name: 'Isolated',
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      }],
    });

    // WHEN
    new elbv2.NetworkLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: false,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'VPCIsolatedSubnet1SubnetEBD00FC6' },
        { Ref: 'VPCIsolatedSubnet2Subnet4B1C8CAA' },
      ],
      Type: 'network',
    });
  });

  describe('security groups', () => {
    describe('`@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault` is disabled', () => {
      let app: cdk.App;
      beforeEach(() => {
        app = new cdk.App({
          context: {
            '@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault': false,
          },
        });
      });

      test('Trivial construction: security groups', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');
        const sg1 = new ec2.SecurityGroup(stack, 'SG1', { vpc });
        const sg2 = new ec2.SecurityGroup(stack, 'SG2', { vpc });

        // WHEN
        const nlb = new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          securityGroups: [sg1],
        });
        nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));
        nlb.addSecurityGroup(sg2);

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          Subnets: [
            { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
            { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
          ],
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg1.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg2.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
          ],
          Type: 'network',
        });
        template.resourcePropertiesCountIs('AWS::EC2::SecurityGroup', {
          SecurityGroupIngress: [
            {
              CidrIp: '0.0.0.0/0',
              Description: 'from 0.0.0.0/0:80',
              FromPort: 80,
              IpProtocol: 'tcp',
              ToPort: 80,
            },
          ],
        }, 2);
      });

      test('Trivial construction: no security groups', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        const nlb = new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
        });
        nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          Subnets: [
            { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
            { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
          ],
          SecurityGroups: Match.absent(),
        });
        template.resourceCountIs('AWS::EC2::SecurityGroup', 0);
        expect(nlb.securityGroups).toBeUndefined();
      });

      test('Trivial construction: empty security groups', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        const nlb = new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          securityGroups: [],
        });
        nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          Subnets: [
            { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
            { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
          ],
          SecurityGroups: [],
        });
        template.resourceCountIs('AWS::EC2::SecurityGroup', 0);
        expect(nlb.securityGroups).toStrictEqual([]);
      });

      test('Can add a security groups from no security groups', () =>{
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');
        const sg1 = new ec2.SecurityGroup(stack, 'SG1', { vpc });
        const sg2 = new ec2.SecurityGroup(stack, 'SG2', { vpc });

        // WHEN
        const nlb = new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
        });
        nlb.addSecurityGroup(sg1);
        nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));
        nlb.addSecurityGroup(sg2);

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          Subnets: [
            { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
            { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
          ],
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg1.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg2.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
          ],
          Type: 'network',
        });
        template.resourcePropertiesCountIs('AWS::EC2::SecurityGroup', {
          SecurityGroupIngress: [
            {
              CidrIp: '0.0.0.0/0',
              Description: 'from 0.0.0.0/0:80',
              FromPort: 80,
              IpProtocol: 'tcp',
              ToPort: 80,
            },
          ],
        }, 2);
      });
    });

    describe('`@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault` is enabled', () => {
      let app: cdk.App;
      beforeEach(() => {
        app = new cdk.App({
          postCliContext: {
            '@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault': true,
          },
        });
      });

      test('creates NLB with auto-generated security group', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
        });

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('LBSecurityGroup.*'),
                'GroupId',
              ],
            },
          ],
          Type: 'network',
        });

        template.hasResourceProperties('AWS::EC2::SecurityGroup', {
          GroupDescription: Match.stringLikeRegexp('Automatically created Security Group for ELB.*'),
          VpcId: { Ref: Match.stringLikeRegexp('Stack.*') },
          SecurityGroupEgress: [{
            CidrIp: '255.255.255.255/32',
            Description: 'Disallow all traffic',
            FromPort: 252,
            IpProtocol: 'icmp',
            ToPort: 86,
          }],
        });
      });

      test('uses provided security groups instead of auto-generated', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');
        const sg1 = new ec2.SecurityGroup(stack, 'SG1', { vpc });
        const sg2 = new ec2.SecurityGroup(stack, 'SG2', { vpc });

        // WHEN
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          securityGroups: [sg1, sg2],
        });

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg1.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
            {
              'Fn::GetAtt': [
                stack.getLogicalId(sg2.node.findChild('Resource') as cdk.CfnElement),
                'GroupId',
              ],
            },
          ],
          Type: 'network',
        });
      });

      test('Empty security groups array: creates NLB with empty security groups', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          securityGroups: [],
        });

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          SecurityGroups: [],
          Type: 'network',
        });
      });

      test('disableSecurityGroups: true - creates NLB without security groups', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          disableSecurityGroups: true,
        });

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          SecurityGroups: Match.absent(),
          Type: 'network',
        });
      });

      test('disableSecurityGroups: false - creates NLB with auto-generated security group (explicit)', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');

        // WHEN
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          disableSecurityGroups: false,
        });

        // THEN
        const template = Template.fromStack(stack);
        template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
          Scheme: 'internet-facing',
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('LBSecurityGroup.*'),
                'GroupId',
              ],
            },
          ],
          Type: 'network',
        });
      });

      test('throw error for disableSecurityGroups set to true and security groups provided', () => {
        // GIVEN
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'Stack');
        const sg1 = new ec2.SecurityGroup(stack, 'SG1', { vpc });

        // THEN
        expect(() => {
          new elbv2.NetworkLoadBalancer(stack, 'LB', {
            vpc,
            internetFacing: true,
            securityGroups: [sg1],
            disableSecurityGroups: true,
          });
        }).toThrow('Cannot specify both `securityGroups` and `disableSecurityGroups` properties.');
      });
    });
  });

  describe('lookup', () => {
    test('Can look up a NetworkLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      // WHEN
      const loadBalancer = elbv2.NetworkLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::NetworkLoadBalancer', 0);
      expect(loadBalancer.loadBalancerArn).toEqual('arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/network/my-load-balancer/50dc6c495c0c9188');
      expect(loadBalancer.loadBalancerCanonicalHostedZoneId).toEqual('Z3DZXE0EXAMPLE');
      expect(loadBalancer.loadBalancerDnsName).toEqual('my-load-balancer-123456789012.us-west-2.elb.amazonaws.com');
      expect(loadBalancer.env.region).toEqual('us-west-2');
    });

    test('Can add listeners to a looked-up NetworkLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      const loadBalancer = elbv2.NetworkLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      const targetGroup = new elbv2.NetworkTargetGroup(stack, 'tg', {
        vpc: loadBalancer.vpc,
        port: 3000,
      });

      // WHEN
      loadBalancer.addListener('listener', {
        protocol: elbv2.Protocol.TCP,
        port: 3000,
        defaultAction: elbv2.NetworkListenerAction.forward([targetGroup]),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::NetworkLoadBalancer', 0);
      Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::Listener', 1);
    });
    test('Can create metrics from a looked-up NetworkLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      const loadBalancer = elbv2.NetworkLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      // WHEN
      const metric = loadBalancer.metrics.custom('MetricName');

      // THEN
      expect(metric.namespace).toEqual('AWS/NetworkELB');
      expect(stack.resolve(metric.dimensions)).toEqual({
        LoadBalancer: 'network/my-load-balancer/50dc6c495c0c9188',
      });
    });

    test('can look up security groups', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      // WHEN
      const nlb = elbv2.NetworkLoadBalancer.fromLookup(stack, 'LB', {
        loadBalancerTags: {
          some: 'tag',
        },
      });
      nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(80));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        CidrIp: '0.0.0.0/0',
        Description: 'from 0.0.0.0/0:80',
        FromPort: 80,
        // ID of looked-up security group is dummy value (defined by ec2.SecurityGroup.fromLookupAttributes)
        GroupId: 'sg-12345678',
        IpProtocol: 'tcp',
        ToPort: 80,
      });
      // IDs of looked-up nlb security groups are dummy values (defined by elbv2.BaseLoadBalancer._queryContextProvider)
      expect(nlb.securityGroups).toEqual(['sg-1234']);
    });
  });

  // test cases for crossZoneEnabled
  describe('crossZoneEnabled', () => {
    test('crossZoneEnabled can be true', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'nlb', {
        vpc,
        crossZoneEnabled: true,
      });
      const t = Template.fromStack(stack);
      t.resourceCountIs('AWS::ElasticLoadBalancingV2::LoadBalancer', 1);
      t.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'load_balancing.cross_zone.enabled',
            Value: 'true',
          },
        ],
      });
    });
    test('crossZoneEnabled can be false', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'nlb', {
        vpc,
        crossZoneEnabled: false,
      });
      const t = Template.fromStack(stack);
      t.resourceCountIs('AWS::ElasticLoadBalancingV2::LoadBalancer', 1);
      t.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'load_balancing.cross_zone.enabled',
            Value: 'false',
          },
        ],
      });
    });
    test('crossZoneEnabled can be undefined', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'nlb', {
        vpc,
      });
      const t = Template.fromStack(stack);
      t.resourceCountIs('AWS::ElasticLoadBalancingV2::LoadBalancer', 1);
      t.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
        ],
      });
    });
  });
  describe('dualstack', () => {
    test('Can create internet-facing dualstack NetworkLoadBalancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        internetFacing: true,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internet-facing',
        Type: 'network',
        IpAddressType: 'dualstack',
      });
    });

    test('Can create internet-facing dualstack NetworkLoadBalancer with denyAllIgwTraffic set to false', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        denyAllIgwTraffic: false,
        internetFacing: true,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internet-facing',
        Type: 'network',
        IpAddressType: 'dualstack',
      });
    });

    test.each([undefined, false])('Can create internal dualstack NetworkLoadBalancer with denyAllIgwTraffic set to true', (internetFacing) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'LB', {
        vpc,
        denyAllIgwTraffic: true,
        internetFacing: internetFacing,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internal',
        Type: 'network',
        IpAddressType: 'dualstack',
      });
    });
  });

  describe('enable prefix for ipv6 source nat', () => {
    test.each([
      { config: true, value: 'on' },
      { config: false, value: 'off' },
    ])('specify EnablePrefixForIpv6SourceNat', ({ config, value }) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.NetworkLoadBalancer(stack, 'Lb', {
        vpc,
        enablePrefixForIpv6SourceNat: config,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internal',
        Type: 'network',
        IpAddressType: 'dualstack',
        EnablePrefixForIpv6SourceNat: value,
      });
    });

    test.each([false, undefined])('throw error for disabling `enablePrefixForIpv6SourceNat` and add UDP listener', (enablePrefixForIpv6SourceNat) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');
      const lb = new elbv2.NetworkLoadBalancer(stack, 'Lb', {
        vpc,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
        enablePrefixForIpv6SourceNat,
      });

      // THEN
      expect(() => {
        lb.addListener('Listener', {
          port: 80,
          protocol: elbv2.Protocol.UDP,
          defaultTargetGroups: [new elbv2.NetworkTargetGroup(stack, 'Group', { vpc, port: 80 })],
        });
      }).toThrow('To add a listener with UDP protocol to a dual stack NLB, \'enablePrefixForIpv6SourceNat\' must be set to true.');
    });
  });

  describe('dualstack without public ipv4', () => {
    test('Throws when creating a dualstack without public ipv4 and a NetworkLoadBalancer', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      expect(() => {
        new elbv2.NetworkLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: true,
          ipAddressType: elbv2.IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4,
        });
      }).toThrow('\'ipAddressType\' DUAL_STACK_WITHOUT_PUBLIC_IPV4 can only be used with Application Load Balancer, got network');
    });
  });

  describe('imported NLB lookup patterns', () => {
    test('imported NLB with hardcoded ARN and security groups can create alarms and allow connections', () => {
      const stack = new cdk.Stack();
      const nlb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
        loadBalancerArn: 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/network/my-load-balancer/50dc6c495c0c9188',
        loadBalancerSecurityGroups: ['sg-12345678'],
      });

      nlb.metrics.activeFlowCount().createAlarm(stack, 'AlarmFlowCount', {
        evaluationPeriods: 1,
        threshold: 0,
      });
      nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(443));

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::CloudWatch::Alarm', {
        Namespace: 'AWS/NetworkELB',
        MetricName: 'ActiveFlowCount',
      });
      template.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        IpProtocol: 'tcp',
        FromPort: 443,
        ToPort: 443,
        GroupId: 'sg-12345678',
      });
    });

    test('imported NLB via Fn::importValue can create alarms and allow connections', () => {
      const stack = new cdk.Stack();
      const nlb = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack, 'NLB', {
        loadBalancerArn: cdk.Fn.importValue('NlbArn'),
        loadBalancerSecurityGroups: [cdk.Fn.importValue('SgId')],
      });

      nlb.metrics.activeFlowCount().createAlarm(stack, 'AlarmFlowCount', {
        evaluationPeriods: 1,
        threshold: 0,
      });
      nlb.connections.allowFromAnyIpv4(ec2.Port.tcp(443));

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::CloudWatch::Alarm', {
        Namespace: 'AWS/NetworkELB',
        MetricName: 'ActiveFlowCount',
      });
      template.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        IpProtocol: 'tcp',
        FromPort: 443,
        ToPort: 443,
        GroupId: { 'Fn::ImportValue': 'SgId' },
      });
    });

    test('imported NLB via cross-stack reference can create alarms and allow connections', () => {
      const app = new cdk.App();
      const stack1 = new cdk.Stack(app, 'Stack1', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new ec2.Vpc(stack1, 'VPC', { maxAzs: 2, restrictDefaultSecurityGroup: false });
      const sg = new ec2.SecurityGroup(stack1, 'SG', { vpc });
      const lb = new elbv2.NetworkLoadBalancer(stack1, 'LB', {
        vpc,
        internetFacing: true,
        securityGroups: [sg],
      });

      const stack2 = new cdk.Stack(app, 'Stack2', { env: { account: '123456789012', region: 'us-west-2' } });
      const imported = elbv2.NetworkLoadBalancer.fromNetworkLoadBalancerAttributes(stack2, 'NLB', {
        loadBalancerArn: lb.loadBalancerArn,
        loadBalancerSecurityGroups: lb.securityGroups,
      });

      imported.metrics.activeFlowCount().createAlarm(stack2, 'AlarmFlowCount', {
        evaluationPeriods: 1,
        threshold: 0,
      });
      imported.connections.allowFromAnyIpv4(ec2.Port.tcp(443));

      const template = Template.fromStack(stack2);
      template.hasResourceProperties('AWS::CloudWatch::Alarm', {
        Namespace: 'AWS/NetworkELB',
        MetricName: 'ActiveFlowCount',
      });
      template.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        IpProtocol: 'tcp',
        FromPort: 443,
        ToPort: 443,
      });
    });

    test('imported target group with loadBalancerArns can create healthyHostCount alarm', () => {
      const stack = new cdk.Stack();
      const tg = elbv2.NetworkTargetGroup.fromTargetGroupAttributes(stack, 'TG', {
        targetGroupArn: 'arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/my-target-group/50dc6c495c0c9188',
        loadBalancerArns: 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/net/my-load-balancer/50dc6c495c0c9188',
      });

      tg.metrics.healthyHostCount().createAlarm(stack, 'HealthyHostCount', {
        evaluationPeriods: 1,
        threshold: 0,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
        Namespace: 'AWS/NetworkELB',
        MetricName: 'HealthyHostCount',
      });
    });

    test('imported target group via Fn::importValue can create healthyHostCount alarm', () => {
      const stack = new cdk.Stack();
      const tg = elbv2.NetworkTargetGroup.fromTargetGroupAttributes(stack, 'TG', {
        targetGroupArn: cdk.Fn.importValue('TgArn'),
        loadBalancerArns: cdk.Fn.importValue('NlbArn'),
      });

      tg.metrics.healthyHostCount().createAlarm(stack, 'HealthyHostCount', {
        evaluationPeriods: 1,
        threshold: 0,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
        Namespace: 'AWS/NetworkELB',
        MetricName: 'HealthyHostCount',
      });
    });
  });
});
