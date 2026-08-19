import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { acknowledgeTestValidationRules } from './util';
import { Annotations, Match, Template } from '../../assertions';
import { App, CfnOutput, CfnResource, Fn, Lazy, Stack, Tags } from '../../core';
import { EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY, EC2_RESTRICT_DEFAULT_SECURITY_GROUP } from '../../cx-api';
import type { PublicSubnet } from '../lib';
import {
  AclCidr,
  AclTraffic,
  BastionHostLinux,
  CfnSubnet,
  CfnVPC,
  SubnetFilter,
  DefaultInstanceTenancy,
  GenericLinuxImage,
  InstanceType,
  InterfaceVpcEndpoint,
  InterfaceVpcEndpointService,
  NatProvider,
  NatGatewayProvider,
  NatInstanceProvider,
  NatTrafficDirection,
  NetworkAcl,
  NetworkAclEntry,
  Peer,
  Port,
  PrivateSubnet,
  RouterType,
  Subnet,
  SubnetType,
  TrafficDirection,
  Vpc,
  IpAddresses,
  Ipv6Addresses,
  InterfaceVpcEndpointAwsService,
  IpProtocol,
  AmazonLinuxImage,
  CpuCredits,
  InstanceClass,
  InstanceSize,
  KeyPair,
  UserData,
} from '../lib';

describe('vpc', () => {
  describe('When creating a VPC', () => {
    testDeprecated('SubnetType.PRIVATE_WITH_NAT is equivalent to SubnetType.PRIVATE_WITH_EGRESS', () => {
      const stack1 = getTestStack();
      const stack2 = getTestStack();
      new Vpc(stack1, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_WITH_NAT,
            name: 'subnet',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
          },
        ],
      });

      new Vpc(stack2, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'subnet',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
          },
        ],
      });

      const t1 = Template.fromStack(stack1);
      const t2 = Template.fromStack(stack2);

      expect(t1.toJSON()).toEqual(t2.toJSON());
    });

    testDeprecated('SubnetType.PRIVATE is equivalent to SubnetType.PRIVATE_WITH_NAT', () => {
      const stack1 = getTestStack();
      const stack2 = getTestStack();
      new Vpc(stack1, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE,
            name: 'subnet',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
          },
        ],
      });

      new Vpc(stack2, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_WITH_NAT,
            name: 'subnet',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
          },
        ],
      });

      const t1 = Template.fromStack(stack1);
      const t2 = Template.fromStack(stack2);

      expect(t1.toJSON()).toEqual(t2.toJSON());
    });

    testDeprecated('SubnetType.ISOLATED is equivalent to SubnetType.PRIVATE_ISOLATED', () => {
      const stack1 = getTestStack();
      const stack2 = getTestStack();
      new Vpc(stack1, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.ISOLATED,
            name: 'subnet',
          },
        ],
      });

      new Vpc(stack2, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'subnet',
          },
        ],
      });
      const t1 = Template.fromStack(stack1);
      const t2 = Template.fromStack(stack2);

      expect(t1.toJSON()).toEqual(t2.toJSON());
    });

    describe('with the default CIDR range', () => {
      test('vpc.vpcId returns a token to the VPC ID', () => {
        const stack = getTestStack();
        const vpc = new Vpc(stack, 'TheVPC');
        expect(stack.resolve(vpc.vpcId)).toEqual({ Ref: 'TheVPC92636AB0' });
      });

      test('vpc.vpcArn returns a token to the VPC ID', () => {
        const stack = getTestStack();
        const vpc = new Vpc(stack, 'TheVPC');
        expect(stack.resolve(vpc.vpcArn)).toEqual({ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':ec2:us-east-1:123456789012:vpc/', { Ref: 'TheVPC92636AB0' }]] });
      });

      test('it uses the correct network range', () => {
        const stack = getTestStack();
        new Vpc(stack, 'TheVPC');
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
          CidrBlock: Vpc.DEFAULT_CIDR_RANGE,
          EnableDnsHostnames: true,
          EnableDnsSupport: true,
          InstanceTenancy: DefaultInstanceTenancy.DEFAULT,
        });
      });
      test('the Name tag is defaulted to path', () => {
        const stack = getTestStack();
        new Vpc(stack, 'TheVPC');
        Template.fromStack(stack).hasResource('AWS::EC2::VPC',
          hasTags([{ Key: 'Name', Value: 'TestStack/TheVPC' }]),
        );
        Template.fromStack(stack).hasResource('AWS::EC2::InternetGateway',
          hasTags([{ Key: 'Name', Value: 'TestStack/TheVPC' }]),
        );
      });
    });

    test('with all of the properties set, it successfully sets the correct VPC properties', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        ipAddresses: IpAddresses.cidr('192.168.0.0/16'),
        enableDnsHostnames: false,
        enableDnsSupport: false,
        defaultInstanceTenancy: DefaultInstanceTenancy.DEDICATED,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
        CidrBlock: '192.168.0.0/16',
        EnableDnsHostnames: false,
        EnableDnsSupport: false,
        InstanceTenancy: DefaultInstanceTenancy.DEDICATED,
      });
    });

    describe('dns getters correspond to CFN properties', () => {
      const inputs = [
        { dnsSupport: false, dnsHostnames: false },
        // {dnsSupport: false, dnsHostnames: true} - this configuration is illegal so its not part of the permutations.
        { dnsSupport: true, dnsHostnames: false },
        { dnsSupport: true, dnsHostnames: true },
      ];

      for (const input of inputs) {
        test(`[dnsSupport=${input.dnsSupport},dnsHostnames=${input.dnsHostnames}]`, () => {
          const stack = getTestStack();
          const vpc = new Vpc(stack, 'TheVPC', {
            ipAddresses: IpAddresses.cidr('192.168.0.0/16'),
            enableDnsHostnames: input.dnsHostnames,
            enableDnsSupport: input.dnsSupport,
            defaultInstanceTenancy: DefaultInstanceTenancy.DEDICATED,
          });

          Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
            CidrBlock: '192.168.0.0/16',
            EnableDnsHostnames: input.dnsHostnames,
            EnableDnsSupport: input.dnsSupport,
            InstanceTenancy: DefaultInstanceTenancy.DEDICATED,
          });

          expect(input.dnsSupport).toEqual(vpc.dnsSupportEnabled);
          expect(input.dnsHostnames).toEqual(vpc.dnsHostnamesEnabled);
        });
      }
    });

    test('contains the correct number of subnets', () => {
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC');
      const zones = stack.availabilityZones.length;
      expect(vpc.publicSubnets.length).toEqual(zones);
      expect(vpc.privateSubnets.length).toEqual(zones);
      expect(stack.resolve(vpc.vpcId)).toEqual({ Ref: 'TheVPC92636AB0' });
    });

    test('can refer to the internet gateway', () => {
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC');
      expect(stack.resolve(vpc.internetGatewayId)).toEqual({ Ref: 'TheVPCIGWFA25CC08' });
    });

    test('with only isolated subnets, the VPC should not contain an IGW or NAT Gateways', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'Isolated',
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 0);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: false,
      });
    });

    test('with no private subnets, the VPC should have an IGW but no NAT Gateways', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PUBLIC,
            name: 'Public',
          },
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'Isolated',
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
    });

    test('with createInternetGateway: false, the VPC should not have an IGW nor NAT Gateways', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        createInternetGateway: false,
        subnetConfiguration: [
          {
            subnetType: SubnetType.PUBLIC,
            name: 'Public',
          },
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'Isolated',
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 0);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
    });

    test('with private subnets and custom networkAcl.', () => {
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PUBLIC,
            name: 'Public',
          },
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'private',
          },
        ],
      });

      const nacl1 = new NetworkAcl(stack, 'myNACL1', {
        vpc,
        subnetSelection: { subnetType: SubnetType.PRIVATE_WITH_EGRESS },
      });

      new NetworkAclEntry(stack, 'AllowDNSEgress', {
        networkAcl: nacl1,
        ruleNumber: 100,
        traffic: AclTraffic.udpPort(53),
        direction: TrafficDirection.EGRESS,
        cidr: AclCidr.ipv4('10.0.0.0/16'),
      });

      new NetworkAclEntry(stack, 'AllowDNSIngress', {
        networkAcl: nacl1,
        ruleNumber: 100,
        traffic: AclTraffic.udpPort(53),
        direction: TrafficDirection.INGRESS,
        cidr: AclCidr.anyIpv4(),
      });

      Template.fromStack(stack).resourceCountIs('AWS::EC2::NetworkAcl', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NetworkAclEntry', 2);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::SubnetNetworkAclAssociation', 3);
    });

    test('with no subnets defined, the VPC should have an IGW, and a NAT Gateway per AZ', () => {
      const stack = getTestStack();
      const zones = stack.availabilityZones.length;
      new Vpc(stack, 'TheVPC', {});
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', zones);
    });

    test('with isolated and public subnet, should be able to use the internet gateway to define routes', () => {
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'isolated',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
          },
        ],
      });
      (vpc.isolatedSubnets[0] as Subnet).addRoute('TheRoute', {
        routerId: vpc.internetGatewayId!,
        routerType: RouterType.GATEWAY,
        destinationCidrBlock: '8.8.8.8/32',
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationCidrBlock: '8.8.8.8/32',
        GatewayId: {},
      });
    });

    test('with only reserved subnets as public subnets, should not create the internet gateway', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'isolated',
          },
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
            reserved: true,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 0);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::VPCGatewayAttachment', 0);
    });

    test('with only reserved subnets as private subnets with egress, should not create the internet gateway', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
            name: 'isolated',
          },
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'egress',
            reserved: true,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 0);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::VPCGatewayAttachment', 0);
    });

    test('with no public subnets and natGateways > 0, should throw an error', () => {
      const stack = getTestStack();
      expect(() => new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'egress',
          },
        ],
        natGateways: 1,
      })).toThrow(/If you configure PRIVATE subnets in 'subnetConfiguration', you must also configure PUBLIC subnets to put the NAT gateways into \(got \[{"subnetType":"Private","name":"egress"}\]./);
    });

    test('with only reserved subnets as public subnets and natGateways > 0, should throw an error', () => {
      const stack = getTestStack();
      expect(() => new Vpc(stack, 'TheVPC', {
        subnetConfiguration: [
          {
            subnetType: SubnetType.PUBLIC,
            name: 'public',
            reserved: true,
          },
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'egress',
          },
        ],
        natGateways: 1,
      })).toThrow(/If you configure PRIVATE subnets in 'subnetConfiguration', you must also configure PUBLIC subnets to put the NAT gateways into \(got \[{"subnetType":"Public","name":"public","reserved":true},{"subnetType":"Private","name":"egress"}\]./);
    });

    test('with subnets and reserved subnets defined, VPC subnet count should not contain reserved subnets ', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        subnetConfiguration: [
          {
            cidrMask: 24,
            subnetType: SubnetType.PUBLIC,
            name: 'Public',
          },
          {
            cidrMask: 24,
            name: 'reserved',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            reserved: true,
          },
          {
            cidrMask: 28,
            name: 'rds',
            subnetType: SubnetType.PRIVATE_ISOLATED,
          },
        ],
        maxAzs: 3,
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 6);
    });
    test('with reserved subnets, any other subnets should not have cidrBlock from within reserved space', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'reserved',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            reserved: true,
          },
          {
            cidrMask: 24,
            name: 'rds',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
        maxAzs: 3,
      });

      const template = Template.fromStack(stack);

      for (let i = 0; i < 3; i++) {
        template.hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i}.0/24`,
        });
      }
      for (let i = 3; i < 6; i++) {
        const matchingSubnets = template.findResources('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i}.0/24`,
        });
        expect(Object.keys(matchingSubnets).length).toBe(0);
      }
      for (let i = 6; i < 9; i++) {
        template.hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i}.0/24`,
        });
      }
    });
    test('with custom subnets, the VPC should have the right number of subnets, an IGW, and a NAT Gateway per AZ', () => {
      const stack = getTestStack();
      const zones = stack.availabilityZones.length;
      new Vpc(stack, 'TheVPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/21'),
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'application',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
          {
            cidrMask: 28,
            name: 'rds',
            subnetType: SubnetType.PRIVATE_ISOLATED,
          },
        ],
        maxAzs: 3,
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', zones);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 9);
      for (let i = 0; i < 6; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i}.0/24`,
        });
      }
      for (let i = 0; i < 3; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.6.${i * 16}/28`,
        });
      }
    });
    test('with custom subnets and natGateways = 2 there should be only two NATGW', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/21'),
        natGateways: 2,
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'application',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
          {
            cidrMask: 28,
            name: 'rds',
            subnetType: SubnetType.PRIVATE_ISOLATED,
          },
        ],
        maxAzs: 3,
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 2);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 9);
      for (let i = 0; i < 6; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i}.0/24`,
        });
      }
      for (let i = 0; i < 3; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.6.${i * 16}/28`,
        });
      }
    });
    test('with enableDnsHostnames enabled but enableDnsSupport disabled, should throw an Error', () => {
      const stack = getTestStack();
      expect(() => new Vpc(stack, 'TheVPC', {
        enableDnsHostnames: true,
        enableDnsSupport: false,
      })).toThrow();
    });
    test('with public subnets MapPublicIpOnLaunch is true', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        maxAzs: 1,
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: true,
      });
    });

    test('with public subnets MapPublicIpOnLaunch is true if parameter mapPublicIpOnLaunch is true', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        maxAzs: 1,
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
            mapPublicIpOnLaunch: true,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: true,
      });
    });
    test('with public subnets MapPublicIpOnLaunch is false if parameter mapPublicIpOnLaunch is false', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        maxAzs: 1,
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
            mapPublicIpOnLaunch: false,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: false,
      });
    });
    test('with private subnets throw exception if parameter mapPublicIpOnLaunch is defined', () => {
      const stack = getTestStack();
      expect(() => {
        new Vpc(stack, 'VPC', {
          maxAzs: 1,
          subnetConfiguration: [
            {
              name: 'public',
              subnetType: SubnetType.PUBLIC,
            },
            {
              name: 'private',
              subnetType: SubnetType.PRIVATE_WITH_EGRESS,
              mapPublicIpOnLaunch: true,
            },
          ],
        });
      }).toThrow(/subnet cannot include mapPublicIpOnLaunch parameter/);
    });
    test('with isolated subnets throw exception if parameter mapPublicIpOnLaunch is defined', () => {
      const stack = getTestStack();
      expect(() => {
        new Vpc(stack, 'VPC', {
          maxAzs: 1,
          subnetConfiguration: [
            {
              name: 'public',
              subnetType: SubnetType.PUBLIC,
            },
            {
              name: 'private',
              subnetType: SubnetType.PRIVATE_ISOLATED,
              mapPublicIpOnLaunch: true,
            },
          ],
        });
      }).toThrow(/subnet cannot include mapPublicIpOnLaunch parameter/);
    });

    test('verify the Default VPC name', () => {
      const stack = getTestStack();
      const tagName = { Key: 'Name', Value: `${stack.node.path}/VPC` };
      new Vpc(stack, 'VPC', {
        maxAzs: 1,
        subnetConfiguration: [
          {
            name: 'public',
            subnetType: SubnetType.PUBLIC,
          },
          {
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 2);
      Template.fromStack(stack).hasResource('AWS::EC2::NatGateway', Match.anyValue());
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: true,
      });
      Template.fromStack(stack).hasResource('AWS::EC2::VPC', hasTags([tagName]));
    });

    test('verify the assigned VPC name passing the "vpcName" prop', () => {
      const stack = getTestStack();
      const tagName = { Key: 'Name', Value: 'CustomVPCName' };
      new Vpc(stack, 'VPC', {
        maxAzs: 1,
        subnetConfiguration: [
          {
            name: 'public',
            subnetType: SubnetType.PUBLIC,
          },
          {
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
        vpcName: 'CustomVPCName',
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 2);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', Match.anyValue());
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        MapPublicIpOnLaunch: true,
      });
      Template.fromStack(stack).hasResource('AWS::EC2::VPC', hasTags([tagName]));
    });
    test('maxAZs defaults to 3 if unset', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC');
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 6);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Route', 6);
      for (let i = 0; i < 6; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i * 32}.0/19`,
        });
      }
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationCidrBlock: '0.0.0.0/0',
        NatGatewayId: {},
      });
    });

    test('with maxAZs set to 2', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', { maxAzs: 2 });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 4);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Route', 4);
      for (let i = 0; i < 4; i++) {
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
          CidrBlock: `10.0.${i * 64}.0/18`,
        });
      }
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationCidrBlock: '0.0.0.0/0',
        NatGatewayId: {},
      });
    });

    test('throws error when both availabilityZones and maxAzs are set', () => {
      const stack = getTestStack();
      expect(() => {
        new Vpc(stack, 'VPC', {
          availabilityZones: stack.availabilityZones,
          maxAzs: 1,
        });
      }).toThrow(/Vpc supports 'availabilityZones' or 'maxAzs', but not both./);
    });

    test('with availabilityZones set correctly', () => {
      const stack = getTestStack();
      const specificAz = stack.availabilityZones[1]; // not the first item
      new Vpc(stack, 'VPC', {
        availabilityZones: [specificAz],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 2);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        AvailabilityZone: specificAz,
      });
    });

    test('with availabilityZones set to zones different from stack', () => {
      const stack = getTestStack();
      // we need to create a context with availability zones, otherwise we're checking against dummy values
      stack.node.setContext(
        `availability-zones:account=${stack.account}:region=${stack.region}`,
        ['us-east-1a', 'us-east1b', 'us-east-1c'],
      );
      expect(() => {
        new Vpc(stack, 'VPC', {
          availabilityZones: [stack.availabilityZones[0] + 'invalid'],
        });
      }).toThrow(/must be a subset of the stack/);
    });

    test('does not throw with availability zones set without context in non-agnostic stack', () => {
      const stack = getTestStack();
      expect(() => {
        new Vpc(stack, 'VPC', {
          availabilityZones: ['us-east-1a'],
        });
      }).not.toThrow();
    });

    test('agnostic stack without context with defined vpc AZs', () => {
      const stack = new Stack(undefined, 'TestStack');
      new Vpc(stack, 'VPC', {
        availabilityZones: ['us-east-1a'],
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 2);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Subnet', {
        AvailabilityZone: 'us-east-1a',
      });
    });

    test('with natGateway set to 1', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        natGateways: 1,
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Subnet', 6);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Route', 6);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationCidrBlock: '0.0.0.0/0',
        NatGatewayId: {},
      });
    });
    test('with natGateway subnets defined', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'egress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
        natGatewaySubnets: {
          subnetGroupName: 'egress',
        },
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 3);
      for (let i = 1; i < 4; i++) {
        Template.fromStack(stack).hasResource('AWS::EC2::Subnet', hasTags([{
          Key: 'aws-cdk:subnet-name',
          Value: 'egress',
        }, {
          Key: 'Name',
          Value: `TestStack/VPC/egressSubnet${i}`,
        }]));
      }
    });

    testDeprecated('natGateways = 0 throws if PRIVATE_WITH_NAT subnets configured', () => {
      const stack = getTestStack();
      expect(() => {
        new Vpc(stack, 'VPC', {
          natGateways: 0,
          subnetConfiguration: [
            {
              name: 'public',
              subnetType: SubnetType.PUBLIC,
            },
            {
              name: 'private',
              subnetType: SubnetType.PRIVATE_WITH_NAT,
            },
          ],
        });
      }).toThrow(/make sure you don't configure any PRIVATE/);
    });

    test('natGateways = 0 succeeds if PRIVATE_WITH_EGRESS subnets configured', () => {
      const stack = getTestStack();

      new Vpc(stack, 'VPC', {
        natGateways: 0,
        subnetConfiguration: [
          {
            name: 'public',
            subnetType: SubnetType.PUBLIC,
          },
          {
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
      });

      Template.fromStack(stack).resourceCountIs('AWS::EC2::InternetGateway', 1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::NatGateway', 0);
    });

    test('natGateway = 0 defaults with ISOLATED subnet', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        natGateways: 0,
      });
      Template.fromStack(stack).hasResource('AWS::EC2::Subnet', hasTags([{
        Key: 'aws-cdk:subnet-type',
        Value: 'Isolated',
      }]));
    });

    test('unspecified natGateways constructs with PRIVATE subnet', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC');
      Template.fromStack(stack).hasResource('AWS::EC2::Subnet', hasTags([{
        Key: 'aws-cdk:subnet-type',
        Value: 'Private',
      }]));
    });

    test('natGateways = 0 allows RESERVED PRIVATE subnets', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        subnetConfiguration: [
          {
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            reserved: true,
          },
        ],
        natGateways: 0,
      });
      Template.fromStack(stack).hasResource('AWS::EC2::Subnet', hasTags([{
        Key: 'aws-cdk:subnet-name',
        Value: 'ingress',
      }]));
    });

    test('EIP passed with NAT gateway does not create duplicate EIP', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'application',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
        natGatewayProvider: NatProvider.gateway({ eipAllocationIds: ['b'] }),
        natGateways: 1,
      });
      Template.fromStack(stack).resourceCountIs('AWS::EC2::EIP', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', {
        AllocationId: 'b',
      });
    });

    test('with mis-matched nat and subnet configs it throws', () => {
      const stack = getTestStack();
      expect(() => new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          {
            cidrMask: 24,
            name: 'ingress',
            subnetType: SubnetType.PUBLIC,
          },
          {
            cidrMask: 24,
            name: 'private',
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
        ],
        natGatewaySubnets: {
          subnetGroupName: 'notthere',
        },
      })).toThrow();
    });
    test('with a vpn gateway', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        vpnGateway: true,
        vpnGatewayAsn: 65000,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGateway', {
        AmazonSideAsn: 65000,
        Type: 'ipsec.1',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCGatewayAttachment', {
        VpcId: {
          Ref: 'VPCB9E5F0B4',
        },
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGatewayRoutePropagation', {
        RouteTableIds: [
          {
            Ref: 'VPCPrivateSubnet1RouteTableBE8A6027',
          },
          {
            Ref: 'VPCPrivateSubnet2RouteTable0A19E10E',
          },
          {
            Ref: 'VPCPrivateSubnet3RouteTable192186F8',
          },
        ],
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });
    });
    test('with a vpn gateway and route propagation on isolated subnets', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'Public' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'Isolated' },
        ],
        vpnGateway: true,
        vpnRoutePropagation: [
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGatewayRoutePropagation', {
        RouteTableIds: [
          {
            Ref: 'VPCIsolatedSubnet1RouteTableEB156210',
          },
          {
            Ref: 'VPCIsolatedSubnet2RouteTable9B4F78DC',
          },
          {
            Ref: 'VPCIsolatedSubnet3RouteTableCB6A1FDA',
          },
        ],
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });
    });
    test('with a vpn gateway and route propagation on private and isolated subnets', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'Public' },
          { subnetType: SubnetType.PRIVATE_WITH_EGRESS, name: 'Private' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'Isolated' },
        ],
        vpnGateway: true,
        vpnRoutePropagation: [
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          },
          {
            subnetType: SubnetType.PRIVATE_ISOLATED,
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGatewayRoutePropagation', {
        RouteTableIds: [
          {
            Ref: 'VPCPrivateSubnet1RouteTableBE8A6027',
          },
          {
            Ref: 'VPCPrivateSubnet2RouteTable0A19E10E',
          },
          {
            Ref: 'VPCPrivateSubnet3RouteTable192186F8',
          },
          {
            Ref: 'VPCIsolatedSubnet1RouteTableEB156210',
          },
          {
            Ref: 'VPCIsolatedSubnet2RouteTable9B4F78DC',
          },
          {
            Ref: 'VPCIsolatedSubnet3RouteTableCB6A1FDA',
          },
        ],
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });
    });
    test('route propagation defaults to isolated subnets when there are no private subnets', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'Public' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'Isolated' },
        ],
        vpnGateway: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGatewayRoutePropagation', {
        RouteTableIds: [
          {
            Ref: 'VPCIsolatedSubnet1RouteTableEB156210',
          },
          {
            Ref: 'VPCIsolatedSubnet2RouteTable9B4F78DC',
          },
          {
            Ref: 'VPCIsolatedSubnet3RouteTableCB6A1FDA',
          },
        ],
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });
    });
    test('route propagation defaults to public subnets when there are no private/isolated subnets', () => {
      const stack = getTestStack();
      new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'Public' },
        ],
        vpnGateway: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPNGatewayRoutePropagation', {
        RouteTableIds: [
          {
            Ref: 'VPCPublicSubnet1RouteTableFEE4B781',
          },
          {
            Ref: 'VPCPublicSubnet2RouteTable6F1A15F1',
          },
          {
            Ref: 'VPCPublicSubnet3RouteTable98AE0E14',
          },
        ],
        VpnGatewayId: {
          Ref: 'VPCVpnGatewayB5ABAE68',
        },
      });
    });
    test('fails when specifying vpnConnections with vpnGateway set to false', () => {
      // GIVEN
      const stack = new Stack();

      expect(() => new Vpc(stack, 'VpcNetwork', {
        vpnGateway: false,
        vpnConnections: {
          VpnConnection: {
            asn: 65000,
            ip: '192.0.2.1',
          },
        },
      })).toThrow(/`vpnConnections`.+`vpnGateway`.+false/);
    });
    test('fails when specifying vpnGatewayAsn with vpnGateway set to false', () => {
      // GIVEN
      const stack = new Stack();

      expect(() => new Vpc(stack, 'VpcNetwork', {
        vpnGateway: false,
        vpnGatewayAsn: 65000,
      })).toThrow(/`vpnGatewayAsn`.+`vpnGateway`.+false/);
    });

    test('Subnets have a defaultChild', () => {
      // GIVEN
      const stack = new Stack();

      const vpc = new Vpc(stack, 'VpcNetwork');

      expect(vpc.publicSubnets[0].node.defaultChild instanceof CfnSubnet).toEqual(true);
    });

    test('CIDR cannot be a Token', () => {
      const stack = new Stack();
      expect(() => {
        new Vpc(stack, 'Vpc', {
          ipAddresses: IpAddresses.cidr(Lazy.string({ produce: () => 'abc' })),
        });
      }).toThrow(/property must be a concrete CIDR string/);
    });

    test('Default NAT gateway provider', () => {
      const stack = new Stack();
      const natGatewayProvider = NatProvider.gateway();
      new Vpc(stack, 'VpcNetwork', { natGatewayProvider });

      expect(natGatewayProvider.configuredGateways.length).toBeGreaterThan(0);
    });

    test('Default NAT gateway provider can be instantiated directly with new', () => {
      const stack = new Stack();
      const natGatewayProvider = new NatGatewayProvider();
      new Vpc(stack, 'VpcNetwork', { natGatewayProvider });

      expect(natGatewayProvider.configuredGateways.length).toBeGreaterThan(0);
    });

    test('NAT gateway provider with EIP allocations', () => {
      const stack = new Stack();
      const natGatewayProvider = NatProvider.gateway({
        eipAllocationIds: ['a', 'b', 'c', 'd'],
      });
      new Vpc(stack, 'VpcNetwork', { natGatewayProvider });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', {
        AllocationId: 'a',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', {
        AllocationId: 'b',
      });
    });

    test('NAT gateway provider with insufficient EIP allocations', () => {
      const stack = new Stack();
      const natGatewayProvider = NatProvider.gateway({ eipAllocationIds: ['a'] });
      expect(() => new Vpc(stack, 'VpcNetwork', { natGatewayProvider }))
        .toThrow(/Not enough NAT gateway EIP allocation IDs \(1 provided\) for the requested subnet count \(\d+ needed\)/);
    });

    test('NAT gateway provider with token EIP allocations', () => {
      const stack = new Stack();
      const eipAllocationIds = Fn.split(',', Fn.importValue('myVpcId'));
      const natGatewayProvider = NatProvider.gateway({ eipAllocationIds });
      new Vpc(stack, 'VpcNetwork', { natGatewayProvider });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', {
        AllocationId: stack.resolve(Fn.select(0, eipAllocationIds)),
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::NatGateway', {
        AllocationId: stack.resolve(Fn.select(1, eipAllocationIds)),
      });
    });

    test('Can add an IPv6 route', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const vpc = new Vpc(stack, 'VPC');
      (vpc.publicSubnets[0] as PublicSubnet).addRoute('SomeRoute', {
        destinationIpv6CidrBlock: '2001:4860:4860::8888/32',
        routerId: 'router-1',
        routerType: RouterType.NETWORK_INTERFACE,
      });

      // THEN

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationIpv6CidrBlock: '2001:4860:4860::8888/32',
        NetworkInterfaceId: 'router-1',
      });
    });
    test('Can add an IPv4 route', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const vpc = new Vpc(stack, 'VPC');
      (vpc.publicSubnets[0] as PublicSubnet).addRoute('SomeRoute', {
        destinationCidrBlock: '0.0.0.0/0',
        routerId: 'router-1',
        routerType: RouterType.NETWORK_INTERFACE,
      });

      // THEN

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        DestinationCidrBlock: '0.0.0.0/0',
        NetworkInterfaceId: 'router-1',
      });
    });
    test('can restrict access to the default security group', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      stack.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, true);
      new Vpc(stack, 'Vpc');

      // THEN
      Template.fromStack(stack).hasResourceProperties('Custom::VpcRestrictDefaultSG', {
        DefaultSecurityGroupId: {
          'Fn::GetAtt': ['Vpc8378EB38', 'DefaultSecurityGroup'],
        },
        ServiceToken: {
          'Fn::GetAtt': ['CustomVpcRestrictDefaultSGCustomResourceProviderHandlerDC833E5E', 'Arn'],
        },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        Policies: [
          {
            PolicyDocument: {
              Statement: [{
                Effect: 'Allow',
                Action: [
                  'ec2:AuthorizeSecurityGroupIngress',
                  'ec2:AuthorizeSecurityGroupEgress',
                  'ec2:RevokeSecurityGroupIngress',
                  'ec2:RevokeSecurityGroupEgress',
                ],
                Resource: [{
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':ec2:us-east-1:123456789012:security-group/',
                      {
                        'Fn::GetAtt': [
                          'Vpc8378EB38',
                          'DefaultSecurityGroup',
                        ],
                      },
                    ],
                  ],
                }],
              }],
            },
          },
        ],
      });
    });

    test('will not restrict access to the default security group when feature flag is false', () => {
      // GIVEN
      const stack = getTestStack();
      stack.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, false);
      new Vpc(stack, 'Vpc');

      Template.fromStack(stack).resourceCountIs('Custom::VpcRestrictDefaultSG', 0);
    });

    test('can disable restrict access to the default security group when feature flag is true', () => {
      // GIVEN
      const stack = getTestStack();
      stack.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, true);
      new Vpc(stack, 'Vpc', { restrictDefaultSecurityGroup: false });

      Template.fromStack(stack).resourceCountIs('Custom::VpcRestrictDefaultSG', 0);
    });

    test.each(
      [
        {
          subnetType: SubnetType.PRIVATE_ISOLATED,
        },
        {
          subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          additionalSubnetConfig: [{ subnetType: SubnetType.PUBLIC, name: 'public' }],
        },
        {
          subnetType: SubnetType.PUBLIC,
        },
      ],
    )('subnet has dependent on the CIDR block when ipv6AssignAddressOnCreation is set to true, ', (testData) => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', {
        ipProtocol: IpProtocol.DUAL_STACK,
        maxAzs: 1,
        subnetConfiguration: [
          {
            subnetType: testData.subnetType,
            name: 'subnetName',
            ipv6AssignAddressOnCreation: true,
          },
          ...testData.additionalSubnetConfig ?? [],
        ],
      });
      Template.fromStack(stack).hasResource('AWS::EC2::Subnet', {
        DependsOn: ['TheVPCipv6cidrF3E84E30'],
      });
    });
  });

  describe('fromVpcAttributes', () => {
    test('passes region correctly', () => {
      // GIVEN
      const stack = getTestStack();

      const vpcId = Fn.importValue('myVpcId');

      // WHEN
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId,
        availabilityZones: ['region-12345a', 'region-12345b', 'region-12345c'],
        region: 'region-12345',
      });

      // THEN
      expect(vpc.env.region).toEqual('region-12345');
    });

    test('passes subnet IPv4 CIDR blocks correctly', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        publicSubnetIds: ['pub-1', 'pub-2', 'pub-3'],
        publicSubnetIpv4CidrBlocks: ['10.0.0.0/18', '10.0.64.0/18', '10.0.128.0/18'],
        privateSubnetIds: ['pri-1', 'pri-2', 'pri-3'],
        privateSubnetIpv4CidrBlocks: ['10.10.0.0/18', '10.10.64.0/18', '10.10.128.0/18'],
        isolatedSubnetIds: ['iso-1', 'iso-2', 'iso-3'],
        isolatedSubnetIpv4CidrBlocks: ['10.20.0.0/18', '10.20.64.0/18', '10.20.128.0/18'],
      });

      // WHEN
      const public1 = vpc.publicSubnets.find(({ subnetId }) => subnetId === 'pub-1');
      const public2 = vpc.publicSubnets.find(({ subnetId }) => subnetId === 'pub-2');
      const public3 = vpc.publicSubnets.find(({ subnetId }) => subnetId === 'pub-3');
      const private1 = vpc.privateSubnets.find(({ subnetId }) => subnetId === 'pri-1');
      const private2 = vpc.privateSubnets.find(({ subnetId }) => subnetId === 'pri-2');
      const private3 = vpc.privateSubnets.find(({ subnetId }) => subnetId === 'pri-3');
      const isolated1 = vpc.isolatedSubnets.find(({ subnetId }) => subnetId === 'iso-1');
      const isolated2 = vpc.isolatedSubnets.find(({ subnetId }) => subnetId === 'iso-2');
      const isolated3 = vpc.isolatedSubnets.find(({ subnetId }) => subnetId === 'iso-3');

      // THEN
      expect(public1?.ipv4CidrBlock).toEqual('10.0.0.0/18');
      expect(public2?.ipv4CidrBlock).toEqual('10.0.64.0/18');
      expect(public3?.ipv4CidrBlock).toEqual('10.0.128.0/18');
      expect(private1?.ipv4CidrBlock).toEqual('10.10.0.0/18');
      expect(private2?.ipv4CidrBlock).toEqual('10.10.64.0/18');
      expect(private3?.ipv4CidrBlock).toEqual('10.10.128.0/18');
      expect(isolated1?.ipv4CidrBlock).toEqual('10.20.0.0/18');
      expect(isolated2?.ipv4CidrBlock).toEqual('10.20.64.0/18');
      expect(isolated3?.ipv4CidrBlock).toEqual('10.20.128.0/18');
    });

    test('throws on incorrect number of subnet names', () => {
      const stack = new Stack();

      expect(() =>
        Vpc.fromVpcAttributes(stack, 'VPC', {
          vpcId: 'vpc-1234',
          availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
          publicSubnetIds: ['s-12345', 's-34567', 's-56789'],
          publicSubnetNames: ['Public 1', 'Public 2'],
        }),
      ).toThrow(/publicSubnetNames must have an entry for every corresponding subnet group/);
    });

    test('throws on incorrect number of route table ids', () => {
      const stack = new Stack();

      expect(() =>
        Vpc.fromVpcAttributes(stack, 'VPC', {
          vpcId: 'vpc-1234',
          availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
          publicSubnetIds: ['s-12345', 's-34567', 's-56789'],
          publicSubnetRouteTableIds: ['rt-12345'],
        }),
      ).toThrow('Number of publicSubnetRouteTableIds (1) must be equal to the amount of publicSubnetIds (3).');
    });

    test('throws on incorrect number of subnet IPv4 CIDR blocks', () => {
      const stack = new Stack();

      expect(() =>
        Vpc.fromVpcAttributes(stack, 'VPC', {
          vpcId: 'vpc-1234',
          availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
          publicSubnetIds: ['s-12345', 's-34567', 's-56789'],
          publicSubnetIpv4CidrBlocks: ['10.0.0.0/18', '10.0.64.0/18'],
        }),
      ).toThrow('Number of publicSubnetIpv4CidrBlocks (2) must be equal to the amount of publicSubnetIds (3).');
    });
  });

  describe('NAT instances', () => {
    test('Can configure NAT instances instead of NAT gateways', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const natGatewayProvider = NatProvider.instance({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
      });
      new Vpc(stack, 'TheVPC', { natGatewayProvider });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 3);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        ImageId: 'ami-1',
        InstanceType: 'q86.mega',
        SourceDestCheck: false,
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        RouteTableId: { Ref: 'TheVPCPrivateSubnet1RouteTableF6513BC2' },
        DestinationCidrBlock: '0.0.0.0/0',
        InstanceId: { Ref: 'TheVPCPublicSubnet1NatInstanceCC514192' },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        SecurityGroupIngress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'from 0.0.0.0/0:ALL TRAFFIC',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('Can configure NAT instances V2 instead of NAT gateways', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const natGatewayProvider = NatProvider.instanceV2({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
      });
      new Vpc(stack, 'TheVPC', { natGatewayProvider });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 3);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        ImageId: 'ami-1',
        InstanceType: 'q86.mega',
        SourceDestCheck: false,
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        RouteTableId: { Ref: 'TheVPCPrivateSubnet1RouteTableF6513BC2' },
        DestinationCidrBlock: '0.0.0.0/0',
        InstanceId: { Ref: 'TheVPCPublicSubnet1NatInstanceCC514192' },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        SecurityGroupIngress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'from 0.0.0.0/0:ALL TRAFFIC',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('Can customize NAT instances V2 properties', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const keyPair = KeyPair.fromKeyPairName(stack, 'KeyPair', 'KeyPairName');
      const userData = UserData.forLinux();
      userData.addCommands('echo "hello world!"');

      const natGatewayProvider = NatProvider.instanceV2({
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.SMALL),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),

        creditSpecification: CpuCredits.UNLIMITED,
        defaultAllowedTraffic: NatTrafficDirection.OUTBOUND_ONLY,
        keyPair,
        userData,

        // Unusuable in its current state
        // The VPC is required to create the security group,
        // but the NAT Provider is required to create the VPC
        // See https://github.com/aws/aws-cdk/issues/27527
        // securityGroup,
      });
      new Vpc(stack, 'TheVPC', { natGatewayProvider });

      // THEN
      expect(natGatewayProvider.gatewayInstances.length).toBe(3);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 3);
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        ImageId: 'ami-1',
        InstanceType: 't3.small',
        SourceDestCheck: false,
        CreditSpecification: { CPUCredits: 'unlimited' },
        KeyName: 'KeyPairName',
        UserData: { 'Fn::Base64': '#!/bin/bash\necho "hello world!"' },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        RouteTableId: { Ref: 'TheVPCPrivateSubnet1RouteTableF6513BC2' },
        DestinationCidrBlock: '0.0.0.0/0',
        InstanceId: { Ref: 'TheVPCPublicSubnet1NatInstanceCC514192' },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('throws if both defaultAllowedTraffic and allowAllTraffic are set', () => {
      // GIVEN
      const stack = getTestStack();

      // THEN
      expect(() => {
        new Vpc(stack, 'TheVPC', {
          natGatewayProvider: NatProvider.instanceV2({
            instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.SMALL),
            defaultAllowedTraffic: NatTrafficDirection.OUTBOUND_ONLY,
            allowAllTraffic: true,
          }),
          natGateways: 1,
        });
      }).toThrow("Can not specify both of 'defaultAllowedTraffic' and 'defaultAllowedTraffic'; prefer 'defaultAllowedTraffic'");
    });

    test('throws if both keyName and keyPair are set', () => {
      // GIVEN
      const stack = getTestStack();

      // THEN
      expect(() => {
        new Vpc(stack, 'TheVPC', {
          natGatewayProvider: NatProvider.instanceV2({
            instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.SMALL),
            keyPair: KeyPair.fromKeyPairName(stack, 'KeyPair', 'KeyPairName'),
            keyName: 'KeyPairName',
          }),
          natGateways: 1,
        });
      }).toThrow("Cannot specify both of 'keyName' and 'keyPair'; prefer 'keyPair'");
    });

    test('throws if creditSpecification is set with a non-burstable instance type', () => {
      // GIVEN
      const stack = getTestStack();

      // THEN
      expect(() => {
        new Vpc(stack, 'TheVPC', {
          natGatewayProvider: NatProvider.instanceV2({
            instanceType: InstanceType.of(InstanceClass.C3, InstanceSize.SMALL),
            creditSpecification: CpuCredits.UNLIMITED,
          }),
          natGateways: 1,
        });
      }).toThrow(/creditSpecification is supported only for .* instance type/);
    });

    test('natGateways controls amount of NAT instances', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: NatProvider.instance({
          instanceType: new InstanceType('q86.mega'),
          machineImage: new GenericLinuxImage({
            'us-east-1': 'ami-1',
          }),
        }),
        natGateways: 1,
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 1);
    });

    test.each([
      [true, true],
      [false, false],
    ])('Can instantiate NatInstanceProviderV2 with associatePublicIpAddress', (input, value) => {
      const stack = getTestStack();
      new Vpc(stack, 'Vpc', {
        natGatewayProvider: NatProvider.instanceV2({
          instanceType: InstanceType.of(InstanceClass.T4G, InstanceSize.MICRO),
          associatePublicIpAddress: input,
        }),
        subnetConfiguration: [
          {
            subnetType: SubnetType.PUBLIC,
            name: 'Public',
            // NAT instance does not work when this set to false.
            mapPublicIpOnLaunch: false,
          },
          {
            subnetType: SubnetType.PRIVATE_WITH_EGRESS,
            name: 'Private',
          },
        ],
      });

      Template.fromStack(stack).hasResource('AWS::EC2::Instance', Match.objectLike({
        Properties: {
          NetworkInterfaces: [{
            AssociatePublicIpAddress: value,
          }],
        },
      }));
    });

    test('Can instantiate NatInstanceProvider directly with new', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: new NatInstanceProvider({
          instanceType: new InstanceType('q86.mega'),
          machineImage: new GenericLinuxImage({
            'us-east-1': 'ami-1',
          }),
        }),
        natGateways: 1,
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 1);
    });

    test('natGateways controls amount of NAT instances V2', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const natGatewayProvider = NatProvider.instanceV2({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
      });
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider,
        natGateways: 1,
      });

      // THEN
      expect(natGatewayProvider.gatewayInstances.length).toBe(1);
      Template.fromStack(stack).resourceCountIs('AWS::EC2::Instance', 1);
    });

    testDeprecated('can configure Security Groups of NAT instances with allowAllTraffic false', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const provider = NatProvider.instance({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
        allowAllTraffic: false,
      });
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: provider,
      });
      provider.connections.allowFrom(Peer.ipv4('1.2.3.4/32'), Port.tcp(86));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        SecurityGroupIngress: [
          {
            CidrIp: '1.2.3.4/32',
            Description: 'from 1.2.3.4/32:86',
            FromPort: 86,
            IpProtocol: 'tcp',
            ToPort: 86,
          },
        ],
      });
    });

    test('can configure Security Groups of NAT instances with defaultAllowAll INBOUND_AND_OUTBOUND', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const provider = NatProvider.instance({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
        defaultAllowedTraffic: NatTrafficDirection.INBOUND_AND_OUTBOUND,
      });
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: provider,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        SecurityGroupIngress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'from 0.0.0.0/0:ALL TRAFFIC',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('can configure Security Groups of NAT instances with defaultAllowAll OUTBOUND_ONLY', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const provider = NatProvider.instance({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
        defaultAllowedTraffic: NatTrafficDirection.OUTBOUND_ONLY,
      });
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: provider,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('can configure Security Groups of NAT instances with defaultAllowAll NONE', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const provider = NatProvider.instance({
        instanceType: new InstanceType('q86.mega'),
        machineImage: new GenericLinuxImage({
          'us-east-1': 'ami-1',
        }),
        defaultAllowedTraffic: NatTrafficDirection.NONE,
      });
      new Vpc(stack, 'TheVPC', {
        natGatewayProvider: provider,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '255.255.255.255/32',
            Description: 'Disallow all traffic',
            FromPort: 252,
            IpProtocol: 'icmp',
            ToPort: 86,
          },
        ],
      });
    });

    test('burstable instance with explicit credit specification', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const natInstanceProvider = NatProvider.instance({
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        machineImage: new AmazonLinuxImage(),
        creditSpecification: CpuCredits.STANDARD,
      });
      new Vpc(stack, 'VPC', {
        natGatewayProvider: natInstanceProvider,
        // The 'natGateways' parameter now controls the number of NAT instances
        natGateways: 1,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        InstanceType: 't3.large',
        CreditSpecification: {
          CPUCredits: 'standard',
        },
      });
    });
  });

  describe('Network ACL association', () => {
    test('by default uses default ACL reference', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const vpc = new Vpc(stack, 'TheVPC', { ipAddresses: IpAddresses.cidr('192.168.0.0/16') });
      new CfnOutput(stack, 'Output', {
        value: (vpc.publicSubnets[0] as Subnet).subnetNetworkAclAssociationId,
      });

      Template.fromStack(stack).templateMatches({
        Outputs: {
          Output: {
            Value: { 'Fn::GetAtt': ['TheVPCPublicSubnet1Subnet770D4FF2', 'NetworkAclAssociationId'] },
          },
        },
      });
    });

    test('if ACL is replaced new ACL reference is returned', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC', { ipAddresses: IpAddresses.cidr('192.168.0.0/16') });

      // WHEN
      new CfnOutput(stack, 'Output', {
        value: (vpc.publicSubnets[0] as Subnet).subnetNetworkAclAssociationId,
      });
      new NetworkAcl(stack, 'ACL', {
        vpc,
        subnetSelection: { subnetType: SubnetType.PUBLIC },
      });

      Template.fromStack(stack).templateMatches({
        Outputs: {
          Output: {
            Value: { Ref: 'ACLDBD1BB49' },
          },
        },
      });
    });

    test('with networkAclName, adds Name tag with the name', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC', { ipAddresses: IpAddresses.cidr('192.168.0.0/16') });

      // WHEN
      new NetworkAcl(stack, 'ACL', {
        vpc,
        networkAclName: 'CustomNetworkAclName',
      });

      Template.fromStack(stack).hasResource('AWS::EC2::NetworkAcl', hasTags([{
        Key: 'Name',
        Value: 'CustomNetworkAclName',
      }]));
    });
  });

  describe('When creating a VPC with a custom CIDR range', () => {
    test('vpc.vpcCidrBlock is the correct network range', () => {
      const stack = getTestStack();
      new Vpc(stack, 'TheVPC', { ipAddresses: IpAddresses.cidr('192.168.0.0/16') });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
        CidrBlock: '192.168.0.0/16',
      });
    });
  });
  describe('When tagging', () => {
    test('VPC propagated tags will be on subnet, IGW, routetables, NATGW', () => {
      const stack = getTestStack();
      const tags = {
        VpcType: 'Good',
      };
      const noPropTags = {
        BusinessUnit: 'Marketing',
      };
      const allTags = { ...noPropTags, ...tags };

      const vpc = new Vpc(stack, 'TheVPC');
      // overwrite to set propagate
      Tags.of(vpc).add('BusinessUnit', 'Marketing', { includeResourceTypes: [CfnVPC.CFN_RESOURCE_TYPE_NAME] });
      Tags.of(vpc).add('VpcType', 'Good');
      Template.fromStack(stack).hasResource('AWS::EC2::VPC', hasTags(toCfnTags(allTags)));
      const taggables = ['Subnet', 'InternetGateway', 'NatGateway', 'RouteTable'];
      const propTags = toCfnTags(tags);
      const noProp = toCfnTags(noPropTags);
      for (const resource of taggables) {
        Template.fromStack(stack).hasResourceProperties(`AWS::EC2::${resource}`, {
          Tags: Match.arrayWith(propTags),
        });
        const matchingResources = Template.fromStack(stack).findResources(`AWS::EC2::${resource}`, hasTags(noProp));
        expect(Object.keys(matchingResources).length).toBe(0);
      }
    });
    test('Subnet Name will propagate to route tables and NATGW', () => {
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'TheVPC');
      for (const subnet of vpc.publicSubnets) {
        const tag = { Key: 'Name', Value: subnet.node.path };
        Template.fromStack(stack).hasResource('AWS::EC2::NatGateway', hasTags([tag]));
        Template.fromStack(stack).hasResource('AWS::EC2::RouteTable', hasTags([tag]));
      }
      for (const subnet of vpc.privateSubnets) {
        const tag = { Key: 'Name', Value: subnet.node.path };
        Template.fromStack(stack).hasResource('AWS::EC2::RouteTable', hasTags([tag]));
      }
    });
    test('Tags can be added after the Vpc is created with `vpc.tags.setTag(...)`', () => {
      const stack = getTestStack();

      const vpc = new Vpc(stack, 'TheVPC');
      const tag = { Key: 'Late', Value: 'Adder' };
      Tags.of(vpc).add(tag.Key, tag.Value);
      Template.fromStack(stack).hasResource('AWS::EC2::VPC', hasTags([tag]));
    });
  });

  describe('subnet selection', () => {
    test('selecting default subnets returns the private ones', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      const { subnetIds } = vpc.selectSubnets();

      // THEN
      expect(subnetIds).toEqual(vpc.privateSubnets.map(s => s.subnetId));
    });

    test('can select public subnets', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ subnetType: SubnetType.PUBLIC });

      // THEN
      expect(subnetIds).toEqual(vpc.publicSubnets.map(s => s.subnetId));
    });

    test('can select isolated subnets', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'Public' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'Isolated' },
        ],
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ subnetType: SubnetType.PRIVATE_ISOLATED });

      // THEN
      expect(subnetIds).toEqual(vpc.isolatedSubnets.map(s => s.subnetId));
    });

    test('can select subnets by name', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'BlaBla' },
          { subnetType: SubnetType.PRIVATE_WITH_EGRESS, name: 'DontTalkToMe' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'DontTalkAtAll' },
        ],
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ subnetGroupName: 'DontTalkToMe' });

      // THEN
      expect(subnetIds).toEqual(vpc.privateSubnets.map(s => s.subnetId));
    });

    test('subnetName is an alias for subnetGroupName (backwards compat)', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC', {
        subnetConfiguration: [
          { subnetType: SubnetType.PUBLIC, name: 'BlaBla' },
          { subnetType: SubnetType.PRIVATE_WITH_EGRESS, name: 'DontTalkToMe' },
          { subnetType: SubnetType.PRIVATE_ISOLATED, name: 'DontTalkAtAll' },
        ],
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ subnetName: 'DontTalkToMe' });

      // THEN
      expect(subnetIds).toEqual(vpc.privateSubnets.map(s => s.subnetId));
    });

    test('selecting default subnets in a VPC with only isolated subnets returns the isolateds', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        isolatedSubnetIds: ['iso-1', 'iso-2', 'iso-3'],
        isolatedSubnetRouteTableIds: ['rt-1', 'rt-2', 'rt-3'],
      });

      // WHEN
      const subnets = vpc.selectSubnets();

      // THEN
      expect(subnets.subnetIds).toEqual(['iso-1', 'iso-2', 'iso-3']);
    });

    test('selecting default subnets in a VPC with only public subnets returns the publics', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        publicSubnetIds: ['pub-1', 'pub-2', 'pub-3'],
        publicSubnetRouteTableIds: ['rt-1', 'rt-2', 'rt-3'],
      });

      // WHEN
      const subnets = vpc.selectSubnets();

      // THEN
      expect(subnets.subnetIds).toEqual(['pub-1', 'pub-2', 'pub-3']);
    });

    test('selecting subnets by name fails if the name is unknown', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VPC');

      expect(() => {
        vpc.selectSubnets({ subnetGroupName: 'Toot' });
      }).toThrow(/There are no subnet groups with name 'Toot' in this VPC. Available names: Public,Private/);
    });

    test('select subnets with az restriction', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VpcNetwork', {
        maxAzs: 1,
        subnetConfiguration: [
          { name: 'lb', subnetType: SubnetType.PUBLIC },
          { name: 'app', subnetType: SubnetType.PRIVATE_WITH_EGRESS },
          { name: 'db', subnetType: SubnetType.PRIVATE_WITH_EGRESS },
        ],
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ onePerAz: true });

      // THEN
      expect(subnetIds.length).toEqual(1);
      expect(subnetIds[0]).toEqual(vpc.privateSubnets[0].subnetId);
    });

    test('fromVpcAttributes using unknown-length list tokens', () => {
      // GIVEN
      const stack = getTestStack();

      const vpcId = Fn.importValue('myVpcId');
      const availabilityZones = Fn.split(',', Fn.importValue('myAvailabilityZones'));
      const publicSubnetIds = Fn.split(',', Fn.importValue('myPublicSubnetIds'));

      // WHEN
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId,
        availabilityZones,
        publicSubnetIds,
      });

      new CfnResource(stack, 'Resource', {
        type: 'Some::Resource',
        properties: {
          subnetIds: vpc.selectSubnets().subnetIds,
        },
      });

      // THEN - No exception
      Template.fromStack(stack).hasResourceProperties('Some::Resource', {
        subnetIds: { 'Fn::Split': [',', { 'Fn::ImportValue': 'myPublicSubnetIds' }] },
      });

      Annotations.fromStack(stack).hasWarning('/TestStack/VPC', "fromVpcAttributes: 'availabilityZones' is a list token: the imported VPC will not work with constructs that require a list of subnets at synthesis time. Use 'Vpc.fromLookup()' or 'Fn.importListValue' instead. [ack: @aws-cdk/aws-ec2:vpcAttributeIsListTokenavailabilityZones]");
    });

    test('fromVpcAttributes using fixed-length list tokens', () => {
      // GIVEN
      const stack = getTestStack();

      const vpcId = Fn.importValue('myVpcId');
      const availabilityZones = Fn.importListValue('myAvailabilityZones', 2);
      const publicSubnetIds = Fn.importListValue('myPublicSubnetIds', 2);

      // WHEN
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId,
        availabilityZones,
        publicSubnetIds,
      });

      new CfnResource(stack, 'Resource', {
        type: 'Some::Resource',
        properties: {
          subnetIds: vpc.selectSubnets().subnetIds,
        },
      });

      // THEN - No exception

      const publicSubnetList = { 'Fn::Split': [',', { 'Fn::ImportValue': 'myPublicSubnetIds' }] };
      Template.fromStack(stack).hasResourceProperties('Some::Resource', {
        subnetIds: [
          { 'Fn::Select': [0, publicSubnetList] },
          { 'Fn::Select': [1, publicSubnetList] },
        ],
      });
    });

    test('select explicitly defined subnets', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        publicSubnetIds: ['pub-1', 'pub-2', 'pub-3'],
        publicSubnetRouteTableIds: ['rt-1', 'rt-2', 'rt-3'],
      });
      const subnet = new PrivateSubnet(stack, 'Subnet', {
        availabilityZone: vpc.availabilityZones[0],
        cidrBlock: '10.0.0.0/28',
        vpcId: vpc.vpcId,
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets({ subnets: [subnet] });

      // THEN
      expect(subnetIds.length).toEqual(1);
      expect(subnetIds[0]).toEqual(subnet.subnetId);
    });

    test('subnet created from subnetId', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const subnet = Subnet.fromSubnetId(stack, 'subnet1', 'pub-1');

      // THEN
      expect(subnet.subnetId).toEqual('pub-1');
    });

    test('Referencing AZ throws error when subnet created from subnetId', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const subnet = Subnet.fromSubnetId(stack, 'subnet1', 'pub-1');

      // THEN
      expect(() => subnet.availabilityZone).toThrow("You cannot reference a Subnet's availability zone if it was not supplied. Add the availabilityZone when importing using Subnet.fromSubnetAttributes()");
    });

    test('Referencing AZ throws error when subnet created from attributes without az', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const subnet = Subnet.fromSubnetAttributes(stack, 'subnet1', { subnetId: 'pub-1', availabilityZone: '' });

      // THEN
      expect(subnet.subnetId).toEqual('pub-1');
      expect(() => subnet.availabilityZone).toThrow("You cannot reference a Subnet's availability zone if it was not supplied. Add the availabilityZone when importing using Subnet.fromSubnetAttributes()");
    });

    test('AZ have value when subnet created from attributes with az', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const subnet = Subnet.fromSubnetAttributes(stack, 'subnet1', { subnetId: 'pub-1', availabilityZone: 'az-1234' });

      // THEN
      expect(subnet.subnetId).toEqual('pub-1');
      expect(subnet.availabilityZone).toEqual('az-1234');
    });

    test('Can select subnets by type and AZ', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC', {
        maxAzs: 3,
      });

      // WHEN
      new InterfaceVpcEndpoint(stack, 'VPC Endpoint', {
        vpc,
        privateDnsEnabled: false,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
        subnets: {
          subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          availabilityZones: ['dummy1a', 'dummy1c'],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        SubnetIds: [
          {
            Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
          },
          {
            Ref: 'VPCPrivateSubnet3Subnet3EDCD457',
          },
        ],
      });
    });

    test('SubnetSelection filtered on az uses default subnetType when no subnet type specified', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VPC', {
        maxAzs: 3,
      });

      // WHEN
      new InterfaceVpcEndpoint(stack, 'VPC Endpoint', {
        vpc,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
        subnets: {
          availabilityZones: ['dummy1a', 'dummy1c'],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        SubnetIds: [
          {
            Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
          },
          {
            Ref: 'VPCPrivateSubnet3Subnet3EDCD457',
          },
        ],
      });
    });
    test('SubnetSelection doesnt throw error when selecting imported subnets', () => {
      // GIVEN
      const stack = getTestStack();

      // WHEN
      const vpc = new Vpc(stack, 'VPC');

      // THEN
      expect(() => vpc.selectSubnets({
        subnets: [
          Subnet.fromSubnetId(stack, 'Subnet', 'sub-1'),
        ],
      })).not.toThrow();
    });

    test('can filter by single IP address', () => {
      // GIVEN
      const stack = getTestStack();

      // IP space is split into 6 pieces, one public/one private per AZ
      const vpc = new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        maxAzs: 3,
      });

      // WHEN
      // We want to place this bastion host in the same subnet as this IPv4
      // address.
      new BastionHostLinux(stack, 'Bastion', {
        vpc,
        subnetSelection: {
          subnetFilters: [SubnetFilter.containsIpAddresses(['10.0.160.0'])],
        },
      });

      // THEN
      // 10.0.160.0/19 is the third subnet, sequentially, if you split
      // 10.0.0.0/16 into 6 pieces
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        SubnetId: {
          Ref: 'VPCPrivateSubnet3Subnet3EDCD457',
        },
      });
    });

    test('can filter by multiple IP addresses', () => {
      // GIVEN
      const stack = getTestStack();

      // IP space is split into 6 pieces, one public/one private per AZ
      const vpc = new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        maxAzs: 3,
      });

      // WHEN
      // We want to place this endpoint in the same subnets as these IPv4
      // address.
      new InterfaceVpcEndpoint(stack, 'VPC Endpoint', {
        vpc,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
        subnets: {
          subnetFilters: [SubnetFilter.containsIpAddresses(['10.0.96.0', '10.0.160.0'])],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        SubnetIds: [
          {
            Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
          },
          {
            Ref: 'VPCPrivateSubnet3Subnet3EDCD457',
          },
        ],
      });
    });

    test('can filter by Subnet Ids', () => {
      // GIVEN
      const stack = getTestStack();

      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        vpcCidrBlock: '192.168.0.0/16',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        privateSubnetIds: ['priv-1', 'priv-2', 'priv-3'],
      });

      // WHEN
      new InterfaceVpcEndpoint(stack, 'VPC Endpoint', {
        vpc,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
        subnets: {
          subnetFilters: [SubnetFilter.byIds(['priv-1', 'priv-2'])],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        SubnetIds: ['priv-1', 'priv-2'],
      });
    });

    test('can filter by Subnet Ids via selectSubnets', () => {
      // GIVEN
      const stack = getTestStack();

      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'vpc-1234',
        vpcCidrBlock: '192.168.0.0/16',
        availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'],
        privateSubnetIds: ['subnet-1', 'subnet-2', 'subnet-3'],
      });

      // WHEN
      const subnets = vpc.selectSubnets({
        subnetFilters: [SubnetFilter.byIds(['subnet-1'])],
      });

      // THEN
      expect(subnets.subnetIds.length).toEqual(1);
    });

    test('can filter by Cidr Netmask', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'VpcNetwork', {
        maxAzs: 1,
        subnetConfiguration: [
          { name: 'normalSn1', subnetType: SubnetType.PUBLIC, cidrMask: 20 },
          { name: 'normalSn2', subnetType: SubnetType.PUBLIC, cidrMask: 20 },
          { name: 'smallSn', subnetType: SubnetType.PUBLIC, cidrMask: 28 },
        ],
      });

      // WHEN
      const { subnetIds } = vpc.selectSubnets(
        { subnetFilters: [SubnetFilter.byCidrMask(20)] },
      );

      // THEN
      expect(subnetIds.length).toEqual(2);
      const expected = vpc.publicSubnets.filter(s => s.ipv4CidrBlock.endsWith('/20'));
      expect(subnetIds).toEqual(expected.map(s => s.subnetId));
    });

    test('can filter by CIDR Range', () => {
      // GIVEN
      const stack = getTestStack();

      // IP space is split into 6 pieces, one public/one private per AZ
      const vpc = new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        maxAzs: 3,
      });

      // WHEN
      // We want to place this endpoint in subnets that are within a given CIDR range
      new InterfaceVpcEndpoint(stack, 'VPC Endpoint', {
        vpc,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
        subnets: {
          subnetFilters: [SubnetFilter.byCidrRanges(['10.0.0.0/16'])],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        SubnetIds: [
          {
            Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
          },
          {
            Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
          },
          {
            Ref: 'VPCPrivateSubnet3Subnet3EDCD457',
          },
        ],
      });
    });

    test('can filter by CIDR Range if CIDR is associated with VPC', () => {
      // GIVEN
      const stack = getTestStack();

      // IP space is split into 6 pieces, one public/one private per AZ
      const vpc = new Vpc(stack, 'VPC', {
        ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
        maxAzs: 3,
      });

      // WHEN
      const subnets = vpc.selectSubnets({
        subnetFilters: [SubnetFilter.byCidrRanges(['100.64.0.0/16'])],
      });

      // THEN
      expect(subnets.subnetIds.length).toEqual(0);
    });

    test('tests router types', () => {
      // GIVEN
      const stack = getTestStack();
      const vpc = new Vpc(stack, 'Vpc');

      // WHEN
      (vpc.publicSubnets[0] as Subnet).addRoute('TransitRoute', {
        routerType: RouterType.TRANSIT_GATEWAY,
        routerId: 'transit-id',
      });
      (vpc.publicSubnets[0] as Subnet).addRoute('CarrierRoute', {
        routerType: RouterType.CARRIER_GATEWAY,
        routerId: 'carrier-gateway-id',
      });
      (vpc.publicSubnets[0] as Subnet).addRoute('LocalGatewayRoute', {
        routerType: RouterType.LOCAL_GATEWAY,
        routerId: 'local-gateway-id',
      });
      (vpc.publicSubnets[0] as Subnet).addRoute('VpcEndpointRoute', {
        routerType: RouterType.VPC_ENDPOINT,
        routerId: 'vpc-endpoint-id',
      });
      (vpc.publicSubnets[0] as Subnet).addRoute('CoreNetworkRoute', {
        routerType: RouterType.CORE_NETWORK,
        routerId: 'core-network-arn',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        TransitGatewayId: 'transit-id',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        LocalGatewayId: 'local-gateway-id',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        CarrierGatewayId: 'carrier-gateway-id',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        VpcEndpointId: 'vpc-endpoint-id',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Route', {
        CoreNetworkArn: 'core-network-arn',
      });
    });
  });

  describe('Using reserved azs', () => {
    test.each([
      [{ maxAzs: 2, reservedAzs: 1 }, { maxAzs: 3 }],
      [{ maxAzs: 2, reservedAzs: 2 }, { maxAzs: 3, reservedAzs: 1 }],
      [{ maxAzs: 2, reservedAzs: 1, subnetConfiguration: [{ cidrMask: 22, name: 'Public', subnetType: SubnetType.PUBLIC }, { cidrMask: 23, name: 'Private', subnetType: SubnetType.PRIVATE_WITH_EGRESS }] },
        { maxAzs: 3, subnetConfiguration: [{ cidrMask: 22, name: 'Public', subnetType: SubnetType.PUBLIC }, { cidrMask: 23, name: 'Private', subnetType: SubnetType.PRIVATE_WITH_EGRESS }] }],
      [{ maxAzs: 2, reservedAzs: 1, subnetConfiguration: [{ cidrMask: 22, name: 'Public', subnetType: SubnetType.PUBLIC }, { cidrMask: 23, name: 'Private', subnetType: SubnetType.PRIVATE_WITH_EGRESS, reserved: true }] },
        { maxAzs: 3, subnetConfiguration: [{ cidrMask: 22, name: 'Public', subnetType: SubnetType.PUBLIC }, { cidrMask: 23, name: 'Private', subnetType: SubnetType.PRIVATE_WITH_EGRESS, reserved: true }] }],
      [{ maxAzs: 2, reservedAzs: 1, ipAddresses: IpAddresses.cidr('192.168.0.0/16') }, { maxAzs: 3, ipAddresses: IpAddresses.cidr('192.168.0.0/16') }],
      [{ availabilityZones: ['dummy1a', 'dummy1b'], reservedAzs: 1 }, { availabilityZones: ['dummy1a', 'dummy1b', 'dummy1c'] }],
    ])('subnets should remain the same going from %p to %p', (propsWithReservedAz, propsWithUsedReservedAz) => {
      const stackWithReservedAz = getTestStack();
      const stackWithUsedReservedAz = getTestStack();

      new Vpc(stackWithReservedAz, 'Vpc', propsWithReservedAz);
      new Vpc(stackWithUsedReservedAz, 'Vpc', propsWithUsedReservedAz);

      const templateWithReservedAz = Template.fromStack(stackWithReservedAz);
      const templateWithUsedReservedAz = Template.fromStack(stackWithUsedReservedAz);

      const subnetsOfTemplateWithReservedAz = templateWithReservedAz.findResources('AWS::EC2::Subnet');
      const subnetsOfTemplateWithUsedReservedAz = templateWithUsedReservedAz.findResources('AWS::EC2::Subnet');
      for (const [logicalId, subnetOfTemplateWithReservedAz] of Object.entries(subnetsOfTemplateWithReservedAz)) {
        const subnetOfTemplateWithUsedReservedAz = subnetsOfTemplateWithUsedReservedAz[logicalId];
        expect(subnetOfTemplateWithUsedReservedAz).toEqual(subnetOfTemplateWithReservedAz);
      }
    });
  });

  describe('can reference vpcEndpointDnsEntries across stacks', () => {
    test('can reference an actual string list across stacks', () => {
      const app = new App();
      const stack1 = new Stack(app, 'Stack1');
      const vpc = new Vpc(stack1, 'Vpc');
      const endpoint = new InterfaceVpcEndpoint(stack1, 'interfaceVpcEndpoint', {
        vpc,
        service: InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      });

      const stack2 = new Stack(app, 'Stack2');
      new CfnOutput(stack2, 'endpoint', {
        value: Fn.select(0, endpoint.vpcEndpointDnsEntries),
      });

      const assembly = app.synth();
      const template1 = assembly.getStackByName(stack1.stackName).template;
      const template2 = assembly.getStackByName(stack2.stackName).template;

      // THEN
      expect(template1).toMatchObject({
        Outputs: {
          ExportsOutputFnGetAttinterfaceVpcEndpoint89C99945DnsEntriesB1872F7A: {
            Value: {
              'Fn::Join': [
                '||', {
                  'Fn::GetAtt': [
                    'interfaceVpcEndpoint89C99945',
                    'DnsEntries',
                  ],
                },
              ],
            },
            Export: { Name: 'Stack1:ExportsOutputFnGetAttinterfaceVpcEndpoint89C99945DnsEntriesB1872F7A' },
          },
        },
      });

      expect(template2).toMatchObject({
        Outputs: {
          endpoint: {
            Value: {
              'Fn::Select': [
                0,
                {
                  'Fn::Split': [
                    '||',
                    {
                      'Fn::ImportValue': 'Stack1:ExportsOutputFnGetAttinterfaceVpcEndpoint89C99945DnsEntriesB1872F7A',
                    },
                  ],
                },
              ],
            },
          },
        },
      });
    });
  });

  test('dual-stack default', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');

    // WHEN
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCCidrBlock', {
      AmazonProvidedIpv6CidrBlock: true,
      VpcId: {
        Ref: Match.stringLikeRegexp('^Vpc.*'),
      },
    });
  });
  test('dual-stack IPv6 default route has DependsOn VPCGatewayAttachment', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');

    // WHEN
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
      maxAzs: 1,
    });

    // THEN - IPv6 default route (::/0) must depend on VPCGatewayAttachment;
    // CloudFormation rejects routes whose IGW has not yet been attached to the VPC.
    Template.fromStack(stack).hasResource('AWS::EC2::Route', {
      Properties: {
        DestinationIpv6CidrBlock: '::/0',
      },
      DependsOn: Match.arrayWith([Match.stringLikeRegexp('VPCGW')]),
    });
  });
  test('EgressOnlyIGW is created if no private subnet configured in dual stack and feature flag EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY is not enabled', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');

    // WHEN
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
      subnetConfiguration: [
        {
          subnetType: SubnetType.PUBLIC,
          name: 'public',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::EC2::EgressOnlyInternetGateway', 1);
  });
  test('EgressOnlyIGW is created if a private subnet is configured in dual stack and feature flag EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY is not enabled', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');

    // WHEN
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
      subnetConfiguration: [
        {
          subnetType: SubnetType.PUBLIC,
          name: 'public',
        },
        {
          subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          name: 'private',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::EC2::EgressOnlyInternetGateway', 1);
  });

  test('EgressOnlyIGW is created if a private subnet is configured in dual stack and feature flag EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY is enabled', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');
    // WHEN
    stack.node.setContext(EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY, true);
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
      subnetConfiguration: [
        {
          subnetType: SubnetType.PUBLIC,
          name: 'public',
        },
        {
          subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          name: 'private',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::EC2::EgressOnlyInternetGateway', 1);
  });
  test('EgressOnlyIGW is not created if no private subnet is configured in dual stack and feature flag EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY is enabled', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'DualStackStack');
    stack.node.setContext(EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY, true);
    // WHEN
    new Vpc(stack, 'Vpc', {
      ipProtocol: IpProtocol.DUAL_STACK,
      subnetConfiguration: [
        {
          subnetType: SubnetType.PUBLIC,
          name: 'public',
        },
      ],
    });
    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::EC2::EgressOnlyInternetGateway', 0);
  });

  test('error should occur if IPv6 properties are provided for a non-dual-stack VPC', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'NonDualStackStack');

    // WHEN
    expect(() => new Vpc(stack, 'Vpc', {
      ipv6Addresses: Ipv6Addresses.amazonProvided(),
    })).toThrow();
  });
});

function getTestStack(): Stack {
  const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });
  acknowledgeTestValidationRules(stack);
  return stack;
}

function toCfnTags(tags: any): Array<{Key: string; Value: string}> {
  return Object.keys(tags).map( key => {
    return { Key: key, Value: tags[key] };
  });
}

function hasTags(expectedTags: Array<{Key: string; Value: string}>) {
  return {
    Properties: {
      Tags: Match.arrayWith(expectedTags),
    },
  };
}
