import { Match, Template } from '../../assertions';
import { AnyPrincipal, PolicyStatement } from '../../aws-iam';
import * as cxschema from '../../cloud-assembly-schema';
import { ContextProvider, Fn, Stack } from '../../core';

import {
  GatewayVpcEndpoint,
  GatewayVpcEndpointAwsService,
  InterfaceVpcEndpoint,
  InterfaceVpcEndpointAwsService,
  InterfaceVpcEndpointService,
  SecurityGroup,
  SubnetFilter,
  SubnetType,
  Vpc,
  VpcEndpointDnsRecordIpType,
  VpcEndpointIpAddressType,
} from '../lib';

describe('vpc endpoint', () => {
  describe('gateway endpoint', () => {
    test('add an endpoint to a vpc', () => {
      // GIVEN
      const stack = new Stack();

      // WHEN
      new Vpc(stack, 'VpcNetwork', {
        gatewayEndpoints: {
          S3: {
            service: GatewayVpcEndpointAwsService.S3,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: {
          'Fn::Join': [
            '',
            [
              'com.amazonaws.',
              {
                Ref: 'AWS::Region',
              },
              '.s3',
            ],
          ],
        },
        VpcId: {
          Ref: 'VpcNetworkB258E83A',
        },
        RouteTableIds: [
          { Ref: 'VpcNetworkPrivateSubnet1RouteTableCD085FF1' },
          { Ref: 'VpcNetworkPrivateSubnet2RouteTableE97B328B' },
          { Ref: 'VpcNetworkPublicSubnet1RouteTable25CCC53F' },
          { Ref: 'VpcNetworkPublicSubnet2RouteTableE5F348DF' },
        ],
        VpcEndpointType: 'Gateway',
      });
    });

    test('routing on private and public subnets', () => {
      // GIVEN
      const stack = new Stack();

      // WHEN
      new Vpc(stack, 'VpcNetwork', {
        gatewayEndpoints: {
          S3: {
            service: GatewayVpcEndpointAwsService.S3,
            subnets: [
              {
                subnetType: SubnetType.PUBLIC,
              },
              {
                subnetType: SubnetType.PRIVATE_WITH_EGRESS,
              },
            ],
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: {
          'Fn::Join': [
            '',
            [
              'com.amazonaws.',
              {
                Ref: 'AWS::Region',
              },
              '.s3',
            ],
          ],
        },
        VpcId: {
          Ref: 'VpcNetworkB258E83A',
        },
        RouteTableIds: [
          {
            Ref: 'VpcNetworkPublicSubnet1RouteTable25CCC53F',
          },
          {
            Ref: 'VpcNetworkPublicSubnet2RouteTableE5F348DF',
          },
          {
            Ref: 'VpcNetworkPrivateSubnet1RouteTableCD085FF1',
          },
          {
            Ref: 'VpcNetworkPrivateSubnet2RouteTableE97B328B',
          },
        ],
        VpcEndpointType: 'Gateway',
      });
    });

    test('add statements to policy', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');
      const endpoint = vpc.addGatewayEndpoint('S3', {
        service: GatewayVpcEndpointAwsService.S3,
      });

      // WHEN
      endpoint.addToPolicy(new PolicyStatement({
        principals: [new AnyPrincipal()],
        actions: ['s3:GetObject', 's3:ListBucket'],
        resources: ['*'],
      }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                's3:GetObject',
                's3:ListBucket',
              ],
              Effect: 'Allow',
              Principal: { AWS: '*' },
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    test('throws when adding a statement without a principal', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');
      const endpoint = vpc.addGatewayEndpoint('S3', {
        service: GatewayVpcEndpointAwsService.S3,
      });

      // THEN
      expect(() => endpoint.addToPolicy(new PolicyStatement({
        actions: ['s3:GetObject', 's3:ListBucket'],
        resources: ['*'],
      }))).toThrow(/`Principal`/);
    });

    test('add an endpoint to a vpc and check that by default the properties are absent', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('EcrDocker', {
        service: InterfaceVpcEndpointAwsService.ECR_DOCKER,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        IpAddressType: Match.absent(),
        DnsOptions: Match.absent(),
      });
    });

    test('check ipAddressType and dnsOptions are present when specified', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('EcrDocker', {
        service: InterfaceVpcEndpointAwsService.ECR_DOCKER,
        ipAddressType: VpcEndpointIpAddressType.DUALSTACK,
        dnsRecordIpType: VpcEndpointDnsRecordIpType.DUALSTACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        IpAddressType: VpcEndpointIpAddressType.DUALSTACK,
        DnsOptions: { DnsRecordIpType: VpcEndpointDnsRecordIpType.DUALSTACK },
      });
    });

    test('import/export', () => {
      // GIVEN
      const stack2 = new Stack();

      // WHEN
      const ep = GatewayVpcEndpoint.fromGatewayVpcEndpointId(stack2, 'ImportedEndpoint', 'endpoint-id');

      // THEN
      expect(ep.vpcEndpointId).toEqual('endpoint-id');
    });

    test('works with an imported vpc', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'id',
        privateSubnetIds: ['1', '2', '3'],
        privateSubnetRouteTableIds: ['rt1', 'rt2', 'rt3'],
        availabilityZones: ['a', 'b', 'c'],
      });

      // THEN
      vpc.addGatewayEndpoint('Gateway', { service: GatewayVpcEndpointAwsService.S3 });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: { 'Fn::Join': ['', ['com.amazonaws.', { Ref: 'AWS::Region' }, '.s3']] },
        VpcId: 'id',
        RouteTableIds: ['rt1', 'rt2', 'rt3'],
        VpcEndpointType: 'Gateway',
      });
    });

    test('throws with an imported vpc without route tables ids', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = Vpc.fromVpcAttributes(stack, 'VPC', {
        vpcId: 'id',
        privateSubnetIds: ['1', '2', '3'],
        availabilityZones: ['a', 'b', 'c'],
      });

      expect(() => vpc.addGatewayEndpoint('Gateway', { service: GatewayVpcEndpointAwsService.S3 })).toThrow(/route table/);
    });

    test('gateway endpoint with ipAddressType', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      new GatewayVpcEndpoint(stack, 'Endpoint', {
        vpc,
        service: GatewayVpcEndpointAwsService.S3,
        ipAddressType: VpcEndpointIpAddressType.DUALSTACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        VpcEndpointType: 'Gateway',
        IpAddressType: 'dualstack',
      });
    });

    test('gateway endpoint with dnsRecordIpType', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      new GatewayVpcEndpoint(stack, 'Endpoint', {
        vpc,
        service: GatewayVpcEndpointAwsService.S3,
        dnsRecordIpType: VpcEndpointDnsRecordIpType.IPV6,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        VpcEndpointType: 'Gateway',
        DnsOptions: { DnsRecordIpType: 'ipv6' },
      });
    });

    test('gateway endpoint without ipAddressType or dnsRecordIpType omits those fields', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      new GatewayVpcEndpoint(stack, 'Endpoint', {
        vpc,
        service: GatewayVpcEndpointAwsService.S3,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        VpcEndpointType: 'Gateway',
        IpAddressType: Match.absent(),
        DnsOptions: Match.absent(),
      });
    });
  });

  describe('interface endpoint', () => {
    test('add an endpoint to a vpc', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('EcrDocker', {
        service: InterfaceVpcEndpointAwsService.ECR_DOCKER,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: {
          'Fn::Join': [
            '',
            [
              'com.amazonaws.',
              {
                Ref: 'AWS::Region',
              },
              '.ecr.dkr',
            ],
          ],
        },
        VpcId: {
          Ref: 'VpcNetworkB258E83A',
        },
        PrivateDnsEnabled: true,
        SecurityGroupIds: [
          {
            'Fn::GetAtt': [
              'VpcNetworkEcrDockerSecurityGroup7C91D347',
              'GroupId',
            ],
          },
        ],
        SubnetIds: [
          {
            Ref: 'VpcNetworkPrivateSubnet1Subnet07BA143B',
          },
          {
            Ref: 'VpcNetworkPrivateSubnet2Subnet5E4189D6',
          },
        ],
        VpcEndpointType: 'Interface',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Default/VpcNetwork/EcrDocker/SecurityGroup',
        VpcId: {
          Ref: 'VpcNetworkB258E83A',
        },
      });
    });

    describe('interface endpoint retains service name in shortName property', () => {
      test('shortName property', () => {
        expect(InterfaceVpcEndpointAwsService.ECS.shortName).toBe('ecs');
        expect(InterfaceVpcEndpointAwsService.ECR_DOCKER.shortName).toBe('ecr.dkr');
      });
    });

    describe('add interface endpoint to looked-up VPC', () => {
      test('initial run', () => {
        // GIVEN
        const stack = new Stack(undefined, undefined, { env: { account: '1234', region: 'us-east-1' } });
        const vpc = Vpc.fromLookup(stack, 'Vpc', {
          vpcId: 'doesnt-matter',
        });

        // THEN: doesn't throw
        vpc.addInterfaceEndpoint('SecretsManagerEndpoint', {
          service: InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
          subnets: {
            subnetFilters: [SubnetFilter.byIds(['1234'])],
          },
        });
      });
    });

    test('import/export', () => {
      // GIVEN
      const stack2 = new Stack();

      // WHEN
      const importedEndpoint = InterfaceVpcEndpoint.fromInterfaceVpcEndpointAttributes(stack2, 'ImportedEndpoint', {
        securityGroups: [SecurityGroup.fromSecurityGroupId(stack2, 'SG', 'security-group-id')],
        vpcEndpointId: 'vpc-endpoint-id',
        port: 80,
      });
      importedEndpoint.connections.allowDefaultPortFromAnyIpv4();

      // THEN
      Template.fromStack(stack2).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        GroupId: 'security-group-id',
      });
      expect(importedEndpoint.vpcEndpointId).toEqual('vpc-endpoint-id');
    });

    test('import/export without security group', () => {
      // GIVEN
      const stack2 = new Stack();

      // WHEN
      const importedEndpoint = InterfaceVpcEndpoint.fromInterfaceVpcEndpointAttributes(stack2, 'ImportedEndpoint', {
        vpcEndpointId: 'vpc-endpoint-id',
        port: 80,
      });
      importedEndpoint.connections.allowDefaultPortFromAnyIpv4();

      // THEN
      expect(importedEndpoint.vpcEndpointId).toEqual('vpc-endpoint-id');
      expect(importedEndpoint.connections.securityGroups.length).toEqual(0);
    });

    test('with existing security groups', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('EcrDocker', {
        service: InterfaceVpcEndpointAwsService.ECR_DOCKER,
        securityGroups: [SecurityGroup.fromSecurityGroupId(stack, 'SG', 'existing-id')],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        SecurityGroupIds: ['existing-id'],
      });
    });
    test('with existing security groups for efs', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('Efs', {
        service: InterfaceVpcEndpointAwsService.ELASTIC_FILESYSTEM,
        securityGroups: [SecurityGroup.fromSecurityGroupId(stack, 'SG', 'existing-id')],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        SecurityGroupIds: ['existing-id'],
      });
    });
    test('security group has ingress by default', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('SecretsManagerEndpoint', {
        service: InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupIngress: [
          Match.objectLike({
            CidrIp: { 'Fn::GetAtt': ['VpcNetworkB258E83A', 'CidrBlock'] },
            FromPort: 443,
            IpProtocol: 'tcp',
            ToPort: 443,
          }),
        ],
      });
    });
    test('non-AWS service interface endpoint', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('YourService', {
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc', 443),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
        PrivateDnsEnabled: false,
      });
    });
    test('marketplace partner service interface endpoint', () => {
      // GIVEN
      const stack = new Stack();
      const vpc = new Vpc(stack, 'VpcNetwork');

      // WHEN
      vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-mktplacesvcwprdns',
          port: 443,
          privateDnsDefault: true,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-mktplacesvcwprdns',
        PrivateDnsEnabled: true,
      });
    });
    test('test endpoint service context azs discovered', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });

      // Setup context for stack AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.AVAILABILITY_ZONE_PROVIDER,
        }).key,
        ['us-east-1a', 'us-east-1b', 'us-east-1c']);
      // Setup context for endpoint service AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.ENDPOINT_SERVICE_AVAILABILITY_ZONE_PROVIDER,
          props: {
            serviceName: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          },
        }).key,
        ['us-east-1a', 'us-east-1c']);

      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        lookupSupportedAzs: true,
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
    test('endpoint service setup with stack AZ context but no endpoint context', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });

      // Setup context for stack AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.AVAILABILITY_ZONE_PROVIDER,
        }).key,
        ['us-east-1a', 'us-east-1b', 'us-east-1c']);

      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        lookupSupportedAzs: true,
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
    test('test endpoint service context with aws service', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });

      // Setup context for stack AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.AVAILABILITY_ZONE_PROVIDER,
        }).key,
        ['us-east-1a', 'us-east-1b', 'us-east-1c']);
      // Setup context for endpoint service AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.ENDPOINT_SERVICE_AVAILABILITY_ZONE_PROVIDER,
          props: {
            serviceName: 'com.amazonaws.us-east-1.execute-api',
          },
        }).key,
        ['us-east-1a', 'us-east-1c']);

      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('API Gateway', {
        service: InterfaceVpcEndpointAwsService.APIGATEWAY,
        lookupSupportedAzs: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-east-1.execute-api',
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
    test('lookupSupportedAzs fails if account is unresolved', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { region: 'us-east-1' } });
      const vpc = new Vpc(stack, 'VPC');
      // WHEN
      expect(() => vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        lookupSupportedAzs: true,
      })).toThrow();
    });
    test('lookupSupportedAzs fails if region is unresolved', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012' } });
      const vpc = new Vpc(stack, 'VPC');
      // WHEN
      expect(() => vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        lookupSupportedAzs: true,
      })).toThrow();
    });
    test('lookupSupportedAzs fails if subnet AZs are tokens', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });
      const tokenAZs = [
        'us-east-1a',
        Fn.select(1, Fn.getAzs()),
        Fn.select(2, Fn.getAzs()),
      ];
      // Setup context for stack AZs
      stack.node.setContext(
        ContextProvider.getKey(stack, {
          provider: cxschema.ContextProvider.AVAILABILITY_ZONE_PROVIDER,
        }).key,
        tokenAZs);
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      expect(() => vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        lookupSupportedAzs: true,
      })).toThrow();
    });
    test('vpc endpoint fails if no subnets provided', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });
      const vpc = new Vpc(stack, 'VPC');
      // WHEN
      expect(() => vpc.addInterfaceEndpoint('YourService', {
        service: {
          name: 'com.amazonaws.vpce.us-east-1.vpce-svc-uuddlrlrbastrtsvc',
          port: 443,
        },
        subnets: vpc.selectSubnets({
          subnets: [],
        }),
      })).toThrow();
    });
    test('test vpc interface endpoint with cn.com.amazonaws prefix can be created correctly in cn-north-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-north-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECR Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'cn.com.amazonaws.cn-north-1.ecr.api',
      });
    });
    test('test vpc interface endpoint with cn.com.amazonaws prefix can be created correctly in cn-northwest-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-northwest-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Lambda Endpoint', {
        service: InterfaceVpcEndpointAwsService.LAMBDA,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'cn.com.amazonaws.cn-northwest-1.lambda',
      });
    });
    test('test vpc interface endpoint without cn.com.amazonaws prefix can be created correctly in cn-north-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-north-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.cn-north-1.ecs',
      });
    });
    test('test vpc interface endpoint without cn.com.amazonaws prefix can be created correctly in cn-northwest-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-northwest-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Glue Endpoint', {
        service: InterfaceVpcEndpointAwsService.GLUE,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.cn-northwest-1.glue',
      });
    });
    test('test vpc interface endpoint for transcribe can be created correctly in non-china regions', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-east-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Transcribe Endpoint', {
        service: InterfaceVpcEndpointAwsService.TRANSCRIBE,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-east-1.transcribe',
      });
    });
    test.each([
      ['transcribe', InterfaceVpcEndpointAwsService.TRANSCRIBE],
    ])('test vpc interface endpoint with .cn suffix for %s can be created correctly in China regions', (name: string, given: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack1 = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-north-1' } });
      const stack2 = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-northwest-1' } });
      const vpc1 = new Vpc(stack1, 'VPC');
      const vpc2 = new Vpc(stack2, 'VPC');

      // WHEN
      vpc1.addInterfaceEndpoint('Endpoint', { service: given });
      vpc2.addInterfaceEndpoint('Endpoint', { service: given });

      // THEN
      Template.fromStack(stack1).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-north-1.${name}.cn`,
      });
      Template.fromStack(stack2).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-northwest-1.${name}.cn`,
      });
    });
    test.each([
      ['application-autoscaling', InterfaceVpcEndpointAwsService.APPLICATION_AUTOSCALING],
      ['appmesh-envoy-management', InterfaceVpcEndpointAwsService.APP_MESH],
      ['athena', InterfaceVpcEndpointAwsService.ATHENA],
      ['autoscaling', InterfaceVpcEndpointAwsService.AUTOSCALING],
      ['awsconnector', InterfaceVpcEndpointAwsService.SERVER_MIGRATION_SERVICE_AWSCONNECTOR],
      ['backup', InterfaceVpcEndpointAwsService.BACKUP],
      ['batch', InterfaceVpcEndpointAwsService.BATCH],
      ['cassandra', InterfaceVpcEndpointAwsService.KEYSPACES],
      ['cloudcontrolapi', InterfaceVpcEndpointAwsService.CLOUD_CONTROL_API],
      ['cloudformation', InterfaceVpcEndpointAwsService.CLOUDFORMATION],
      ['cloudformation', InterfaceVpcEndpointAwsService.CLOUDFORMATION],
      ['codedeploy-commands-secure', InterfaceVpcEndpointAwsService.CODEDEPLOY_COMMANDS_SECURE],
      ['databrew', InterfaceVpcEndpointAwsService.GLUE_DATABREW],
      ['dms', InterfaceVpcEndpointAwsService.DATABASE_MIGRATION_SERVICE],
      ['ebs', InterfaceVpcEndpointAwsService.EBS_DIRECT],
      ['ec2', InterfaceVpcEndpointAwsService.EC2],
      ['ecr.api', InterfaceVpcEndpointAwsService.ECR],
      ['ecr.dkr', InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ['eks', InterfaceVpcEndpointAwsService.EKS],
      ['elasticache', InterfaceVpcEndpointAwsService.ELASTICACHE],
      ['elasticbeanstalk', InterfaceVpcEndpointAwsService.ELASTIC_BEANSTALK],
      ['elasticfilesystem', InterfaceVpcEndpointAwsService.ELASTIC_FILESYSTEM],
      ['elasticfilesystem-fips', InterfaceVpcEndpointAwsService.ELASTIC_FILESYSTEM_FIPS],
      ['emr-containers', InterfaceVpcEndpointAwsService.EMR_EKS],
      ['execute-api', InterfaceVpcEndpointAwsService.APIGATEWAY],
      ['fsx', InterfaceVpcEndpointAwsService.FSX],
      ['imagebuilder', InterfaceVpcEndpointAwsService.IMAGE_BUILDER],
      ['iot.data', InterfaceVpcEndpointAwsService.IOT_CORE],
      ['kinesis-streams', InterfaceVpcEndpointAwsService.KINESIS_STREAMS],
      ['lambda', InterfaceVpcEndpointAwsService.LAMBDA],
      ['license-manager', InterfaceVpcEndpointAwsService.LICENSE_MANAGER],
      ['monitoring', InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING],
      ['rds', InterfaceVpcEndpointAwsService.RDS],
      ['redshift', InterfaceVpcEndpointAwsService.REDSHIFT],
      ['redshift-data', InterfaceVpcEndpointAwsService.REDSHIFT_DATA],
      ['s3', InterfaceVpcEndpointAwsService.S3],
      ['sagemaker.api', InterfaceVpcEndpointAwsService.SAGEMAKER_API],
      ['sagemaker.featurestore-runtime', InterfaceVpcEndpointAwsService.SAGEMAKER_FEATURESTORE_RUNTIME],
      ['sagemaker.runtime', InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME],
      ['securityhub', InterfaceVpcEndpointAwsService.SECURITYHUB],
      ['servicecatalog', InterfaceVpcEndpointAwsService.SERVICE_CATALOG],
      ['sms', InterfaceVpcEndpointAwsService.SERVER_MIGRATION_SERVICE],
      ['sqs', InterfaceVpcEndpointAwsService.SQS],
      ['states', InterfaceVpcEndpointAwsService.STEP_FUNCTIONS],
      ['states', InterfaceVpcEndpointAwsService.STEP_FUNCTIONS],
      ['sts', InterfaceVpcEndpointAwsService.STS],
      ['sync-states', InterfaceVpcEndpointAwsService.STEP_FUNCTIONS_SYNC],
      ['synthetics', InterfaceVpcEndpointAwsService.CLOUDWATCH_SYNTHETICS],
      ['transcribestreaming', InterfaceVpcEndpointAwsService.TRANSCRIBE_STREAMING],
      ['transfer', InterfaceVpcEndpointAwsService.TRANSFER],
      ['xray', InterfaceVpcEndpointAwsService.XRAY],
    ])('test vpc interface endpoint for %s can be created correctly in China regions', (name: string, given: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack1 = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-north-1' } });
      const stack2 = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-northwest-1' } });
      const vpc1 = new Vpc(stack1, 'VPC');
      const vpc2 = new Vpc(stack2, 'VPC');

      // WHEN
      vpc1.addInterfaceEndpoint('Endpoint', { service: given });
      vpc2.addInterfaceEndpoint('Endpoint', { service: given });

      // THEN
      Template.fromStack(stack1).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-north-1.${name}`,
      });
      Template.fromStack(stack2).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-northwest-1.${name}`,
      });
    });
    test.each([
      ['iotsitewise.api', InterfaceVpcEndpointAwsService.IOT_SITEWISE_API],
      ['iotsitewise.data', InterfaceVpcEndpointAwsService.IOT_SITEWISE_DATA],
    ])('test vpc interface endpoint for %s can be created correctly in cn-north-1 only', (name: string, given: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-north-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Endpoint', { service: given });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-north-1.${name}`,
      });
    });
    test.each([
      ['account', InterfaceVpcEndpointAwsService.ACCOUNT_MANAGEMENT],
      ['workspaces', InterfaceVpcEndpointAwsService.WORKSPACES],
    ])('test vpc interface endpoint for %s can be created correctly in cn-northwest-1 only', (name: string, given: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'cn-northwest-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Endpoint', { service: given });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `cn.com.amazonaws.cn-northwest-1.${name}`,
      });
    });

    test('test vpc interface endpoint with eu.amazonaws prefix can be created correctly in eusc-de-east-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'eusc-de-east-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECR Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'eu.amazonaws.eusc-de-east-1.ecr.api',
      });
    });

    test('test vpc interface endpoint without eu.amazonaws prefix can be created correctly in eusc-de-east-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'eusc-de-east-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.eusc-de-east-1.ecs',
      });
    });

    test.each([
      ['ecr.api', InterfaceVpcEndpointAwsService.ECR],
      ['ecr.dkr', InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ['execute-api', InterfaceVpcEndpointAwsService.APIGATEWAY],
      ['securityhub', InterfaceVpcEndpointAwsService.SECURITYHUB],
    ])('test vpc interface endpoint for %s can be created correctly in eusc-de-east-1', (name: string, given: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'eusc-de-east-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Endpoint', { service: given });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `eu.amazonaws.eusc-de-east-1.${name}`,
      });
    });

    test.each([
      ['us-iso-east-1', 'gov.ic.c2s'],
      ['us-iso-west-1', 'gov.ic.c2s'],
    ])('test vpc interface endpoint with %s prefix can be created correctly in %s', (region: string, prefix: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECR Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `${prefix}.${region}.ecr.api`,
      });
    });

    test.each([
      ['us-iso-east-1'],
      ['us-iso-west-1'],
    ])('test vpc interface endpoint without gov.ic.c2s prefix can be created correctly in %s', (region: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `com.amazonaws.${region}.ecs`,
      });
    });

    test.each([
      ['us-isob-east-1', 'gov.sgov.sc2s'],
      ['us-isob-west-1', 'gov.sgov.sc2s'],
    ])('test vpc interface endpoint with %s prefix can be created correctly in %s', (region: string, prefix: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECR Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `${prefix}.${region}.ecr.api`,
      });
    });

    test.each([
      ['us-isob-east-1'],
      ['us-isob-west-1'],
    ])('test vpc interface endpoint without gov.sgov.sc2s prefix can be created correctly in %s', (region: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `com.amazonaws.${region}.ecs`,
      });
    });

    test.each([
      ['us-isof-south-1', 'gov.ic.hci.csp'],
      ['us-isof-east-1', 'gov.ic.hci.csp'],
    ])('test vpc interface endpoint with %s prefix can be created correctly in %s', (region: string, prefix: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECR Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `${prefix}.${region}.ecr.api`,
      });
    });

    test.each([
      ['us-isof-south-1'],
      ['us-isof-east-1'],
    ])('test vpc interface endpoint without gov.ic.hci.csp prefix can be created correctly in %s', (region: string) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `com.amazonaws.${region}.ecs`,
      });
    });

    test.each([
      ['ebs', InterfaceVpcEndpointAwsService.EBS_DIRECT],
      ['ecr.api', InterfaceVpcEndpointAwsService.ECR],
      ['ecr.dkr', InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ['execute-api', InterfaceVpcEndpointAwsService.APIGATEWAY],
    ])('test vpc interface endpoint for %s can be created correctly in eu-isoe-west-1', (name: string, service: InterfaceVpcEndpointAwsService) => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'eu-isoe-west-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Endpoint', { service });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: `uk.adc-e.cloud.eu-isoe-west-1.${name}`,
      });
    });

    test('test vpc interface endpoint without uk.adc-e.cloud prefix can be created correctly in eu-isoe-west-1', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'eu-isoe-west-1' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('ECS Endpoint', {
        service: InterfaceVpcEndpointAwsService.ECS,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.eu-isoe-west-1.ecs',
      });
    });

    test('test codeartifact vpc interface endpoint in us-west-2', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('CodeArtifact API Endpoint', {
        service: InterfaceVpcEndpointAwsService.CODEARTIFACT_API,
      });

      vpc.addInterfaceEndpoint('CodeArtifact Repositories Endpoint', {
        service: InterfaceVpcEndpointAwsService.CODEARTIFACT_REPOSITORIES,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.codeartifact.repositories',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.codeartifact.api',
      });
    });

    test('test s3 vpc interface endpoint in us-west-2', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('CodeArtifact API Endpoint', {
        service: InterfaceVpcEndpointAwsService.S3,
      });

      // THEN

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.s3',
      });
    });

    test('test batch vpc interface endpoint in us-west-2', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('CodeArtifact API Endpoint', {
        service: InterfaceVpcEndpointAwsService.BATCH,
      });

      // THEN

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.batch',
      });
    });

    test('test autoscaling vpc interface endpoint in us-west-2', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Autoscaling API Endpoint', {
        service: InterfaceVpcEndpointAwsService.AUTOSCALING,
      });

      vpc.addInterfaceEndpoint('Autoscaling-plan API Endpoint', {
        service: InterfaceVpcEndpointAwsService.AUTOSCALING_PLANS,
      });

      vpc.addInterfaceEndpoint('Application-Autoscaling API Endpoint', {
        service: InterfaceVpcEndpointAwsService.APPLICATION_AUTOSCALING,
      });

      // THEN

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.autoscaling',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.autoscaling-plans',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.application-autoscaling',
      });
    });

    test('global vpc interface endpoints', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('Global S3 API Endpoint', {
        service: InterfaceVpcEndpointAwsService.S3_MULTI_REGION_ACCESS_POINTS,
      });
      vpc.addInterfaceEndpoint('Global CodeCatalyst API Endpoint', {
        service: InterfaceVpcEndpointAwsService.CODECATALYST,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.s3-global.accesspoint',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'aws.api.global.codecatalyst',
      });
    });

    test('test vpc interface endpoint with private dns disabled', () => {
      // GIVEN
      const stack = new Stack(undefined, 'TestStack', { env: { account: '123456789012', region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      vpc.addInterfaceEndpoint('DynamoDB Endpoint', {
        service: InterfaceVpcEndpointAwsService.DYNAMODB,
        privateDnsEnabled: false,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceName: 'com.amazonaws.us-west-2.dynamodb',
        VpcId: stack.resolve(vpc.vpcId),
        PrivateDnsEnabled: false,
        VpcEndpointType: 'Interface',
      });
    });

    test('same-region endpoint does not include ServiceRegion property', () => {
      // GIVEN
      const stack = new Stack(undefined, undefined, { env: { region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN - Same region endpoint without serviceRegion
      new InterfaceVpcEndpoint(stack, 'Endpoint', {
        vpc,
        service: InterfaceVpcEndpointAwsService.S3,
        // No serviceRegion specified - same region default
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        VpcEndpointType: 'Interface',
        ServiceName: 'com.amazonaws.us-west-2.s3',
        ServiceRegion: Match.absent(),
      });
    });

    test('cross-region endpoint includes correct ServiceRegion property', () => {
      // GIVEN
      const stack = new Stack(undefined, undefined, { env: { region: 'us-west-2' } });
      const vpc = new Vpc(stack, 'VPC');

      // WHEN
      new InterfaceVpcEndpoint(stack, 'Endpoint', {
        vpc,
        service: new InterfaceVpcEndpointService('com.amazonaws.vpce.us-east-1.vpce-svc-123456', 443),
        serviceRegion: 'us-east-1',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPCEndpoint', {
        ServiceRegion: 'us-east-1',
      });
    });
  });
});
