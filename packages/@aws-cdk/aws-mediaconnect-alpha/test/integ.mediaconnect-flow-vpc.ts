import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';

import { Flow } from '../lib/flow';
import { SourceConfiguration, NetworkConfiguration } from '../lib/flow-source-configuration';
import { VpcInterface } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-vpc');

const vpc = new Vpc(stack, 'testvpc', {
  ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [{
    name: 'subnet1',
    subnetType: SubnetType.PRIVATE_ISOLATED,
    cidrMask: 24,
  }],
});
const subnet = new PrivateSubnet(stack, 'mysubnet', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.172.0/24',
  vpcId: vpc.vpcId,
});
const role = new Role(stack, 'my-role', {
  assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  inlinePolicies: {
    default: new PolicyDocument({
      statements: [
        new PolicyStatement({
          actions: ['ec2:CreateNetworkInterface', 'ec2:CreateNetworkInterfacePermission', 'ec2:DeleteNetworkInterface', 'ec2:DescribeSubnets'],
          resources: ['*'],
          effect: Effect.ALLOW,
        }),
      ],
    }),
  },
});

const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'test',
  role,
  securityGroups: [new SecurityGroup(stack, 'sg', {
    vpc,
  })],
  subnet,
});

new Flow(stack, 'flow', {
  vpcInterfaces: [vpcInterface],
  source: SourceConfiguration.rtpFec({
    flowSourceName: 'my-flow',
    description: 'my test flow',
    maxBitrate: cdk.Bitrate.mbps(10),
    port: 2099,
    network: NetworkConfiguration.vpc(vpcInterface),
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-vpc', {
  testCases: [stack],
});

app.synth();
