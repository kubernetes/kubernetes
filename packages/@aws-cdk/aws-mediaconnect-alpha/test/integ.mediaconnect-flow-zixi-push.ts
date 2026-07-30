/**
 * Integration test: Zixi Push source.
 *
 * MediaConnect assigns the Zixi Push ingest port automatically:
 * - Standard (public) sources use port 2088.
 * - VPC sources are auto-assigned a port in 2090-2099 (reserved exclusively for
 *   VPC Zixi).
 *
 * The L2 hardcodes `ingestPort: 2088` on the CFN shape and does not expose a port
 * field. This test deploys both the standard and VPC variants to confirm both
 * paths are accepted by the service.
 *
 * @see https://docs.aws.amazon.com/mediaconnect/latest/ug/source-ports.html
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';

import { Flow } from '../lib/flow';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { VpcInterface } from '../lib/shared';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-zixi-push');

// Standard Zixi Push source on the public internet.
new Flow(stack, 'StandardFlow', {
  flowName: 'zixi-push-standard-flow',
  source: SourceConfiguration.zixiPush({
    flowSourceName: 'zixi-push-standard-source',
    network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    maxLatency: cdk.Duration.millis(6000),
    description: 'Zixi Push standard (port 2088) integ test',
  }),
});

// VPC Zixi Push source — MediaConnect auto-assigns an ingest port in 2090-2099.
const vpc = new Vpc(stack, 'Vpc', {
  ipAddresses: IpAddresses.cidr('10.1.0.0/16'),
  subnetConfiguration: [{
    name: 'subnet1',
    subnetType: SubnetType.PRIVATE_ISOLATED,
    cidrMask: 24,
  }],
});
const subnet = new PrivateSubnet(stack, 'Subnet', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.1.172.0/24',
  vpcId: vpc.vpcId,
});
const role = new Role(stack, 'VpcInterfaceRole', {
  assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  inlinePolicies: {
    default: new PolicyDocument({
      statements: [
        new PolicyStatement({
          actions: [
            'ec2:CreateNetworkInterface',
            'ec2:CreateNetworkInterfacePermission',
            'ec2:DeleteNetworkInterface',
            'ec2:DescribeSubnets',
          ],
          resources: ['*'],
          effect: Effect.ALLOW,
        }),
      ],
    }),
  },
});
const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'vpc-iface',
  role,
  securityGroups: [new SecurityGroup(stack, 'Sg', { vpc })],
  subnet,
});

new Flow(stack, 'VpcFlow', {
  flowName: 'zixi-push-vpc-flow',
  vpcInterfaces: [vpcInterface],
  source: SourceConfiguration.zixiPush({
    flowSourceName: 'zixi-push-vpc-source',
    network: NetworkConfiguration.vpc(vpcInterface),
    maxLatency: cdk.Duration.millis(6000),
    description: 'Zixi Push VPC (service-assigned port) integ test',
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-zixi-push', {
  testCases: [stack],
});

app.synth();
