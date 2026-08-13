/**
 * Integration test: output protocols not already covered by the kitchen-sink.
 *
 * Exercises the remaining unencrypted output protocols end-to-end on both
 * public and VPC network paths:
 * - RTP
 * - RIST
 * - SRT Caller
 *
 * (Zixi Pull, Zixi Push, SRT Listener, NDI, and RTP-FEC outputs are already
 * covered by the kitchen-sink integ.)
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';

import { Flow, FlowOutput, OutputConfiguration } from '../lib';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { VpcInterface } from '../lib/shared';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-outputs');

// VPC and interface shared by all three VPC outputs.
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
const vpcRole = new Role(stack, 'VpcInterfaceRole', {
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
  vpcInterfaceName: 'outputs-vpc-iface',
  role: vpcRole,
  securityGroups: [new SecurityGroup(stack, 'Sg', { vpc })],
  subnet,
});

const flow = new Flow(stack, 'Flow', {
  flowName: 'outputs_integ_flow',
  vpcInterfaces: [vpcInterface],
  source: SourceConfiguration.rtp({
    flowSourceName: 'ingest',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
  }),
});

// Public outputs.
new FlowOutput(stack, 'RtpOutput', {
  flow,
  output: OutputConfiguration.rtp({
    destination: '198.51.100.10',
    port: 5100,
  }),
});

new FlowOutput(stack, 'RistOutput', {
  flow,
  output: OutputConfiguration.rist({
    destination: '198.51.100.20',
    port: 5200,
  }),
});

new FlowOutput(stack, 'SrtCallerOutput', {
  flow,
  output: OutputConfiguration.srtCaller({
    destination: '198.51.100.30',
    port: 5300,
    minLatency: cdk.Duration.millis(120),
  }),
});

// VPC outputs — same protocols, routed through the VPC interface.
new FlowOutput(stack, 'RtpVpcOutput', {
  flow,
  output: OutputConfiguration.rtp({
    destination: '10.1.172.10',
    port: 6100,
    vpcInterfaceAttachmentName: vpcInterface.name,
  }),
});

new FlowOutput(stack, 'RistVpcOutput', {
  flow,
  output: OutputConfiguration.rist({
    destination: '10.1.172.20',
    port: 6200,
    vpcInterfaceAttachmentName: vpcInterface.name,
  }),
});

new FlowOutput(stack, 'SrtCallerVpcOutput', {
  flow,
  output: OutputConfiguration.srtCaller({
    destination: '10.1.172.30',
    port: 6300,
    minLatency: cdk.Duration.millis(120),
    vpcInterfaceAttachmentName: vpcInterface.name,
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-outputs', {
  testCases: [stack],
});

app.synth();
