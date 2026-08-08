/**
 * Integration test for MediaConnect Flow with Router source
 *
 * Scenario 4: Router source
 * Outputs: Router Output / Entitlement / VPC interface output / Output with Encryption
 */

import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { App, Bitrate, Stack } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as mediaconnect from '../lib';

const app = new App();
const stack = new Stack(app, 'aws-cdk-mediaconnect-router-source');

// VPC Setup
const vpc = new ec2.Vpc(stack, 'Vpc', {
  ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [{
    name: 'isolated',
    subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
    cidrMask: 24,
  }],
});
const subnet = new ec2.PrivateSubnet(stack, 'Subnet', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.172.0/24',
  vpcId: vpc.vpcId,
});
const securityGroup = new ec2.SecurityGroup(stack, 'SG', { vpc });

// IAM Roles and Secrets
const vpcRole = new iam.Role(stack, 'VpcRole', {
  assumedBy: new iam.ServicePrincipal('mediaconnect.amazonaws.com'),
  inlinePolicies: {
    eni: new iam.PolicyDocument({
      statements: [new iam.PolicyStatement({
        actions: ['ec2:CreateNetworkInterface', 'ec2:CreateNetworkInterfacePermission', 'ec2:DeleteNetworkInterface', 'ec2:DescribeSubnets'],
        resources: ['*'],
      })],
    }),
  },
});

const encryptionSecret = new secretsmanager.Secret(stack, 'EncryptionSecret');
const encryptionRole = new iam.Role(stack, 'EncryptionRole', {
  assumedBy: new iam.ServicePrincipal('mediaconnect.amazonaws.com'),
  inlinePolicies: {
    secrets: new iam.PolicyDocument({
      statements: [new iam.PolicyStatement({
        actions: ['secretsmanager:GetSecretValue'],
        resources: [encryptionSecret.secretArn],
      })],
    }),
  },
});

const outputVpcInterface = mediaconnect.VpcInterface.define({
  vpcInterfaceName: 'output-vpc-iface',
  role: vpcRole,
  securityGroups: [securityGroup],
  subnet,
});

// Router Output that feeds the flow source
const sourceRouterNI = new mediaconnect.RouterNetworkInterface(stack, 'SourceRouterNI', {
  routerNetworkInterfaceName: 'source-router-ni',
  configuration: mediaconnect.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

const sourceRouterOutput = new mediaconnect.RouterOutput(stack, 'SourceRouterOutput', {
  routerOutputName: 'source-router-output',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.GLOBAL,
  tier: mediaconnect.RouterOutputTier.OUTPUT_100,
  configuration: mediaconnect.RouterOutputConfiguration.standard({
    protocol: mediaconnect.RouterOutputProtocol.rtp({
      destinationAddress: '203.0.113.50',
      port: 5000,
    }),
    networkInterface: sourceRouterNI,
  }),
  tags: { Environment: 'test' },
});

// Flow with Router Source (with transit encryption)
const flow = new mediaconnect.Flow(stack, 'RouterSourceFlow', {
  flowName: 'router-source-flow',
  availabilityZone: stack.availabilityZones[0],
  vpcInterfaces: [outputVpcInterface],
  maintenanceConfiguration: {
    day: mediaconnect.MaintenanceDay.FRIDAY,
    time: '02:00',
  },
  source: mediaconnect.SourceConfiguration.router({
    routerOutput: sourceRouterOutput,
    decryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

// Output 1: Router Output (flow → router input)
const routerFlowOutput = flow.addOutput('RouterFlowOutput', {
  output: mediaconnect.OutputConfiguration.router({
    encryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

const destRouterNI = new mediaconnect.RouterNetworkInterface(stack, 'DestRouterNI', {
  routerNetworkInterfaceName: 'dest-router-ni',
  configuration: mediaconnect.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new mediaconnect.RouterInput(stack, 'RouterInput', {
  routerInputName: 'router-source-input',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterInputTier.INPUT_20,
  transitEncryption: {
    role: encryptionRole,
    secret: encryptionSecret,
  },
  configuration: mediaconnect.RouterInputConfiguration.mediaConnectFlow({
    flow,
    flowOutput: routerFlowOutput,
    sourceTransitDecryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

new mediaconnect.RouterOutput(stack, 'DestRouterOutput', {
  routerOutputName: 'dest-router-output',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterOutputTier.OUTPUT_100,
  configuration: mediaconnect.RouterOutputConfiguration.standard({
    protocol: mediaconnect.RouterOutputProtocol.rtp({
      destinationAddress: '203.0.113.20',
      port: 5001,
    }),
    networkInterface: destRouterNI,
  }),
});

// Output 2: Entitlement with encryption
new mediaconnect.FlowEntitlement(stack, 'Entitlement', {
  flow,
  description: 'Entitlement from router-sourced flow',
  subscribers: ['111122223333'],
  encryption: {
    algorithm: mediaconnect.EncryptionAlgorithm.AES256,
    role: encryptionRole,
    secret: encryptionSecret,
  },
});

// Output 3: VPC Interface Output
new mediaconnect.FlowOutput(stack, 'VpcOutput', {
  flowOutputName: 'vpc-interface-output',
  flow,
  output: mediaconnect.OutputConfiguration.rist({
    destination: '10.0.1.100',
    port: 6000,
    vpcInterfaceAttachmentName: outputVpcInterface.name,
  }),
});

// Output 4: Encrypted SRT Output
new mediaconnect.FlowOutput(stack, 'EncryptedOutput', {
  flowOutputName: 'encrypted-srt-output',
  flow,
  output: mediaconnect.OutputConfiguration.srtCaller({
    destination: '203.0.113.100',
    port: 7000,
    encryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

new IntegTest(app, 'cdk-integ-mediaconnect-router-source', {
  testCases: [stack],
});

app.synth();
