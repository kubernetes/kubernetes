/**
 * Integration test for MediaConnect Flow with dual sources
 *
 * Scenario 1: Two sources (one encrypted, one not, one on VPC)
 * Outputs: Router Output / Entitlement / VPC interface / Output with Encryption
 * Also exercises: maintenance window, source failover, source monitoring
 */

import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { App, Bitrate, Duration, Stack } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as mediaconnect from '../lib';

const app = new App();
const stack = new Stack(app, 'aws-cdk-mediaconnect-dual-source');

// VPC Setup. Empty subnetConfiguration skips auto-generated subnets, which resolve AZs at synth
// time and hit the integ-runner AZ bug (aws-cdk#38268); the VPC interfaces use the explicit
// `subnet` below instead.
const vpc = new ec2.Vpc(stack, 'Vpc', {
  ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [],
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

// VPC Interfaces
const sourceVpcInterface = mediaconnect.VpcInterface.define({
  vpcInterfaceName: 'source-vpc-iface',
  role: vpcRole,
  securityGroups: [securityGroup],
  subnet,
});
const outputVpcInterface = mediaconnect.VpcInterface.define({
  vpcInterfaceName: 'output-vpc-iface',
  role: vpcRole,
  securityGroups: [securityGroup],
  subnet,
});

// Flow with Primary Source (VPC, not encrypted) + maintenance + source monitoring
const flow = new mediaconnect.Flow(stack, 'DualSourceFlow', {
  flowName: 'dual-source-flow',
  // Matches the explicit subnet's AZ; resolves at deploy time via the region token.
  availabilityZone: `${stack.region}a`,
  vpcInterfaces: [sourceVpcInterface, outputVpcInterface],
  maintenance: {
    maintenanceDay: mediaconnect.MaintenanceDay.TUESDAY,
    maintenanceStartHour: '03:00',
  },
  sourceFailoverConfig: mediaconnect.FailoverConfig.failover(),
  sourceMonitoringConfig: {
    thumbnailState: mediaconnect.State.ENABLED,
    contentQualityAnalysisState: mediaconnect.State.ENABLED,
  },
  source: mediaconnect.SourceConfiguration.srtListener({
    flowSourceName: 'primary-source-vpc',
    description: 'Primary source on VPC (not encrypted)',
    port: 5000,
    minLatency: Duration.millis(200),
    maxBitrate: Bitrate.mbps(50),
    network: mediaconnect.NetworkConfiguration.vpc(sourceVpcInterface),
  }),
});

// Secondary Source (encrypted, not on VPC)
new mediaconnect.FlowSource(stack, 'SecondarySource', {
  flow,
  source: mediaconnect.SourceConfiguration.srtListener({
    flowSourceName: 'secondary-source-encrypted',
    description: 'Secondary encrypted source (not on VPC)',
    port: 5001,
    maxBitrate: Bitrate.mbps(50),
    network: mediaconnect.NetworkConfiguration.publicNetwork('10.1.0.0/16'),
    decryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

// Output 1: Router Output (flow → router)
const routerFlowOutput = flow.addOutput('RouterFlowOutput', mediaconnect.OutputConfiguration.router());

const routerNetworkInterface = new mediaconnect.RouterNetworkInterface(stack, 'RouterNI', {
  routerNetworkInterfaceName: 'dual-source-router-ni',
  configuration: mediaconnect.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new mediaconnect.RouterInput(stack, 'RouterInput', {
  routerInputName: 'dual-source-router-input',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterInputTier.INPUT_20,
  configuration: mediaconnect.RouterInputConfiguration.mediaConnectFlow({
    flow,
    flowOutput: routerFlowOutput,
  }),
});

new mediaconnect.RouterOutput(stack, 'RouterOutput', {
  routerOutputName: 'dual-source-router-output',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterOutputTier.OUTPUT_100,
  configuration: mediaconnect.RouterOutputConfiguration.standard({
    protocol: mediaconnect.RouterOutputProtocol.rtp({
      destinationAddress: '203.0.113.10',
      port: 5000,
    }),
    networkInterface: routerNetworkInterface,
  }),
});

// Output 2: Entitlement with encryption
new mediaconnect.FlowEntitlement(stack, 'Entitlement', {
  flow,
  description: 'Entitlement for another AWS account',
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
    vpcInterfaceAttachment: outputVpcInterface,
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

new IntegTest(app, 'cdk-integ-mediaconnect-dual-source', {
  testCases: [stack],
});

app.synth();
