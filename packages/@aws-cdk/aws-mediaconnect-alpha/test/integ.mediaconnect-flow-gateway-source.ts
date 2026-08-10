/**
 * Integration test for MediaConnect Flow with Gateway Bridge source
 *
 * Scenario 3: Gateway source
 * Outputs: Router Output / Entitlement / VPC interface output / Output with Encryption
 */

import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { App, Bitrate, Stack } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as mediaconnect from '../lib';

const app = new App();
const stack = new Stack(app, 'aws-cdk-mediaconnect-gateway-source');

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

// Gateway + Ingress Bridge (network source → cloud flow)
const gwNetwork = mediaconnect.GatewayNetwork.define({
  cidrBlock: '192.168.1.0/24',
  name: 'gw-network',
});

const gateway = new mediaconnect.Gateway(stack, 'Gateway', {
  gatewayName: 'gateway-source-gw',
  egressCidrBlocks: ['10.0.0.0/16'],
  networks: [gwNetwork],
});

const bridge = new mediaconnect.Bridge(stack, 'IngressBridge', {
  bridgeName: 'ingress-bridge',
  gateway,
  config: mediaconnect.BridgeConfiguration.ingress({
    maxBitrate: Bitrate.mbps(10),
    maxOutputs: 2,
    networkSources: [{
      name: 'ingress-network-source',
      source: {
        multicastIp: '239.0.0.1',
        network: gwNetwork,
        port: 5000,
        protocol: mediaconnect.BridgeProtocol.RTP,
      },
    }],
  }),
});

// Flow with Gateway Bridge Source
const flow = new mediaconnect.Flow(stack, 'GatewaySourceFlow', {
  flowName: 'gateway-source-flow',
  availabilityZone: stack.availabilityZones[0],
  vpcInterfaces: [outputVpcInterface],
  maintenanceConfiguration: {
    day: mediaconnect.MaintenanceDay.THURSDAY,
    time: '05:00',
  },
  source: mediaconnect.SourceConfiguration.gatewayBridge({
    bridge,
  }),
});

// Output 1: Router Output (flow → router)
const routerFlowOutput = flow.addOutput('RouterFlowOutput', { output: mediaconnect.OutputConfiguration.router() });

const routerNetworkInterface = new mediaconnect.RouterNetworkInterface(stack, 'RouterNI', {
  routerNetworkInterfaceName: 'gateway-router-ni',
  configuration: mediaconnect.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new mediaconnect.RouterInput(stack, 'RouterInput', {
  routerInputName: 'gateway-router-input',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterInputTier.INPUT_20,
  configuration: mediaconnect.RouterInputConfiguration.mediaConnectFlow({
    flow,
    flowOutput: routerFlowOutput,
  }),
});

new mediaconnect.RouterOutput(stack, 'RouterOutput', {
  routerOutputName: 'gateway-router-output',
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
  description: 'Entitlement from gateway-sourced flow',
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

new IntegTest(app, 'cdk-integ-mediaconnect-gateway-source', {
  testCases: [stack],
});

app.synth();
