import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Vpc, IpAddresses, PrivateSubnet, SecurityGroup } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { RouterOutput, RoutingScope, RouterOutputTier, RouterOutputConfiguration, RouterOutputProtocol, RouterNetworkInterface, RouterNetworkConfiguration, ForwardErrorCorrection } from '../lib';

import { Bridge, BridgeConfiguration, BridgeOutputConfiguration } from '../lib/bridge';
import { Flow } from '../lib/flow';
import { SourceConfiguration } from '../lib/flow-source-configuration';
import { Gateway } from '../lib/gateway';
import { BridgeProtocol, GatewayNetwork, VpcInterface } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-bridge-vpc');

const network = GatewayNetwork.define({
  cidrBlock: '10.0.0.0/23',
  name: 'network-1',
});

const gateway = new Gateway(stack, 'gateway', {
  egressCidrBlocks: ['10.0.0.0/23'],
  networks: [network],
});

// Empty subnetConfiguration avoids auto-generated subnets that would resolve AZs at synth time
// (see aws-cdk#38268); the VPC interface uses the explicit `subnet` below instead.
const vpc = new Vpc(stack, 'testvpc', {
  ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [],
});
const subnet = new PrivateSubnet(stack, 'mysubnet', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.172.0/24',
  vpcId: vpc.vpcId,
});

const securityGroup = new SecurityGroup(stack, 'BridgeSecurityGroup', {
  vpc,
  description: 'Security group for MediaConnect bridge',
});

const role = new Role(stack, 'BridgeRole', {
  assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  inlinePolicies: {
    policy: new PolicyDocument({
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
  vpcInterfaceName: 'bridge-vpc-interface',
  role,
  securityGroups: [securityGroup],
  subnet,
});

const networkInterface = new RouterNetworkInterface(stack, 'network', {
  routerNetworkInterfaceName: 'test-output-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

// Test 1: Standard RTP Output
const router = new RouterOutput(stack, 'rtpOutput', {
  routerOutputName: 'test-rtp-output',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.rtp({
      destinationAddress: '198.51.100.10',
      port: 5000,
      forwardErrorCorrection: ForwardErrorCorrection.ENABLED,
    }),
    networkInterface: networkInterface,
  }),
});

// Create a flow to use as source for egress bridge
const sourceFlow = new Flow(stack, 'SourceFlow', {
  flowName: 'bridge-source-flow',
  availabilityZone: `${stack.region}a`,
  vpcInterfaces: [vpcInterface],
  source: SourceConfiguration.router({
    routerOutput: router,
  }),
});

// Egress Bridge with Flow Source and VPC Interface
new Bridge(stack, 'egressBridgeWithVpc', {
  bridgeName: 'test-egress-bridge-vpc',
  config: BridgeConfiguration.egress({
    maxBitrate: cdk.Bitrate.mbps(10),
    flowSources: [{
      name: 'egress-flow-source-vpc',
      source: {
        flow: sourceFlow,
        vpcInterface: vpcInterface,
      },
    }],
    networkOutputs: [{
      name: 'bridge-vpc-output',
      output: BridgeOutputConfiguration.network({
        ipAddress: '192.168.1.200',
        port: 5001,
        network,
        protocol: BridgeProtocol.UDP,
        ttl: 200,
      }),
    }],
  }),
  gateway,
});

new IntegTest(app, 'cdk-integ-emx-bridge-vpc', {
  testCases: [stack],
});

app.synth();
