import { IntegTest, ExpectedResult, Match } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';

import { EncodingProfile, Flow, FlowSize } from '../lib/flow';
import { OutputConfiguration } from '../lib/flow-output';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { State, VpcInterface } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-ndi');

const vpc = new Vpc(stack, 'vpc', {
  ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [{
    name: 'private',
    subnetType: SubnetType.PRIVATE_ISOLATED,
    cidrMask: 24,
  }],
});

const subnet = new PrivateSubnet(stack, 'subnet', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.172.0/24',
  vpcId: vpc.vpcId,
});

const role = new Role(stack, 'role', {
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
  vpcInterfaceName: 'ndi-vpc',
  role,
  securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
  subnet,
});

new Flow(stack, 'flow', {
  flowSize: FlowSize.LARGE,
  vpcInterfaces: [vpcInterface],
  ndiConfig: {
    ndiState: State.ENABLED,
    ndiDiscoveryServers: [{
      discoveryServerAddress: '10.0.172.10',
      vpcInterface,
    }],
  },
  encodingConfig: {
    encodingProfile: EncodingProfile.CONTRIBUTION_H264_DEFAULT,
  },
  source: SourceConfiguration.ndi({
    flowSourceName: 'ndi-source',
    description: 'NDI source integ test',
  }),
});

// Defaults flow #1 — NDI source with `encodingConfig` fields omitted. Verifies the
// service applies EncodingProfile.DISTRIBUTION_H264_DEFAULT and 20 Mbps videoMaxBitrate,
// plus a generated 12-character NDI machineName when omitted.
const sourceDefaultsFlow = new Flow(stack, 'sourceDefaultsFlow', {
  flowName: 'ndi-source-defaults',
  flowSize: FlowSize.LARGE,
  vpcInterfaces: [vpcInterface],
  ndiConfig: {
    ndiState: State.ENABLED,
    ndiDiscoveryServers: [{
      discoveryServerAddress: '10.0.172.10',
      vpcInterface,
    }],
  },
  // Empty encodingConfig satisfies the L2 NDI-source guard while letting both
  // encodingProfile and videoMaxBitrate default at the service.
  encodingConfig: {},
  source: SourceConfiguration.ndi({
    flowSourceName: 'ndi-source-defaults',
    description: 'NDI source defaults integ test',
  }),
});

// Defaults flow #2 — RTP source with an NDI output (omitted ndiSpeedHqQuality and
// ndiProgramName). Verifies the service applies 100 for ndiSpeedHqQuality.
//
// NDI source + NDI output cannot coexist (the service rejects "NDI passthrough"),
// so this flow uses a non-NDI source. NDI outputs still require ndiState ENABLED
// and FlowSize.LARGE on the parent flow.
const outputDefaultsFlow = new Flow(stack, 'outputDefaultsFlow', {
  flowName: 'ndi-output-defaults',
  flowSize: FlowSize.LARGE,
  vpcInterfaces: [vpcInterface],
  ndiConfig: {
    ndiState: State.ENABLED,
    ndiDiscoveryServers: [{
      discoveryServerAddress: '10.0.172.10',
      vpcInterface,
    }],
  },
  source: SourceConfiguration.rtp({
    flowSourceName: 'rtp-source',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});

outputDefaultsFlow.addOutput('NdiOutput', { output: OutputConfiguration.ndi() });

const test = new IntegTest(app, 'cdk-integ-emx-flow-ndi', {
  testCases: [stack],
});

// Verify NDI source-side defaults that the API surfaces.
//
// EncodingConfig (encodingProfile, videoMaxBitrate) is documented to default to
// DISTRIBUTION_H264_DEFAULT and 20 Mbps, but DescribeFlow returns Flow.EncodingConfig
// as undefined when the field was omitted on input — same omission semantics as
// FlowSize and SourceZixiPush.maxLatency. Documented in JSDoc, not asserted here.
test.assertions
  .awsApiCall('MediaConnect', 'describeFlow', { FlowArn: sourceDefaultsFlow.flowArn })
  .expect(ExpectedResult.objectLike({
    Flow: {
      NdiConfig: {
        MachineName: Match.stringLikeRegexp('^.{12}$'),
      },
    },
  }));

// Verify NDI output-side default.
test.assertions
  .awsApiCall('MediaConnect', 'describeFlow', { FlowArn: outputDefaultsFlow.flowArn })
  .expect(ExpectedResult.objectLike({
    Flow: {
      Outputs: [{
        Transport: { NdiSpeedHqQuality: 100 },
      }],
    },
  }));

app.synth();
