/**
 * Kitchen sink integration test for MediaConnect Flow
 *
 * Exercises features not covered by other integ tests:
 * - Audio + video monitoring settings
 * - NDI config + NDI output + NDI discovery server (LARGE flow)
 * - Zixi Pull output (static key encryption)
 * - Zixi Push output
 * - SRT Listener output (SRT password encryption)
 * - RTP-FEC output
 * - Maintenance window
 * - Entitlement with dataTransferSubscriberFeePercent
 *
 * Note: CDI/JPEG XS (LARGE_4X) and NDI (LARGE) are mutually exclusive flow sizes,
 * so media streams and CDI are tested in separate integ tests.
 */

import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { App, Duration, Stack } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as mediaconnect from '../lib';

const app = new App();
const stack = new Stack(app, 'aws-cdk-mediaconnect-kitchen-sink');

// VPC Setup (needed for NDI discovery server)
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

const ndiVpcInterface = mediaconnect.VpcInterface.define({
  vpcInterfaceName: 'ndi-vpc-iface',
  role: vpcRole,
  securityGroups: [securityGroup],
  subnet,
});

// Encryption
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

// LARGE Flow with RTP source, NDI config + discovery server, full monitoring
const flow = new mediaconnect.Flow(stack, 'KitchenSinkFlow', {
  flowName: 'kitchen-sink-flow',
  availabilityZone: stack.availabilityZones[0],
  flowSize: mediaconnect.FlowSize.LARGE,
  vpcInterfaces: [ndiVpcInterface],
  maintenanceConfiguration: {
    day: mediaconnect.MaintenanceDay.SUNDAY,
    time: '06:00',
  },
  ndiConfig: {
    ndiState: mediaconnect.State.ENABLED,
    machineName: 'kitchen-sink',
    ndiDiscoveryServers: [{
      discoveryServerAddress: '10.0.1.50',
      discoveryServerPort: 5959,
      vpcInterface: ndiVpcInterface,
    }],
  },
  sourceMonitoringConfig: {
    thumbnailState: mediaconnect.State.ENABLED,
    contentQualityAnalysisState: mediaconnect.State.ENABLED,
    silentAudio: {
      state: mediaconnect.State.ENABLED,
      threshold: Duration.seconds(15),
    },
    blackFrames: {
      state: mediaconnect.State.ENABLED,
      threshold: Duration.seconds(15),
    },
    frozenFrames: {
      state: mediaconnect.State.ENABLED,
      threshold: Duration.seconds(15),
    },
  },
  source: mediaconnect.SourceConfiguration.rtp({
    flowSourceName: 'rtp-source',
    port: 5000,
    network: mediaconnect.NetworkConfiguration.publicNetwork('10.0.0.0/16'),
  }),
});

// Output 1: NDI Output
new mediaconnect.FlowOutput(stack, 'NdiOutput', {
  flowOutputName: 'ndi-output',
  flow,
  output: mediaconnect.OutputConfiguration.ndi({
    ndiProgramName: 'kitchen-sink-ndi',
    ndiSpeedHqQuality: 100,
  }),
});

// Output 2: RTP-FEC Output
new mediaconnect.FlowOutput(stack, 'RtpFecOutput', {
  flowOutputName: 'rtp-fec-output',
  flow,
  output: mediaconnect.OutputConfiguration.rtpFec({
    destination: '203.0.113.10',
    port: 6000,
  }),
});

// Output 3: Zixi Pull Output with static key encryption
new mediaconnect.FlowOutput(stack, 'ZixiPullOutput', {
  flowOutputName: 'zixi-pull-output',
  flow,
  output: mediaconnect.OutputConfiguration.zixiPull({
    streamId: 'kitchen-sink-zixi-pull',
    remoteId: 'remote-receiver-1',
    maxLatency: Duration.millis(1000),
    cidrAllowList: ['10.0.0.0/16'],
    encryption: {
      algorithm: mediaconnect.EncryptionAlgorithm.AES256,
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

// Output 4: Zixi Push Output with static key encryption
new mediaconnect.FlowOutput(stack, 'ZixiPushOutput', {
  flowOutputName: 'zixi-push-output',
  flow,
  output: mediaconnect.OutputConfiguration.zixiPush({
    streamId: 'kitchen-sink-zixi-push',
    destination: '203.0.113.20',
    port: 2088,
    maxLatency: Duration.millis(500),
    encryption: {
      algorithm: mediaconnect.EncryptionAlgorithm.AES128,
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

// Output 5: SRT Listener Output with SRT password encryption
new mediaconnect.FlowOutput(stack, 'SrtListenerOutput', {
  flowOutputName: 'srt-listener-output',
  flow,
  output: mediaconnect.OutputConfiguration.srtListener({
    port: 7000,
    minLatency: Duration.millis(100),
    cidrAllowList: ['10.0.0.0/8'],
    encryption: {
      role: encryptionRole,
      secret: encryptionSecret,
    },
  }),
});

// Output 6: Entitlement
new mediaconnect.FlowEntitlement(stack, 'Entitlement', {
  flow,
  description: 'Kitchen sink entitlement',
  subscribers: ['111122223333'],
  dataTransferSubscriberFeePercent: 50,
  encryption: {
    algorithm: mediaconnect.EncryptionAlgorithm.AES256,
    role: encryptionRole,
    secret: encryptionSecret,
  },
});

new IntegTest(app, 'cdk-integ-mediaconnect-kitchen-sink', {
  testCases: [stack],
});

app.synth();
