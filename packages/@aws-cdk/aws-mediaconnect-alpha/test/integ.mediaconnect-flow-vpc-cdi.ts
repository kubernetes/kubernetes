import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Colorimetry, Flow, FlowSize, MediaStream, MediaVideoFormat, ScanMode, Tcs, VideoRange } from '../lib/flow';
import { Encoding, SourceConfiguration } from '../lib/flow-source-configuration';
import { Framerate, NetworkInterface, PixelAspectRatio, VpcInterface } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-vpc-cdi');

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
const role = new Role(stack, 'myrole', {
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
const videoMediaStream = MediaStream.video({
  mediaStreamId: 1,
  mediaStreamName: 'mystream',
  videoFormat: MediaVideoFormat.HD_1080P,
  fmtp: {
    colorimetry: Colorimetry.BT709,
    exactFramerate: Framerate.FPS_59_94,
    par: PixelAspectRatio.SQUARE,
    scanMode: ScanMode.PROGRESSIVE,
    videoRange: VideoRange.NARROW,
    tcs: Tcs.SDR,
  },
});
const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'myvpcinterface',
  role,
  securityGroups: [new SecurityGroup(stack, 'sg', {
    vpc,
  })],
  subnet,
  networkInterfaceType: NetworkInterface.EFA,
});
new Flow(stack, 'flow', {
  flowSize: FlowSize.LARGE_4X,
  mediaStreams: [videoMediaStream],
  vpcInterfaces: [vpcInterface],
  source: SourceConfiguration.cdi({
    flowSourceName: 'my-flow',
    vpcInterface: vpcInterface,
    port: 5000,
    maxSyncBuffer: 10,
    mediaStreamSourceConfigurations: [{
      encoding: Encoding.RAW,
      mediaStream: videoMediaStream,
    }],
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-vpc', {
  testCases: [stack],
});

app.synth();

