import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { IpAddresses, PrivateSubnet, SecurityGroup, SubnetType, Vpc } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Framerate, NetworkInterface, PixelAspectRatio, VpcInterface } from '../lib';
import { Colorimetry, Flow, FlowSize, MediaStream, MediaVideoFormat, ScanMode, Tcs, VideoRange } from '../lib/flow';
import { SourceConfiguration, Encoding } from '../lib/flow-source-configuration';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-jpegxs');

const vpc = new Vpc(stack, 'testvpc', {
  ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  subnetConfiguration: [{
    name: 'subnet1',
    subnetType: SubnetType.PRIVATE_ISOLATED,
    cidrMask: 24,
  }],
});
const subnet = new PrivateSubnet(stack, 'mysubnet1', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.172.0/24',
  vpcId: vpc.vpcId,
});
const subnet2 = new PrivateSubnet(stack, 'mysubnet2', {
  availabilityZone: `${stack.region}a`,
  cidrBlock: '10.0.173.0/24',
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

const interface1 = VpcInterface.define({
  vpcInterfaceName: 'myvpcinterface',
  role,
  securityGroups: [new SecurityGroup(stack, 'sg', {
    vpc,
  })],
  subnet,
  networkInterfaceType: NetworkInterface.ENA,
});

const interface2 = VpcInterface.define({
  vpcInterfaceName: 'myvpcinterface2',
  role,
  securityGroups: [new SecurityGroup(stack, 'sg2', {
    vpc,
  })],
  subnet: subnet2,
  networkInterfaceType: NetworkInterface.ENA,
});

const videoMediaStream = MediaStream.video({
  mediaStreamId: 1,
  mediaStreamName: 'video-stream',
  videoFormat: MediaVideoFormat.UHD_2160P,
  clockRate: 90000,
  fmtp: {
    colorimetry: Colorimetry.BT2020,
    videoRange: VideoRange.FULL,
    scanMode: ScanMode.PROGRESSIVE,
    tcs: Tcs.PQ,
    exactFramerate: Framerate.FPS_59_94,
    par: PixelAspectRatio.SQUARE,
  },
});

new Flow(stack, 'flow', {
  flowSize: FlowSize.LARGE_4X,
  vpcInterfaces: [interface1, interface2],
  mediaStreams: [videoMediaStream],
  source: SourceConfiguration.jpegXs({
    flowSourceName: 'my-flow',
    maxSyncBuffer: 100,
    mediaStreamSourceConfigurations: [{
      encoding: Encoding.JXSV,
      mediaStream: videoMediaStream,
      port: 5000,
      inputInterface: [interface1, interface2],
    }],
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-jpegxs', {
  testCases: [stack],
});

app.synth();
