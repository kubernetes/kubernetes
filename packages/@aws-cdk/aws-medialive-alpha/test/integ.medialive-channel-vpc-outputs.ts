// Tests channel-level VPC output settings that place output endpoints inside a VPC.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-vpc-outputs');

// STANDARD channels need subnets in two different availability zones.
const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 2 });
const outputSecurityGroup = new ec2.SecurityGroup(stack, 'OutputSg', { vpc });

const input = new medialive.Input(stack, 'Input', {
  inputName: 'vpc-outputs-srt-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
    { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
  ]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'h264-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'vpc-outputs-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  // All output endpoints are created within this VPC instead of the MediaLive-managed network.
  vpc: {
    subnets: vpc.privateSubnets,
    securityGroups: [outputSecurityGroup],
  },
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-to-s3',
      // STANDARD channels require one destination per pipeline.
      destinations: [
        medialive.OutputDestination.toBucket(outputBucket, 'pipeline-0/stream'),
        medialive.OutputDestination.toBucket(outputBucket, 'pipeline-1/stream'),
      ],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [{ encodes: [video, audio], outputName: 'hls_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-channel-vpc-outputs', {
  testCases: [stack],
});

app.synth();
