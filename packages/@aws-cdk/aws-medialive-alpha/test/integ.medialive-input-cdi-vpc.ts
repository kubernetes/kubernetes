// Tests CDI (uncompressed) VPC input with auto-created IAM role and STANDARD channel output.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-cdi-vpc');

// CDI inputs attach network interfaces in two subnets across two availability zones.
const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 2 });

// A CDI input carries uncompressed video across its two pipeline ENIs, which must talk to each
// other — so MediaLive requires the security group to allow all traffic both to and from itself.
// A self-referencing rule is scoped to members of this security group; it is not open to the world.
const securityGroup = new ec2.SecurityGroup(stack, 'CdiSg', { vpc, allowAllOutbound: false });
securityGroup.addIngressRule(securityGroup, ec2.Port.allTraffic(), 'Allow all traffic from itself');
securityGroup.addEgressRule(securityGroup, ec2.Port.allTraffic(), 'Allow all traffic to itself');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// No `role` — the input auto-creates one with the medialive.amazonaws.com service principal and
// grants it the EC2 ENI permissions, applied before the input is created.
const input = new medialive.Input(stack, 'Input', {
  inputName: 'cdi-vpc-input',
  input: medialive.InputConfiguration.cdi({
    subnets: vpc.privateSubnets,
    securityGroups: [securityGroup],
  }),
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

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'cdi-vpc-channel',
  // CDI inputs are STANDARD class, so the channel must be STANDARD (two pipelines).
  channelClass: medialive.ChannelClass.STANDARD,
  // CDI inputs use the CDI input specification (sets cdiInputSpecification on the channel).
  inputSpecification: medialive.InputSpecification.cdi({
    cdiResolution: medialive.CdiInputResolution.UHD,
    codec: medialive.InputCodec.HEVC,
    maximumBitrate: medialive.InputMaximumBitrate.MAX_50_MBPS,
    resolution: medialive.InputResolution.UHD,
  }),
  inputs: [{ input }],
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

new IntegTest(app, 'cdk-integ-medialive-input-cdi-vpc', {
  testCases: [stack],
});

app.synth();
