// Tests MediaConnect flow input configuration with auto-granted flow-management IAM actions.
import * as mediaconnect from '@aws-cdk/aws-mediaconnect-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-mediaconnect-flow');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// MediaConnect flow with an RTP source — the upstream contribution feed.
const flow = new mediaconnect.Flow(stack, 'Flow', {
  flowName: 'medialive-source-flow',
  source: mediaconnect.SourceConfiguration.rtp({
    flowSourceName: 'primary',
    port: 5000,
    network: mediaconnect.NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});

// No `role` is provided — the input auto-creates one with the medialive.amazonaws.com service
// principal and grants it the managed flow-management actions for the flow.
const input = new medialive.Input(stack, 'Input', {
  inputName: 'mediaconnect-flow-input',
  input: medialive.InputConfiguration.mediaConnect({
    flows: [flow],
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
  channelName: 'mediaconnect-flow-channel',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-to-s3',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'live/stream')],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [{ encodes: [video, audio], outputName: 'hls_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });
new cdk.CfnOutput(stack, 'FlowArn', { value: flow.flowArn });

new IntegTest(app, 'cdk-integ-medialive-input-mediaconnect-flow', {
  testCases: [stack],
});

app.synth();
