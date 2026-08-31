// Tests linked channel settings (primary and follower) on SINGLE_PIPELINE channels.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-linked');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

function encodes() {
  const video = medialive.EncodeConfiguration.video({
    name: 'video',
    width: 1280,
    height: 720,
    codec: medialive.VideoCodecSettings.h264({
      rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
      framerate: medialive.Framerate.FPS_29_97,
    }),
  });
  const audio = medialive.EncodeConfiguration.audio({ name: 'audio', codec: medialive.AudioCodecSettings.aac() });
  return { video, audio };
}

// --- Primary channel ---
const primaryInput = new medialive.Input(stack, 'PrimaryInput', {
  inputName: 'linked-primary-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
  ]),
});

const primaryEncodes = encodes();
const primaryChannel = new medialive.Channel(stack, 'PrimaryChannel', {
  channelName: 'linked-primary-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  linkedChannelSettings: medialive.LinkedChannelSettings.primary(),
  inputs: [{ input: primaryInput }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-primary',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'primary')],
      outputs: [{ encodes: [primaryEncodes.video, primaryEncodes.audio], outputName: 'hls_output' }],
    }),
  ],
});

// --- Follower channel — references the primary above ---
const followerInput = new medialive.Input(stack, 'FollowerInput', {
  inputName: 'linked-follower-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
  ]),
});

const followerEncodes = encodes();
const followerChannel = new medialive.Channel(stack, 'FollowerChannel', {
  channelName: 'linked-follower-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  linkedChannelSettings: medialive.LinkedChannelSettings.follower(primaryChannel),
  inputs: [{ input: followerInput }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-follower',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'follower')],
      outputs: [{ encodes: [followerEncodes.video, followerEncodes.audio], outputName: 'hls_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'PrimaryChannelArn', { value: primaryChannel.channelArn });
new cdk.CfnOutput(stack, 'FollowerChannelArn', { value: followerChannel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-channel-linked', {
  testCases: [stack],
});

app.synth();
