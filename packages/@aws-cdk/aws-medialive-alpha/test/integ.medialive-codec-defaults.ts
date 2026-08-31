/**
 * Integration test: Minimal channel per codec type to discover required fields.
 * Run this to find out what CFN requires for each codec with no rate control set.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-codec-defaults');

const input1 = new medialive.Input(stack, 'Input1', {
  inputName: 'codec-defaults-input-h264',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 9000,
  }]),
});

const input2 = new medialive.Input(stack, 'Input2', {
  inputName: 'codec-defaults-input-h265',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.101',
    srtListenerPort: 9000,
  }]),
});

const role = new iam.Role(stack, 'Role', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// H.264 — no rate control set
const h264 = medialive.EncodeConfiguration.video({
  name: 'h264',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264(),
});

// H.265 — no rate control set
const h265 = medialive.EncodeConfiguration.video({
  name: 'h265',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h265({
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({ name: 'aac', codec: medialive.AudioCodecSettings.aac() });

new medialive.Channel(stack, 'H264Channel', {
  channelName: 'codec-defaults-h264',
  role,
  inputs: [{ input: input1 }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.url('s3ssl://my-bucket/h264')],
      outputs: [{ encodes: [h264, audio], outputName: 'out' }],
    }),
  ],
});

new medialive.Channel(stack, 'H265Channel', {
  channelName: 'codec-defaults-h265',
  role,
  inputs: [{ input: input2 }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.url('s3ssl://my-bucket/h265')],
      outputs: [{ encodes: [h265, audio], outputName: 'out' }],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-codec-defaults', {
  testCases: [stack],
});

app.synth();
