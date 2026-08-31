// Tests channel-wide global configuration, Nielsen configuration, and thumbnail configuration settings.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-global-config');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const input = new medialive.Input(stack, 'Input', {
  inputName: 'global-config-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
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
  channelName: 'global-config-channel',
  inputs: [{ input }],
  globalConfiguration: {
    initialAudioGain: -6,
    inputEndAction: medialive.InputEndAction.SWITCH_AND_LOOP_INPUTS,
    // Epoch locking requires the output timing source to be the input clock.
    outputLocking: medialive.OutputLocking.epoch(),
    outputTimingSource: medialive.OutputTimingSource.INPUT_CLOCK,
    supportLowFramerateInputs: true,
  },
  nielsenConfiguration: {
    distributorId: 'ACME123',
    nielsenPcmToId3Tagging: medialive.NielsenPcmToId3TaggingState.ENABLED,
  },
  thumbnailConfiguration: {
    state: medialive.ThumbnailState.AUTO,
  },
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

new IntegTest(app, 'cdk-integ-medialive-channel-global-config', {
  testCases: [stack],
});

app.synth();
