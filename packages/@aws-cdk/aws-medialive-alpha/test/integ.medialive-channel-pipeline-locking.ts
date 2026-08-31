// Tests STANDARD channel pipeline output locking with videoAlignment method and timecode configuration.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-pipeline-locking');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// STANDARD channels need a STANDARD input — an SRT caller with one source per pipeline.
const input = new medialive.Input(stack, 'Input', {
  inputName: 'pipeline-locking-srt-input',
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

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'pipeline-locking-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  globalConfiguration: {
    // Video alignment locks frames by content match — no embedded timecodes required.
    outputLocking: medialive.OutputLocking.pipeline({
      method: medialive.PipelineLockingMethod.VIDEO_ALIGNMENT,
    }),
  },
  // System-clock timecode with a resync threshold, instead of the default EMBEDDED source.
  timecodeConfig: {
    source: medialive.TimecodeSource.SYSTEMCLOCK,
    syncThreshold: 1,
  },
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-to-s3',
      // STANDARD channel: one destination per pipeline.
      destinations: [
        medialive.OutputDestination.toBucket(outputBucket, 'live/pipeline0'),
        medialive.OutputDestination.toBucket(outputBucket, 'live/pipeline1'),
      ],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [{ encodes: [video, audio], outputName: 'hls_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-pipeline-locking', {
  testCases: [stack],
});

app.synth();
