/**
 * Integration test: MP4 file from S3 → MediaLive Channel → HLS to S3.
 * Tests InputSource.fromBucket for input and OutputDestination.fromBucket for output,
 * with auto-created channel role and S3 grants.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-s3-to-s3');

// --- S3 Buckets ---
const sourceBucket = new s3.Bucket(stack, 'SourceBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

// --- Input: MP4 file from S3 ---
const input = new medialive.Input(stack, 'FileInput', {
  inputName: 's3-file-input',
  input: medialive.InputConfiguration.mp4File([
    medialive.InputSource.fromBucket(sourceBucket, 'videos/test.mp4'),
  ]),
});

// --- Encodes ---
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

// --- Channel: no role provided — auto-created ---
new medialive.Channel(stack, 'Channel', {
  channelName: 's3-to-s3-channel',
  inputs: [{ input, sourceEndBehavior: medialive.SourceEndBehavior.LOOP }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-to-s3',
      destinations: [
        medialive.OutputDestination.toBucket(outputBucket, 'live/stream'),
      ],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [
        { encodes: [video, audio], outputName: 'hls_output' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-s3-to-s3', {
  testCases: [stack],
});

app.synth();
