/**
 * Integration test: MediaLive Channel → Frame Capture (S3)
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-frame-capture');

// --- S3 Bucket for frame capture output ---
const bucket = new s3.Bucket(stack, 'FrameCaptureBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// --- Input ---
const input = new medialive.Input(stack, 'SrtInput', {
  inputName: 'integ-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 9000,
  }]),
});

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});
bucket.grantReadWrite(role);

// --- Video + audio encodes for the required video output group ---
const videoEncode = medialive.EncodeConfiguration.video({
  name: 'video-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audioEncode = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac(),
});

// --- Frame capture encode with interval ---
const frameCapture = medialive.EncodeConfiguration.video({
  name: 'frame-capture',
  width: 640,
  height: 360,
  codec: medialive.VideoCodecSettings.frameCapture({
    captureInterval: cdk.Duration.seconds(5),
  }),
});

// --- Channel with Archive (video) + Frame Capture output groups ---
new medialive.Channel(stack, 'FrameCaptureChannel', {
  channelName: 'integ-frame-capture-channel',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.archive({
      name: 'archive-output',
      destinations: [medialive.S3OutputDestination.toBucket(bucket, 'archive')],
      outputs: [
        { encodes: [videoEncode, audioEncode], outputName: 'archive_out' },
      ],
    }),
    medialive.OutputGroupConfiguration.frameCapture({
      name: 'frame-capture-output',
      destinations: [medialive.S3OutputDestination.toBucket(bucket, 'frames')],
      outputs: [
        { encodes: [frameCapture], outputName: 'frame_out' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-frame-capture', {
  testCases: [stack],
});

app.synth();
