/**
 * Integration test: MediaLive Channel → HLS (S3)
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-hls');

// --- S3 Bucket for HLS output ---
const bucket = new s3.Bucket(stack, 'HlsBucket', {
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

// --- Encodes ---
const hd = medialive.EncodeConfiguration.video({
  name: 'hd-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac(),
});

// --- Channel with HLS output ---
new medialive.Channel(stack, 'HlsChannel', {
  channelName: 'integ-hls-channel',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls_output',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live')],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [
        { encodes: [hd, audio], outputName: 'hd_output', nameModifier: '_hd' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-hls', {
  testCases: [stack],
});

app.synth();
