/**
 * Integration test: MediaLive Channel → Archive (S3)
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-archive');

// --- S3 Bucket for archive output ---
const bucket = new s3.Bucket(stack, 'ArchiveBucket', {
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

// --- Channel with Archive output ---
new medialive.Channel(stack, 'ArchiveChannel', {
  channelName: 'integ-archive-channel',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.archive({
      name: 'archive-output',
      destinations: [medialive.S3OutputDestination.toBucket(bucket, 'archive')],
      rolloverInterval: cdk.Duration.seconds(300),
      outputs: [
        { encodes: [hd, audio], outputName: 'archive_out' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-archive', {
  testCases: [stack],
});

app.synth();
