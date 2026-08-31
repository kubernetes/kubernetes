// Tests multi-input channel with main SRT, backup URL-pull, and S3 MP4 slate inputs attached to a single channel.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-multi-input');

// --- S3 Buckets ---
const slateBucket = new s3.Bucket(stack, 'SlateBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

// --- Main input: SRT caller (live camera) ---
const mainInput = new medialive.Input(stack, 'MainInput', {
  inputName: 'main-srt-feed',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
});

// --- Backup input: URL pull (redundant HLS) ---
const backupInput = new medialive.Input(stack, 'BackupInput', {
  inputName: 'backup-hls-feed',
  input: medialive.InputConfiguration.urlPull([
    medialive.InputSource.url('https://example.com/backup/stream.m3u8'),
  ]),
});

// --- Slate input: MP4 file from S3 (looping) ---
const slateInput = new medialive.Input(stack, 'SlateInput', {
  inputName: 'slate-mp4',
  input: medialive.InputConfiguration.mp4File([
    medialive.InputSource.fromBucket(slateBucket, 'slate/holding-card.mp4'),
  ]),
});

// --- Encodes ---
const video = medialive.EncodeConfiguration.video({
  name: 'video',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'audio', codec: medialive.AudioCodecSettings.aac() });

// --- Channel: 3 inputs, HLS output to S3 ---
const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'multi-input-channel',
  inputs: [
    { input: mainInput, inputAttachmentName: 'main-feed' },
    {
      input: backupInput,
      inputAttachmentName: 'backup-feed',
      // Mark the HLS URL-pull as a LIVE input (bufferSegments <= 10). Without this,
      // MediaLive treats it as VOD, which can't be attached to a multi-input channel.
      networkInputSettings: {
        hlsInputSettings: { bufferSegments: 3 },
      },
    },
    { input: slateInput, inputAttachmentName: 'slate' },
  ],
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

// S3 read grant for slate is automatic via InputSource.fromBucket
// S3 write grant for output is automatic via OutputDestination.toBucket
// Only the backup URL pull needs no grant (public URL)

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-multi-input', {
  testCases: [stack],
});

app.synth();
