// Tests multiple audio codecs (AC3, EAC3, EAC3 Atmos, MP2) in a single archive output to S3.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-audio-codecs');

const bucket = new s3.Bucket(stack, 'ArchiveBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const input = new medialive.Input(stack, 'Input', {
  inputName: 'audio-codecs-srt-input',
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

// Dolby Digital (AC3) 5.1
const ac3 = medialive.EncodeConfiguration.audio({
  name: 'ac3-surround',
  codec: medialive.AudioCodecSettings.ac3({
    bitrate: cdk.Bitrate.kbps(384),
    codingMode: medialive.Ac3CodingMode.CODING_MODE_3_2_LFE,
  }),
});
// Dolby Digital Plus (EAC3) 5.1
const eac3 = medialive.EncodeConfiguration.audio({
  name: 'eac3-surround',
  codec: medialive.AudioCodecSettings.eac3({
    bitrate: cdk.Bitrate.kbps(256),
    codingMode: medialive.Eac3CodingMode.CODING_MODE_3_2,
  }),
});
// Dolby Atmos
const atmos = medialive.EncodeConfiguration.audio({
  name: 'eac3-atmos',
  codec: medialive.AudioCodecSettings.eac3Atmos({
    codingMode: medialive.Eac3AtmosCodingMode.CODING_MODE_5_1_4,
  }),
});
// MPEG-1 Layer II
const mp2 = medialive.EncodeConfiguration.audio({
  name: 'mp2-stereo',
  codec: medialive.AudioCodecSettings.mp2({
    bitrate: cdk.Bitrate.kbps(192),
    codingMode: medialive.Mp2CodingMode.CODING_MODE_2_0,
  }),
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'audio-codecs-channel',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.archive({
      name: 'archive',
      // toBucket() auto-grants the channel's role scoped write access.
      destinations: [medialive.S3OutputDestination.toBucket(bucket, 'archive')],
      rolloverInterval: cdk.Duration.minutes(5),
      outputs: [
        { outputName: 'archive-output', encodes: [video, ac3, eac3, atmos, mp2] },
      ],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-audio-codecs', {
  testCases: [stack],
});

app.synth();
