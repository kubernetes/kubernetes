// Tests global configuration input-loss behavior, output locking, and ESAM ad avail handling on an HLS channel.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { StringParameter } from 'aws-cdk-lib/aws-ssm';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-global-avail');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'global-avail-srt-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
  ]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'h264-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const slateBucket = new s3.Bucket(stack, 'SlateBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const poisPassword = new StringParameter(stack, 'PoisPassword', {
  stringValue: 'placeholder-pois-password',
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'global-avail',
  inputs: [{ input }],
  globalConfiguration: {
    inputLossBehavior: {
      blackFrame: cdk.Duration.seconds(1),
      repeatFrame: cdk.Duration.seconds(5),
      imageType: medialive.InputLossImageType.SLATE,
      imageSlate: medialive.FileLocation.fromBucket(slateBucket, 'slates/offline.png'),
    },
    outputLocking: medialive.OutputLocking.disabled(),
  },
  availSettings: medialive.AvailSettings.esam({
    pois: {
      url: 'https://pois.example.com/esam',
      username: 'pois-user',
      password: poisPassword,
    },
    acquisitionPointId: 'acquisition-point-1',
    adAvailOffset: cdk.Duration.millis(200),
    zoneIdentity: 'zone-1',
  }),
  scte35SegmentationScope: medialive.Scte35SegmentationScope.SCTE35_ENABLED_OUTPUT_GROUPS,
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'hls')],
      outputs: [{ encodes: [video, audio], outputName: 'video' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-global-avail', {
  testCases: [stack],
});

app.synth();
