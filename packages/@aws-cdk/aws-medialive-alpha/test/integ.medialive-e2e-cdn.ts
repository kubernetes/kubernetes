// Tests end-to-end live-streaming pipeline: MediaConnect Router → MediaLive → MediaPackage V2 + HLS to S3 via CloudFront.
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Bitrate } from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-e2e-cdn');
const azs = cdk.Stack.of(stack).availabilityZones;

// S3 bucket for the HLS output. toBucket() below auto-grants the channel role scoped write.
const hlsBucket = new s3.Bucket(stack, 'HlsBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// MediaPackage V2 channel (downstream)
const channelGroup = new mediapackagev2.ChannelGroup(stack, 'ChannelGroup');
const mpChannel = channelGroup.addChannel('MpChannel', {
  input: mediapackagev2.InputConfiguration.cmaf(),
});

// MediaLive input — MediaConnect Router (two AZs → STANDARD pipeline redundancy)
const input = new medialive.Input(stack, 'Input', {
  inputName: 'e2e-cdn-router-input',
  input: medialive.InputConfiguration.mediaConnectRouter({
    availabilityZones: [azs[0], azs[1]],
  }),
});

// Encode configurations — defined once, shared across output groups
const video1080 = medialive.EncodeConfiguration.video({
  name: 'video_1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    profile: medialive.H264Profile.HIGH,
    rateControl: medialive.H264RateControl.cbr({ bitrate: Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
    gopSize: medialive.GopSize.seconds(2),
    gopNumBFrames: 3,
    adaptiveQuantization: medialive.H264AdaptiveQuantization.HIGH,
  }),
});

const video720 = medialive.EncodeConfiguration.video({
  name: 'video_720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    profile: medialive.H264Profile.HIGH,
    rateControl: medialive.H264RateControl.cbr({ bitrate: Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_29_97,
    gopSize: medialive.GopSize.seconds(2),
    gopNumBFrames: 3,
    adaptiveQuantization: medialive.H264AdaptiveQuantization.HIGH,
  }),
});

const audioStereo = medialive.EncodeConfiguration.audio({
  name: 'audio_aac_stereo',
  codec: medialive.AudioCodecSettings.aac({
    bitrate: Bitrate.kbps(192),
    profile: medialive.AacProfile.LC,
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
  }),
});

// MediaLive Channel — encodes auto-derived from outputs, deduped by name
new medialive.Channel(stack, 'Channel', {
  channelName: 'e2e-cdn-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  logLevel: medialive.LogLevel.INFO,
  inputs: [{ input }],
  inputSpecification: medialive.InputSpecification.standard({
    codec: medialive.InputCodec.AVC,
    maximumBitrate: medialive.InputMaximumBitrate.MAX_20_MBPS,
    resolution: medialive.InputResolution.HD,
  }),
  outputGroups: [
    // MediaPackage V2 — one track per output (CMAF)
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'mp2',
      channel: mpChannel,
      outputs: [
        { outputName: 'mp2-1080', encode: video1080 },
        { outputName: 'mp2-720', encode: video720 },
        { outputName: 'mp2-audio', encode: audioStereo },
      ],
    }),
    // HLS to S3 — multiple encodes per output. toBucket() auto-grants scoped write.
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [
        medialive.OutputDestination.toBucket(hlsBucket, 'pipeline-0/index'),
        medialive.OutputDestination.toBucket(hlsBucket, 'pipeline-1/index'),
      ],
      segment: medialive.Segment.seconds(6),
      keepSegments: 21,
      indexNSegments: 7,
      outputs: [
        { outputName: 'hls-1080', nameModifier: '_1080p', encodes: [video1080, audioStereo] },
        { outputName: 'hls-720', nameModifier: '_720p', encodes: [video720, audioStereo] },
      ],
    }),
  ],
});

// Serve the HLS output through CloudFront, locked to the bucket with Origin Access Control.
new cloudfront.Distribution(stack, 'Cdn', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withOriginAccessControl(hlsBucket),
    viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
  },
});

new IntegTest(app, 'cdk-integ-medialive-e2e-cdn', {
  testCases: [stack],
});

app.synth();
