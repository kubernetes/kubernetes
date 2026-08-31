// Tests MediaLive channel with burn-in and embedded captions on MediaPackage V2 outputs.
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-captions');

// --- MediaPackage V2 ---
const empGroup = new mediapackagev2.ChannelGroup(stack, 'EmpGroup', {
  channelGroupName: 'integ-captions-group',
});

const empChannel = new mediapackagev2.Channel(stack, 'EmpChannel', {
  channelGroup: empGroup,
  channelName: 'integ-captions-channel',
});

new mediapackagev2.OriginEndpoint(stack, 'HlsEndpoint', {
  channel: empChannel,
  originEndpointName: 'integ-captions-endpoint',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});

// --- MediaLive Input ---
const input = new medialive.Input(stack, 'SrtInput', {
  inputName: 'integ-captions-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.10',
    srtListenerPort: 5000,
  }]),
});

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- Encodes ---
const video = medialive.EncodeConfiguration.video({
  name: 'video-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    profile: medialive.H264Profile.HIGH,
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const video720 = medialive.EncodeConfiguration.video({
  name: 'video-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    profile: medialive.H264Profile.MAIN,
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac({ bitrate: cdk.Bitrate.kbps(192) }),
});

const burnIn = medialive.EncodeConfiguration.caption({
  name: 'burn-in-eng',
  captionSelectorName: 'captions',
  destination: medialive.CaptionDestination.burnIn({
    alignment: medialive.CaptionAlignment.CENTERED,
    fontColor: medialive.CaptionFontColor.WHITE,
    outlineColor: medialive.CaptionOutlineColor.BLACK,
    fontSize: medialive.CaptionFontSize.AUTO,
  }),
});

const embedded = medialive.EncodeConfiguration.caption({
  name: 'embedded-passthrough',
  captionSelectorName: 'captions',
  destination: medialive.CaptionDestination.embedded(),
});

// --- MediaLive Channel ---
new medialive.Channel(stack, 'CaptionChannel', {
  channelName: 'integ-captions-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  role,
  inputs: [{
    input,
    captionSelectors: [medialive.CaptionSelector.embedded('captions')],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'emp',
      channel: empChannel,
      outputs: [
        // Video with burn-in captions rendered into the pixels
        { encode: video, captions: [burnIn], outputName: 'video_burnin' },
        // Video with embedded passthrough captions
        { encode: video720, captions: [embedded], outputName: 'video_embedded' },
        // Audio track
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-captions', {
  testCases: [stack],
});
