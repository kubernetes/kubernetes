/**
 * Integration test: Camera (SRT) → MediaLive → MediaPackage V2 (ABR HLS)
 */
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-camera-to-mediapackage');

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- MediaPackage V2 ---
const channelGroup = new mediapackagev2.ChannelGroup(stack, 'Group');
const mpChannel = new mediapackagev2.Channel(stack, 'MpChannel', {
  channelGroup,
  channelName: 'camera-channel',
});
new mediapackagev2.OriginEndpoint(stack, 'HlsEndpoint', {
  channel: mpChannel,
  originEndpointName: 'hls-endpoint',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});

mpChannel.grants.ingest(role);

// --- SRT Input ---
const input = new medialive.Input(stack, 'CameraInput', {
  inputName: 'camera-feed',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
});

// --- Encodes ---
const hd = medialive.EncodeConfiguration.video({
  name: 'hd',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const sd = medialive.EncodeConfiguration.video({
  name: 'sd',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(2) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({ name: 'audio', codec: medialive.AudioCodecSettings.aac() });

// --- Channel ---
new medialive.Channel(stack, 'LiveChannel', {
  channelName: 'camera-to-clients',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'to-mediapackage',
      channel: mpChannel,
      outputs: [
        { encode: hd, outputName: 'hd' },
        { encode: sd, outputName: 'sd' },
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-camera-to-mediapackage', {
  testCases: [stack],
});

app.synth();
