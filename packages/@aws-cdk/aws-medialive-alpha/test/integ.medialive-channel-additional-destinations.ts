/**
 * Integration test: SINGLE_PIPELINE channel with additional MediaPackage V2 destinations.
 * Tests that a channel can send to multiple MediaPackage V2 endpoints from the same pipeline.
 */
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-additional-destinations');

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- MediaPackage V2: Primary channel ---
const channelGroup = new mediapackagev2.ChannelGroup(stack, 'Group');
const primaryChannel = new mediapackagev2.Channel(stack, 'PrimaryMpChannel', {
  channelGroup,
  channelName: 'primary-channel',
});
new mediapackagev2.OriginEndpoint(stack, 'PrimaryEndpoint', {
  channel: primaryChannel,
  originEndpointName: 'primary-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});
primaryChannel.grants.ingest(role);

// --- MediaPackage V2: Additional (backup) channel ---
const backupChannel = new mediapackagev2.Channel(stack, 'BackupMpChannel', {
  channelGroup,
  channelName: 'backup-channel',
});
new mediapackagev2.OriginEndpoint(stack, 'BackupEndpoint', {
  channel: backupChannel,
  originEndpointName: 'backup-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});
backupChannel.grants.ingest(role);

// --- SRT Input ---
const input = new medialive.Input(stack, 'Input', {
  inputName: 'additional-dest-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
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

// --- Channel with additional destinations ---
// SINGLE_PIPELINE channel with 2 destinations (1 primary + 1 additional)
new medialive.Channel(stack, 'Channel', {
  channelName: 'additional-dest-channel',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2PerPipeline({
      name: 'to-mediapackage',
      destinations: [
        medialive.MediaPackageV2Destination.channel(primaryChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
      ],
      additionalDestinations: [
        medialive.MediaPackageV2Destination.channel(backupChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
      ],
      outputs: [
        { encode: video, outputName: 'video' },
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-additional-destinations', {
  testCases: [stack],
});

app.synth();
