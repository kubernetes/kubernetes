// Tests a STANDARD channel with additional MediaPackage V2 destinations across multiple pipelines.
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-standard-additional-destinations');

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- MediaPackage V2: Channel Group ---
const channelGroup = new mediapackagev2.ChannelGroup(stack, 'Group');

// --- MediaPackage V2: Primary channel (both pipelines ingest here) ---
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

// --- MediaPackage V2: Additional destination channel ---
const additionalChannel = new mediapackagev2.Channel(stack, 'AdditionalMpChannel', {
  channelGroup,
  channelName: 'additional-channel',
});
new mediapackagev2.OriginEndpoint(stack, 'AdditionalEndpoint', {
  channel: additionalChannel,
  originEndpointName: 'additional-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});
additionalChannel.grants.ingest(role);

// --- SRT Input with 2 sources (one per pipeline for STANDARD) ---
const input = new medialive.Input(stack, 'Input', {
  inputName: 'standard-additional-dest-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
    { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
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

// --- STANDARD Channel: 2 pipelines → same MediaPackage channel, + 1 additional ---
new medialive.Channel(stack, 'Channel', {
  channelName: 'standard-additional-dest-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'to-mediapackage',
      // Single channel reference — MediaLive maps both STANDARD pipelines to its endpoints.
      channel: primaryChannel,
      additionalDestinations: [
        // Fan out a copy to a separate channel
        medialive.MediaPackageV2Destination.channel(additionalChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
      ],
      outputs: [
        { encode: video, outputName: 'video' },
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-standard-additional-destinations', {
  testCases: [stack],
});

app.synth();
