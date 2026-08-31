// Tests multi-region MediaLive to MediaPackage V2 with a cross-region additional destination.
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();

// --- Secondary stack (us-west-2): create the cross-region MediaPackage channel ---
const secondaryStack = new cdk.Stack(app, 'aws-cdk-medialive-multi-region-secondary', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: 'us-west-2' },
});

const secondaryGroup = new mediapackagev2.ChannelGroup(secondaryStack, 'SecondaryGroup', {
  channelGroupName: 'secondary-group',
});
const secondaryChannelReal = new mediapackagev2.Channel(secondaryStack, 'SecondaryMpChannel', {
  channelGroup: secondaryGroup,
  channelName: 'secondary-mp',
});
new mediapackagev2.OriginEndpoint(secondaryStack, 'SecondaryEndpoint', {
  channel: secondaryChannelReal,
  originEndpointName: 'secondary-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});

// --- Primary stack (us-east-1): MediaLive + primary MediaPackage ---
const primaryStack = new cdk.Stack(app, 'aws-cdk-medialive-multi-region-primary', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: 'us-east-1' },
});

const role = new iam.Role(primaryStack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

const primaryGroup = new mediapackagev2.ChannelGroup(primaryStack, 'PrimaryGroup');
const primaryChannel = new mediapackagev2.Channel(primaryStack, 'PrimaryMpChannel', {
  channelGroup: primaryGroup,
  channelName: 'primary-mp',
});
new mediapackagev2.OriginEndpoint(primaryStack, 'PrimaryEndpoint', {
  channel: primaryChannel,
  originEndpointName: 'primary-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});
primaryChannel.grants.ingest(role);

// --- Import the secondary channel cross-region ---
// Use fromChannelAttributes (not fromChannelArn) so the import works when the
// account is an unresolved token at synth time; pass the region explicitly since
// the secondary channel lives in us-west-2 while this stack is us-east-1.
const secondaryChannel = mediapackagev2.Channel.fromChannelAttributes(primaryStack, 'ImportedSecondaryChannel', {
  channelName: 'secondary-mp',
  channelGroupName: 'secondary-group',
  region: 'us-west-2',
});

// --- SRT Input ---
const input = new medialive.Input(primaryStack, 'Input', {
  inputName: 'multi-region-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
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

// --- MediaLive Channel: primary + cross-region imported additional destination ---
new medialive.Channel(primaryStack, 'MultiRegionChannel', {
  channelName: 'multi-region-channel',
  role,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'to-mediapackage',
      channel: primaryChannel,
      additionalDestinations: [
        medialive.MediaPackageV2Destination.channel(secondaryChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
      ],
      outputs: [
        { encode: video, outputName: 'video' },
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-multi-region', {
  testCases: [secondaryStack, primaryStack],
});

app.synth();
