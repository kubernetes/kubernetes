// Tests MediaConnect Router input to MediaLive channel with AV1 video encode output to MediaPackage V2.
import * as mediaconnect from '@aws-cdk/aws-mediaconnect-alpha';
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-mediapackage');
const az = cdk.Stack.of(stack).availabilityZones[0];

// --- MediaPackage V2 ---
const empGroup = new mediapackagev2.ChannelGroup(stack, 'EmpGroup', {
  channelGroupName: 'integ-group',
});

const empChannel = new mediapackagev2.Channel(stack, 'EmpChannel', {
  channelGroup: empGroup,
  channelName: 'integ-channel',
});

new mediapackagev2.OriginEndpoint(stack, 'HlsEndpoint', {
  channel: empChannel,
  originEndpointName: 'integ-hls-endpoint',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});

// --- MediaLive Input: MediaConnect Router ---
const emlInput = new medialive.Input(stack, 'RouterInput', {
  inputName: 'integ-router-input',
  input: medialive.InputConfiguration.mediaConnectRouter({
    availabilityZones: [az],
  }),
});

// --- MediaConnect: Router Output → MediaLive ---
new mediaconnect.RouterOutput(stack, 'RouterOutput', {
  routerOutputName: 'to-medialive',
  maximumBitrate: cdk.Bitrate.mbps(20),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterOutputTier.OUTPUT_100,
  configuration: mediaconnect.RouterOutputConfiguration.mediaLiveInput({
    input: emlInput,
    pipeline: mediaconnect.MediaLivePipeline.PIPELINE_0,
  }),
});

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- Encode Profiles ---
const hd = medialive.EncodeConfiguration.video({
  name: 'hd-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.qvbr({ maxBitrate: cdk.Bitrate.mbps(8), qvbrQualityLevel: 8 }),
    profile: medialive.H264Profile.HIGH,
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const sd = medialive.EncodeConfiguration.video({
  name: 'sd-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(2) }),
    profile: medialive.H264Profile.MAIN,
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const av1 = medialive.EncodeConfiguration.video({
  name: 'av1-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.av1({
    rateControl: medialive.Av1RateControl.qvbr({ maxBitrate: cdk.Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac({
    bitrate: cdk.Bitrate.kbps(192),
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
  }),
});

// --- MediaLive Channel ---
new medialive.Channel(stack, 'LiveChannel', {
  channelName: 'integ-live-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  role,
  inputs: [{ input: emlInput, inputAttachmentName: 'router-feed' }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'emp-output',
      channel: empChannel,
      outputs: [
        { encode: hd, outputName: 'hd_output' },
        { encode: sd, outputName: 'sd_output' },
        // CMAF Ingest requires one track per output — the AV1 encode gets its own output.
        { encode: av1, outputName: 'av1_output' },
        { encode: audio, outputName: 'audio_output' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-mediapackage', {
  testCases: [stack],
});

app.synth();
