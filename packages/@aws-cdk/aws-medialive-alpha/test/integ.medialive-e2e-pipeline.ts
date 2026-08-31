/**
 * End-to-end integration test:
 * MediaConnect Router Output → MediaLive Channel → MediaPackage V2 (CMAF)
 *                                                → UDP → MediaConnect Router Input
 */
import * as emx from '@aws-cdk/aws-mediaconnect-alpha';
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-e2e-pipeline');
const az = cdk.Stack.of(stack).availabilityZones[0];

// --- MediaLive Input: MediaConnect Router ---
const emlInput = new medialive.Input(stack, 'RouterInput', {
  inputName: 'e2e-router-input',
  input: medialive.InputConfiguration.mediaConnectRouter({
    availabilityZones: [az],
  }),
});

// --- MediaConnect: Router Output (feeds into MediaLive) ---
new emx.RouterOutput(stack, 'RouterOutput', {
  routerOutputName: 'to-medialive',
  maximumBitrate: cdk.Bitrate.mbps(20),
  routingScope: emx.RoutingScope.REGIONAL,
  tier: emx.RouterOutputTier.OUTPUT_100,
  configuration: emx.RouterOutputConfiguration.mediaLiveInput({
    input: emlInput,
    pipeline: emx.MediaLivePipeline.PIPELINE_0,
  }),
});

// --- MediaConnect: Router Input (destination from MediaLive UDP output) ---
const destInterface = new emx.RouterNetworkInterface(stack, 'DestInterface', {
  routerNetworkInterfaceName: 'dest-interface',
  configuration: emx.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new emx.RouterInput(stack, 'RouterDestInput', {
  routerInputName: 'dest-rtp-input',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: emx.RoutingScope.REGIONAL,
  tier: emx.RouterInputTier.INPUT_50,
  configuration: emx.RouterInputConfiguration.standard({
    networkInterface: destInterface,
    protocol: emx.RouterInputProtocol.rtp({ port: 6000 }),
  }),
});

// --- MediaPackage V2: packaging + delivery ---
const empGroup = new mediapackagev2.ChannelGroup(stack, 'EmpGroup', {
  channelGroupName: 'e2e-group',
});

const empChannel = new mediapackagev2.Channel(stack, 'EmpChannel', {
  channelGroup: empGroup,
  channelName: 'e2e-channel',
});

new mediapackagev2.OriginEndpoint(stack, 'HlsEndpoint', {
  channel: empChannel,
  originEndpointName: 'e2e-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
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

const audio = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac({
    bitrate: cdk.Bitrate.kbps(192),
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
  }),
});

// --- MediaLive Channel ---
const channel = new medialive.Channel(stack, 'LiveChannel', {
  channelName: 'e2e-live-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  logLevel: medialive.LogLevel.INFO,
  role,
  inputSpecification: medialive.InputSpecification.standard({
    codec: medialive.InputCodec.AVC,
    maximumBitrate: medialive.InputMaximumBitrate.MAX_20_MBPS,
    resolution: medialive.InputResolution.HD,
  }),
  inputs: [{ input: emlInput, inputAttachmentName: 'router-feed' }],
  outputGroups: [
    // Output 1: MediaPackage V2 (CMAF ingest, one track per output)
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'emp-output',
      channel: empChannel,
      outputs: [
        { encode: hd, outputName: 'hd_output' },
        { encode: sd, outputName: 'sd_output' },
        { encode: audio, outputName: 'audio_output' },
      ],
    }),
  ],
});

// Output 2: UDP back to MediaConnect Router Input
channel.addOutputGroup(
  medialive.OutputGroupConfiguration.udp({
    name: 'udp-to-router',
    destinations: [medialive.UdpOutputDestination.rtp({ address: '10.0.0.100', port: 6000 })],
    outputs: [
      { encodes: [hd, audio], outputName: 'udp_output' },
    ],
  }),
);

// Grant MediaLive permission to ingest into MediaPackage
empChannel.grants.ingest(role);

new IntegTest(app, 'cdk-integ-medialive-e2e-pipeline', {
  testCases: [stack],
});

app.synth();
