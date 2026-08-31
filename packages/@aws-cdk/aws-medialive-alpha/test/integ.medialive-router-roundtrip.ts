// Tests MediaConnect Router round-trip: Router Output → MediaLive Channel → RTP → Router Input.
import * as emx from '@aws-cdk/aws-mediaconnect-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-router-roundtrip');
const az = cdk.Stack.of(stack).availabilityZones[0];

// --- MediaLive Input: MediaConnect Router ---
const emlInput = new medialive.Input(stack, 'RouterInput', {
  inputName: 'roundtrip-router-input',
  input: medialive.InputConfiguration.mediaConnectRouter({
    availabilityZones: [az],
  }),
});

// --- MediaConnect: Router Output feeding the MediaLive input ---
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

// --- MediaConnect: Router Input that receives the MediaLive RTP output ---
const destInterface = new emx.RouterNetworkInterface(stack, 'DestInterface', {
  routerNetworkInterfaceName: 'dest-interface',
  configuration: emx.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new emx.RouterInput(stack, 'RouterDestInput', {
  routerInputName: 'from-medialive-rtp',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: emx.RoutingScope.REGIONAL,
  tier: emx.RouterInputTier.INPUT_50,
  configuration: emx.RouterInputConfiguration.standard({
    networkInterface: destInterface,
    protocol: emx.RouterInputProtocol.rtp({ port: 6000 }),
  }),
});

// --- IAM Role for MediaLive ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- Encode profiles ---
const video = medialive.EncodeConfiguration.video({
  name: 'hd-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(8) }),
    profile: medialive.H264Profile.HIGH,
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

// --- MediaLive Channel: router in, RTP back to a router input ---
new medialive.Channel(stack, 'LiveChannel', {
  channelName: 'roundtrip-live-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  role,
  inputSpecification: medialive.InputSpecification.standard({
    codec: medialive.InputCodec.AVC,
    maximumBitrate: medialive.InputMaximumBitrate.MAX_20_MBPS,
    resolution: medialive.InputResolution.HD,
  }),
  inputs: [{ input: emlInput, inputAttachmentName: 'router-feed' }],
  outputGroups: [
    medialive.OutputGroupConfiguration.udp({
      name: 'udp-to-router',
      destinations: [medialive.UdpOutputDestination.rtp({ address: '10.0.0.100', port: 6000 })],
      outputs: [
        { encodes: [video, audio], outputName: 'udp_output' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-router-roundtrip', {
  testCases: [stack],
});

app.synth();
