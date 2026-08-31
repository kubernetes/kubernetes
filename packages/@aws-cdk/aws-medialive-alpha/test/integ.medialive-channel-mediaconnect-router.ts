// Tests STANDARD channel with MediaConnect Router output group using mixed per-pipeline transit encryption.
import * as mediaconnect from '@aws-cdk/aws-mediaconnect-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-mediaconnect-router');
const [az0, az1] = cdk.Stack.of(stack).availabilityZones;

const transitSecret = new Secret(stack, 'TransitSecret');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'mcr-srt-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
    { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
  ]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'h264-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const routerOutput: medialive.MediaConnectRouterOutputDefinition = {
  encodes: [video, audio],
  outputName: 'router-ts',
};

const routerOutputGroup = medialive.OutputGroupConfiguration.mediaConnectRouter({
  name: 'router-out',
  availabilityZones: [az0, az1],
  routerSettings: medialive.MediaConnectRouterSettings.perPipeline({
    pipeline0: { encryptionSecret: transitSecret },
  }),
  outputs: [routerOutput],
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'mcr-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  outputGroups: [routerOutputGroup],
});

new mediaconnect.RouterInput(stack, 'RouterInputPipeline0', {
  routerInputName: 'mcr-router-input-p0',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterInputTier.INPUT_50,
  configuration: mediaconnect.RouterInputConfiguration.mediaLiveChannel({
    channel,
    outputName: routerOutput.outputName,
    pipeline: mediaconnect.MediaLivePipeline.PIPELINE_0,
    sourceTransitDecryption: { secret: transitSecret },
  }),
});

new mediaconnect.RouterInput(stack, 'RouterInputPipeline1', {
  routerInputName: 'mcr-router-input-p1',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: mediaconnect.RoutingScope.REGIONAL,
  tier: mediaconnect.RouterInputTier.INPUT_50,
  configuration: mediaconnect.RouterInputConfiguration.mediaLiveChannel({
    channel,
    outputName: routerOutput.outputName,
    pipeline: mediaconnect.MediaLivePipeline.PIPELINE_1,
  }),
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-mediaconnect-router', {
  testCases: [stack],
});

app.synth();
