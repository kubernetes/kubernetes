// Tests MediaPackage V2 output group with custom per-pipeline endpoint mapping on a STANDARD channel.
import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-mediapackage-custom-endpoints');

const empGroup = new mediapackagev2.ChannelGroup(stack, 'EmpGroup', {
  channelGroupName: 'custom-ep-group',
});
const empChannel = new mediapackagev2.Channel(stack, 'EmpChannel', {
  channelGroup: empGroup,
  channelName: 'custom-ep-channel',
});
new mediapackagev2.OriginEndpoint(stack, 'HlsEndpoint', {
  channel: empChannel,
  originEndpointName: 'custom-ep-hls',
  segment: mediapackagev2.Segment.cmaf(),
  manifests: [mediapackagev2.Manifest.hls({ manifestName: 'index' })],
});

const input = new medialive.Input(stack, 'Input', {
  inputName: 'custom-ep-srt-input',
  // Two sources → STANDARD input class, required to match the STANDARD channel.
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
    { srtListenerAddress: '203.0.113.101', srtListenerPort: 5001 },
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

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'custom-ep-channel',
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2PerPipeline({
      name: 'emp-output',
      // Per-pipeline path: deliberately flip the mapping — pipeline 0 → ENDPOINT_2, pipeline 1 → ENDPOINT_1
      // (the opposite of what auto would assign), proving explicit per-pipeline control.
      destinations: [
        medialive.MediaPackageV2Destination.channel(empChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_2),
        medialive.MediaPackageV2Destination.channel(empChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
      ],
      outputs: [
        { encode: video, outputName: 'hd_output' },
        { encode: audio, outputName: 'audio_output' },
      ],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-mediapackage-custom-endpoints', {
  testCases: [stack],
});

app.synth();
