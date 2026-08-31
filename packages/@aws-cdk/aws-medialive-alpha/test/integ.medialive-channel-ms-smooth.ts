// Tests MS Smooth (Microsoft Smooth Streaming) output group with SMPTE-TT and EBU-TT-D caption destinations.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-ms-smooth');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'smooth-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
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

// SMPTE-TT — a "Stream" caption category, riding alongside video/audio in the same output.
const smpteTt = medialive.EncodeConfiguration.caption({
  name: 'eng-smptett',
  captionSelectorName: 'embedded-cc',
  languageCode: 'eng',
  destination: medialive.CaptionDestination.smpteTt(),
});

// EBU-TT-D — a "Sidecar" caption category; must be its own caption-only output.
const ebuTtD = medialive.EncodeConfiguration.caption({
  name: 'eng-ebuttd',
  captionSelectorName: 'embedded-cc',
  languageCode: 'eng',
  destination: medialive.CaptionDestination.ebuTtD({ copyrightHolder: 'Acme Corp' }),
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'ms-smooth-channel',
  inputs: [{
    input,
    captionSelectors: [medialive.CaptionSelector.embedded('embedded-cc')],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.msSmooth({
      name: 'smooth',
      // SINGLE_PIPELINE channel — exactly one destination.
      destinations: [medialive.OutputDestination.url('https://smooth.example.com/live.isml')],
      fragmentLength: cdk.Duration.seconds(2),
      outputs: [
        { outputName: 'smooth-output', encodes: [video, audio], nameModifier: '_video' },
        { outputName: 'captions-smptett', encodes: [smpteTt], nameModifier: '_smptett' },
        { outputName: 'captions-ebuttd', encodes: [ebuTtD], nameModifier: '_ebuttd' },
      ],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-channel-ms-smooth', {
  testCases: [stack],
});

app.synth();
