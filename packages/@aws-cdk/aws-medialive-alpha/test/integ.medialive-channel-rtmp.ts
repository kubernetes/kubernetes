// Tests RTMP output group with RTMP caption info destination.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-rtmp');

// --- Input ---
const input = new medialive.Input(stack, 'SrtInput', {
  inputName: 'integ-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 9000,
  }]),
});

// --- IAM Role ---
const role = new iam.Role(stack, 'MediaLiveRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- Encodes ---
const hd = medialive.EncodeConfiguration.video({
  name: 'hd-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'aac-stereo',
  codec: medialive.AudioCodecSettings.aac(),
});

// RTMP CaptionInfo — an object-style caption format supported by RTMP outputs, riding alongside
// the video/audio in the same output (unlike WebVTT/EBU-TT-D sidecar formats).
const captionInfo = medialive.EncodeConfiguration.caption({
  name: 'eng-caption-info',
  captionSelectorName: 'embedded-cc',
  languageCode: 'eng',
  destination: medialive.CaptionDestination.rtmpCaptionInfo(),
});

// --- Channel with RTMP output ---
new medialive.Channel(stack, 'RtmpChannel', {
  channelName: 'integ-rtmp-channel',
  role,
  inputs: [{
    input,
    captionSelectors: [medialive.CaptionSelector.embedded('embedded-cc')],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.rtmp({
      name: 'rtmp-output',
      authenticationScheme: medialive.RtmpAuthenticationScheme.COMMON,
      restartDelay: cdk.Duration.seconds(15),
      outputs: [
        {
          destinations: [medialive.RtmpDestination.url('rtmp://203.0.113.200/live', 'stream')],
          encodes: [hd, audio, captionInfo],
          outputName: 'rtmp_out',
        },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-rtmp', {
  testCases: [stack],
});

app.synth();
