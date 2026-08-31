/**
 * Integration test: MediaLive SRT output group with directly-constructed destinations —
 * `SrtDestination.listener()` (LISTENER mode) and `SrtDestination.caller()` (CALLER mode).
 * SRT output is always encrypted, so both destinations carry a passphrase secret.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-srt-outputs');

const passphrase = new Secret(stack, 'SrtPassphrase');

// SRT listener output requires channel security groups (MediaLive opens a listening socket).
const channelSg = new medialive.InputSecurityGroup(stack, 'ChannelSg', {
  allowlistRules: ['203.0.113.0/24'],
});

const input = new medialive.Input(stack, 'Input', {
  inputName: 'srt-outputs-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
  ]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'h264-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

new medialive.Channel(stack, 'Channel', {
  channelName: 'srt-outputs',
  inputs: [{ input }],
  channelSecurityGroups: [channelSg],
  outputGroups: [
    medialive.OutputGroupConfiguration.srt({
      name: 'srt',
      outputs: [
        // LISTENER mode — MediaLive opens a socket and waits for the downstream to connect.
        {
          encodes: [video, audio],
          outputName: 'srt_listener',
          destinations: [medialive.SrtDestination.listener({ listenerPort: 5000, encryptionPassphraseSecret: passphrase })],
        },
        // CALLER mode — MediaLive dials out to a remote listener.
        {
          encodes: [video, audio],
          outputName: 'srt_caller',
          destinations: [medialive.SrtDestination.caller({ address: '203.0.113.20', port: 5001, encryptionPassphraseSecret: passphrase })],
        },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-channel-srt-outputs', {
  testCases: [stack],
});

app.synth();
