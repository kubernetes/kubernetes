// Tests MediaLive SRT caller output targeting a MediaConnect Router Input SRT listener with shared passphrase.
import * as emx from '@aws-cdk/aws-mediaconnect-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-srt-to-router');

// Shared SRT passphrase: the router input listener decrypts with it, and the MediaLive
// SRT caller must encrypt with the same secret to connect.
const srtPassphrase = new Secret(stack, 'SrtPassphrase');

// --- MediaConnect: an SRT-listener Router Input that MediaLive will push to ---
const networkInterface = new emx.RouterNetworkInterface(stack, 'NetInterface', {
  routerNetworkInterfaceName: 'srt-net-interface',
  configuration: emx.RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

const routerInput = new emx.RouterInput(stack, 'SrtRouterInput', {
  routerInputName: 'from-medialive-srt',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: emx.RoutingScope.REGIONAL,
  tier: emx.RouterInputTier.INPUT_50,
  // Passphrase encryption (not automatic) so an external SRT caller can match the secret.
  transitEncryption: { secret: srtPassphrase },
  configuration: emx.RouterInputConfiguration.standard({
    networkInterface,
    protocol: emx.RouterInputProtocol.srtListener({
      port: 5000,
      minimumLatency: cdk.Duration.millis(200),
    }),
  }),
});

// --- MediaLive input (SRT caller source feeding the channel) ---
const emlInput = new medialive.Input(stack, 'Input', {
  inputName: 'srt-to-router-source',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 9000 },
  ]),
});

// --- Encode profiles ---
const video = medialive.EncodeConfiguration.video({
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
  codec: medialive.AudioCodecSettings.aac({
    bitrate: cdk.Bitrate.kbps(192),
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
  }),
});

// --- MediaLive Channel: SRT output targeting the MediaConnect router input (L2) ---
new medialive.Channel(stack, 'Channel', {
  channelName: 'srt-to-router-channel',
  channelClass: medialive.ChannelClass.SINGLE_PIPELINE,
  inputs: [{ input: emlInput }],
  inputSpecification: medialive.InputSpecification.standard({
    codec: medialive.InputCodec.AVC,
    maximumBitrate: medialive.InputMaximumBitrate.MAX_20_MBPS,
    resolution: medialive.InputResolution.HD,
  }),
  outputGroups: [
    medialive.OutputGroupConfiguration.srt({
      name: 'srt',
      // Exercises the SrtInputLossAction fix from the enum audit (previously had the wrong
      // values entirely — DROP_TS/DROP_PROGRAM/EMIT_PROGRAM are the real allowed values).
      inputLossAction: medialive.SrtInputLossAction.DROP_PROGRAM,
      outputs: [{
        encodes: [video, audio],
        outputName: 'srt_out',
        // Target the router input's ingest endpoint explicitly, sharing the same passphrase
        // secret the router input decrypts with.
        destinations: [medialive.SrtDestination.callerUrl(routerInput.endpoints[0].url, {
          encryptionPassphraseSecret: srtPassphrase,
        })],
        latency: cdk.Duration.millis(1000),
        // Exercises SrtEncryptionType.AES192, added during the enum audit (previously only
        // AES128/AES256 were modeled).
        encryptionType: medialive.SrtEncryptionType.AES192,
      }],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-srt-to-router', {
  testCases: [stack],
});

app.synth();
