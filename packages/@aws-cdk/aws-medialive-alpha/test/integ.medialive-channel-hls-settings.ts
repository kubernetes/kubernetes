// Tests per-output HLS settings (M3U8 container, audio-only rendition) and group-level CDN/encryption configuration.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-hls-settings');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'hls-settings-srt-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
  ]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'h264-720p',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'hls-settings',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'hls')],
      adMarkers: [medialive.HlsAdMarkers.ELEMENTAL_SCTE35],
      // Group-level CDN + static-key encryption settings — previously had zero integ coverage.
      // The destination below uses s3ssl://, which requires hlsCdnSettings to be s3 or basicPut.
      hlsCdnSettings: medialive.HlsCdnSettings.s3({
        cannedAcl: medialive.S3CannedAcl.BUCKET_OWNER_FULL_CONTROL,
      }),
      encryptionType: medialive.HlsEncryptionType.AES128,
      keyProviderSettings: medialive.HlsKeyProviderSettings.staticKey({
        keyProviderServerUrl: 'https://license.example.com/key',
        staticKeyValue: cdk.SecretValue.unsafePlainText('11111111111111111111111111111111'),
      }),
      outputs: [
        // Standard video output with explicit M3U8 transport-stream settings.
        {
          encodes: [video],
          outputName: 'video',
          hlsSettings: medialive.HlsSettings.standard({
            audioRenditionSets: 'program-audio',
            m3u8Settings: medialive.M3u8Settings.of({
              scte35Behavior: medialive.M3u8Scte35Behavior.PASSTHROUGH,
              scte35Pid: '500',
              pcrControl: medialive.M3u8PcrControl.PCR_EVERY_PES_PACKET,
              pcrPeriod: cdk.Duration.millis(40),
              timedMetadataBehavior: medialive.M3u8TimedMetadataBehavior.PASSTHROUGH,
              videoPid: '481',
              audioPids: '482-498',
              programNum: 1,
            }),
          }),
        },
        // Audio-only AAC rendition that the video output's rendition set references.
        {
          encodes: [audio],
          outputName: 'audio',
          hlsSettings: medialive.HlsSettings.audioOnly({
            audioGroupId: 'program-audio',
            audioTrackType: medialive.HlsAudioTrackType.ALTERNATE_AUDIO_AUTO_SELECT_DEFAULT,
            segmentType: medialive.HlsAudioOnlySegmentType.AAC,
          }),
        },
      ],
    }),
    // Second HLS group pushing to an https CDN destination. An https URL requires hlsCdnSettings
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-cdn',
      destinations: [medialive.OutputDestination.url('https://cdn.example.com/live/stream')],
      hlsCdnSettings: medialive.HlsCdnSettings.basicPut({
        numRetries: 5,
        connectionRetryInterval: 2,
      }),
      outputs: [{ encodes: [video], outputName: 'cdn_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-hls-settings', {
  testCases: [stack],
});

app.synth();
