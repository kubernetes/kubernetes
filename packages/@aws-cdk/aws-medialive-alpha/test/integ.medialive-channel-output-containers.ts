/**
 * Integration test: MediaLive output container settings — UDP with FEC and explicit
 * M2tsSettings, plain UDP, and Archive with a raw WAV container.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-output-containers');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'output-containers-srt-input',
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

const rawAudio = medialive.EncodeConfiguration.audio({
  name: 'wav-raw',
  codec: medialive.AudioCodecSettings.wav({
    codingMode: medialive.WavCodingMode.CODING_MODE_2_0,
  }),
});

const archiveBucket = new s3.Bucket(stack, 'ArchiveBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});
const m2tsSettings = medialive.M2tsSettings.of({
  bitrate: cdk.Bitrate.mbps(8),
  rateMode: medialive.M2tsRateMode.CBR,
  bufferModel: medialive.M2tsBufferModel.MULTIPLEX,
  pcrControl: medialive.M2tsPcrControl.PCR_EVERY_PES_PACKET,
  videoPid: '481',
  audioPids: '482-498',
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'output-containers',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.udp({
      name: 'udp-fec',
      destinations: [medialive.UdpOutputDestination.rtp({ address: '203.0.113.5', port: 5000 })],
      outputs: [{
        encodes: [video, audio],
        outputName: 'ts',
        fec: { mode: medialive.FecMode.COLUMN_AND_ROW, columnDepth: 10, rowLength: 10 },
        m2tsSettings,
      }],
    }),
    medialive.OutputGroupConfiguration.udp({
      name: 'udp-plain',
      destinations: [medialive.UdpOutputDestination.udp({ address: '203.0.113.6', port: 5001 })],
      outputs: [{ encodes: [video, audio], outputName: 'ts-plain' }],
    }),
    medialive.OutputGroupConfiguration.archive({
      name: 'archive',
      destinations: [medialive.S3OutputDestination.url(`s3ssl://${archiveBucket.bucketName}/archive/recording`)],
      outputs: [
        {
          encodes: [video, audio],
          outputName: 'ts-archive',
          nameModifier: '_ts',
          container: medialive.ArchiveContainer.m2ts(m2tsSettings),
        },
        {
          encodes: [rawAudio],
          outputName: 'raw',
          nameModifier: '_raw',
          extension: 'wav',
          container: medialive.ArchiveContainer.raw(),
        },
      ],
    }),
  ],
});

archiveBucket.grantReadWrite(channel.role);

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-output-containers', {
  testCases: [stack],
});

app.synth();
