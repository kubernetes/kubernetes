// Tests CMAF Ingest output group with an AV1 video encode.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-channel-cmaf-ingest');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'cmaf-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
});

// AV1 video — only supported in CMAF Ingest output groups.
const video = medialive.EncodeConfiguration.video({
  name: 'av1-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.av1({
    rateControl: medialive.Av1RateControl.qvbr({ maxBitrate: cdk.Bitrate.mbps(4), qvbrQualityLevel: 7 }),
    framerate: medialive.Framerate.FPS_30,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'cmaf-ingest-channel',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.cmafIngest({
      name: 'cmaf',
      // SINGLE_PIPELINE channel — exactly one destination. CMAF ingest URLs must end with '/'.
      destinations: [medialive.OutputDestination.url('https://ingest.example.com/v1/channel/pipeline-0/')],
      segment: medialive.Segment.seconds(6),
      // Map captions channels to languages for this CMAF Ingest output group.
      captionLanguageMappings: [
        { captionChannel: 1, languageCode: 'eng' },
        { captionChannel: 2, languageCode: 'spa' },
      ],
      // CMAF Ingest requires one track per output, each with a unique name modifier.
      outputs: [
        { outputName: 'cmaf-video', nameModifier: '_video', encode: video },
        { outputName: 'cmaf-audio', nameModifier: '_audio', encode: audio },
      ],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-channel-cmaf-ingest', {
  testCases: [stack],
});

app.synth();
