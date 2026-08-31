// Tests S3-backed file locations (avail blanking, blackout slate, burn-in font, audio-only image) and network input settings.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-file-locations');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'file-locations-url-pull',
  input: medialive.InputConfiguration.urlPull([
    medialive.InputSource.url('https://example.com/stream.m3u8'),
  ]),
});

// Bucket holding the slate/blackout images, burn-in font, and cover-art image.
const assetsBucket = new s3.Bucket(stack, 'AssetsBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});
const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
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

// Burn-in captions with a font sourced from S3 (auto-grants read on the bucket).
const burnIn = medialive.EncodeConfiguration.caption({
  name: 'burnin-eng',
  captionSelectorName: 'embedded-cc',
  languageCode: 'eng',
  destination: medialive.CaptionDestination.burnIn({
    alignment: medialive.CaptionAlignment.CENTERED,
    fontColor: medialive.CaptionFontColor.WHITE,
    font: medialive.FileLocation.fromBucket(assetsBucket, 'fonts/caption-font.ttf'),
  }),
});

const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'file-locations',
  inputs: [{
    input,
    captionSelectors: [medialive.CaptionSelector.embedded('embedded-cc')],
    networkInputSettings: {
      serverValidation: medialive.ServerValidation.CHECK_CRYPTOGRAPHY_AND_VALIDATE_NAME,
      hlsInputSettings: {
        bufferSegments: 3,
        scte35Source: medialive.HlsScte35Source.MANIFEST,
      },
    },
  }],
  availBlanking: {
    state: medialive.AvailBlankingState.ENABLED,
    image: medialive.FileLocation.fromBucket(assetsBucket, 'slates/avail.png'),
  },
  blackoutSlate: {
    state: medialive.BlackoutSlateState.ENABLED,
    image: medialive.FileLocation.fromBucket(assetsBucket, 'slates/blackout.png'),
  },
  // Color correction with a 3D LUT read from S3 (auto-grants read on the bucket).
  colorCorrections: [{
    inputColorSpace: medialive.ColorSpace.REC_601,
    outputColorSpace: medialive.ColorSpace.REC_709,
    lut: medialive.Lut.fromBucket(assetsBucket, 'luts/rec601-to-rec709.cube'),
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'hls')],
      outputs: [
        // Burn-in captions ride the video output; the video references the audio rendition set.
        {
          encodes: [video, burnIn],
          outputName: 'video',
          hlsSettings: medialive.HlsSettings.standard({ audioRenditionSets: 'program-audio' }),
        },
        // Audio-only rendition with cover-art image sourced from S3.
        {
          encodes: [audio],
          outputName: 'audio',
          hlsSettings: medialive.HlsSettings.audioOnly({
            audioGroupId: 'program-audio',
            audioOnlyImage: medialive.FileLocation.fromBucket(assetsBucket, 'art/cover.png'),
          }),
        },
      ],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-file-locations', {
  testCases: [stack],
});

app.synth();
