// Tests video and audio encode override settings (H.264, H.265, AV1, AAC, AC3, EAC3) on a CMAF Ingest channel.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-encode-overrides');

const input = new medialive.Input(stack, 'Input', {
  inputName: 'overrides-srt-input',
  input: medialive.InputConfiguration.srtCaller([
    { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
  ]),
});

// H.264: all adaptive-quantization and scene-change detection disabled, with a fixed sub-GOP.
const h264 = medialive.EncodeConfiguration.video({
  name: 'h264-overrides',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    profile: medialive.H264Profile.HIGH,
    sceneChangeDetect: medialive.H264SceneChangeDetect.DISABLED,
    spatialAq: medialive.H264SpatialAq.DISABLED,
    temporalAq: medialive.H264TemporalAq.DISABLED,
    subgopLength: medialive.SubgopLength.FIXED,
  }),
});

// H.264 High 4:2:2 — exercises H264Profile.HIGH_422 / HIGH_422_10BIT, added during the enum
// audit (the enum previously stopped at HIGH_10BIT).
const h264High422 = medialive.EncodeConfiguration.video({
  name: 'h264-high422',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h264({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    profile: medialive.H264Profile.HIGH_422,
  }),
});

// H.265: scene-change detection disabled, HLG 2020 color space.
const h265 = medialive.EncodeConfiguration.video({
  name: 'h265-overrides',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.h265({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.H265RateControl.cbr({ bitrate: cdk.Bitrate.mbps(3) }),
    profile: medialive.H265Profile.MAIN_10BIT,
    sceneChangeDetect: medialive.H265SceneChangeDetect.DISABLED,
    colorSpaceSettings: medialive.H265ColorSpaceSettings.hlg2020(),
  }),
});

// AV1: HLG 2020 color space with explicit AQ / scene-change toggles, plus metadata-OBU timecode
// insertion — exercises Av1TimecodeInsertion.METADATA_OBU, fixed during the enum audit (the
// enum previously had the H.264/H.265 timecode values copy-pasted in by mistake).
const av1 = medialive.EncodeConfiguration.video({
  name: 'av1-overrides',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.av1({
    framerate: medialive.Framerate.FPS_29_97,
    rateControl: medialive.Av1RateControl.qvbr({ maxBitrate: cdk.Bitrate.mbps(3), qvbrQualityLevel: 7 }),
    bitDepth: medialive.Av1BitDepth.BIT_DEPTH_10,
    colorSpaceSettings: medialive.Av1ColorSpaceSettings.hlg2020(),
    sceneChangeDetect: medialive.Av1SceneChangeDetect.DISABLED,
    spatialAq: medialive.Av1SpatialAq.ENABLED,
    temporalAq: medialive.Av1TemporalAq.ENABLED,
    timecodeInsertion: medialive.Av1TimecodeInsertion.METADATA_OBU,
  }),
});

// AAC: codec-level settings + configured audio type + DVB DASH accessibility + DASH roles +
// language/stream naming + a stereo channel remix.
const aac = medialive.EncodeConfiguration.audio({
  name: 'aac-accessible',
  codec: medialive.AudioCodecSettings.aac({
    bitrate: cdk.Bitrate.kbps(192),
    profile: medialive.AacProfile.LC,
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
    rateControlMode: medialive.AacRateControlMode.CBR,
    sampleRate: medialive.AudioSampleRate.HZ_48000,
  }),
  languageCode: 'eng',
  streamName: 'English (described)',
  audioType: medialive.AudioType.VISUAL_IMPAIRED_COMMENTARY,
  dvbDashAccessibility: medialive.DvbDashAccessibility.VISUALLY_IMPAIRED,
  audioDashRoles: [medialive.AudioDashRole.ALTERNATE, medialive.AudioDashRole.COMMENTARY],
  remixSettings: {
    channelsIn: 2,
    channelsOut: 2,
    channelMappings: [
      { outputChannel: 0, inputChannelLevels: [{ inputChannel: 1, gain: -3 }] },
      { outputChannel: 1, inputChannelLevels: [{ inputChannel: 0, gain: -3 }] },
    ],
  },
});

// AC3: surround codec settings + explicit audioTypeControl override + loudness normalization.
const ac3 = medialive.EncodeConfiguration.audio({
  name: 'ac3-main',
  codec: medialive.AudioCodecSettings.ac3({
    bitrate: cdk.Bitrate.kbps(384),
    codingMode: medialive.Ac3CodingMode.CODING_MODE_3_2_LFE,
    dialNorm: 24,
    bitstreamMode: medialive.Ac3BitstreamMode.COMPLETE_MAIN,
  }),
  audioTypeControl: medialive.AudioTypeControl.USE_CONFIGURED,
  audioType: medialive.AudioType.CLEAN_EFFECTS,
  audioNormalization: {
    algorithm: medialive.AudioNormalizationAlgorithm.ITU_1770_1,
    algorithmControl: medialive.AudioNormalizationAlgorithmControl.CORRECT_AUDIO,
    targetLkfs: -24,
  },
});

// EAC3: a second-language track with its default (surround) coding mode + naming.
const eac3 = medialive.EncodeConfiguration.audio({
  name: 'eac3-main',
  codec: medialive.AudioCodecSettings.eac3(),
  languageCode: 'spa',
  streamName: 'Espanol',
});

// EAC3 Atmos: immersive codec + Nielsen NAES II/NW watermarking.
const eac3Atmos = medialive.EncodeConfiguration.audio({
  name: 'eac3-atmos',
  codec: medialive.AudioCodecSettings.eac3Atmos(),
  audioWatermarkSettings: {
    nielsenWatermarks: {
      distributionType: medialive.NielsenDistributionType.PROGRAM_CONTENT,
      naesIiNwSettings: { checkDigitString: 'CD', sid: 123 },
    },
  },
});

// Frame capture with a sub-second interval — exercises the MILLISECONDS capture-interval unit
// (whole-second Durations render as SECONDS; sub-second as MILLISECONDS).
const frameCapture = medialive.EncodeConfiguration.video({
  name: 'thumbnails',
  width: 1280,
  height: 720,
  codec: medialive.VideoCodecSettings.frameCapture({
    captureInterval: cdk.Duration.millis(500),
  }),
});

const thumbnailBucket = new s3.Bucket(stack, 'ThumbnailBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'encode-overrides',
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.cmafIngest({
      name: 'cmaf',
      destinations: [medialive.OutputDestination.url('https://ingest.example.com/v1/channel/')],
      outputs: [
        { outputName: 'h264', nameModifier: '_h264', encode: h264 },
        { outputName: 'h264-high422', nameModifier: '_h264_422', encode: h264High422 },
        { outputName: 'h265', nameModifier: '_h265', encode: h265 },
        { outputName: 'av1', nameModifier: '_av1', encode: av1 },
        { outputName: 'aac', nameModifier: '_aac', encode: aac },
        { outputName: 'ac3', nameModifier: '_ac3', encode: ac3 },
        { outputName: 'eac3', nameModifier: '_eac3', encode: eac3 },
        { outputName: 'eac3-atmos', nameModifier: '_atmos', encode: eac3Atmos },
      ],
    }),
    medialive.OutputGroupConfiguration.frameCapture({
      name: 'thumbnails',
      destinations: [medialive.S3OutputDestination.toBucket(thumbnailBucket, 'thumbs')],
      outputs: [{ encodes: [frameCapture], outputName: 'thumbnail', nameModifier: '_thumb' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-encode-overrides', {
  testCases: [stack],
});

app.synth();
