import * as mediapackagev2 from '@aws-cdk/aws-mediapackagev2-alpha';
import { App, Stack, Bitrate, Duration, Lazy } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import type { Metric } from 'aws-cdk-lib/aws-cloudwatch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { StringParameter } from 'aws-cdk-lib/aws-ssm';
import {
  Channel,
  ChannelClass,
  EncodeConfiguration,
  type IChannel,
  InputConfiguration,
  Input,
  InputPreference,
  FailoverCondition,
  AudioSelector,
  AudioLanguageSelectionPolicy,
  AudioPreMixerSettings,
  DolbyEProgramSelection,
  CaptionSelector,
  Convert608To708,
  OcrLanguage,
  Scte20Detection,
  VideoSelection,
  VideoColorSpace,
  VideoColorSpaceUsage,
  type AutomaticInputFailover,
  LogLevel,
  OutputGroupConfiguration,
  Pipeline,
  VideoCodecSettings,
  H264Profile,
  H264RateControl,
  H265Profile,
  H265Tier,
  H265RateControl,
  GopSize,
  ScalingBehavior,
  InputSecurityGroup,
  Framerate,
  TimecodeSource,
  OutputTimingSource,
  AvailBlankingState,
  AvailSettings,
  LinkedChannelSettings,
  BlackoutSlateState,
  NetworkEndBlackout,
  UdpInputLossAction,
  HlsMode,
  HlsProgramDateTimeClock,
  RtmpAuthenticationScheme,
  MediaPackageV2EndpointId,
  MediaPackageV2Destination,
  MediaConnectRouterSettings,
  M2tsSettings,
  M2tsRateMode,
  M2tsScte35Control,
  AudioCodecSettings,
  WavCodingMode,
  ArchiveContainer,
  FecMode,
  DvbSdtOutputMode,
  SrtDestination,
  OutputDestination,
  S3OutputDestination,
  UdpOutputDestination,
  RtmpDestination,
  InputSource,
  Segment,
  Id3Behavior,
  KlvBehavior,
  NielsenId3Behavior,
  Scte35Type,
  TimedMetadataId3Frame,
  TimedMetadataPassthrough,
  SdiSource,
  SdiType,
  Cluster,
  ThumbnailState,
  H264SceneChangeDetect,
  H264SpatialAq,
  H264TemporalAq,
  DvbDashAccessibility,
  AudioType,
  AudioLanguageCodeControl,
  AudioDashRole,
  MediaPackageV2HlsSetting,
  HlsSettings,
  M3u8Settings,
  M3u8Scte35Behavior,
  M3u8PcrControl,
  CaptionDestination,
  CaptionAccessibility,
  CaptionDashRole,
  CaptionFontColor,
  CaptionOutlineColor,
  CaptionAlignment,
  CaptionFontSize,
  InputLossImageType,
  OutputLocking,
  PipelineLockingMethod,
  Scte35SegmentationScope,
  FileLocation,
  Lut,
  ColorSpace,
  HlsScte35Source,
  ServerValidation,
  type VpcOutputSettings,
  Scte35FlagBehavior,
  NielsenDistributionType,
  NielsenCbetStepaside,
  NielsenWatermarkTimezone,
  SrtInputLossAction,
  SrtEncryptionType,
  NielsenPcmToId3TaggingState,
  Av1RateControl,
  InputNetworkLocation,
  InputSpecification,
} from '../lib';

let app: App;
let stack: Stack;
let defaultInput: Input;
let empChannel: mediapackagev2.IChannel;
// Shared passphrase for SRT output destinations (SRT output is always encrypted).
let srtSecret: secretsmanager.ISecret;
beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
  const sg = new InputSecurityGroup(stack, 'DefaultSG', {
    allowlistRules: ['0.0.0.0/0'],
  });
  defaultInput = new Input(stack, 'DefaultInput', {
    inputName: 'test-input',
    input: InputConfiguration.srtListener({ inputSecurityGroups: [sg] }),
  });
  srtSecret = new secretsmanager.Secret(stack, 'SrtSecret');
  const empGroup = new mediapackagev2.ChannelGroup(stack, 'EmpGroup', {
    channelGroupName: 'emp-group',
  });
  empChannel = new mediapackagev2.Channel(stack, 'EmpChannel', {
    channelGroup: empGroup,
    channelName: 'emp-channel',
  });
});

describe('Channel', () => {
  test('creates a minimal channel with required input and output group', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'aac-stereo', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      channelName: 'my-channel',
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'mediapackage',
          channel: empChannel,
          outputs: [
            { encode: hd, outputName: 'hd_output' },
            { encode: audio, outputName: 'audio_output' },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Name: 'my-channel',
      ChannelClass: 'SINGLE_PIPELINE',
      LogLevel: 'DISABLED',
      EncoderSettings: {
        VideoDescriptions: [Match.objectLike({ Name: 'hd', Width: 1920, Height: 1080 })],
        AudioDescriptions: [{ Name: 'aac-stereo' }],
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({ VideoDescriptionName: 'hd' }),
              Match.objectLike({ AudioDescriptionNames: ['aac-stereo'] }),
            ]),
          }),
        ]),
      },
      InputAttachments: Match.arrayWith([
        Match.objectLike({}),
      ]),
    });
  });

  test('creates a standard class channel with role and log level', () => {
    const role = new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
    });
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      channelClass: ChannelClass.SINGLE_PIPELINE,
      logLevel: LogLevel.INFO,
      role,
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      ChannelClass: 'SINGLE_PIPELINE',
      LogLevel: 'INFO',
      RoleArn: { 'Fn::GetAtt': [Match.anyValue(), 'Arn'] },
    });
  });

  test('H264 AQ and scene-change flags are overrideable', () => {
    const video = EncodeConfiguration.video({
      name: 'video',
      width: 1280,
      height: 720,
      codec: VideoCodecSettings.h264({
        framerate: Framerate.FPS_29_97,
        sceneChangeDetect: H264SceneChangeDetect.DISABLED,
        spatialAq: H264SpatialAq.DISABLED,
        temporalAq: H264TemporalAq.DISABLED,
      }),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            H264Settings: Match.objectLike({
              SceneChangeDetect: 'DISABLED',
              SpatialAq: 'DISABLED',
              TemporalAq: 'DISABLED',
            }),
          },
        })],
      }),
    });
  });

  test('audioTypeControl defaults to USE_CONFIGURED when audioType is set, and DASH accessibility is wired', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({
      name: 'aac',
      audioType: AudioType.CLEAN_EFFECTS,
      audioDashRoles: [AudioDashRole.MAIN],
      dvbDashAccessibility: DvbDashAccessibility.VISUALLY_IMPAIRED,
      codec: AudioCodecSettings.aac(),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          AudioType: 'CLEAN_EFFECTS',
          AudioTypeControl: 'USE_CONFIGURED',
          AudioDashRoles: ['MAIN'],
          DvbDashAccessibility: 'DVBDASH_1_VISUALLY_IMPAIRED',
        })],
      }),
    });
  });

  test('audioTypeControl defaults to FOLLOW_INPUT when audioType is unset', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({ AudioTypeControl: 'FOLLOW_INPUT' })],
      }),
    });
  });

  test('languageCodeControl defaults to USE_CONFIGURED when languageCode is set', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'aac', languageCode: 'eng', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          LanguageCode: 'eng',
          LanguageCodeControl: 'USE_CONFIGURED',
        })],
      }),
    });
  });

  test('languageCodeControl defaults to FOLLOW_INPUT when languageCode is unset', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({ LanguageCodeControl: 'FOLLOW_INPUT' })],
      }),
    });
  });

  test('languageCodeControl can be FOLLOW_INPUT while a fallback languageCode is configured', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({
      name: 'aac',
      languageCode: 'eng',
      languageCodeControl: AudioLanguageCodeControl.FOLLOW_INPUT,
      codec: AudioCodecSettings.aac(),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          LanguageCode: 'eng',
          LanguageCodeControl: 'FOLLOW_INPUT',
        })],
      }),
    });
  });

  test('MediaPackageV2 audio output hlsDefault/hlsAutoSelect are overrideable', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [
            { encode: hd, outputName: 'hd_output' },
            {
              encode: audio,
              outputName: 'audio_output',
              hlsDefault: MediaPackageV2HlsSetting.YES,
              hlsAutoSelect: MediaPackageV2HlsSetting.YES,
            },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  MediaPackageOutputSettings: {
                    MediaPackageV2DestinationSettings: Match.objectLike({
                      HlsDefault: 'YES',
                      HlsAutoSelect: 'YES',
                    }),
                  },
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('HLS output M3U8 settings are configurable', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [video],
            outputName: 'video_output',
            hlsSettings: HlsSettings.standard({
              audioRenditionSets: 'programAudio',
              m3u8Settings: M3u8Settings.of({
                scte35Behavior: M3u8Scte35Behavior.PASSTHROUGH,
                scte35Pid: '500',
                pcrControl: M3u8PcrControl.PCR_EVERY_PES_PACKET,
                videoPid: '481',
                programNum: 1,
              }),
            }),
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  HlsOutputSettings: Match.objectLike({
                    HlsSettings: {
                      StandardHlsSettings: {
                        AudioRenditionSets: 'programAudio',
                        M3u8Settings: Match.objectLike({
                          Scte35Behavior: 'PASSTHROUGH',
                          Scte35Pid: '500',
                          PcrControl: 'PCR_EVERY_PES_PACKET',
                          VideoPid: '481',
                          ProgramNum: 1,
                        }),
                      },
                    },
                  }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('HLS output defaults to standard settings with empty M3U8 settings', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  HlsOutputSettings: Match.objectLike({
                    HlsSettings: { StandardHlsSettings: { M3u8Settings: {} } },
                  }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('caption output with burn-in destination, accessibility, and DASH roles', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const caption = EncodeConfiguration.caption({
      name: 'eng-burnin',
      captionSelectorName: 'english',
      languageCode: 'eng',
      destination: CaptionDestination.burnIn({
        alignment: CaptionAlignment.CENTERED,
        fontColor: CaptionFontColor.WHITE,
        outlineColor: CaptionOutlineColor.BLACK,
        fontSize: CaptionFontSize.AUTO,
        backgroundOpacity: 128,
      }),
      accessibility: CaptionAccessibility.IMPLEMENTS_ACCESSIBILITY_FEATURES,
      captionDashRoles: [CaptionDashRole.CAPTION, CaptionDashRole.MAIN],
      dvbDashAccessibility: DvbDashAccessibility.HARD_OF_HEARING,
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, caption], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: Match.arrayWith([
          Match.objectLike({
            Name: 'eng-burnin',
            Accessibility: 'IMPLEMENTS_ACCESSIBILITY_FEATURES',
            CaptionDashRoles: ['CAPTION', 'MAIN'],
            DvbDashAccessibility: 'DVBDASH_2_HARD_OF_HEARING',
            DestinationSettings: {
              BurnInDestinationSettings: Match.objectLike({
                Alignment: 'CENTERED',
                FontColor: 'WHITE',
                OutlineColor: 'BLACK',
                FontSize: 'auto',
                BackgroundOpacity: 128,
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('caption output with WebVTT sidecar destination', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const caption = EncodeConfiguration.caption({
      name: 'eng-webvtt',
      captionSelectorName: 'english',
      destination: CaptionDestination.webvtt(),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, caption], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: Match.arrayWith([
          Match.objectLike({
            Name: 'eng-webvtt',
            DestinationSettings: { WebvttDestinationSettings: {} },
          }),
        ]),
      }),
    });
  });

  test('initial input attachment with name', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput, inputAttachmentName: 'primary' }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: [Match.objectLike({
        InputAttachmentName: 'primary',
      })],
    });
  });

  test('adds additional input via addInput', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const secondSg = new InputSecurityGroup(stack, 'SecondSG', {
      allowlistRules: ['0.0.0.0/0'],
    });
    const secondInput = new Input(stack, 'SecondInput', {
      inputName: 'backup-input',
      input: InputConfiguration.srtListener({ inputSecurityGroups: [secondSg] }),
    });

    const channel = new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput, inputAttachmentName: 'primary' }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    channel.addInput({ input: secondInput, inputAttachmentName: 'backup' });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({ InputAttachmentName: 'primary' }),
        Match.objectLike({ InputAttachmentName: 'backup' }),
      ]),
    });
  });

  test('adds additional output group via addOutputGroup', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const sd = EncodeConfiguration.video({
      name: 'sd',
      width: 1280,
      height: 720,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });

    const channel = new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'mediapackage',
          channel: empChannel,
          outputs: [
            { encode: hd, outputName: 'hd_output' },
            { encode: sd, outputName: 'sd_output' },
            { encode: audio, outputName: 'audio_output' },
          ],
        }),
      ],
    });

    // Add a second output group via helper
    channel.addOutputGroup(
      OutputGroupConfiguration.hls({
        name: 'hls',
        destinations: [OutputDestination.url('s3ssl://bucket/archive')],
        outputs: [
          { encodes: [hd, audio], outputName: 'archive_output' },
        ],
      }),
    );

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: {
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: { MediaPackageGroupSettings: Match.anyValue() },
            Outputs: Match.arrayWith([
              Match.objectLike({ VideoDescriptionName: 'hd' }),
              Match.objectLike({ VideoDescriptionName: 'sd' }),
              Match.objectLike({ AudioDescriptionNames: ['aac'] }),
            ]),
          }),
          Match.objectLike({
            OutputGroupSettings: { HlsGroupSettings: Match.anyValue() },
            Outputs: [
              Match.objectLike({ VideoDescriptionName: 'hd', AudioDescriptionNames: ['aac'] }),
            ],
          }),
        ]),
      },
    });
  });

  test('imports from arn', () => {
    const imported = Channel.fromChannelArn(stack, 'Imported', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    expect(imported.channelArn).toBe('arn:aws:medialive:us-east-1:123456789012:channel:1234567');
    expect(imported.channelId).toBe('1234567');
  });
});

describe('Channel metrics', () => {
  test('imported channels can build CloudWatch metrics', () => {
    const imported = Channel.fromChannelArn(stack, 'Imported', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    const metric = imported.metricNetworkIn(Pipeline.PIPELINE_0);
    expect(metric.namespace).toBe('AWS/MediaLive');
    expect(metric.metricName).toBe('NetworkIn');
    expect(metric.statistic).toBe('Average');
    expect(metric.dimensions).toEqual({ ChannelId: '1234567', Pipeline: '0' });
  });

  test.each<[string, string, string, (ch: IChannel, p: Pipeline) => Metric]>([
    ['metricActiveAlerts', 'ActiveAlerts', 'Maximum', (ch, p) => ch.metricActiveAlerts(p)],
    ['metricNetworkIn', 'NetworkIn', 'Average', (ch, p) => ch.metricNetworkIn(p)],
    ['metricNetworkOut', 'NetworkOut', 'Average', (ch, p) => ch.metricNetworkOut(p)],
    ['metricInputVideoFrameRate', 'InputVideoFrameRate', 'Maximum', (ch, p) => ch.metricInputVideoFrameRate(p)],
    ['metricFillMsec', 'FillMsec', 'Maximum', (ch, p) => ch.metricFillMsec(p)],
    ['metricInputLossSeconds', 'InputLossSeconds', 'Sum', (ch, p) => ch.metricInputLossSeconds(p)],
    ['metricDroppedFrames', 'DroppedFrames', 'Sum', (ch, p) => ch.metricDroppedFrames(p)],
    ['metricSvqTime', 'SvqTime', 'Maximum', (ch, p) => ch.metricSvqTime(p)],
  ])('%s emits the AWS-published name and recommended statistic', (helper, expectedName, expectedStat, build) => {
    const imported = Channel.fromChannelArn(stack, `Ch-${helper}`, 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    const metric = build(imported, Pipeline.PIPELINE_0);
    expect(metric.metricName).toBe(expectedName);
    expect(metric.statistic).toBe(expectedStat);
    expect(metric.namespace).toBe('AWS/MediaLive');
    expect(metric.dimensions).toEqual({ ChannelId: '1234567', Pipeline: '0' });
  });

  test('users can scope a metric to Pipeline 1', () => {
    const imported = Channel.fromChannelArn(stack, 'PipelineImport', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    const metric = imported.metricNetworkIn(Pipeline.PIPELINE_1);
    expect(metric.dimensions).toEqual({ ChannelId: '1234567', Pipeline: '1' });
  });

  test('metric() builds a custom-named metric in the AWS/MediaLive namespace', () => {
    const imported = Channel.fromChannelArn(stack, 'CustomMetric', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    const metric = imported.metric('Output4xxErrors', Pipeline.PIPELINE_0, { statistic: 'sum' });
    expect(metric.namespace).toBe('AWS/MediaLive');
    expect(metric.metricName).toBe('Output4xxErrors');
    expect(metric.statistic).toBe('Sum');
  });

  test('a caller-supplied dimensionsMap cannot drop the required ChannelId/Pipeline dimensions', () => {
    const imported = Channel.fromChannelArn(stack, 'DimMetric', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    const metric = imported.metric('NetworkIn', Pipeline.PIPELINE_0, {
      dimensionsMap: { CustomDimension: 'value' },
    });
    // Caller's extra dimension is kept, but ChannelId/Pipeline are applied last and survive.
    expect(metric.dimensions).toEqual({
      CustomDimension: 'value',
      ChannelId: '1234567',
      Pipeline: '0',
    });
  });

  test('SINGLE_PIPELINE channels reject Pipeline.PIPELINE_1', () => {
    const video = EncodeConfiguration.video({
      name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
    });
    const audio = EncodeConfiguration.audio({ name: 'audio', codec: AudioCodecSettings.aac() });
    const channel = new Channel(stack, 'SingleChannel', {
      // channelClass defaults to SINGLE_PIPELINE
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    expect(() => channel.metricNetworkIn(Pipeline.PIPELINE_1)).toThrow(
      /Pipeline\.PIPELINE_1 is not available on SINGLE_PIPELINE channels/,
    );
  });

  test('STANDARD channels accept Pipeline.PIPELINE_1', () => {
    const input = new Input(stack, 'Input', {
      inputName: 'test-input',
      input: InputConfiguration.srtCaller([{
        srtListenerAddress: '10.10.10.10/10',
        srtListenerPort: 5000,
      }, {
        srtListenerAddress: '10.10.10.11/10',
        srtListenerPort: 5000,
      }]),
    });
    const video = EncodeConfiguration.video({
      name: 'video',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'audio', codec: AudioCodecSettings.aac() });
    const channel = new Channel(stack, 'StandardChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [
            OutputDestination.url('s3ssl://bucket/p0'),
            OutputDestination.url('s3ssl://bucket/p1'),
          ],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    const metric = channel.metricNetworkIn(Pipeline.PIPELINE_1);
    expect(metric.dimensions).toEqual({ ChannelId: channel.channelId, Pipeline: '1' });
  });

  test('imported channels accept Pipeline.PIPELINE_1 (channel class unknown)', () => {
    const imported = Channel.fromChannelArn(stack, 'ImportedStandard', 'arn:aws:medialive:us-east-1:123456789012:channel:1234567');

    expect(() => imported.metricNetworkIn(Pipeline.PIPELINE_1)).not.toThrow();
  });
});

describe('Default codec settings (golden)', () => {
  // Exact-match (Match.objectEquals) golden records of the COMPLETE payload each codec factory
  // emits with no configuration. Any field not listed is proven absent — i.e. deliberately left
  // unset so the MediaLive service default applies. Update intentionally when a default changes;
  // a failure here means a default moved. (H.264's golden lives in the 'H.264 with defaults' test.)
  function goldenTemplate(...encodes: EncodeConfiguration[]): Template {
    new Channel(stack, 'GoldenChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes, outputName: 'out' }],
        }),
      ],
    });
    return Template.fromStack(stack);
  }

  test('h265({ framerate }) emits only these defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h265({ framerate: Framerate.FPS_29_97 }),
    });
    goldenTemplate(video).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            H265Settings: Match.objectEquals({
              Profile: 'MAIN',
              Tier: 'MAIN',
              GopSize: 1,
              GopSizeUnits: 'SECONDS',
              FramerateNumerator: 30000,
              FramerateDenominator: 1001,
              ParNumerator: 1,
              ParDenominator: 1,
              SceneChangeDetect: 'ENABLED',
              AdaptiveQuantization: 'AUTO',
              AfdSignaling: 'NONE',
              ScanType: 'PROGRESSIVE',
            }),
          },
        })],
      }),
    });
  });

  test('av1({ rateControl: qvbr }) emits only these defaults', () => {
    // AV1 is only valid in a MediaPackage V2 / CMAF Ingest group (not HLS), and MediaPackage V2
    // requires an explicit framerate — so this golden includes the framerate fields.
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.av1({
        rateControl: Av1RateControl.qvbr({ maxBitrate: Bitrate.mbps(4) }),
        framerate: Framerate.FPS_29_97,
      }),
    });
    new Channel(stack, 'GoldenChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: video, outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            Av1Settings: Match.objectEquals({
              MaxBitrate: 4_000_000,
              RateControlMode: 'QVBR',
              GopSize: 1,
              GopSizeUnits: 'SECONDS',
              FramerateNumerator: 30000,
              FramerateDenominator: 1001,
              AfdSignaling: 'NONE',
              Level: 'AV1_LEVEL_AUTO',
              LookAheadRateControl: 'HIGH',
              SceneChangeDetect: 'ENABLED',
              SpatialAq: 'ENABLED',
              TemporalAq: 'ENABLED',
            }),
          },
        })],
      }),
    });
  });

  test('aac() emits only these defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
    });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.aac() });
    goldenTemplate(video, audio).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          CodecSettings: {
            AacSettings: Match.objectEquals({
              Bitrate: 192_000,
              Profile: 'LC',
              CodingMode: 'CODING_MODE_2_0',
              RateControlMode: 'CBR',
              SampleRate: 48_000,
              RawFormat: 'NONE',
              Spec: 'MPEG4',
              InputType: 'NORMAL',
            }),
          },
        })],
      }),
    });
  });

  test('mp2() emits only these defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
    });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.mp2() });
    // MP2 is not a valid HLS audio codec, so this uses an Archive (M2TS) output group rather than
    // the shared goldenTemplate() helper (which builds an HLS group).
    new Channel(stack, 'Mp2Golden', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          CodecSettings: {
            // mp2() forces CodingMode and SampleRate to the console defaults; bitrate passes
            // through undefined so the MediaLive service default applies.
            Mp2Settings: Match.objectEquals({
              CodingMode: 'CODING_MODE_2_0',
              SampleRate: 48_000,
            }),
          },
        })],
      }),
    });
  });

  test('wav() emits only these defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
    });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.wav() });
    // WAV is only valid in a raw-container Archive output (audio-only), which the archive group
    // must pair with a video output — so this can't use the shared goldenTemplate() helper.
    new Channel(stack, 'WavGolden', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [
            { encodes: [video], outputName: 'video_out', nameModifier: '_ts' },
            { encodes: [audio], outputName: 'raw_out', nameModifier: '_raw', extension: 'wav', container: ArchiveContainer.raw() },
          ],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: [Match.objectLike({
          CodecSettings: {
            // wav() forces BitDepth, CodingMode, and SampleRate to the console defaults.
            WavSettings: Match.objectEquals({
              BitDepth: 16,
              CodingMode: 'CODING_MODE_2_0',
              SampleRate: 48_000,
            }),
          },
        })],
      }),
    });
  });

  // ac3(), eac3(), eac3Atmos(), and passthrough() impose no forced defaults: every field passes
  // through undefined (passthrough() emits an empty PassThroughSettings object), so there is no
  // golden default set to record for them.

  test('frameCapture() forces no defaults (no fixed capture interval)', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.frameCapture(),
    });
    goldenTemplate(video).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            FrameCaptureSettings: Match.objectEquals({}),
          },
        })],
      }),
    });
  });

  test('frameCapture() with a whole-second interval emits SECONDS units', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.frameCapture({ captureInterval: Duration.seconds(5) }),
    });
    goldenTemplate(video).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            FrameCaptureSettings: Match.objectEquals({
              CaptureInterval: 5,
              CaptureIntervalUnits: 'SECONDS',
            }),
          },
        })],
      }),
    });
  });

  test('frameCapture() with a sub-second interval emits MILLISECONDS units', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.frameCapture({ captureInterval: Duration.millis(500) }),
    });
    goldenTemplate(video).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            FrameCaptureSettings: Match.objectEquals({
              CaptureInterval: 500,
              CaptureIntervalUnits: 'MILLISECONDS',
            }),
          },
        })],
      }),
    });
  });
});

describe('Default configuration (golden)', () => {
  // Exact-match golden records of the COMPLETE payload each non-codec configuration surface emits
  // with minimal/no config. Any field not listed is proven absent — deliberately left unset so the
  // MediaLive service default applies. A failure here means a forced default moved.
  const video = () => EncodeConfiguration.video({
    name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
  });

  test('InputSpecification.standard() (the channel default) emits AVC / 20 Mbps / HD', () => {
    // No inputSpecification prop => Channel falls back to InputSpecification.standard().
    new Channel(stack, 'InputSpecGolden', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputSpecification: Match.objectEquals({
        Codec: 'AVC',
        MaximumBitrate: 'MAX_20_MBPS',
        Resolution: 'HD',
      }),
      // standard() emits no CdiInputSpecification.
      CdiInputSpecification: Match.absent(),
    });
  });

  test('InputSpecification.cdi() emits the standard defaults plus a CDI resolution of HD', () => {
    new Channel(stack, 'CdiSpecGolden', {
      inputs: [{ input: defaultInput }],
      inputSpecification: InputSpecification.cdi(),
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputSpecification: Match.objectEquals({
        Codec: 'AVC',
        MaximumBitrate: 'MAX_20_MBPS',
        Resolution: 'HD',
      }),
      CdiInputSpecification: Match.objectEquals({
        Resolution: 'HD',
      }),
    });
  });

  test('InputSpecification.elementalLink() emits neither specification', () => {
    new Channel(stack, 'ElementalLinkSpecGolden', {
      inputs: [{ input: defaultInput }],
      inputSpecification: InputSpecification.elementalLink(),
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputSpecification: Match.absent(),
      CdiInputSpecification: Match.absent(),
    });
  });

  test('OutputGroupConfiguration.hls(...) emits only these group-settings defaults', () => {
    new Channel(stack, 'HlsGroupGolden', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          OutputGroupSettings: {
            HlsGroupSettings: Match.objectEquals({
              Destination: { DestinationRefId: 'hls' },
              SegmentLength: 2,
              KeepSegments: 21,
              IndexNSegments: 10,
              Mode: 'LIVE',
              InputLossAction: 'EMIT_OUTPUT',
              ClientCache: 'ENABLED',
              CodecSpecification: 'RFC_4281',
              DirectoryStructure: 'SINGLE_DIRECTORY',
              DiscontinuityTags: 'INSERT',
              HlsId3SegmentTagging: 'DISABLED',
              IFrameOnlyPlaylists: 'DISABLED',
              IncompleteSegmentBehavior: 'AUTO',
              ManifestCompression: 'NONE',
              ManifestDurationFormat: 'FLOATING_POINT',
              OutputSelection: 'MANIFESTS_AND_SEGMENTS',
              ProgramDateTime: 'INCLUDE',
              ProgramDateTimeClock: 'SYSTEM_CLOCK',
              ProgramDateTimePeriod: 600,
              RedundantManifest: 'DISABLED',
              SegmentationMode: 'USE_SEGMENT_DURATION',
              SegmentsPerSubdirectory: 10_000,
              StreamInfResolution: 'INCLUDE',
              TimedMetadataId3Frame: 'PRIV',
              TimedMetadataId3Period: 10,
              TsFileMode: 'SEGMENTED_FILES',
            }),
          },
        })],
      }),
    });
  });

  test('OutputGroupConfiguration.rtmp(...) emits only these group-settings defaults', () => {
    new Channel(stack, 'RtmpGroupGolden', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          outputs: [{
            encodes: [video()],
            outputName: 'out',
            destinations: [RtmpDestination.url('rtmp://203.0.113.100/live', 'key')],
          }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          OutputGroupSettings: {
            RtmpGroupSettings: Match.objectEquals({
              AuthenticationScheme: 'COMMON',
              RestartDelay: 1,
            }),
          },
        })],
      }),
    });
  });
});

describe('VideoCodecSettings', () => {
  test('H.264 with defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'video',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264(),
    });
    const audio = EncodeConfiguration.audio({ name: 'audio', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'video_audio_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: {
        VideoDescriptions: [Match.objectLike({
          Name: 'video',
          Width: 1920,
          Height: 1080,
          CodecSettings: {
            // Golden default set for `h264()` with no config. Exact match: any field NOT listed
            // here is deliberately left unset so the MediaLive service default applies.
            H264Settings: Match.objectEquals({
              Profile: 'MAIN',
              GopSize: 1,
              GopSizeUnits: 'SECONDS',
              AdaptiveQuantization: 'AUTO',
              FramerateControl: 'INITIALIZE_FROM_SOURCE',
              ParControl: 'INITIALIZE_FROM_SOURCE',
              SceneChangeDetect: 'ENABLED',
              SpatialAq: 'ENABLED',
              TemporalAq: 'ENABLED',
              AfdSignaling: 'NONE',
              ScanType: 'PROGRESSIVE',
              FlickerAq: 'ENABLED',
              Level: 'H264_LEVEL_AUTO',
              LookAheadRateControl: 'HIGH',
              Syntax: 'DEFAULT',
            }),
          },
        })],
      },
    });
  });

  test('H.264 with custom settings', () => {
    const video = EncodeConfiguration.video({
      name: 'custom-h264',
      width: 1280,
      height: 720,
      codec: VideoCodecSettings.h264({
        rateControl: H264RateControl.cbr({ bitrate: Bitrate.mbps(3) }),
        profile: H264Profile.MAIN,
        gopSize: GopSize.frames(60),
        gopNumBFrames: 2,
        framerate: Framerate.FPS_29_97,
      }),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: {
        VideoDescriptions: [Match.objectLike({
          Name: 'custom-h264',
          CodecSettings: {
            H264Settings: Match.objectLike({
              Bitrate: 3_000_000,
              RateControlMode: 'CBR',
              Profile: 'MAIN',
              GopSize: 60,
              GopSizeUnits: 'FRAMES',
              GopNumBFrames: 2,
              FramerateControl: 'SPECIFIED',
              FramerateNumerator: 30000,
              FramerateDenominator: 1001,
            }),
          },
        })],
      },
    });
  });

  test('ScalingBehavior.SMART_CROP renders on video description', () => {
    const video = EncodeConfiguration.video({
      name: 'vertical_crop',
      width: 1080,
      height: 1920,
      scalingBehavior: ScalingBehavior.SMART_CROP,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'audio', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'SmartCropChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: [Match.objectLike({
          Name: 'vertical_crop',
          Width: 1080,
          Height: 1920,
          ScalingBehavior: 'SMART_CROP',
        })],
      }),
    });
  });

  test('H.265 with defaults', () => {
    const video = EncodeConfiguration.video({
      name: 'video',
      width: 3840,
      height: 2160,
      codec: VideoCodecSettings.h265({
        framerate: Framerate.FPS_29_97,
      }),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: {
        VideoDescriptions: [Match.objectLike({
          Name: 'video',
          CodecSettings: {
            H265Settings: Match.objectLike({
              Profile: 'MAIN',
              Tier: 'MAIN',
              GopSize: 1,
              GopSizeUnits: 'SECONDS',
              SceneChangeDetect: 'ENABLED',
            }),
          },
        })],
      },
    });
  });

  test('H.265 with custom settings', () => {
    const video = EncodeConfiguration.video({
      name: 'video',
      width: 3840,
      height: 2160,
      codec: VideoCodecSettings.h265({
        rateControl: H265RateControl.vbr({ bitrate: Bitrate.mbps(15), maxBitrate: Bitrate.mbps(15) }),
        profile: H265Profile.MAIN_10BIT,
        tier: H265Tier.HIGH,
        gopSize: GopSize.frames(90),
        framerate: Framerate.FPS_59_94,
      }),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: {
        VideoDescriptions: [Match.objectLike({
          CodecSettings: {
            H265Settings: Match.objectLike({
              Bitrate: 15_000_000,
              RateControlMode: 'VBR',
              Profile: 'MAIN_10BIT',
              Tier: 'HIGH',
              GopSize: 90,
              GopSizeUnits: 'FRAMES',
              FramerateNumerator: 60000,
              FramerateDenominator: 1001,
            }),
          },
        })],
      },
    });
  });
});

describe('Video dimension validation', () => {
  test.each([1281, 721])('fails for odd dimension %d', (dim) => {
    expect(() => EncodeConfiguration.video({ name: 'v', width: dim, height: 720, codec: VideoCodecSettings.h264() })).toThrow(/even number/);
    expect(() => EncodeConfiguration.video({ name: 'v', width: 1280, height: dim, codec: VideoCodecSettings.h264() })).toThrow(/even number/);
  });

  test('does not validate tokenized width or height', () => {
    // Odd numeric values would normally throw; tokens must skip the even-number check.
    expect(() => EncodeConfiguration.video({
      name: 'v',
      width: Lazy.number({ produce: () => 1281 }),
      height: Lazy.number({ produce: () => 721 }),
      codec: VideoCodecSettings.h264(),
    })).not.toThrow();
  });
});

describe('Explicit encode names', () => {
  test('video encode uses explicit name', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    expect(video.name).toBe('video');
  });

  test('audio encode uses explicit name', () => {
    const audio1 = EncodeConfiguration.audio({ name: 'audio1', codec: AudioCodecSettings.aac() });
    const audio2 = EncodeConfiguration.audio({ name: 'audio2', codec: AudioCodecSettings.aac() });
    expect(audio1.name).toBe('audio1');
    expect(audio2.name).toBe('audio2');
    expect(audio1.name).not.toBe(audio2.name);
  });

  test('explicit name is set correctly', () => {
    const video = EncodeConfiguration.video({ name: 'my-hd', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'my-audio', codec: AudioCodecSettings.aac() });
    expect(video.name).toBe('my-hd');
    expect(audio.name).toBe('my-audio');
  });
});

describe('Full channel snapshot', () => {
  test('end-to-end channel with all settings produces correct CloudFormation', () => {
    const input = new Input(stack, 'LiveInput', {
      inputName: 'live-encoder',
      input: InputConfiguration.srtCaller([
        {
          srtListenerAddress: '203.0.113.100',
          srtListenerPort: 9000,
          minimumLatency: Duration.millis(1000),
        },
        {
          srtListenerAddress: '203.0.113.101',
          srtListenerPort: 9000,
          minimumLatency: Duration.millis(1000),
        },
      ]),
    });

    const hd = EncodeConfiguration.video({
      name: 'hd-1080p',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({
        rateControl: H264RateControl.qvbr({ maxBitrate: Bitrate.mbps(8), qvbrQualityLevel: 8 }),
        profile: H264Profile.HIGH,
        gopSize: GopSize.seconds(2),
        gopNumBFrames: 3,
        framerate: Framerate.FPS_29_97,
      }),
    });

    const sd = EncodeConfiguration.video({
      name: 'sd-720p',
      width: 1280,
      height: 720,
      codec: VideoCodecSettings.h264({
        rateControl: H264RateControl.qvbr({ maxBitrate: Bitrate.mbps(3), qvbrQualityLevel: 7 }),
        profile: H264Profile.MAIN,
        framerate: Framerate.FPS_29_97,
      }),
    });

    const uhd = EncodeConfiguration.video({
      name: 'uhd-4k',
      width: 3840,
      height: 2160,
      codec: VideoCodecSettings.h265({
        rateControl: H265RateControl.vbr({ bitrate: Bitrate.mbps(15), maxBitrate: Bitrate.mbps(15) }),
        profile: H265Profile.MAIN_10BIT,
        tier: H265Tier.HIGH,
        gopSize: GopSize.frames(90),
        framerate: Framerate.FPS_59_94,
      }),
    });

    const audio = EncodeConfiguration.audio({ name: 'aac-stereo', codec: AudioCodecSettings.aac() });

    const role = new Role(stack, 'ChannelRole', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
    });

    const channel = new Channel(stack, 'LiveChannel', {
      channelName: 'production-live',
      channelClass: ChannelClass.STANDARD,
      logLevel: LogLevel.WARNING,
      role,
      inputs: [{ input, inputAttachmentName: 'primary-feed' }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'mediapackage',
          channel: empChannel,
          outputs: [
            { encode: hd, outputName: 'hd_output' },
            { encode: sd, outputName: 'sd_output' },
            { encode: uhd, outputName: 'uhd_output' },
            { encode: audio, outputName: 'audio_output' },
          ],
        }),
      ],
    });

    // Add HLS archive output group via helper
    channel.addOutputGroup(
      OutputGroupConfiguration.hls({
        name: 'hls-archive',
        destinations: [
          OutputDestination.url('s3ssl://my-bucket2/archive'), OutputDestination.url('s3ssl://my-bucket2/archive2'),
        ],
        outputs: [
          { encodes: [uhd, audio], outputName: 'archive_output' },
        ],
      }),
    );

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::MediaLive::Channel', {
      Name: 'production-live',
      ChannelClass: 'STANDARD',
      LogLevel: 'WARNING',
      RoleArn: { 'Fn::GetAtt': [Match.anyValue(), 'Arn'] },
      Destinations: [
        {
          Id: 'mediapackage',
          MediaPackageSettings: [
            {
              ChannelName: 'emp-channel',
              ChannelGroup: 'emp-group',
              ChannelEndpointId: 'ENDPOINT_1',
            },
            {
              ChannelName: 'emp-channel',
              ChannelGroup: 'emp-group',
              ChannelEndpointId: 'ENDPOINT_2',
            },
          ],
        },
        {
          Id: 'hls-archive',
          Settings: [
            { Url: 's3ssl://my-bucket2/archive' },
            { Url: 's3ssl://my-bucket2/archive2' },
          ],
        },
      ],
      InputAttachments: [
        Match.objectLike({ InputAttachmentName: 'primary-feed' }),
      ],
      EncoderSettings: {
        VideoDescriptions: [
          {
            Name: 'hd-1080p',
            Width: 1920,
            Height: 1080,
            CodecSettings: {
              H264Settings: {
                MaxBitrate: 8_000_000,
                RateControlMode: 'QVBR',
                QvbrQualityLevel: 8,
                Profile: 'HIGH',
                GopSize: 2,
                GopSizeUnits: 'SECONDS',
                GopNumBFrames: 3,
                AdaptiveQuantization: 'AUTO',
                FramerateControl: 'SPECIFIED',
                FramerateNumerator: 30000,
                FramerateDenominator: 1001,
                ParControl: 'SPECIFIED',
                ParNumerator: 1,
                ParDenominator: 1,
                SceneChangeDetect: 'ENABLED',
                SpatialAq: 'ENABLED',
                TemporalAq: 'ENABLED',
              },
            },
          },
          {
            Name: 'sd-720p',
            Width: 1280,
            Height: 720,
            CodecSettings: {
              H264Settings: {
                MaxBitrate: 3_000_000,
                RateControlMode: 'QVBR',
                QvbrQualityLevel: 7,
                Profile: 'MAIN',
                GopSize: 1,
                GopSizeUnits: 'SECONDS',
                AdaptiveQuantization: 'AUTO',
                FramerateControl: 'SPECIFIED',
                FramerateNumerator: 30000,
                FramerateDenominator: 1001,
                ParControl: 'SPECIFIED',
                ParNumerator: 1,
                ParDenominator: 1,
                SceneChangeDetect: 'ENABLED',
                SpatialAq: 'ENABLED',
                TemporalAq: 'ENABLED',
              },
            },
          },
          {
            Name: 'uhd-4k',
            Width: 3840,
            Height: 2160,
            CodecSettings: {
              H265Settings: {
                Bitrate: 15_000_000,
                MaxBitrate: 15_000_000,
                RateControlMode: 'VBR',
                Profile: 'MAIN_10BIT',
                Tier: 'HIGH',
                GopSize: 90,
                GopSizeUnits: 'FRAMES',
                FramerateNumerator: 60000,
                FramerateDenominator: 1001,
                SceneChangeDetect: 'ENABLED',
              },
            },
          },
        ],
        AudioDescriptions: [
          { Name: 'aac-stereo' },
        ],
        OutputGroups: [
          {
            OutputGroupSettings: {
              MediaPackageGroupSettings: {
                Destination: {
                  DestinationRefId: 'mediapackage',
                },
                MediapackageV2GroupSettings: {
                  SegmentLength: 1,
                  SegmentLengthUnits: 'SECONDS',
                  Id3Behavior: 'DISABLED',
                  KlvBehavior: 'NO_PASSTHROUGH',
                  NielsenId3Behavior: 'NO_PASSTHROUGH',
                  Scte35Type: 'SCTE_35_WITHOUT_SEGMENTATION',
                  TimedMetadataId3Frame: 'NONE',
                  TimedMetadataId3Period: 10,
                  TimedMetadataPassthrough: 'DISABLED',
                },
              },
            },
            Outputs: [
              {
                OutputSettings: {
                  MediaPackageOutputSettings: {
                    MediaPackageV2DestinationSettings: {
                      HlsAutoSelect: 'OMIT',
                      HlsDefault: 'OMIT',
                    },
                  },
                },
                VideoDescriptionName: 'hd-1080p',
              },
              {
                OutputSettings: {
                  MediaPackageOutputSettings: {
                    MediaPackageV2DestinationSettings: {
                      HlsAutoSelect: 'OMIT',
                      HlsDefault: 'OMIT',
                    },
                  },
                },
                VideoDescriptionName: 'sd-720p',
              },
              {
                OutputSettings: {
                  MediaPackageOutputSettings: {
                    MediaPackageV2DestinationSettings: {
                      HlsAutoSelect: 'OMIT',
                      HlsDefault: 'OMIT',
                    },
                  },
                },
                VideoDescriptionName: 'uhd-4k',
              },
              {
                OutputSettings: {
                  MediaPackageOutputSettings: {
                    MediaPackageV2DestinationSettings: {
                      HlsAutoSelect: 'OMIT',
                      HlsDefault: 'OMIT',
                    },
                  },
                },
                AudioDescriptionNames: ['aac-stereo'],
              },
            ],
          },
          {
            OutputGroupSettings: {
              HlsGroupSettings: {
                Destination: {
                  DestinationRefId: 'hls-archive',
                },
              },
            },
            Outputs: [
              {
                OutputSettings: {
                  HlsOutputSettings: {
                    HlsSettings: {
                      StandardHlsSettings: {
                        M3u8Settings: {},
                      },
                    },
                  },
                },
                VideoDescriptionName: 'uhd-4k',
                AudioDescriptionNames: ['aac-stereo'],
              },
            ],
          },
        ],
      },
    });
  });
});

describe('Codec validation', () => {
  test('fails when a modelled video codec is not supported by the output group', () => {
    // RTMP output groups support only H.264 video; an H.265 encode must be rejected at synth.
    const h265 = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h265({ framerate: Framerate.FPS_29_97 }),
    });

    new Channel(stack, 'BadCodecChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          outputs: [{
            encodes: [h265],
            outputName: 'out',
            destinations: [RtmpDestination.url('rtmp://203.0.113.100/live', 'key')],
          }],
        }),
      ],
    });

    // Codec validation runs at synth time (Output._bind), so assert on synthesis.
    expect(() => Template.fromStack(stack)).toThrow("does not support video codec 'H265'. Supported: H264.");
  });

  test('fails when a modelled audio codec is not supported by the output group', () => {
    // RTMP output groups support only AAC audio; an AC3 encode must be rejected at synth.
    const video = EncodeConfiguration.video({
      name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264(),
    });
    const ac3 = EncodeConfiguration.audio({ name: 'ac3', codec: AudioCodecSettings.ac3() });

    new Channel(stack, 'BadAudioChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          outputs: [{
            encodes: [video, ac3],
            outputName: 'out',
            destinations: [RtmpDestination.url('rtmp://203.0.113.100/live', 'key')],
          }],
        }),
      ],
    });

    expect(() => Template.fromStack(stack)).toThrow("does not support audio codec 'AC3'. Supported: AAC.");
  });
});

describe('Validation', () => {
  test('MediaPackage V2 output group accepts single track per output', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'GoodChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'mediapackage',
          channel: empChannel,
          outputs: [
            { encode: hd, outputName: 'hd_output' },
            { encode: audio, outputName: 'audio_output' },
          ],
        }),
      ],
    });

    Template.fromStack(stack);
  });

  test('STANDARD channel requires exactly 2 primary MediaPackage V2 destinations (custom)', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const stdInput2 = new Input(stack, 'StdInput2', {
      inputName: 'std-input-2',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: stdInput2 }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2PerPipeline({
            name: 'emp',
            destinations: [MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_1)],
            outputs: [{ encode: hd, outputName: 'hd_output' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/exactly 2 primary destination/);
  });

  test('SINGLE_PIPELINE channel rejects more than 1 additional MediaPackage V2 destination', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    expect(() => {
      new Channel(stack, 'BadChannel', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            additionalDestinations: [
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_2),
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_1),
            ],
            outputs: [{ encode: hd, outputName: 'hd_output' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/at most 1 additional destination/);
  });

  test('STANDARD HLS channel requires exactly 2 destinations', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const stdInput3 = new Input(stack, 'StdInput3', {
      inputName: 'std-input-3',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: stdInput3 }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/exactly 2 primary destination/);
  });

  test('epoch locking with system clock timing source throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => {
      new Channel(stack, 'EpochChannel', {
        inputs: [{ input: defaultInput }],
        globalConfiguration: {
          outputLocking: OutputLocking.epoch(),
          outputTimingSource: OutputTimingSource.SYSTEM_CLOCK,
        },
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
    }).toThrow(/must be INPUT_CLOCK when using epoch output locking/);
  });

  test('epoch locking with input clock timing source is accepted', () => {
    // Epoch locking requires an explicit frame rate on H.264 encodes.
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }) });

    new Channel(stack, 'EpochOkChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        outputLocking: OutputLocking.epoch(),
        outputTimingSource: OutputTimingSource.INPUT_CLOCK,
      },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({
          OutputLockingMode: 'EPOCH_LOCKING',
          OutputLockingSettings: { EpochLockingSettings: {} },
          OutputTimingSource: 'INPUT_CLOCK',
        }),
        // Under epoch locking, the HLS program-date-time clock auto-corrects from the SYSTEM_CLOCK
        // default to INITIALIZE_FROM_OUTPUT_TIMECODE (the service rejects SYSTEM_CLOCK).
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                ProgramDateTimeClock: 'INITIALIZE_FROM_OUTPUT_TIMECODE',
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('epoch locking with an explicit HLS SYSTEM_CLOCK program-date-time clock throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => {
      new Channel(stack, 'EpochPdtChannel', {
        inputs: [{ input: defaultInput }],
        globalConfiguration: {
          outputLocking: OutputLocking.epoch(),
          outputTimingSource: OutputTimingSource.INPUT_CLOCK,
        },
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            programDateTimeClock: HlsProgramDateTimeClock.SYSTEM_CLOCK,
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/programDateTimeClock must be INITIALIZE_FROM_OUTPUT_TIMECODE when using epoch output locking/);
  });

  test('epoch locking with an H.264 encode that follows the source frame rate throws', () => {
    // No framerate on the H.264 encode => framerateControl INITIALIZE_FROM_SOURCE, which epoch
    // locking rejects at deploy.
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => {
      new Channel(stack, 'EpochFramerateChannel', {
        inputs: [{ input: defaultInput }],
        globalConfiguration: {
          outputLocking: OutputLocking.epoch(),
          outputTimingSource: OutputTimingSource.INPUT_CLOCK,
        },
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/epoch output locking requires an explicit frame rate on H.264 video encodes/);
  });

  test('epoch locking with an H.264 encode that has an explicit frame rate is accepted', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    new Channel(stack, 'EpochFramerateOkChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        outputLocking: OutputLocking.epoch(),
        outputTimingSource: OutputTimingSource.INPUT_CLOCK,
      },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // A fractional but explicit rate (29.97) is fine — epoch locking only rejects source-derived rates.
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({ OutputLockingMode: 'EPOCH_LOCKING' }),
      }),
    });
  });

  // Epoch-locking checks are deferred to synth so they also cover groups added post-construction
  // via addOutputGroup(), not just those passed in props.
  function epochChannel(): Channel {
    const video = EncodeConfiguration.video({
      name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
    });
    return new Channel(stack, 'EpochAddChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        outputLocking: OutputLocking.epoch(),
        outputTimingSource: OutputTimingSource.INPUT_CLOCK,
      },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
  }

  test('epoch locking auto-corrects the program-date-time clock on a group added via addOutputGroup()', () => {
    const channel = epochChannel();
    const video = EncodeConfiguration.video({
      name: 'v2', width: 1280, height: 720, codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
    });
    channel.addOutputGroup(OutputGroupConfiguration.hls({
      name: 'hls-added',
      destinations: [OutputDestination.url('s3ssl://bucket/added')],
      outputs: [{ encodes: [video], outputName: 'out2' }],
    }));

    // The later-added group's clock is auto-corrected to INITIALIZE_FROM_OUTPUT_TIMECODE, same as
    // the initial group.
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Name: 'hls-added',
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({ ProgramDateTimeClock: 'INITIALIZE_FROM_OUTPUT_TIMECODE' }),
            },
          }),
        ]),
      }),
    });
  });

  test('epoch locking validates a group added via addOutputGroup() (H.264 without frame rate throws)', () => {
    const channel = epochChannel();
    const video = EncodeConfiguration.video({ name: 'v2', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    channel.addOutputGroup(OutputGroupConfiguration.hls({
      name: 'hls-added',
      destinations: [OutputDestination.url('s3ssl://bucket/added')],
      outputs: [{ encodes: [video], outputName: 'out2' }],
    }));

    expect(() => Template.fromStack(stack)).toThrow(/epoch output locking requires an explicit frame rate on H.264 video encodes/);
  });
});

describe('TimecodeConfig', () => {
  test('custom timecode source and syncThreshold', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      timecodeConfig: {
        source: TimecodeSource.SYSTEMCLOCK,
        syncThreshold: 5,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        TimecodeConfig: {
          Source: 'SYSTEMCLOCK',
          SyncThreshold: 5,
        },
      }),
    });
  });

  test('default timecode uses EMBEDDED', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        TimecodeConfig: Match.objectLike({
          Source: 'EMBEDDED',
        }),
      }),
    });
  });
});

describe('AvailBlanking and AvailSettings', () => {
  test('avail blanking enabled with image URL', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      availBlanking: {
        state: AvailBlankingState.ENABLED,
        image: FileLocation.url('s3://bucket/slate.png'),
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailBlanking: {
          State: 'ENABLED',
          AvailBlankingImage: { Uri: 's3://bucket/slate.png' },
        },
      }),
    });
  });

  test('avail settings with splice insert', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      availSettings: AvailSettings.spliceInsert({ adAvailOffset: 1000 }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailConfiguration: {
          AvailSettings: {
            Scte35SpliceInsert: Match.objectLike({
              AdAvailOffset: 1000,
            }),
          },
        },
      }),
    });
  });

  test('avail settings with splice insert honors the regional blackout and web delivery flags', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      availSettings: AvailSettings.spliceInsert({
        noRegionalBlackoutFlag: Scte35FlagBehavior.IGNORE,
        webDeliveryAllowedFlag: Scte35FlagBehavior.IGNORE,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailConfiguration: {
          AvailSettings: {
            Scte35SpliceInsert: Match.objectLike({
              NoRegionalBlackoutFlag: 'IGNORE',
              WebDeliveryAllowedFlag: 'IGNORE',
            }),
          },
        },
      }),
    });
  });

  test('avail settings with time signal APOS', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      availSettings: AvailSettings.timeSignalApos({
        adAvailOffset: 500,
        noRegionalBlackoutFlag: Scte35FlagBehavior.FOLLOW,
        webDeliveryAllowedFlag: Scte35FlagBehavior.IGNORE,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailConfiguration: {
          AvailSettings: {
            Scte35TimeSignalApos: Match.objectLike({
              AdAvailOffset: 500,
              NoRegionalBlackoutFlag: 'FOLLOW',
              WebDeliveryAllowedFlag: 'IGNORE',
            }),
          },
        },
      }),
    });
  });
});

describe('Nielsen audio watermarking', () => {
  test('CBET and NAES II/NW watermark settings are wired through', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({
      name: 'aac-stereo',
      audioWatermarkSettings: {
        nielsenWatermarks: {
          distributionType: NielsenDistributionType.FINAL_DISTRIBUTOR,
          cbetSettings: {
            cbetCheckDigitString: '12345',
            csid: 'CSID1',
            cbetStepaside: NielsenCbetStepaside.ENABLED,
          },
          naesIiNwSettings: {
            checkDigitString: '67890',
            sid: 123.4,
            timezone: NielsenWatermarkTimezone.US_EASTERN,
          },
        },
      },
      codec: AudioCodecSettings.aac(),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, audio], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: Match.arrayWith([
          Match.objectLike({
            Name: 'aac-stereo',
            AudioWatermarkingSettings: {
              NielsenWatermarksSettings: Match.objectLike({
                NielsenDistributionType: 'FINAL_DISTRIBUTOR',
                NielsenCbetSettings: Match.objectLike({
                  CbetCheckDigitString: '12345',
                  Csid: 'CSID1',
                  CbetStepaside: 'ENABLED',
                }),
                NielsenNaesIiNwSettings: Match.objectLike({
                  CheckDigitString: '67890',
                  Sid: 123.4,
                  Timezone: 'US_EASTERN',
                }),
              }),
            },
          }),
        ]),
      }),
    });
  });
});

describe('Caption encode', () => {
  test('caption encode wired to output', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const caption = EncodeConfiguration.caption({
      name: 'eng-captions',
      captionSelectorName: 'english',
      languageCode: 'eng',
      destination: CaptionDestination.embedded(),
    });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video, caption], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: Match.arrayWith([
          Match.objectLike({
            Name: 'eng-captions',
            CaptionSelectorName: 'english',
            LanguageCode: 'eng',
          }),
        ]),
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                CaptionDescriptionNames: ['eng-captions'],
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('Output group types', () => {
  test('Archive output group', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          rolloverInterval: Duration.seconds(600),
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              ArchiveGroupSettings: Match.objectLike({
                RolloverInterval: 600,
                Destination: { DestinationRefId: 'archive' },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('RTMP output group', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          authenticationScheme: RtmpAuthenticationScheme.COMMON,
          restartDelay: Duration.seconds(15),
          outputs: [{ destinations: [RtmpDestination.url('rtmp://live.example.com/app', 'stream')], encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              RtmpGroupSettings: Match.objectLike({
                AuthenticationScheme: 'COMMON',
                RestartDelay: 15,
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('UDP output group with settings', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          destinations: [UdpOutputDestination.rtp({ address: '239.10.10.10', port: 5001 })],
          inputLossAction: UdpInputLossAction.DROP_TS,
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({ Settings: [Match.objectLike({ Url: 'rtp://239.10.10.10:5001' })] }),
      ]),
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              UdpGroupSettings: Match.objectLike({
                InputLossAction: 'DROP_TS',
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('HLS output group with settings', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          segment: Segment.seconds(6),
          mode: HlsMode.VOD,
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                SegmentLength: 6,
                Mode: 'VOD',
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('an enum-like class of() escape-hatch value passes through to the template', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          // A value not modelled by CDK — the escape hatch must pass it through unchanged.
          mode: HlsMode.of('TEST'),
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                Mode: 'TEST',
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('output group emits its name as the display name', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'my-hls-group',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({ Name: 'my-hls-group' }),
        ]),
      }),
    });
  });
});

describe('CMAF Ingest output group', () => {
  test('accepts a destination URL that ends with a slash', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'audio', codec: AudioCodecSettings.aac() });

    new Channel(stack, 'CmafChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.cmafIngest({
          name: 'cmaf',
          destinations: [OutputDestination.url('https://ingest.example.com/v1/channel/')],
          outputs: [
            { outputName: 'cmaf_video', nameModifier: '_video', encode: video },
            { outputName: 'cmaf_audio', nameModifier: '_audio', encode: audio },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({ OutputGroupSettings: { CmafIngestGroupSettings: Match.anyValue() } }),
        ]),
      }),
    });
  });

  test('fails when a destination URL does not end with a slash', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => new Channel(stack, 'CmafBadChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.cmafIngest({
          name: 'cmaf',
          destinations: [OutputDestination.url('https://ingest.example.com/v1/channel')],
          outputs: [{ outputName: 'cmaf_video', nameModifier: '_video', encode: video }],
        }),
      ],
    })).toThrow(/must end with '\/'/);
  });
});

describe('Per-output settings', () => {
  test('HLS output with nameModifier', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output', nameModifier: '_hd' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  HlsOutputSettings: Match.objectLike({
                    NameModifier: '_hd',
                  }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('Archive output with extension', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [{ encodes: [video], outputName: 'video_output', extension: 'mp4' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  ArchiveOutputSettings: Match.objectLike({
                    Extension: 'mp4',
                  }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('Linked channel settings', () => {
  test('linked channel on STANDARD class throws', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => {
      new Channel(stack, 'MyChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'video_output' }],
          }),
        ],
        linkedChannelSettings: LinkedChannelSettings.primary(),
      });
    }).toThrow(/SINGLE_PIPELINE/);
  });

  test('linked channel primary on SINGLE_PIPELINE', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      channelClass: ChannelClass.SINGLE_PIPELINE,
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      linkedChannelSettings: LinkedChannelSettings.primary(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      LinkedChannelSettings: Match.objectLike({
        PrimaryChannelSettings: {
          LinkedChannelType: 'PRIMARY_CHANNEL',
        },
      }),
    });
  });
});

describe('BlackoutSlate and ColorCorrection', () => {
  test('blackout slate enabled', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      blackoutSlate: {
        state: BlackoutSlateState.ENABLED,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        BlackoutSlate: Match.objectLike({
          State: 'ENABLED',
        }),
      }),
    });
  });

  test('color corrections', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      colorCorrections: [
        { inputColorSpace: ColorSpace.REC_601, outputColorSpace: ColorSpace.REC_709 },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        ColorCorrectionSettings: {
          GlobalColorCorrections: Match.arrayWith([
            Match.objectLike({
              InputColorSpace: 'REC_601',
              OutputColorSpace: 'REC_709',
            }),
          ]),
        },
      }),
    });
  });

  test('color correction with a LUT from a bucket grants read and emits the s3ssl URI', () => {
    const video = EncodeConfiguration.video({ name: 'video', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const lutBucket = new s3.Bucket(stack, 'LutBucket');

    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'video_output' }],
        }),
      ],
      colorCorrections: [
        { inputColorSpace: ColorSpace.REC_601, outputColorSpace: ColorSpace.REC_709, lut: Lut.fromBucket(lutBucket, 'luts/rec709.cube') },
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        ColorCorrectionSettings: {
          GlobalColorCorrections: Match.arrayWith([
            Match.objectLike({
              Uri: { 'Fn::Join': ['', ['s3ssl://', { Ref: Match.stringLikeRegexp('LutBucket') }, '/luts/rec709.cube']] },
            }),
          ]),
        },
      }),
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({ Action: Match.arrayWith(['s3:GetObject*']) }),
        ]),
      }),
    });
  });

  test('Lut.url accepts an s3ssl:// URL', () => {
    expect(() => Lut.url('s3ssl://bucket/luts/rec709.cube')).not.toThrow();
  });

  test('Lut.url rejects a non-S3 URL', () => {
    expect(() => Lut.url('https://example.com/rec709.cube')).toThrow(/must be an s3:\/\/ or s3ssl:\/\/ URL/);
  });
});

describe('NielsenConfiguration and thumbnails', () => {
  function channelWithHlsOutput() {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    return { video };
  }

  test('nielsenConfiguration renders distributorId', () => {
    const { video } = channelWithHlsOutput();

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      nielsenConfiguration: { distributorId: 'ACME123' },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        NielsenConfiguration: Match.objectLike({ DistributorId: 'ACME123' }),
      }),
    });
  });

  test.each([
    NielsenPcmToId3TaggingState.ENABLED,
    NielsenPcmToId3TaggingState.DISABLED,
  ])('nielsenConfiguration renders nielsenPcmToId3Tagging %s', (nielsenPcmToId3Tagging) => {
    const { video } = channelWithHlsOutput();

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      nielsenConfiguration: { nielsenPcmToId3Tagging },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        NielsenConfiguration: Match.objectLike({ NielsenPcmToId3Tagging: nielsenPcmToId3Tagging.value }),
      }),
    });
  });

  test('thumbnailConfiguration explicitly enabled with ThumbnailState.AUTO', () => {
    const { video } = channelWithHlsOutput();

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      thumbnailConfiguration: { state: ThumbnailState.AUTO },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        ThumbnailConfiguration: { State: 'AUTO' },
      }),
    });
  });
});

describe('FileLocation grants', () => {
  function grantsGetObject(template: Template): void {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({ Action: Match.arrayWith(['s3:GetObject*']) }),
        ]),
      }),
    });
  }

  test('avail-blanking image from a bucket grants the channel role read access', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const bucket = new s3.Bucket(stack, 'BlankBucket');
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      availBlanking: { state: AvailBlankingState.ENABLED, image: FileLocation.fromBucket(bucket, 'slate.png') },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
    grantsGetObject(Template.fromStack(stack));
  });

  test('burn-in caption font from a bucket grants the channel role read access', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const fontBucket = new s3.Bucket(stack, 'FontBucket');
    const caption = EncodeConfiguration.caption({
      name: 'eng-burnin',
      captionSelectorName: 'english',
      destination: CaptionDestination.burnIn({
        font: FileLocation.fromBucket(fontBucket, 'fonts/caption-font.ttf'),
      }),
    });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video, caption], outputName: 'out' }],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: Match.arrayWith([
          Match.objectLike({
            DestinationSettings: {
              BurnInDestinationSettings: Match.objectLike({
                // backgroundOpacity defers to the service (blank in the console), so it is not emitted.
                BackgroundOpacity: Match.absent(),
                Font: {
                  Uri: { 'Fn::Join': ['', ['s3ssl://', { Ref: Match.stringLikeRegexp('FontBucket') }, '/fonts/caption-font.ttf']] },
                },
              }),
            },
          }),
        ]),
      }),
    });
    grantsGetObject(template);
  });

  test('audio-only HLS cover-art image from a bucket grants the channel role read access', () => {
    const audio = EncodeConfiguration.audio({ name: 'aac', codec: AudioCodecSettings.aac() });
    const artBucket = new s3.Bucket(stack, 'ArtBucket');
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{
            encodes: [audio],
            outputName: 'audio_only',
            hlsSettings: HlsSettings.audioOnly({
              audioGroupId: 'program',
              audioOnlyImage: FileLocation.fromBucket(artBucket, 'art/cover.png'),
            }),
          }],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  HlsOutputSettings: Match.objectLike({
                    HlsSettings: {
                      AudioOnlyHlsSettings: Match.objectLike({
                        AudioOnlyImage: {
                          Uri: { 'Fn::Join': ['', ['s3ssl://', { Ref: Match.stringLikeRegexp('ArtBucket') }, '/art/cover.png']] },
                        },
                      }),
                    },
                  }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
    grantsGetObject(template);
  });
});

describe('Timecode configuration', () => {
  test('defaults to EMBEDDED', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        TimecodeConfig: { Source: 'EMBEDDED' },
      }),
    });
  });

  test('custom timecode source and sync threshold', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      timecodeConfig: { source: TimecodeSource.ZEROBASED, syncThreshold: 100 },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        TimecodeConfig: { Source: 'ZEROBASED', SyncThreshold: 100 },
      }),
    });
  });
});

describe('Avail blanking and configuration', () => {
  test('avail blanking with image', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      availBlanking: { state: AvailBlankingState.ENABLED, image: FileLocation.url('s3://bucket/slate.png') },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailBlanking: {
          State: 'ENABLED',
          AvailBlankingImage: { Uri: 's3://bucket/slate.png' },
        },
      }),
    });
  });
});

describe('Blackout slate', () => {
  test('blackout slate with network end blackout', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      blackoutSlate: {
        state: BlackoutSlateState.ENABLED,
        image: FileLocation.url('s3://bucket/blackout.png'),
        networkEndBlackout: NetworkEndBlackout.ENABLED,
        networkId: '10.1234/5678-9012-3456-7890-1234-C',
      },
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://b/l')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        BlackoutSlate: {
          State: 'ENABLED',
          BlackoutSlateImage: { Uri: 's3://bucket/blackout.png' },
          NetworkEndBlackout: 'ENABLED',
          NetworkId: '10.1234/5678-9012-3456-7890-1234-C',
        },
      }),
    });
  });
});

describe('Caption descriptions', () => {
  test('caption encode wired to output and channel', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const caption = EncodeConfiguration.caption({
      name: 'eng-captions',
      captionSelectorName: 'embedded',
      languageCode: 'eng',
      languageDescription: 'English',
      destination: CaptionDestination.embedded(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [
            { encode: video, captions: [caption], outputName: 'video_out' },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: [{
          Name: 'eng-captions',
          CaptionSelectorName: 'embedded',
          LanguageCode: 'eng',
          LanguageDescription: 'English',
        }],
      }),
    });
  });
});

describe('MediaPackage V2 in-band captions', () => {
  test('video output with burn-in caption in captions prop', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const burnIn = EncodeConfiguration.caption({
      name: 'burn-in',
      captionSelectorName: 'cc',
      destination: CaptionDestination.burnIn(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: video, captions: [burnIn], outputName: 'hd' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          Outputs: [Match.objectLike({
            VideoDescriptionName: 'v',
            CaptionDescriptionNames: ['burn-in'],
          })],
        })],
      }),
    });
  });

  test('video output with embedded caption in captions prop', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const embedded = EncodeConfiguration.caption({
      name: 'emb',
      captionSelectorName: 'cc',
      destination: CaptionDestination.embedded(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: video, captions: [embedded], outputName: 'hd' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          Outputs: [Match.objectLike({
            VideoDescriptionName: 'v',
            CaptionDescriptionNames: ['emb'],
          })],
        })],
      }),
    });
  });

  test('fails when captions contains an out-of-band type', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const webvtt = EncodeConfiguration.caption({
      name: 'webvtt',
      captionSelectorName: 'cc',
      destination: CaptionDestination.webvtt(),
    });

    expect(() => {
      new Channel(stack, 'Ch', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            outputs: [{ encode: video, captions: [webvtt], outputName: 'hd' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/out-of-band destination/);
  });

  test('fails when captions contains a non-caption encode', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const audio = EncodeConfiguration.audio({
      name: 'a',
      codec: AudioCodecSettings.aac({ bitrate: Bitrate.kbps(192) }),
    });

    expect(() => {
      new Channel(stack, 'Ch', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            outputs: [{ encode: video, captions: [audio], outputName: 'hd' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/caption encodes only/);
  });

  test('fails when more than one embedded caption per output', () => {
    const video = EncodeConfiguration.video({
      name: 'v',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const emb1 = EncodeConfiguration.caption({
      name: 'emb1',
      captionSelectorName: 'cc1',
      destination: CaptionDestination.embedded(),
    });
    const emb2 = EncodeConfiguration.caption({
      name: 'emb2',
      captionSelectorName: 'cc2',
      destination: CaptionDestination.embeddedPlusScte20(),
    });

    expect(() => {
      new Channel(stack, 'Ch', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            outputs: [{ encode: video, captions: [emb1, emb2], outputName: 'hd' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/only one embedded caption/);
  });

  test('fails when caption is the primary encode', () => {
    const webvtt = EncodeConfiguration.caption({
      name: 'webvtt',
      captionSelectorName: 'cc',
      destination: CaptionDestination.webvtt(),
    });

    expect(() => {
      new Channel(stack, 'Ch', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            outputs: [{ encode: webvtt, outputName: 'cap_out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/must be a video or audio encode/);
  });
});

describe('Caption destination variants', () => {
  // Each variant is paired with a caption selector and output group that AWS documents as a
  // valid combination (see the MediaLive captions compatibility tables per output group —
  // https://docs.aws.amazon.com/medialive/latest/ug/supported-formats-ts-output.html and
  // https://docs.aws.amazon.com/medialive/latest/ug/general-information-supported-formats.html).
  // The CDK library does not validate caption-format-vs-output-group compatibility itself (that
  // is a MediaLive service-side check), so these pairings matter for the test to model something
  // that would actually deploy, not just something that synthesizes.
  //
  // Archive/UDP/SRT/MediaConnect Router (MPEG-TS transport-stream groups) support DVB-Sub, ARIB,
  // Embedded+SCTE-20, and Teletext as *outputs* — none of these are valid on HLS/CMAF/MSS/RTMP.
  test.each([
    ['dvbSub', 'embedded-cc', () => CaptionSelector.embedded('embedded-cc'),
      () => CaptionDestination.dvbSub({ fontColor: CaptionFontColor.WHITE }),
      // backgroundOpacity defers to the service (blank in the console), so it is not emitted.
      { DvbSubDestinationSettings: Match.objectLike({ FontColor: 'WHITE', BackgroundOpacity: Match.absent() }) }],
    ['arib', 'arib-cc', () => CaptionSelector.arib('arib-cc'),
      () => CaptionDestination.arib(),
      { AribDestinationSettings: {} }],
    ['embeddedPlusScte20', 'embedded-cc', () => CaptionSelector.embedded('embedded-cc'),
      () => CaptionDestination.embeddedPlusScte20(),
      { EmbeddedPlusScte20DestinationSettings: {} }],
    ['teletext', 'ttx-cc', () => CaptionSelector.teletext('ttx-cc', { pageNumber: '888' }),
      () => CaptionDestination.teletext(),
      { TeletextDestinationSettings: {} }],
  ] as const)('%s caption destination renders on an Archive (transport-stream) output', (_label, selectorName, selectorFactory, destFactory, expected) => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.aac() });
    const caption = EncodeConfiguration.caption({
      name: 'cap',
      captionSelectorName: selectorName,
      languageCode: 'eng',
      destination: destFactory(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput, captionSelectors: [selectorFactory()] }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [{ encodes: [video, audio, caption], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: [Match.objectLike({
          DestinationSettings: expected,
        })],
      }),
    });
  });

  // MS Smooth supports Burn-in, EBU-TT-D, SMPTE-TT, and TTML from an embedded source. Captions
  // must be in their own output, separate from audio/video, regardless of category.
  test.each([
    ['ebuTtD', () => CaptionDestination.ebuTtD({ copyrightHolder: 'Acme Corp' }), { EbuTtDDestinationSettings: Match.objectLike({ CopyrightHolder: 'Acme Corp' }) }],
    ['ttml', () => CaptionDestination.ttml(), { TtmlDestinationSettings: {} }],
  ] as const)('%s caption destination renders on an MS Smooth output', (_label, destFactory, expected) => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.aac() });
    const caption = EncodeConfiguration.caption({
      name: 'cap',
      captionSelectorName: 'embedded-cc',
      languageCode: 'eng',
      destination: destFactory(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput, captionSelectors: [CaptionSelector.embedded('embedded-cc')] }],
      outputGroups: [
        OutputGroupConfiguration.msSmooth({
          name: 'smooth',
          destinations: [OutputDestination.url('https://smooth.example.com/live.isml')],
          outputs: [
            { outputName: 'video-audio', encodes: [video, audio], nameModifier: '_video' },
            { outputName: 'captions', encodes: [caption], nameModifier: '_captions' },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: [Match.objectLike({
          DestinationSettings: expected,
        })],
      }),
    });
  });

  // RTMP CaptionInfo is only valid on an RTMP output.
  test('rtmpCaptionInfo caption destination renders on an RTMP output', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.aac() });
    const caption = EncodeConfiguration.caption({
      name: 'cap',
      captionSelectorName: 'embedded-cc',
      languageCode: 'eng',
      destination: CaptionDestination.rtmpCaptionInfo(),
    });

    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput, captionSelectors: [CaptionSelector.embedded('embedded-cc')] }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          outputs: [{
            destinations: [RtmpDestination.url('rtmp://203.0.113.100/live', 'stream')],
            encodes: [video, audio, caption],
            outputName: 'out',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        CaptionDescriptions: [Match.objectLike({
          DestinationSettings: { RtmpCaptionInfoDestinationSettings: {} },
        })],
      }),
    });
  });
});

describe('Output group types', () => {
  test('archive output group', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          rolloverInterval: Duration.seconds(600),
          outputs: [{ encodes: [video], outputName: 'archive_out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          OutputGroupSettings: {
            ArchiveGroupSettings: {
              Destination: { DestinationRefId: 'archive' },
              RolloverInterval: 600,
            },
          },
        })],
      }),
      Destinations: [Match.objectLike({
        Id: 'archive',
        Settings: [{ Url: 's3ssl://bucket/archive' }],
      })],
    });
  });

  test('rtmp output group', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const audio = EncodeConfiguration.audio({ name: 'a', codec: AudioCodecSettings.aac() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.rtmp({
          name: 'rtmp',
          authenticationScheme: RtmpAuthenticationScheme.AKAMAI,
          restartDelay: Duration.seconds(30),
          outputs: [{ destinations: [RtmpDestination.url('rtmp://live.example.com/app', 'stream')], encodes: [video, audio], outputName: 'rtmp_out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          OutputGroupSettings: {
            RtmpGroupSettings: {
              AuthenticationScheme: 'AKAMAI',
              RestartDelay: 30,
            },
          },
        })],
      }),
    });
  });

  test('udp output group with settings', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          destinations: [UdpOutputDestination.rtp({ address: '239.10.10.10', port: 5001 })],
          buffer: Duration.millis(1000),
          outputs: [{ encodes: [video], outputName: 'udp_out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          Outputs: [Match.objectLike({
            OutputSettings: {
              UdpOutputSettings: Match.objectLike({
                BufferMsec: 1000,
              }),
            },
          })],
        })],
      }),
    });
  });

  test('udp output omits BufferMsec when no buffer is set (defers to service default)', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          destinations: [UdpOutputDestination.rtp({ address: '239.10.10.10', port: 5001 })],
          outputs: [{ encodes: [video], outputName: 'udp_out' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          Outputs: [Match.objectLike({
            OutputSettings: {
              UdpOutputSettings: Match.objectLike({
                BufferMsec: Match.absent(),
              }),
            },
          })],
        })],
      }),
    });
  });

  test('hls output group with segment settings', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          segment: Segment.seconds(6),
          keepSegments: 10,
          indexNSegments: 8,
          mode: HlsMode.LIVE,
          outputs: [{ encodes: [video], outputName: 'hls_out', nameModifier: '_hd' }],
        }),
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: [Match.objectLike({
          OutputGroupSettings: {
            HlsGroupSettings: Match.objectLike({
              SegmentLength: 6,
              KeepSegments: 10,
              IndexNSegments: 8,
              Mode: 'LIVE',
            }),
          },
          Outputs: [Match.objectLike({
            OutputSettings: {
              HlsOutputSettings: Match.objectLike({
                NameModifier: '_hd',
              }),
            },
          })],
        })],
      }),
    });
  });
});

describe('Linked channel settings', () => {
  test('rejects linked channel on STANDARD class', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    expect(() => {
      new Channel(stack, 'Ch', {
        channelClass: ChannelClass.STANDARD,
        linkedChannelSettings: LinkedChannelSettings.primary(),
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://b/l')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
    }).toThrow(/SINGLE_PIPELINE/);
  });
});

describe('Additional destinations validation', () => {
  test('SINGLE_PIPELINE MediaPackage V2 allows 1 additional destination', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    expect(() => {
      new Channel(stack, 'GoodChannel', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            additionalDestinations: [
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_2),
            ],
            outputs: [{ encode: hd, outputName: 'hd_output' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).not.toThrow();
  });

  test('STANDARD MediaPackage V2 allows up to 2 additional destinations', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const standardInput = new Input(stack, 'StandardInput', {
      inputName: 'standard-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'GoodChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            additionalDestinations: [
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_1),
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_2),
            ],
            outputs: [{ encode: hd, outputName: 'hd_output' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).not.toThrow();
  });

  test('STANDARD MediaPackage V2 rejects more than 2 additional destinations', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    const standardInput = new Input(stack, 'StandardInput2', {
      inputName: 'standard-input-2',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.mediaPackageV2({
            name: 'emp',
            channel: empChannel,
            additionalDestinations: [
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_1),
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_2),
              MediaPackageV2Destination.channel(empChannel, MediaPackageV2EndpointId.ENDPOINT_1),
            ],
            outputs: [{ encode: hd, outputName: 'hd_output' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/at most 2 additional destination/);
  });
});

describe('SRT/RTMP STANDARD destination count validation', () => {
  test('SRT output with STANDARD channel requires two destinations (one per pipeline)', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'SrtStdInput', {
      inputName: 'srt-std-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadSrtChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.srt({
            name: 'srt',
            outputs: [{
              encodes: [video],
              outputName: 'out',
              destinations: [SrtDestination.caller({ address: '203.0.113.100', port: 5000, encryptionPassphraseSecret: srtSecret })],
            }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/requires exactly 2 destination.*STANDARD/);
  });

  test('SRT output on STANDARD channel binds two destinations (A and B) under the output name', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'SrtAbInput', {
      inputName: 'srt-ab-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    new Channel(stack, 'SrtAbChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'srt_out',
            destinations: [
              SrtDestination.caller({ address: '203.0.113.10', port: 5000, encryptionPassphraseSecret: srtSecret }),
              SrtDestination.callerUrl('srt://203.0.113.11:5000', { encryptionPassphraseSecret: srtSecret }),
            ],
          }],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    // One channel destination keyed by the output name, carrying both pipeline endpoints A and B.
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'srt-out',
          SrtSettings: [
            Match.objectLike({ Url: 'srt://203.0.113.10:5000', ConnectionMode: 'CALLER' }),
            Match.objectLike({ Url: 'srt://203.0.113.11:5000', ConnectionMode: 'CALLER' }),
          ],
        }),
      ]),
    });
  });

  test('RTMP output with STANDARD channel requires two destinations (one per pipeline)', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'RtmpStdInput', {
      inputName: 'rtmp-std-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadRtmpChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.rtmp({
            name: 'rtmp',
            outputs: [{
              encodes: [video],
              outputName: 'out',
              destinations: [RtmpDestination.url('rtmp://203.0.113.100/live', 'key')],
            }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/requires exactly 2 destination.*STANDARD/);
  });
});

describe('SRT output settings', () => {
  test.each([
    SrtInputLossAction.DROP_TS,
    SrtInputLossAction.DROP_PROGRAM,
    SrtInputLossAction.EMIT_PROGRAM,
  ])('renders SRT group inputLossAction %s', (inputLossAction) => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'SrtLossChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          inputLossAction,
          outputs: [{
            encodes: [video],
            outputName: 'out',
            destinations: [SrtDestination.caller({ address: '203.0.113.100', port: 5000, encryptionPassphraseSecret: srtSecret })],
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              SrtGroupSettings: { InputLossAction: inputLossAction.value },
            },
          }),
        ]),
      }),
    });
  });

  test.each([
    SrtEncryptionType.AES128,
    SrtEncryptionType.AES192,
    SrtEncryptionType.AES256,
  ])('renders SRT output encryptionType %s', (encryptionType) => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'SrtEncChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'out',
            destinations: [SrtDestination.caller({ address: '203.0.113.100', port: 5000, encryptionPassphraseSecret: srtSecret })],
            encryptionType,
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  SrtOutputSettings: Match.objectLike({ EncryptionType: encryptionType.value }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('renders SRT output buffer and latency', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'SrtBufChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'out',
            destinations: [SrtDestination.caller({ address: '203.0.113.100', port: 5000, encryptionPassphraseSecret: srtSecret })],
            buffer: Duration.millis(2000),
            latency: Duration.millis(1500),
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  SrtOutputSettings: Match.objectLike({ BufferMsec: 2000, Latency: 1500 }),
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('Input pipeline class validation', () => {
  test('SINGLE_PIPELINE input on STANDARD channel throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

    expect(() => {
      new Channel(stack, 'BadChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [
              OutputDestination.url('s3ssl://bucket/p0'),
              OutputDestination.url('s3ssl://bucket/p1'),
            ],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/incompatible.*STANDARD/);
  });

  test('STANDARD input on SINGLE_PIPELINE channel throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'StdInput', {
      inputName: 'std-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'BadChannel2', {
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/incompatible.*SINGLE_PIPELINE/);
  });

  test('matching input and channel class succeeds', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'MatchInput', {
      inputName: 'match-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    expect(() => {
      new Channel(stack, 'GoodChannel', {
        channelClass: ChannelClass.STANDARD,
        inputs: [{ input: standardInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [
              OutputDestination.url('s3ssl://bucket/p0'),
              OutputDestination.url('s3ssl://bucket/p1'),
            ],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
    }).not.toThrow();
  });

  test('an input added via addInput() is validated at synth (pipeline class mismatch throws)', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'AddMatchInput', {
      inputName: 'add-match-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5000 },
      ]),
    });

    const channel = new Channel(stack, 'AddInputChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/p0'), OutputDestination.url('s3ssl://bucket/p1')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // defaultInput is SINGLE_PIPELINE — incompatible with the STANDARD channel. The deferred
    // validation must catch it even though it was attached after construction.
    channel.addInput({ input: defaultInput, inputAttachmentName: 'late' });

    expect(() => Template.fromStack(stack)).toThrow(/incompatible.*STANDARD/);
  });
});

describe('Anywhere-only input type validation', () => {
  test('SDI input on cloud channel throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const sdiInput = new Input(stack, 'SdiInput', {
      inputName: 'sdi-input',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.sdi([
        new SdiSource(stack, 'Sdi', { sdiSourceName: 'cam-1', type: SdiType.SINGLE }),
      ]),
    });

    expect(() => {
      new Channel(stack, 'Ch', {
        inputs: [{ input: sdiInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/requires anywhereSettings/);
  });

  test('multicast input on cloud channel throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const multicastInput = new Input(stack, 'McInput', {
      input: InputConfiguration.multicast({ sources: [{ address: '239.0.0.1', port: 5000 }] }),
    });

    expect(() => {
      new Channel(stack, 'McCh', {
        inputs: [{ input: multicastInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/requires anywhereSettings/);
  });

  test('SMPTE 2110 input on cloud channel throws', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const smpteInput = new Input(stack, 'SmpteInput', {
      input: InputConfiguration.smpte2110ReceiverGroup({
        videoSdp: { sdpUrl: 'https://example.com/video.sdp' },
      }),
    });

    expect(() => {
      new Channel(stack, 'SmpteCh', {
        inputs: [{ input: smpteInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/requires anywhereSettings/);
  });

  test('SDI input on Anywhere channel does not throw', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });
    const sdiInput = new Input(stack, 'SdiInput2', {
      inputName: 'sdi-input-2',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.sdi([new SdiSource(stack, 'Sdi2', { sdiSourceName: 'cam-2', type: SdiType.SINGLE })]),
    });
    const cluster = Cluster.fromClusterArn(stack, 'Cluster', 'arn:aws:medialive:us-east-1:123456789012:cluster:cluster-123');

    expect(() => {
      new Channel(stack, 'Ch2', {
        inputs: [{ input: sdiInput }],
        anywhereSettings: { cluster },
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('s3ssl://bucket/live')],
            outputs: [{ encodes: [video], outputName: 'out' }],
          }),
        ],
      });
    }).not.toThrow();
  });
});

describe('S3 destination and source helpers', () => {
  test('OutputDestination.fromBucket builds correct s3ssl URL', () => {
    const bucket = new s3.Bucket(stack, 'Bucket');
    const dest = OutputDestination.toBucket(bucket, 'live/');
    const bound = dest._bind();
    expect(bound.url).toMatch(/^s3ssl:\/\/.*\/live\/$/);
  });

  test('InputSource.fromBucket builds correct s3ssl URL', () => {
    const bucket = new s3.Bucket(stack, 'SrcBucket');
    const source = InputSource.fromBucket(bucket, 'videos/test.mp4');
    expect(source._bind().url).toMatch(/^s3ssl:\/\/.*\/videos\/test\.mp4$/);
  });

  test('OutputDestination.fromBucket auto-grants write to channel role', () => {
    const bucket = new s3.Bucket(stack, 'OutputBucket');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'S3Channel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.toBucket(bucket, 'live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // The auto-created role should have S3 write permissions on the bucket
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:PutObject', 's3:Abort*']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('InputSource.fromBucket auto-grants read to channel role', () => {
    const bucket = new s3.Bucket(stack, 'SourceBucket');
    const fileInput = new Input(stack, 'FileInput', {
      inputName: 'file-input',
      input: InputConfiguration.mp4File([
        InputSource.fromBucket(bucket, 'videos/test.mp4'),
      ]),
    });
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'S3Channel', {
      inputs: [{ input: fileInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://other-bucket/live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // grantRead produces GetObject*, GetBucket*, List* — no PutObject*
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:GetObject*', 's3:GetBucket*', 's3:List*']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('URL pull input with SSM password param auto-grants read to channel role', () => {
    const password = new StringParameter(stack, 'PullPassword', {
      parameterName: '/medialive/pull-password',
      stringValue: 'placeholder',
    });
    const fileInput = new Input(stack, 'UrlInput', {
      inputName: 'url-with-creds',
      input: InputConfiguration.urlPull([
        InputSource.url('https://example.com/stream.m3u8', { username: 'user', password }),
      ]),
    });
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    new Channel(stack, 'PwChannel', {
      inputs: [{ input: fileInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // MediaLive reads the password from SSM at runtime — scoped to the parameter, not
    // the broad AmazonSSMReadOnlyAccess managed policy.
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith(['ssm:GetParameters', 'ssm:GetParameter']),
          }),
        ]),
      },
    });
  });

  test('addOutputGroup auto-grants write to the channel role', () => {
    const bucket = new s3.Bucket(stack, 'AddedOutputBucket');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    const channel = new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'initial',
          destinations: [OutputDestination.url('s3ssl://other-bucket/live')],
          outputs: [{ encodes: [video], outputName: 'initial_out' }],
        }),
      ],
    });

    // Output group added AFTER construction must still grant write to its S3 destination.
    channel.addOutputGroup(
      OutputGroupConfiguration.hls({
        name: 'added',
        destinations: [OutputDestination.toBucket(bucket, 'live/stream')],
        outputs: [{ encodes: [video], outputName: 'added_out' }],
      }),
    );

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:PutObject', 's3:Abort*']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('addInput auto-grants read to the channel role', () => {
    const bucket = new s3.Bucket(stack, 'AddedSourceBucket');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    const channel = new Channel(stack, 'Ch', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://other-bucket/live')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    const fileInput = new Input(stack, 'AddedFileInput', {
      inputName: 'added-file-input',
      input: InputConfiguration.mp4File([InputSource.fromBucket(bucket, 'videos/test.mp4')]),
    });

    // Input added AFTER construction must still grant read to its S3 source.
    channel.addInput({ input: fileInput, inputAttachmentName: 'added' });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:GetObject*', 's3:GetBucket*', 's3:List*']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('channel auto-creates role when none provided', () => {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    const channel = new Channel(stack, 'AutoRoleChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    expect(channel.role).toBeDefined();
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [Match.objectLike({
          Principal: { Service: 'medialive.amazonaws.com' },
          // Confused-deputy conditions on the auto-created channel role.
          Condition: {
            StringEquals: { 'aws:SourceAccount': { Ref: 'AWS::AccountId' } },
            ArnLike: {
              'aws:SourceArn': {
                'Fn::Join': ['', [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':medialive:us-east-1:123456789012:channel:*',
                ]],
              },
            },
          },
        })],
      },
    });
  });
});

describe('Bring-your-own role: no automatic grants', () => {
  function userRole(): Role {
    return new Role(stack, 'UserRole', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
    });
  }

  test('a user-provided role receives no auto-grants, even with S3 destinations, S3 input sources, and logging', () => {
    const role = userRole();
    const bucket = new s3.Bucket(stack, 'OutputBucket');
    const sourceBucket = new s3.Bucket(stack, 'SourceBucket');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const fileInput = new Input(stack, 'FileInput', {
      inputName: 'file-input',
      input: InputConfiguration.mp4File([
        InputSource.fromBucket(sourceBucket, 'videos/test.mp4'),
      ]),
    });

    new Channel(stack, 'ByoChannel', {
      role,
      logLevel: LogLevel.INFO,
      inputs: [{ input: fileInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.toBucket(bucket, 'live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    // The user owns the role, so the channel attaches no inline policy to it at all — no S3
    // read/write, no CloudWatch Logs, nothing.
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  });

  test('addInput and addOutputGroup add no grants when the role is user-provided', () => {
    const role = userRole();
    const outBucket = new s3.Bucket(stack, 'AddedOutputBucket');
    const srcBucket = new s3.Bucket(stack, 'AddedSourceBucket');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });

    const channel = new Channel(stack, 'ByoChannel', {
      role,
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });

    const addedInput = new Input(stack, 'AddedInput', {
      inputName: 'added-input',
      input: InputConfiguration.mp4File([InputSource.fromBucket(srcBucket, 'videos/added.mp4')]),
    });
    channel.addInput({ input: addedInput, inputAttachmentName: 'added' });
    channel.addOutputGroup(
      OutputGroupConfiguration.hls({
        name: 'hls-2',
        destinations: [OutputDestination.toBucket(outBucket, 'live/added')],
        outputs: [{ encodes: [video], outputName: 'added-out' }],
      }),
    );

    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  });
});

describe('Auto-grant: service-role permissions', () => {
  function minimalChannel(props: { logLevel?: LogLevel; thumbnailState?: ThumbnailState; vpc?: VpcOutputSettings } = {}) {
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    return new Channel(stack, 'SvcChannel', {
      inputs: [{ input: defaultInput }],
      logLevel: props.logLevel,
      thumbnailConfiguration: props.thumbnailState ? { state: props.thumbnailState } : undefined,
      vpc: props.vpc,
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live/stream')],
          outputs: [{ encodes: [video], outputName: 'out' }],
        }),
      ],
    });
  }

  test('thumbnails are on by default — grants s3:PutObject on *', () => {
    minimalChannel();

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 's3:PutObject',
            Effect: 'Allow',
            Resource: '*',
          }),
        ]),
      },
    });
  });

  test('thumbnails explicitly disabled — no permissions are granted', () => {
    minimalChannel({ thumbnailState: ThumbnailState.DISABLED });

    // No buckets, secrets, logs, or VPC configured and thumbnails off → no inline policy at all.
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  });

  test('channel logging grants CloudWatch Logs write', () => {
    minimalChannel({ logLevel: LogLevel.ERROR });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'logs:CreateLogGroup',
              'logs:CreateLogStream',
              'logs:PutLogEvents',
            ]),
          }),
        ]),
      },
    });
  });

  test('VPC output grants scoped ENI create/delete plus wildcard describe', () => {
    const vpc = new ec2.Vpc(stack, 'OutputVpc');
    const sg = new ec2.SecurityGroup(stack, 'OutputSg', { vpc });
    minimalChannel({
      vpc: { subnets: vpc.privateSubnets, securityGroups: [sg] },
    });

    const template = Template.fromStack(stack);
    // ENI create/delete — scoped (resource is not a bare '*').
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'ec2:CreateNetworkInterface',
              'ec2:CreateNetworkInterfacePermission',
              'ec2:DeleteNetworkInterface',
            ]),
          }),
        ]),
      },
    });
    // Describe* — EC2 forces wildcard resource.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'ec2:DescribeNetworkInterfaces',
              'ec2:DescribeSubnets',
              'ec2:DescribeSecurityGroups',
            ]),
            Resource: '*',
          }),
        ]),
      },
    });
  });

  test('public address allocations grant ec2:AssociateAddress', () => {
    const vpc = new ec2.Vpc(stack, 'OutputVpc');
    minimalChannel({
      vpc: { subnets: vpc.privateSubnets, publicAddressAllocationIds: ['eipalloc-123', 'eipalloc-456'] },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: 'ec2:AssociateAddress',
            Resource: '*',
          }),
        ]),
      },
    });
  });
});

describe('Auto-grant: MediaPackage V2 ingest', () => {
  test('MediaPackageV2Destination.channel auto-grants ingest to channel role', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    new Channel(stack, 'MpChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: hd, outputName: 'hd_output' }],
        }),
      ],
    });

    // The MediaPackage V2 ingest grant must land on the channel role. Assert the actual actions
    // (not just that synth succeeds) so a change to the ingest action set is caught here.
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: ['mediapackagev2:GetChannel', 'mediapackagev2:PutObject'],
          }),
        ]),
      },
    });
  });
});

describe('Auto-grant: SRT encryption secret', () => {
  test('SrtDestination.caller with encryption secret auto-grants read', () => {
    const secret = srtSecret;
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    const standardInput = new Input(stack, 'SrtInput2', {
      inputName: 'srt-input-2',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
      ]),
    });

    new Channel(stack, 'SrtChannel', {
      inputs: [{ input: standardInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'out',
            destinations: [SrtDestination.caller({
              address: '203.0.113.100',
              port: 5000,
              encryptionPassphraseSecret: secret,
            })],
          }],
        }),
      ],
    });

    // The auto-created role should have Secrets Manager read permissions
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });
});

describe('SrtDestination', () => {
  test('caller and listener destinations carry the encryption passphrase and grant read', () => {
    const secret = new secretsmanager.Secret(stack, 'Passphrase');
    const sg = new InputSecurityGroup(stack, 'ChannelSg', { allowlistRules: ['203.0.113.0/24'] });
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'SrtChannel', {
      inputs: [{ input: defaultInput }],
      // Listener output below requires channel security groups.
      channelSecurityGroups: [sg],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [
            {
              encodes: [video],
              outputName: 'srt_caller',
              destinations: [SrtDestination.caller({ address: '203.0.113.20', port: 5001, encryptionPassphraseSecret: secret })],
            },
            {
              encodes: [video],
              outputName: 'srt_listener',
              destinations: [SrtDestination.listener({ listenerPort: 5000, encryptionPassphraseSecret: secret })],
            },
          ],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'srt-caller',
          SrtSettings: [Match.objectLike({
            ConnectionMode: 'CALLER',
            EncryptionPassphraseSecretArn: { Ref: Match.stringLikeRegexp('Passphrase') },
          })],
        }),
        Match.objectLike({
          Id: 'srt-listener',
          SrtSettings: [Match.objectLike({
            ConnectionMode: 'LISTENER',
            EncryptionPassphraseSecretArn: { Ref: Match.stringLikeRegexp('Passphrase') },
          })],
        }),
      ]),
    });
    // The channel role can read the passphrase secret.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['secretsmanager:GetSecretValue']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('callerUrl targets an explicit endpoint URL with a passphrase', () => {
    const secret = new secretsmanager.Secret(stack, 'Passphrase');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'SrtUrlChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'srt_out',
            destinations: [SrtDestination.callerUrl('srt://203.0.113.30:5000', { encryptionPassphraseSecret: secret })],
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'srt-out',
          SrtSettings: [Match.objectLike({
            ConnectionMode: 'CALLER',
            Url: 'srt://203.0.113.30:5000',
            EncryptionPassphraseSecretArn: { Ref: Match.stringLikeRegexp('Passphrase') },
          })],
        }),
      ]),
    });
  });

  test('fails when an SRT listener output has no channel security groups', () => {
    const secret = new secretsmanager.Secret(stack, 'Passphrase');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    expect(() => {
      new Channel(stack, 'SrtListenerNoSg', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.srt({
            name: 'srt',
            outputs: [{
              encodes: [video],
              outputName: 'srt_listener',
              destinations: [SrtDestination.listener({ listenerPort: 5000, encryptionPassphraseSecret: secret })],
            }],
          }),
        ],
      });
    }).toThrow(/listener connection mode requires channelSecurityGroups/);
  });

  test('caller-only SRT output does not require channel security groups', () => {
    const secret = new secretsmanager.Secret(stack, 'Passphrase');
    const video = EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    new Channel(stack, 'SrtCallerNoSg', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.srt({
          name: 'srt',
          outputs: [{
            encodes: [video],
            outputName: 'srt_caller',
            destinations: [SrtDestination.caller({ address: '203.0.113.20', port: 5001, encryptionPassphraseSecret: secret })],
          }],
        }),
      ],
    });
    // No throw, and no ChannelSecurityGroups emitted.
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      ChannelSecurityGroups: Match.absent(),
    });
  });
});

describe('Multi-region additional destinations', () => {
  test('additional destination resolves region from the MediaPackage channel stack', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    // Primary stack with cross-region references enabled
    const primaryStack = new Stack(app, 'PrimaryStack', {
      env: { account: '123456789012', region: 'us-east-1' },
      crossRegionReferences: true,
    });
    const primarySg = new InputSecurityGroup(primaryStack, 'SG', { allowlistRules: ['0.0.0.0/0'] });
    const primaryInput = new Input(primaryStack, 'Input', {
      inputName: 'mr-input',
      input: InputConfiguration.srtListener({ inputSecurityGroups: [primarySg] }),
    });
    const primaryGroup = new mediapackagev2.ChannelGroup(primaryStack, 'PrimaryGroup');
    const primaryMpChannel = new mediapackagev2.Channel(primaryStack, 'PrimaryMpChannel', {
      channelGroup: primaryGroup,
      channelName: 'primary-mp',
    });

    // Secondary stack in a different region
    const secondaryStack = new Stack(app, 'SecondaryStack', {
      env: { account: '123456789012', region: 'us-west-2' },
      crossRegionReferences: true,
    });
    const secondaryGroup = new mediapackagev2.ChannelGroup(secondaryStack, 'SecondaryGroup');
    const secondaryChannel = new mediapackagev2.Channel(secondaryStack, 'SecondaryChannel', {
      channelGroup: secondaryGroup,
      channelName: 'secondary-channel',
    });

    new Channel(primaryStack, 'MultiRegionChannel', {
      inputs: [{ input: primaryInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'mp',
          channel: primaryMpChannel,
          additionalDestinations: [
            MediaPackageV2Destination.channel(secondaryChannel, MediaPackageV2EndpointId.ENDPOINT_1),
          ],
          outputs: [{ encode: hd, outputName: 'hd_output' }],
        }),
      ],
    });

    const template = Template.fromStack(primaryStack);

    // Primary destination should have us-east-1 region
    // Additional destination should have us-west-2 region
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mp',
          MediaPackageSettings: Match.arrayWith([
            Match.objectLike({ MediaPackageRegionName: 'us-east-1' }),
          ]),
        }),
        Match.objectLike({
          Id: 'mp-additional-0',
          MediaPackageSettings: Match.arrayWith([
            Match.objectLike({ MediaPackageRegionName: 'us-west-2' }),
          ]),
        }),
      ]),
    });

    // Output group settings should reference the additional destination
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MediaPackageGroupSettings: Match.objectLike({
                MediapackageV2GroupSettings: {
                  AdditionalDestinations: [
                    Match.objectLike({
                      Destination: { DestinationRefId: 'mp-additional-0' },
                    }),
                  ],
                },
              }),
            },
          }),
        ]),
      }),
    });
  });
});

describe('MediaPackageV2Destination endpoint', () => {
  function mpChannelOutputGroup(endpointId?: MediaPackageV2EndpointId) {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });
    return OutputGroupConfiguration.mediaPackageV2PerPipeline({
      name: 'mp',
      destinations: [MediaPackageV2Destination.channel(empChannel, endpointId)],
      outputs: [{ encode: hd, outputName: 'hd_output' }],
    });
  }

  test('omitting the endpoint leaves channelEndpointId and region unset (pipeline auto-map)', () => {
    new Channel(stack, 'AutoMapChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [mpChannelOutputGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mp',
          MediaPackageSettings: [
            Match.objectLike({
              ChannelName: Match.anyValue(),
              ChannelGroup: Match.anyValue(),
              ChannelEndpointId: Match.absent(),
              MediaPackageRegionName: Match.absent(),
            }),
          ],
        }),
      ]),
    });
  });

  test('providing the endpoint emits channelEndpointId and the channel region', () => {
    new Channel(stack, 'ExplicitEndpointChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [mpChannelOutputGroup(MediaPackageV2EndpointId.ENDPOINT_1)],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mp',
          MediaPackageSettings: [
            Match.objectLike({
              ChannelEndpointId: 'ENDPOINT_1',
              MediaPackageRegionName: 'us-east-1',
            }),
          ],
        }),
      ]),
    });
  });
});

describe('Output container settings', () => {
  function video() {
    return EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
  }

  test('UDP output with FEC settings', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          // FEC requires an rtp:// destination.
          destinations: [UdpOutputDestination.rtp({ address: '203.0.113.5', port: 5000 })],
          outputs: [{
            encodes: [video()],
            outputName: 'ts',
            fec: { mode: FecMode.COLUMN_AND_ROW, columnDepth: 10, rowLength: 10 },
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  UdpOutputSettings: Match.objectLike({
                    FecOutputSettings: {
                      IncludeFec: 'COLUMN_AND_ROW',
                      ColumnDepth: 10,
                      RowLength: 10,
                    },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('FEC fails when a destination is not rtp://', () => {
    expect(() => {
      new Channel(stack, 'BadFecProtocol', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.udp({
            name: 'udp',
            destinations: [UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
            outputs: [{ encodes: [video()], outputName: 'ts', fec: { mode: FecMode.COLUMN } }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/FEC enabled, which requires every destination in the group to use the rtp:\/\/ protocol/);
  });

  test.each([
    ['columnDepth', { columnDepth: 21 }, /columnDepth must be between 4 and 20/],
    ['rowLength', { rowLength: 0 }, /rowLength must be between 1 and 20/],
  ])('UDP FEC fails for out-of-range %s', (_label, fec, re) => {
    expect(() => {
      new Channel(stack, 'BadFec', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.udp({
            name: 'udp',
            destinations: [UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
            outputs: [{ encodes: [video()], outputName: 'ts', fec }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(re);
  });

  test('Archive output with a raw container', () => {
    const wav = EncodeConfiguration.audio({
      name: 'wav',
      codec: AudioCodecSettings.wav({ codingMode: WavCodingMode.CODING_MODE_2_0 }),
    });
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [
            // An archive group needs a video output; the raw output is audio-only WAV.
            { encodes: [video()], outputName: 'video_out', nameModifier: '_ts' },
            { encodes: [wav], outputName: 'raw_out', nameModifier: '_raw', extension: 'wav', container: ArchiveContainer.raw() },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  ArchiveOutputSettings: Match.objectLike({
                    Extension: 'wav',
                    ContainerSettings: { RawSettings: {} },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('Archive output with passthrough audio', () => {
    // Not integ-tested — the service accepts this shape regardless of whether the source is
    // actually Dolby-encoded, so a deploy would prove nothing without a real Dolby feed.
    const passthrough = EncodeConfiguration.audio({
      name: 'audio-passthrough',
      codec: AudioCodecSettings.passthrough(),
    });
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [
            { encodes: [video()], outputName: 'video_out', nameModifier: '_ts' },
            { encodes: [passthrough], outputName: 'passthrough_out', nameModifier: '_passthrough' },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AudioDescriptions: Match.arrayWith([
          Match.objectLike({
            Name: 'audio-passthrough',
            CodecSettings: { PassThroughSettings: {} },
          }),
        ]),
      }),
    });
  });

  test('raw archive output requires an extension', () => {
    const wav = EncodeConfiguration.audio({
      name: 'wav',
      codec: AudioCodecSettings.wav({ codingMode: WavCodingMode.CODING_MODE_2_0 }),
    });
    expect(() => {
      new Channel(stack, 'BadRawExtension', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.archive({
            name: 'archive',
            destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
            outputs: [
              { encodes: [video()], outputName: 'video_out', nameModifier: '_ts' },
              { encodes: [wav], outputName: 'raw_out', nameModifier: '_raw', container: ArchiveContainer.raw() },
            ],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/raw container, which requires an explicit 'extension'/);
  });

  test('archive group with only a raw (audio-only) output throws', () => {
    const wav = EncodeConfiguration.audio({
      name: 'wav',
      codec: AudioCodecSettings.wav({ codingMode: WavCodingMode.CODING_MODE_2_0 }),
    });
    expect(() => new Channel(stack, 'NoVideoArchive', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [{ encodes: [wav], outputName: 'raw_out', extension: 'wav', container: ArchiveContainer.raw() }],
        }),
      ],
    })).toThrow(/must contain at least one output with a video encode/);
  });

  test('Archive output defaults to an M2TS container', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.archive({
          name: 'archive',
          destinations: [S3OutputDestination.url('s3ssl://bucket/archive')],
          outputs: [{ encodes: [video()], outputName: 'm2ts_out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  ArchiveOutputSettings: Match.objectLike({
                    ContainerSettings: { M2tsSettings: {} },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('CMAF Ingest output group with caption language mappings', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.cmafIngest({
          name: 'cmaf',
          destinations: [OutputDestination.url('https://ingest.example.com/v1/channel/')],
          captionLanguageMappings: [
            { captionChannel: 1, languageCode: 'eng' },
            { captionChannel: 2, languageCode: 'spa' },
          ],
          outputs: [{ outputName: 'cmaf_video', nameModifier: '_video', encode: video() }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              CmafIngestGroupSettings: Match.objectLike({
                CaptionLanguageMappings: [
                  { CaptionChannel: 1, LanguageCode: 'eng' },
                  { CaptionChannel: 2, LanguageCode: 'spa' },
                ],
              }),
            },
          }),
        ]),
      }),
    });
  });
});

describe('MediaPackageV2GroupSettings', () => {
  test('passes segment and metadata settings to MediaPackageV2GroupSettings', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    new Channel(stack, 'MpV2SettingsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          segment: Segment.seconds(4),
          id3Behavior: Id3Behavior.ENABLED,
          klvBehavior: KlvBehavior.PASSTHROUGH,
          nielsenId3Behavior: NielsenId3Behavior.PASSTHROUGH,
          scte35Type: Scte35Type.NONE,
          timedMetadataId3Frame: TimedMetadataId3Frame.PRIV,
          timedMetadataId3Period: Duration.seconds(10),
          timedMetadataPassthrough: TimedMetadataPassthrough.ENABLED,
          captionLanguageMappings: [{
            captionChannel: 1,
            languageCode: 'eng',
            languageDescription: 'English',
          }],
          outputs: [{ encode: hd, outputName: 'hd_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MediaPackageGroupSettings: Match.objectLike({
                MediapackageV2GroupSettings: Match.objectLike({
                  SegmentLength: 4,
                  SegmentLengthUnits: 'SECONDS',
                  Id3Behavior: 'ENABLED',
                  KlvBehavior: 'PASSTHROUGH',
                  NielsenId3Behavior: 'PASSTHROUGH',
                  Scte35Type: 'NONE',
                  TimedMetadataId3Frame: 'PRIV',
                  TimedMetadataId3Period: 10,
                  TimedMetadataPassthrough: 'ENABLED',
                  CaptionLanguageMappings: [{
                    CaptionChannel: 1,
                    LanguageCode: 'eng',
                    LanguageDescription: 'English',
                  }],
                }),
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('defaults segment length to 1 second in MediaPackageV2GroupSettings', () => {
    const hd = EncodeConfiguration.video({
      name: 'hd',
      width: 1920,
      height: 1080,
      codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
    });

    new Channel(stack, 'DefaultSettingsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: hd, outputName: 'hd_output' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MediaPackageGroupSettings: Match.objectLike({
                MediapackageV2GroupSettings: {
                  SegmentLength: 1,
                  SegmentLengthUnits: 'SECONDS',
                  Id3Behavior: 'DISABLED',
                  KlvBehavior: 'NO_PASSTHROUGH',
                  NielsenId3Behavior: 'NO_PASSTHROUGH',
                  Scte35Type: 'SCTE_35_WITHOUT_SEGMENTATION',
                  TimedMetadataId3Frame: 'NONE',
                  TimedMetadataId3Period: 10,
                  TimedMetadataPassthrough: 'DISABLED',
                },
              }),
            },
          }),
        ]),
      }),
    });
  });
});

describe('Automatic input failover', () => {
  function secondaryInput(): Input {
    const sg = new InputSecurityGroup(stack, 'SecondarySG', { allowlistRules: ['203.0.113.0/24'] });
    return new Input(stack, 'SecondaryInput', {
      inputName: 'secondary-input',
      input: InputConfiguration.srtListener({ inputSecurityGroups: [sg] }),
    });
  }

  function failoverChannel(automaticInputFailover: AutomaticInputFailover): void {
    new Channel(stack, 'FailoverChannel', {
      inputs: [
        { input: defaultInput, automaticInputFailover },
        // The secondary of a failover pair must also be attached to the channel.
        { input: automaticInputFailover.secondaryInput, inputAttachmentName: 'secondary' },
      ],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [EncodeConfiguration.video({
              name: 'v',
              width: 1280,
              height: 720,
              codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
            })],
            outputName: 'out',
          }],
        }),
      ],
    });
  }

  test('defaults to a single input-loss condition when none are provided', () => {
    failoverChannel({ secondaryInput: secondaryInput() });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          AutomaticInputFailoverSettings: {
            SecondaryInputId: Match.anyValue(),
            FailoverConditions: [
              { FailoverConditionSettings: { InputLossSettings: {} } },
            ],
          },
        }),
      ]),
    });
  });

  test('renders explicit conditions, preference, and error-clear time', () => {
    failoverChannel({
      secondaryInput: secondaryInput(),
      inputPreference: InputPreference.PRIMARY_INPUT_PREFERRED,
      errorClearTime: Duration.seconds(3),
      failoverConditions: [
        FailoverCondition.inputLoss({ threshold: Duration.millis(1500) }),
        FailoverCondition.audioSilence({ audioSelector: AudioSelector.byLanguage('aac', 'eng'), threshold: Duration.seconds(2) }),
        FailoverCondition.videoBlack({ blackDetectThreshold: 0.1, threshold: Duration.seconds(1) }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          AutomaticInputFailoverSettings: {
            InputPreference: 'PRIMARY_INPUT_PREFERRED',
            ErrorClearTimeMsec: 3000,
            FailoverConditions: [
              { FailoverConditionSettings: { InputLossSettings: { InputLossThresholdMsec: 1500 } } },
              { FailoverConditionSettings: { AudioSilenceSettings: { AudioSelectorName: 'aac', AudioSilenceThresholdMsec: 2000 } } },
              { FailoverConditionSettings: { VideoBlackSettings: { BlackDetectThreshold: 0.1, VideoBlackThresholdMsec: 1000 } } },
            ],
          },
        }),
      ]),
    });
  });
});

describe('AudioSelector', () => {
  test('renders all selector variants on the input attachment', () => {
    new Channel(stack, 'AudioSelectorChannel', {
      inputs: [{
        input: defaultInput,
        audioSelectors: [
          AudioSelector.byLanguage('eng', 'eng', AudioLanguageSelectionPolicy.STRICT),
          AudioSelector.byPid('pid', [{ pid: 256 }]),
          AudioSelector.byTrack('tracks', [{ track: 1 }, { track: 2 }], DolbyEProgramSelection.PROGRAM_1),
          AudioSelector.hlsRendition('hls', { groupId: 'audio', renditionName: 'English' }),
          AudioSelector.default('def'),
        ],
      }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [EncodeConfiguration.video({
              name: 'v',
              width: 1280,
              height: 720,
              codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
            })],
            outputName: 'out',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            AudioSelectors: [
              { Name: 'eng', SelectorSettings: { AudioLanguageSelection: { LanguageCode: 'eng', LanguageSelectionPolicy: 'STRICT' } } },
              { Name: 'pid', SelectorSettings: { AudioPidSelection: { Pids: [{ Pid: 256 }] } } },
              { Name: 'tracks', SelectorSettings: { AudioTrackSelection: { Tracks: [{ Track: 1 }, { Track: 2 }], DolbyEDecode: { ProgramSelection: 'PROGRAM_1' } } } },
              { Name: 'hls', SelectorSettings: { AudioHlsRenditionSelection: { GroupId: 'audio', Name: 'English' } } },
              { Name: 'def' },
            ],
          }),
        }),
      ]),
    });
  });

  test('audio selector with per-PID Dolby E decode and premix settings', () => {
    new Channel(stack, 'PremixChannel', {
      inputs: [{
        input: defaultInput,
        audioSelectors: [
          AudioSelector.byPid('premix-pid', [{
            pid: 100,
            dolbyEDecode: DolbyEProgramSelection.PROGRAM_1,
            premixSettings: AudioPreMixerSettings.of({ gainDb: -3, channels: 2 }),
          }]),
          AudioSelector.byTrack('premix-track', [{
            track: 1,
            premixSettings: AudioPreMixerSettings.of({ gainDb: 6 }),
          }]),
        ],
      }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [EncodeConfiguration.video({
              name: 'v',
              width: 1280,
              height: 720,
              codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
            })],
            outputName: 'out',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            AudioSelectors: Match.arrayWith([
              Match.objectLike({
                Name: 'premix-pid',
                SelectorSettings: {
                  AudioPidSelection: {
                    Pids: [{
                      Pid: 100,
                      DolbyEDecode: { ProgramSelection: 'PROGRAM_1' },
                      PremixSettings: { GainDb: -3, Channels: 2 },
                    }],
                  },
                },
              }),
              Match.objectLike({
                Name: 'premix-track',
                SelectorSettings: {
                  AudioTrackSelection: {
                    Tracks: [{
                      Track: 1,
                      PremixSettings: { GainDb: 6 },
                    }],
                  },
                },
              }),
            ]),
          }),
        }),
      ]),
    });
  });
});

describe('VideoSelector', () => {
  test('renders color space, usage, HDR10 metadata, and program selection', () => {
    new Channel(stack, 'VideoSelChannel', {
      inputs: [{
        input: defaultInput,
        videoSelector: {
          colorSpace: VideoColorSpace.HDR10,
          colorSpaceUsage: VideoColorSpaceUsage.FORCE,
          hdr10: { maxContentLightLevel: 1000, maxFrameAverageLightLevel: 400 },
          selectBy: VideoSelection.byProgramId(3),
        },
      }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [EncodeConfiguration.video({
              name: 'v',
              width: 1280,
              height: 720,
              codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
            })],
            outputName: 'out',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            VideoSelector: {
              ColorSpace: 'HDR10',
              ColorSpaceUsage: 'FORCE',
              ColorSpaceSettings: { Hdr10Settings: { MaxCll: 1000, MaxFall: 400 } },
              SelectorSettings: { VideoSelectorProgramId: { ProgramId: 3 } },
            },
          }),
        }),
      ]),
    });
  });
});

describe('CaptionSelector', () => {
  test('renders source-specific selector settings for each format', () => {
    new Channel(stack, 'CaptionSelChannel', {
      inputs: [{
        input: defaultInput,
        captionSelectors: [
          CaptionSelector.byLanguage('eng', 'eng'),
          CaptionSelector.embedded('emb', {
            convert608To708: Convert608To708.UPCONVERT,
            scte20Detection: Scte20Detection.AUTO,
            source608ChannelNumber: 1,
          }),
          CaptionSelector.ancillary('anc', { sourceChannelNumber: 2 }),
          CaptionSelector.arib('arib'),
          CaptionSelector.dvbSub('dvb', { pid: 256, ocrLanguage: OcrLanguage.ENG }),
          CaptionSelector.scte27('s27', { pid: 257 }),
          CaptionSelector.teletext('ttx', {
            pageNumber: '888',
            outputRectangle: { height: 80, width: 80, leftOffset: 10, topOffset: 10 },
          }),
        ],
      }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          outputs: [{
            encodes: [EncodeConfiguration.video({
              name: 'v',
              width: 1280,
              height: 720,
              codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
            })],
            outputName: 'out',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            CaptionSelectors: [
              { Name: 'eng', LanguageCode: 'eng' },
              { Name: 'emb', SelectorSettings: { EmbeddedSourceSettings: { Convert608To708: 'UPCONVERT', Scte20Detection: 'AUTO', Source608ChannelNumber: 1 } } },
              { Name: 'anc', SelectorSettings: { AncillarySourceSettings: { SourceAncillaryChannelNumber: 2 } } },
              { Name: 'arib', SelectorSettings: { AribSourceSettings: {} } },
              { Name: 'dvb', SelectorSettings: { DvbSubSourceSettings: { Pid: 256, OcrLanguage: 'ENG' } } },
              { Name: 's27', SelectorSettings: { Scte27SourceSettings: { Pid: 257 } } },
              { Name: 'ttx', SelectorSettings: { TeletextSourceSettings: { PageNumber: '888', OutputRectangle: { Height: 80, Width: 80, LeftOffset: 10, TopOffset: 10 } } } },
            ],
          }),
        }),
      ]),
    });
  });
});

describe('CaptionSelector validation', () => {
  test.each([0, 5, -1])('fails for invalid embedded source608ChannelNumber %d', (val) => {
    expect(() => CaptionSelector.embedded('emb', { source608ChannelNumber: val })).toThrow(/source608ChannelNumber must be between 1 and 4/);
  });

  test.each([0, 5, -1])('fails for invalid ancillary sourceChannelNumber %d', (val) => {
    expect(() => CaptionSelector.ancillary('anc', { sourceChannelNumber: val })).toThrow(/sourceChannelNumber must be between 1 and 4/);
  });

  test.each([0, 5, -1])('fails for invalid scte20 source608ChannelNumber %d', (val) => {
    expect(() => CaptionSelector.scte20('s20', { source608ChannelNumber: val })).toThrow(/source608ChannelNumber must be between 1 and 4/);
  });

  test.each([1, 2, 3, 4])('accepts valid source608ChannelNumber %d', (val) => {
    expect(() => CaptionSelector.embedded('emb', { source608ChannelNumber: val })).not.toThrow();
  });
});

describe('Selector and failover edge cases', () => {
  function hlsGroup() {
    return OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [OutputDestination.url('s3ssl://bucket/live')],
      outputs: [{
        encodes: [EncodeConfiguration.video({
          name: 'v',
          width: 1280,
          height: 720,
          codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_30 }),
        })],
        outputName: 'out',
      }],
    });
  }

  test('fails when failover primary and secondary input classes differ', () => {
    const primary = new Input(stack, 'PrimaryDevice', {
      inputName: 'primary-device',
      input: InputConfiguration.inputDevice({ deviceIds: ['hd-0000001'] }), // SINGLE_PIPELINE
    });
    const secondary = new Input(stack, 'SecondaryDevice', {
      inputName: 'secondary-device',
      input: InputConfiguration.inputDevice({ deviceIds: ['hd-0000002', 'hd-0000003'] }), // STANDARD
    });

    expect(() => {
      new Channel(stack, 'MismatchChannel', {
        inputs: [{ input: primary, automaticInputFailover: { secondaryInput: secondary } }],
        outputGroups: [hlsGroup()],
      });
    }).toThrow(/same input class/);
  });

  test('fails when failover secondary input is not attached to the channel', () => {
    const sg = new InputSecurityGroup(stack, 'UnattachedSG', { allowlistRules: ['203.0.113.0/24'] });
    const secondary = new Input(stack, 'UnattachedSecondary', {
      inputName: 'unattached-secondary',
      input: InputConfiguration.srtListener({ inputSecurityGroups: [sg] }),
    });

    expect(() => {
      new Channel(stack, 'UnattachedFailoverChannel', {
        inputs: [{ input: defaultInput, automaticInputFailover: { secondaryInput: secondary } }],
        outputGroups: [hlsGroup()],
      });
      Template.fromStack(stack);
    }).toThrow(/not attached to the channel/);
  });

  test('embedded caption selector with no options emits no selector settings', () => {
    new Channel(stack, 'EmbeddedDefaultChannel', {
      inputs: [{ input: defaultInput, captionSelectors: [CaptionSelector.embedded('emb')] }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            CaptionSelectors: [{ Name: 'emb', SelectorSettings: Match.absent() }],
          }),
        }),
      ]),
    });
  });

  test('byLanguage audio selector omits the policy when not set', () => {
    new Channel(stack, 'AudioLangChannel', {
      inputs: [{ input: defaultInput, audioSelectors: [AudioSelector.byLanguage('eng', 'eng')] }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            AudioSelectors: [{
              Name: 'eng',
              SelectorSettings: { AudioLanguageSelection: { LanguageCode: 'eng', LanguageSelectionPolicy: Match.absent() } },
            }],
          }),
        }),
      ]),
    });
  });

  test('video selector by PID renders VideoSelectorPid', () => {
    new Channel(stack, 'VideoPidChannel', {
      inputs: [{ input: defaultInput, videoSelector: { selectBy: VideoSelection.byPid(256) } }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            VideoSelector: { SelectorSettings: { VideoSelectorPid: { Pid: 256 } } },
          }),
        }),
      ]),
    });
  });
});

describe('Network input settings', () => {
  function hlsGroup() {
    return OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [OutputDestination.url('s3ssl://bucket/live')],
      outputs: [{
        encodes: [EncodeConfiguration.video({ name: 'v', width: 1280, height: 720, codec: VideoCodecSettings.h264() })],
        outputName: 'out',
      }],
    });
  }

  test('renders HLS input settings with scte35Source and server validation', () => {
    new Channel(stack, 'NetChannel', {
      inputs: [{
        input: defaultInput,
        networkInputSettings: {
          serverValidation: ServerValidation.CHECK_CRYPTOGRAPHY_AND_VALIDATE_NAME,
          hlsInputSettings: {
            bandwidth: Bitrate.mbps(5),
            bufferSegments: 3,
            retries: 5,
            retryInterval: Duration.seconds(2),
            scte35Source: HlsScte35Source.MANIFEST,
          },
        },
      }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            NetworkInputSettings: {
              ServerValidation: 'CHECK_CRYPTOGRAPHY_AND_VALIDATE_NAME',
              HlsInputSettings: {
                Bandwidth: 5000000,
                BufferSegments: 3,
                Retries: 5,
                RetryInterval: 2,
                Scte35Source: 'MANIFEST',
              },
            },
          }),
        }),
      ]),
    });
  });

  test('renders multicast source IP under network input settings', () => {
    new Channel(stack, 'MulticastChannel', {
      inputs: [{
        input: defaultInput,
        networkInputSettings: { multicastSourceIp: '203.0.113.42' },
      }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          InputSettings: Match.objectLike({
            NetworkInputSettings: {
              MulticastInputSettings: { SourceIpAddress: '203.0.113.42' },
            },
          }),
        }),
      ]),
    });
  });

  test('renders logical interface names on the input attachment', () => {
    new Channel(stack, 'LogicalIfChannel', {
      inputs: [{
        input: defaultInput,
        logicalInterfaceNames: ['eth0', 'eth1'],
      }],
      outputGroups: [hlsGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      InputAttachments: Match.arrayWith([
        Match.objectLike({
          LogicalInterfaceNames: ['eth0', 'eth1'],
        }),
      ]),
    });
  });
});

// NOTE: AWS's docs also list AV1 and MPEG-2 for MediaConnect Router, but the service rejects
// both in practice, so neither is added to the output's allowed-codec list.

describe('MediaPackage V2 output group codec support', () => {
  // MediaPackage V2 uses CMAF Ingest under the hood, which supports AV1 in addition to
  // H.264/H.265 — see https://docs.aws.amazon.com/mediapackage/latest/userguide/cmaf-ingest.html.
  // This was previously missing from the output's allowed-codec list.
  test('accepts an AV1 video encode', () => {
    const av1Video = EncodeConfiguration.video({
      name: 'av1',
      width: 1280,
      height: 720,
      codec: VideoCodecSettings.av1({
        rateControl: Av1RateControl.qvbr({ maxBitrate: Bitrate.mbps(3) }),
        framerate: Framerate.FPS_29_97,
      }),
    });

    new Channel(stack, 'EmpAv1Channel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaPackageV2({
          name: 'emp',
          channel: empChannel,
          outputs: [{ encode: av1Video, outputName: 'av1_out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        VideoDescriptions: Match.arrayWith([
          Match.objectLike({ Name: 'av1', CodecSettings: Match.objectLike({ Av1Settings: Match.anyValue() }) }),
        ]),
      }),
    });
  });
});

describe('MediaConnect Router output group settings', () => {
  const video = () => EncodeConfiguration.video({
    name: 'v',
    width: 1920,
    height: 1080,
    codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
  });

  function standardInput(): Input {
    return new Input(stack, 'McrStdInput', {
      inputName: 'mcr-std-input',
      input: InputConfiguration.srtCaller([
        { srtListenerAddress: '203.0.113.100', srtListenerPort: 5000 },
        { srtListenerAddress: '203.0.113.101', srtListenerPort: 5001 },
      ]),
    });
  }

  test('omitting routerSettings defaults to AUTOMATIC on a SINGLE_PIPELINE channel', () => {
    new Channel(stack, 'McrChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a'],
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mcr',
          MediaConnectRouterSettings: [{ EncryptionType: 'AUTOMATIC' }],
        }),
      ]),
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: { MediaConnectRouterGroupSettings: {} },
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  MediaConnectRouterOutputSettings: Match.objectLike({
                    Destination: { DestinationRefId: 'mcr' },
                    ContainerSettings: { M2tsSettings: {} },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('STANDARD channel auto-expands to two AUTOMATIC destinations', () => {
    new Channel(stack, 'McrStdChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput() }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a', 'us-east-1b'],
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mcr',
          MediaConnectRouterSettings: [
            { EncryptionType: 'AUTOMATIC' },
            { EncryptionType: 'AUTOMATIC' },
          ],
        }),
      ]),
    });
  });

  test('shared secret applies SECRETS_MANAGER to every pipeline and grants read', () => {
    const secret = new secretsmanager.Secret(stack, 'McrSecret', { secretName: 'mcr-passphrase' });

    new Channel(stack, 'McrSharedChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput() }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a', 'us-east-1b'],
          routerSettings: MediaConnectRouterSettings.shared({ encryptionSecret: secret }),
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mcr',
          MediaConnectRouterSettings: [
            Match.objectLike({ EncryptionType: 'SECRETS_MANAGER', SecretArn: Match.anyValue() }),
            Match.objectLike({ EncryptionType: 'SECRETS_MANAGER', SecretArn: Match.anyValue() }),
          ],
        }),
      ]),
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('perPipeline mixes AUTOMATIC and SECRETS_MANAGER positionally', () => {
    const secret = new secretsmanager.Secret(stack, 'McrPpSecret', { secretName: 'mcr-pp-passphrase' });

    new Channel(stack, 'McrPerPipelineChannel', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput() }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a', 'us-east-1b'],
          routerSettings: MediaConnectRouterSettings.perPipeline({
            pipeline1: { encryptionSecret: secret },
          }),
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({
          Id: 'mcr',
          MediaConnectRouterSettings: [
            { EncryptionType: 'AUTOMATIC' },
            Match.objectLike({ EncryptionType: 'SECRETS_MANAGER', SecretArn: Match.anyValue() }),
          ],
        }),
      ]),
    });
  });

  test('passes availabilityZones to the group settings', () => {
    new Channel(stack, 'McrAzChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a'],
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MediaConnectRouterGroupSettings: { AvailabilityZones: ['us-east-1a'] },
            },
          }),
        ]),
      }),
    });
  });

  test('fails when pipeline1 settings are given on a SINGLE_PIPELINE channel', () => {
    const secret = new secretsmanager.Secret(stack, 'McrSingleSecret', { secretName: 'mcr-single' });

    new Channel(stack, 'McrSingleFail', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a'],
          routerSettings: MediaConnectRouterSettings.perPipeline({
            pipeline1: { encryptionSecret: secret },
          }),
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    expect(() => Template.fromStack(stack)).toThrow(/pipeline1 settings are not valid on a SINGLE_PIPELINE channel/);
  });

  test('fails when the availabilityZones count does not match the channel class', () => {
    new Channel(stack, 'McrAzMismatch', {
      channelClass: ChannelClass.STANDARD,
      inputs: [{ input: standardInput() }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'mcr',
          availabilityZones: ['us-east-1a'], // STANDARD requires two
          outputs: [{ encodes: [video()], outputName: 'ts' }],
        }),
      ],
    });

    expect(() => Template.fromStack(stack)).toThrow(/requires exactly 2 availabilityZone\(s\) for a STANDARD channel/);
  });
});

describe('M2TS container settings', () => {
  const video = () => EncodeConfiguration.video({ name: 'v', width: 1920, height: 1080, codec: VideoCodecSettings.h264() });

  test('UDP output renders configured M2TS settings with strong-type conversions', () => {
    new Channel(stack, 'M2tsUdpChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          destinations: [UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
          outputs: [{
            encodes: [video()],
            outputName: 'ts',
            m2tsSettings: M2tsSettings.of({
              bitrate: Bitrate.mbps(8),
              rateMode: M2tsRateMode.VBR,
              programNum: 3,
              patInterval: Duration.millis(100),
              scte35Control: M2tsScte35Control.PASSTHROUGH,
              dvbSdtSettings: {
                outputSdt: DvbSdtOutputMode.SDT_MANUAL,
                serviceName: 'My Service',
                repInterval: Duration.millis(2000),
              },
            }),
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  UdpOutputSettings: Match.objectLike({
                    ContainerSettings: {
                      M2tsSettings: Match.objectLike({
                        Bitrate: 8_000_000,
                        RateMode: 'VBR',
                        ProgramNum: 3,
                        PatInterval: 100,
                        Scte35Control: 'PASSTHROUGH',
                        DvbSdtSettings: {
                          OutputSdt: 'SDT_MANUAL',
                          ServiceName: 'My Service',
                          RepInterval: 2000,
                        },
                      }),
                    },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('omitting m2tsSettings still produces an empty M2tsSettings object', () => {
    new Channel(stack, 'M2tsDefaultChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.udp({
          name: 'udp',
          destinations: [UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
          outputs: [{
            encodes: [video()],
            outputName: 'ts',
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  UdpOutputSettings: Match.objectLike({
                    ContainerSettings: { M2tsSettings: {} },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('M2TS settings apply to a MediaConnect Router output', () => {
    new Channel(stack, 'M2tsMcrChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.mediaConnectRouter({
          name: 'router-out',
          availabilityZones: ['us-east-1a'],
          outputs: [{
            encodes: [video()],
            outputName: 'router-ts',
            m2tsSettings: M2tsSettings.of({ bitrate: Bitrate.mbps(10) }),
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: Match.objectLike({
                  MediaConnectRouterOutputSettings: Match.objectLike({
                    ContainerSettings: { M2tsSettings: Match.objectLike({ Bitrate: 10_000_000 }) },
                  }),
                }),
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('Global configuration and avail', () => {
  function videoOutputGroup() {
    const video = EncodeConfiguration.video({ name: 'video', width: 1280, height: 720, codec: VideoCodecSettings.h264() });
    return OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [OutputDestination.url('s3ssl://bucket/live')],
      outputs: [{ encodes: [video], outputName: 'video_output' }],
    });
  }

  test('input-loss behavior with a slate image', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        inputLossBehavior: {
          blackFrame: Duration.seconds(1),
          repeatFrame: Duration.millis(2500),
          imageType: InputLossImageType.SLATE,
          imageSlate: FileLocation.url('s3ssl://bucket/slate.png'),
        },
      },
      outputGroups: [videoOutputGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({
          InputLossBehavior: {
            BlackFrameMsec: 1000,
            RepeatFrameMsec: 2500,
            InputLossImageType: 'SLATE',
            InputLossImageSlate: { Uri: 's3ssl://bucket/slate.png' },
          },
        }),
      }),
    });
  });

  test('input-loss slate image from a bucket grants the channel role read access', () => {
    const slateBucket = new s3.Bucket(stack, 'SlateBucket');
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        inputLossBehavior: {
          imageType: InputLossImageType.SLATE,
          imageSlate: FileLocation.fromBucket(slateBucket, 'slates/offline.png'),
        },
      },
      outputGroups: [videoOutputGroup()],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({
          InputLossBehavior: Match.objectLike({
            InputLossImageSlate: {
              Uri: {
                'Fn::Join': ['', ['s3ssl://', { Ref: Match.stringLikeRegexp('SlateBucket') }, '/slates/offline.png']],
              },
            },
          }),
        }),
      }),
    });
    // The channel role is granted read access to the slate bucket.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:GetObject*']),
          }),
        ]),
      }),
    });
  });

  test('granular output locking (disabled with custom epoch)', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        outputLocking: OutputLocking.disabled('2024-01-01T00:00:00Z'),
      },
      outputGroups: [videoOutputGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({
          OutputLockingSettings: { DisabledLockingSettings: { CustomEpoch: '2024-01-01T00:00:00Z' } },
        }),
      }),
    });
  });

  test('granular pipeline output locking with a locking method', () => {
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      globalConfiguration: {
        outputLocking: OutputLocking.pipeline({ method: PipelineLockingMethod.VIDEO_ALIGNMENT }),
      },
      outputGroups: [videoOutputGroup()],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        GlobalConfiguration: Match.objectLike({
          OutputLockingMode: 'PIPELINE_LOCKING',
          OutputLockingSettings: { PipelineLockingSettings: { PipelineLockingMethod: 'VIDEO_ALIGNMENT' } },
        }),
      }),
    });
  });

  test('ESAM avail settings with a POIS endpoint', () => {
    const poisPassword = new StringParameter(stack, 'PoisPassword', { stringValue: 'placeholder' });
    new Channel(stack, 'MyChannel', {
      inputs: [{ input: defaultInput }],
      availSettings: AvailSettings.esam({
        pois: {
          url: 'https://pois.example.com/esam',
          username: 'pois-user',
          password: poisPassword,
        },
        acquisitionPointId: 'acq-1',
        adAvailOffset: Duration.millis(200),
        zoneIdentity: 'zone-1',
      }),
      scte35SegmentationScope: Scte35SegmentationScope.SCTE35_ENABLED_OUTPUT_GROUPS,
      outputGroups: [videoOutputGroup()],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        AvailConfiguration: {
          AvailSettings: {
            Esam: Match.objectLike({
              PoisEndpoint: 'https://pois.example.com/esam',
              AcquisitionPointId: 'acq-1',
              AdAvailOffset: 200,
              ZoneIdentity: 'zone-1',
              Username: 'pois-user',
              PasswordParam: {
                'Fn::Join': ['', ['ssm://', { Ref: Match.stringLikeRegexp('PoisPassword') }]],
              },
            }),
          },
          Scte35SegmentationScope: 'SCTE35_ENABLED_OUTPUT_GROUPS',
        },
      }),
    });
    // The channel role is granted read access to the POIS password parameter.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['ssm:GetParameters']),
          }),
        ]),
      }),
    });
  });
});
