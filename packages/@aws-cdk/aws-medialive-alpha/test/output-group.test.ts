import { App, SecretValue, Stack } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import {
  Channel,
  EncodeConfiguration,
  Input,
  InputConfiguration,
  InputSecurityGroup,
  Framerate,
  VideoCodecSettings,
  OutputGroupConfiguration,
  OutputDestination,
  S3OutputDestination,
  S3CannedAcl,
  HttpTransferMode,
  HlsCdnSettings,
  HlsKeyProviderSettings,
  MsSmoothAudioOnlyTimecodeControl,
  MsSmoothCertificateMode,
  H265PackagingType,
} from '../lib';

let app: App;
let stack: Stack;
let defaultInput: Input;

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
});

function video() {
  return EncodeConfiguration.video({
    name: 'video',
    width: 1280,
    height: 720,
    codec: VideoCodecSettings.h264({ framerate: Framerate.FPS_29_97 }),
  });
}

function frameCaptureVideo() {
  return EncodeConfiguration.video({
    name: 'frame-capture-video',
    width: 1280,
    height: 720,
    codec: VideoCodecSettings.frameCapture(),
  });
}

describe('OutputGroupConfiguration.frameCapture', () => {
  test('creates a frame capture output group with a single destination', () => {
    new Channel(stack, 'FcChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.frameCapture({
          name: 'frame-capture',
          destinations: [S3OutputDestination.url('s3ssl://bucket/frames')],
          outputs: [{ encodes: [frameCaptureVideo()], outputName: 'fc' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({ Id: 'frame-capture' }),
      ]),
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              FrameCaptureGroupSettings: Match.objectLike({
                Destination: { DestinationRefId: 'frame-capture' },
              }),
            },
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputName: 'fc',
                OutputSettings: { FrameCaptureOutputSettings: {} },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('applies a canned ACL to the S3 destination', () => {
    new Channel(stack, 'FcChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.frameCapture({
          name: 'frame-capture',
          destinations: [S3OutputDestination.url('s3ssl://bucket/frames')],
          frameCaptureS3CannedAcl: S3CannedAcl.PUBLIC_READ,
          outputs: [{ encodes: [frameCaptureVideo()], outputName: 'fc' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              FrameCaptureGroupSettings: Match.objectLike({
                FrameCaptureCdnSettings: {
                  FrameCaptureS3Settings: { CannedAcl: 'PUBLIC_READ' },
                },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('a name modifier is applied to the frame capture output', () => {
    new Channel(stack, 'FcChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.frameCapture({
          name: 'frame-capture',
          destinations: [S3OutputDestination.url('s3ssl://bucket/frames')],
          outputs: [{ encodes: [frameCaptureVideo()], outputName: 'fc', nameModifier: '_low' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: { FrameCaptureOutputSettings: { NameModifier: '_low' } },
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('OutputGroupConfiguration.msSmooth', () => {
  test('creates an MS Smooth output group with a single destination', () => {
    new Channel(stack, 'MsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.msSmooth({
          name: 'ms-smooth',
          destinations: [OutputDestination.url('https://iis.example.com/live')],
          outputs: [{ encodes: [video()], outputName: 'ms' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      Destinations: Match.arrayWith([
        Match.objectLike({ Id: 'ms-smooth' }),
      ]),
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MsSmoothGroupSettings: Match.objectLike({
                Destination: { DestinationRefId: 'ms-smooth' },
              }),
            },
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputName: 'ms',
                OutputSettings: { MsSmoothOutputSettings: {} },
              }),
            ]),
          }),
        ]),
      }),
    });
  });

  test('all MS Smooth group settings are wired through', () => {
    new Channel(stack, 'MsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.msSmooth({
          name: 'ms-smooth',
          destinations: [OutputDestination.url('https://iis.example.com/live')],
          audioOnlyTimecodeControl: MsSmoothAudioOnlyTimecodeControl.PASSTHROUGH,
          certificateMode: MsSmoothCertificateMode.VERIFY_AUTHENTICITY,
          acquisitionPointId: 'acq-1',
          eventId: 'evt-1',
          outputs: [{
            encodes: [video()],
            outputName: 'ms',
            h265PackagingType: H265PackagingType.HVC1,
          }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              MsSmoothGroupSettings: Match.objectLike({
                AcquisitionPointId: 'acq-1',
                AudioOnlyTimecodeControl: 'PASSTHROUGH',
                CertificateMode: 'VERIFY_AUTHENTICITY',
                EventId: 'evt-1',
              }),
            },
            Outputs: Match.arrayWith([
              Match.objectLike({
                OutputSettings: {
                  MsSmoothOutputSettings: { H265PackagingType: 'HVC1' },
                },
              }),
            ]),
          }),
        ]),
      }),
    });
  });
});

describe('HlsCdnSettings', () => {
  function hlsChannel(cdnSettings: HlsCdnSettings) {
    new Channel(stack, 'HlsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          hlsCdnSettings: cdnSettings,
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });
  }

  test('s3() applies a canned ACL', () => {
    hlsChannel(HlsCdnSettings.s3({ cannedAcl: S3CannedAcl.BUCKET_OWNER_FULL_CONTROL }));

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                HlsCdnSettings: {
                  HlsS3Settings: { CannedAcl: 'BUCKET_OWNER_FULL_CONTROL' },
                },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('basicPut() wires retry and cache settings', () => {
    hlsChannel(HlsCdnSettings.basicPut({ numRetries: 5, connectionRetryInterval: 2 }));

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                HlsCdnSettings: {
                  HlsBasicPutSettings: Match.objectLike({
                    NumRetries: 5,
                    ConnectionRetryInterval: 2,
                  }),
                },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('akamai() wires authentication and transfer mode settings', () => {
    hlsChannel(HlsCdnSettings.akamai({
      salt: 'my-salt',
      token: 'my-token',
      httpTransferMode: HttpTransferMode.CHUNKED,
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                HlsCdnSettings: {
                  HlsAkamaiSettings: Match.objectLike({
                    Salt: 'my-salt',
                    Token: 'my-token',
                    HttpTransferMode: 'CHUNKED',
                  }),
                },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('webdav() wires transfer mode settings', () => {
    hlsChannel(HlsCdnSettings.webdav({ httpTransferMode: HttpTransferMode.NON_CHUNKED }));

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                HlsCdnSettings: {
                  HlsWebdavSettings: Match.objectLike({ HttpTransferMode: 'NON_CHUNKED' }),
                },
              }),
            },
          }),
        ]),
      }),
    });
  });

  test('fails when an https destination has no hlsCdnSettings', () => {
    expect(() => {
      new Channel(stack, 'HlsHttpsChannel', {
        inputs: [{ input: defaultInput }],
        outputGroups: [
          OutputGroupConfiguration.hls({
            name: 'hls',
            destinations: [OutputDestination.url('https://cdn.example.com/live/stream')],
            outputs: [{ encodes: [video()], outputName: 'out' }],
          }),
        ],
      });
      Template.fromStack(stack);
    }).toThrow(/https destination URL, which requires hlsCdnSettings/);
  });

  test('accepts an https destination when hlsCdnSettings is set', () => {
    new Channel(stack, 'HlsHttpsOkChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('https://cdn.example.com/live/stream')],
          hlsCdnSettings: HlsCdnSettings.basicPut(),
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                HlsCdnSettings: { HlsBasicPutSettings: Match.anyValue() },
              }),
            },
          }),
        ]),
      }),
    });
  });
});

describe('HlsKeyProviderSettings', () => {
  test('staticKey() wires the key value and provider server URL', () => {
    new Channel(stack, 'HlsChannel', {
      inputs: [{ input: defaultInput }],
      outputGroups: [
        OutputGroupConfiguration.hls({
          name: 'hls',
          destinations: [OutputDestination.url('s3ssl://bucket/live')],
          encryptionType: undefined,
          keyProviderSettings: HlsKeyProviderSettings.staticKey({
            staticKeyValue: SecretValue.unsafePlainText('11111111111111111111111111111111'),
            keyProviderServerUrl: 'https://keys.example.com/key',
          }),
          outputs: [{ encodes: [video()], outputName: 'out' }],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Channel', {
      EncoderSettings: Match.objectLike({
        OutputGroups: Match.arrayWith([
          Match.objectLike({
            OutputGroupSettings: {
              HlsGroupSettings: Match.objectLike({
                KeyProviderSettings: {
                  StaticKeySettings: Match.objectLike({
                    StaticKeyValue: '11111111111111111111111111111111',
                    KeyProviderServer: { Uri: 'https://keys.example.com/key' },
                  }),
                },
              }),
            },
          }),
        ]),
      }),
    });
  });
});
