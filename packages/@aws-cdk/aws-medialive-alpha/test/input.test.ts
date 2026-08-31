import * as mediaconnect from '@aws-cdk/aws-mediaconnect-alpha';
import { App, Stack, Duration } from 'aws-cdk-lib';import { Template, Match } from 'aws-cdk-lib/assertions';
import { Vpc } from 'aws-cdk-lib/aws-ec2';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import { StringParameter } from 'aws-cdk-lib/aws-ssm';
import { Input, InputConfiguration, InputSource, MulticastProtocol, InputNetworkLocation, SrtDecryptionAlgorithm } from '../lib/input';
import { InputSecurityGroup } from '../lib/input-security-group';
import { Network } from '../lib/network';
import { SdiSource } from '../lib/sdi-source';

let app: App;
let stack: Stack;
let defaultSg: InputSecurityGroup;
beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
  defaultSg = new InputSecurityGroup(stack, 'DefaultSG', {
    allowlistRules: ['0.0.0.0/0'],
  });
});

describe('URL Pull input', () => {
  test('creates input with single source', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-url-pull',
      input: InputConfiguration.urlPull([InputSource.url('https://example.com/stream.m3u8')]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Name: 'my-url-pull',
      Type: 'URL_PULL',
      Sources: [{ Url: 'https://example.com/stream.m3u8' }],
    });
  });

  test('creates input with credentials', () => {
    const password = StringParameter.fromStringParameterName(stack, 'Param', 'my-secret');
    new Input(stack, 'MyInput', {
      inputName: 'my-url-pull',
      input: InputConfiguration.urlPull([
        InputSource.url('https://example.com/stream.m3u8', { username: 'user', password }),
      ]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'URL_PULL',
      Sources: [Match.objectLike({
        Url: 'https://example.com/stream.m3u8',
        Username: 'user',
        PasswordParam: 'my-secret',
      })],
    });
  });

  test('creates input with redundant sources', () => {
    new Input(stack, 'MyInput', {
      inputName: 'redundant-pull',
      input: InputConfiguration.urlPull([
        InputSource.url('https://primary.example.com/stream.m3u8'),
        InputSource.url('https://backup.example.com/stream.m3u8'),
      ]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'URL_PULL',
      Sources: [
        { Url: 'https://primary.example.com/stream.m3u8' },
        { Url: 'https://backup.example.com/stream.m3u8' },
      ],
    });
  });

  test('throws when more than 2 sources', () => {
    expect(() => {
      InputConfiguration.urlPull([
        InputSource.url('https://source-a.example.com/1'),
        InputSource.url('https://source-b.example.com/2'),
        InputSource.url('https://source-c.example.com/3'),
      ]);
    }).toThrow(/You cannot specify more than 2 input sources/);
  });

  test('throws when no sources', () => {
    expect(() => {
      InputConfiguration.urlPull([]);
    }).toThrow(/You must specify at least 1 input source/);
  });
});

describe('RTMP Pull input', () => {
  test('creates input with correct type', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-rtmp-pull',
      input: InputConfiguration.rtmpPull([InputSource.url('rtmp://example.com/live/stream')]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'RTMP_PULL',
      Sources: [{ Url: 'rtmp://example.com/live/stream' }],
    });
  });
});

describe('MP4 File input', () => {
  test('creates input with correct type', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-mp4',
      input: InputConfiguration.mp4File([InputSource.url('s3://my-bucket/video.mp4')]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MP4_FILE',
      Sources: [{ Url: 's3://my-bucket/video.mp4' }],
    });
  });
});

describe('TS File input', () => {
  test('creates input with correct type', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-ts',
      input: InputConfiguration.tsFile([InputSource.url('s3://my-bucket/video.ts')]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'TS_FILE',
      Sources: [{ Url: 's3://my-bucket/video.ts' }],
    });
  });
});

describe('MediaConnect Router input', () => {
  test('creates input with single pipeline', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-router',
      input: InputConfiguration.mediaConnectRouter({
        availabilityZones: ['us-east-1a'],
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MEDIACONNECT_ROUTER',
      RouterSettings: {
        Destinations: [{ AvailabilityZoneName: 'us-east-1a' }],
      },
    });
  });

  test('creates input with redundant pipelines', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-router',
      input: InputConfiguration.mediaConnectRouter({
        availabilityZones: ['us-east-1a', 'us-east-1b'],
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MEDIACONNECT_ROUTER',
      RouterSettings: {
        Destinations: [
          { AvailabilityZoneName: 'us-east-1a' },
          { AvailabilityZoneName: 'us-east-1b' },
        ],
      },
    });
  });
});

describe('MediaConnect Flow input', () => {
  test('creates input with single flow', () => {
    const role = new Role(stack, 'Role', { assumedBy: new ServicePrincipal('medialive.amazonaws.com') });
    new Input(stack, 'MyInput', {
      inputName: 'my-mc-flow',
      input: InputConfiguration.mediaConnect({
        role,
        flows: [
          mediaconnect.Flow.fromFlowArn(stack, 'Flow', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa:my-flow'),
        ],
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MEDIACONNECT',
      MediaConnectFlows: [{
        FlowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa:my-flow',
      }],
    });
  });

  test('a user-provided role receives no managed flow-management grants', () => {
    const role = new Role(stack, 'Role', { assumedBy: new ServicePrincipal('medialive.amazonaws.com') });
    new Input(stack, 'MyInput', {
      inputName: 'my-mc-flow',
      input: InputConfiguration.mediaConnect({
        role,
        flows: [mediaconnect.Flow.fromFlowArn(stack, 'Flow', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa:my-flow')],
      }),
    });

    // The caller owns the role, so the input attaches no policy to it — no managed
    // flow-management actions, nothing.
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  });

  test('auto-creates a role when none is provided', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-mc-flow',
      input: InputConfiguration.mediaConnect({
        flows: [mediaconnect.Flow.fromFlowArn(stack, 'Flow', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa:my-flow')],
      }),
    });

    const template = Template.fromStack(stack);
    // A role is created with the medialive.amazonaws.com service principal...
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'sts:AssumeRole',
            Principal: { Service: 'medialive.amazonaws.com' },
          }),
        ]),
      },
    });
    // ...wired to the input and granted the managed flow-management actions.
    template.hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MEDIACONNECT',
      RoleArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('MediaConnectRole'), 'Arn'] },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['mediaconnect:ManagedAddOutput', 'mediaconnect:ManagedRemoveOutput']),
            Resource: '*',
          }),
        ]),
      },
    });
  });

  test('throws when more than 2 flows', () => {
    const role = new Role(stack, 'Role2', { assumedBy: new ServicePrincipal('medialive.amazonaws.com') });
    expect(() => {
      InputConfiguration.mediaConnect({
        role,
        flows: [
          mediaconnect.Flow.fromFlowArn(stack, 'F1', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-a:f1'),
          mediaconnect.Flow.fromFlowArn(stack, 'F2', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-b:f2'),
          mediaconnect.Flow.fromFlowArn(stack, 'F3', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-c:f3'),
        ],
      });
    }).toThrow(/You must specify 1 or 2 MediaConnect flows/);
  });

  test('throws when no flows', () => {
    const role = new Role(stack, 'Role3', { assumedBy: new ServicePrincipal('medialive.amazonaws.com') });
    expect(() => {
      InputConfiguration.mediaConnect({
        role,
        flows: [],
      });
    }).toThrow(/You must specify 1 or 2 MediaConnect flows/);
  });
});

describe('SRT Caller input', () => {
  test('creates input with basic config', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-srt-caller',
      input: InputConfiguration.srtCaller([{
        srtListenerAddress: '203.0.113.100',
        srtListenerPort: 9000,
        minimumLatency: Duration.millis(1000),
      }]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SRT_CALLER',
      SrtSettings: {
        SrtCallerSources: [Match.objectLike({
          SrtListenerAddress: '203.0.113.100',
          SrtListenerPort: '9000',
          MinimumLatency: 1000,
        })],
      },
    });
  });

  test('creates input with decryption', () => {
    const secret = Secret.fromSecretCompleteArn(stack, 'Secret', 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret');
    new Input(stack, 'MyInput', {
      inputName: 'my-srt-encrypted',
      input: InputConfiguration.srtCaller([{
        srtListenerAddress: '203.0.113.100',
        srtListenerPort: 9000,
        decryption: {
          algorithm: SrtDecryptionAlgorithm.AES256,
          passphraseSecret: secret,
        },
      }]),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SRT_CALLER',
      SrtSettings: {
        SrtCallerSources: [Match.objectLike({
          Decryption: {
            Algorithm: 'AES256',
            PassphraseSecretArn: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret',
          },
        })],
      },
    });
  });

  test('throws when more than 2 sources', () => {
    expect(() => {
      InputConfiguration.srtCaller([
        { srtListenerAddress: '198.51.100.1', srtListenerPort: 9000 },
        { srtListenerAddress: '198.51.100.2', srtListenerPort: 9001 },
        { srtListenerAddress: '198.51.100.3', srtListenerPort: 9002 },
      ]);
    }).toThrow(/You must specify 1 or 2 SRT caller sources/);
  });
});

describe('SRT Listener input', () => {
  test('creates input with defaults', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-srt-listener',
      input: InputConfiguration.srtListener({
        inputSecurityGroups: [defaultSg],
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SRT_LISTENER',
    });
  });

  test('creates input with decryption and latency', () => {
    const secret = Secret.fromSecretCompleteArn(stack, 'Secret', 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret');
    new Input(stack, 'MyInput', {
      inputName: 'my-srt-listener',
      input: InputConfiguration.srtListener({
        inputSecurityGroups: [defaultSg],
        minimumLatency: Duration.millis(500),
        decryption: {
          algorithm: SrtDecryptionAlgorithm.AES128,
          passphraseSecret: secret,
        },
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SRT_LISTENER',
      SrtSettings: {
        SrtListenerSettings: {
          MinimumLatency: 500,
          Decryption: {
            Algorithm: 'AES128',
            PassphraseSecretArn: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret',
          },
        },
      },
    });
  });

  test('creates input with a stream ID', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-srt-listener',
      input: InputConfiguration.srtListener({
        inputSecurityGroups: [defaultSg],
        streamId: 'my-stream-id',
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SRT_LISTENER',
      SrtSettings: {
        SrtListenerSettings: Match.objectLike({ StreamId: 'my-stream-id' }),
      },
    });
  });
});

describe('MediaLive Anywhere input networking', () => {
  test('input network location ON_PREMISES with push-destination networking', () => {
    const network = new Network(stack, 'Net', {
      networkName: 'on-prem-net',
      ipPools: ['192.168.1.0/24'],
    });

    new Input(stack, 'MyInput', {
      inputName: 'on-prem-rtp',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.rtpPush({
        destinations: [{
          network,
          networkRoutes: [{ cidr: '10.0.0.0/24', gateway: '10.0.0.1' }],
          staticIpAddress: '192.168.1.50',
        }],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'RTP_PUSH',
      InputNetworkLocation: 'ON_PREMISES',
      Destinations: [Match.objectLike({
        Network: { Ref: Match.stringLikeRegexp('Net') },
        NetworkRoutes: [{ Cidr: '10.0.0.0/24', Gateway: '10.0.0.1' }],
        StaticIpAddress: '192.168.1.50',
      })],
    });
  });

  test('on-premises input rejects input security groups', () => {
    expect(() => new Input(stack, 'MyInput', {
      inputName: 'on-prem-bad',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.rtpPush({
        inputSecurityGroups: [defaultSg],
        destinations: [{ staticIpAddress: '192.168.1.50' }],
      }),
    })).toThrow(/on-premises inputs do not support input security groups/);
  });

  test('cloud push input requires at least one security group', () => {
    expect(() => new Input(stack, 'MyInput', {
      inputName: 'cloud-bad',
      input: InputConfiguration.rtpPush({
        destinations: [{}],
      }),
    })).toThrow(/cloud push inputs require at least one input security group/);
  });

  test('UDP push destinations are rendered', () => {
    new Input(stack, 'MyInput', {
      inputName: 'udp-with-dest',
      input: InputConfiguration.udpPush({
        inputSecurityGroups: [defaultSg],
        destinations: [{ staticIpAddress: '192.168.1.51' }],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'UDP_PUSH',
      Destinations: [Match.objectLike({ StaticIpAddress: '192.168.1.51' })],
    });
  });
});

describe('UDP push input', () => {
  test('creates input with correct type and security group', () => {
    new Input(stack, 'MyInput', {
      inputName: 'my-udp-push',
      input: InputConfiguration.udpPush({ inputSecurityGroups: [defaultSg] }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Name: 'my-udp-push',
      Type: 'UDP_PUSH',
      InputSecurityGroups: Match.anyValue(),
    });
  });

  test('single destination is SINGLE_PIPELINE', () => {
    const input = new Input(stack, 'MyInput', {
      inputName: 'udp-single',
      input: InputConfiguration.udpPush({
        inputSecurityGroups: [defaultSg],
        destinations: [{}],
      }),
    });
    expect(input.inputClass).toBe('SINGLE_PIPELINE');
  });

  test('two destinations is STANDARD', () => {
    const input = new Input(stack, 'MyInput', {
      inputName: 'udp-standard',
      input: InputConfiguration.udpPush({
        inputSecurityGroups: [defaultSg],
        destinations: [{}, {}],
      }),
    });
    expect(input.inputClass).toBe('STANDARD');
  });
});

describe('Input import', () => {
  test('imports from arn', () => {
    const imported = Input.fromInputArn(stack, 'Imported', 'arn:aws:medialive:us-east-1:123456789012:input:1234567');
    expect(imported.inputArn).toBe('arn:aws:medialive:us-east-1:123456789012:input:1234567');
    expect(imported.inputId).toBe('1234567');
  });
});

describe('Input name auto-generation', () => {
  test('auto-generates name when not provided', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.srtListener({
        inputSecurityGroups: [defaultSg],
      }),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Name: Match.anyValue(),
      Type: 'SRT_LISTENER',
    });
  });
});

describe('Elemental Link (input device) input', () => {
  test('creates input with device IDs', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.inputDevice({ deviceIds: ['hd-0001', 'hd-0002'] }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'INPUT_DEVICE',
      InputDevices: [{ Id: 'hd-0001' }, { Id: 'hd-0002' }],
    });
  });

  test('single device is SINGLE_PIPELINE', () => {
    const input = new Input(stack, 'MyInput', {
      input: InputConfiguration.inputDevice({ deviceIds: ['hd-0001'] }),
    });
    expect(input.inputClass).toBe('SINGLE_PIPELINE');
  });

  test.each([[[] as string[]], [['a', 'b', 'c']]])('fails for invalid device count %p', (ids) => {
    expect(() => InputConfiguration.inputDevice({ deviceIds: ids })).toThrow(/1 or 2 input devices/);
  });
});

describe('CDI input', () => {
  let vpc: Vpc;
  let role: Role;
  beforeEach(() => {
    vpc = new Vpc(stack, 'Vpc', { maxAzs: 2 });
    role = new Role(stack, 'CdiRole', { assumedBy: new ServicePrincipal('medialive.amazonaws.com') });
  });

  test('creates a VPC input with role and subnets', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.cdi({ subnets: vpc.privateSubnets, role }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'AWS_CDI',
      RoleArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('CdiRole'), 'Arn'] },
      Vpc: { SubnetIds: Match.anyValue() },
    });
  });

  test('a user-provided role receives no EC2 ENI grants', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.cdi({ subnets: vpc.privateSubnets, role }),
    });

    // The caller owns the role, so the input attaches no EC2 ENI policy to it.
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  });

  test('fails when not exactly 2 subnets', () => {
    expect(() => InputConfiguration.cdi({ subnets: [vpc.privateSubnets[0]], role }))
      .toThrow(/exactly 2 subnets/);
  });

  test('auto-creates a role when none is provided', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.cdi({ subnets: vpc.privateSubnets }),
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'sts:AssumeRole',
            Principal: { Service: 'medialive.amazonaws.com' },
          }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'AWS_CDI',
      RoleArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('CdiRole'), 'Arn'] },
    });
    // The auto-created role is granted the EC2 ENI actions it needs.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['ec2:CreateNetworkInterface', 'ec2:DeleteNetworkInterface']),
          }),
          Match.objectLike({
            Action: Match.arrayWith(['ec2:DescribeNetworkInterfaces']),
            Resource: '*',
          }),
        ]),
      },
    });
  });
});

describe('Multicast input', () => {
  test('creates input with multicast sources', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.multicast({
        sources: [
          { address: '239.0.0.1', port: 5000 },
          { address: '239.0.0.2', port: 5000, sourceIp: '10.0.0.5' },
        ],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'MULTICAST',
      MulticastSettings: {
        Sources: [
          { Url: 'udp://239.0.0.1:5000' },
          { Url: 'udp://239.0.0.2:5000', SourceIp: '10.0.0.5' },
        ],
      },
    });
  });

  test('single source is SINGLE_PIPELINE', () => {
    const input = new Input(stack, 'MyInput', {
      input: InputConfiguration.multicast({ sources: [{ address: '239.0.0.1', port: 5000 }] }),
    });
    expect(input.inputClass).toBe('SINGLE_PIPELINE');
  });

  test.each([
    [[] as { address: string; port: number }[]],
    [[{ address: 'a', port: 1 }, { address: 'b', port: 2 }, { address: 'c', port: 3 }]],
  ])('fails for invalid source count %p', (sources) => {
    expect(() => InputConfiguration.multicast({ sources })).toThrow(/1 or 2 multicast sources/);
  });

  test('RTP protocol builds an rtp:// source URL', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.multicast({
        sources: [{ address: '233.252.0.0', port: 20000, protocol: MulticastProtocol.RTP }],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      MulticastSettings: { Sources: [{ Url: 'rtp://233.252.0.0:20000' }] },
    });
  });
});

describe('SDI input', () => {
  test('renders the SDI source ID, not the ARN', () => {
    // MediaLive requires the numeric SDI source ID, not the ARN.
    const source = SdiSource.fromSdiSourceAttributes(stack, 'ImportedSdi', {
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:1234567',
      sdiSourceId: '1234567',
    });

    new Input(stack, 'MyInput', {
      inputName: 'sdi-input',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.sdi([source]),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SDI',
      SdiSources: ['1234567'],
    });
  });

  test('two sources produce a STANDARD input class', () => {
    const source1 = SdiSource.fromSdiSourceAttributes(stack, 'ImportedSdi1', {
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:1234567',
      sdiSourceId: '1234567',
    });
    const source2 = SdiSource.fromSdiSourceAttributes(stack, 'ImportedSdi2', {
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:7654321',
      sdiSourceId: '7654321',
    });

    const input = new Input(stack, 'MyInput', {
      inputName: 'sdi-input-standard',
      inputNetworkLocation: InputNetworkLocation.ON_PREMISES,
      input: InputConfiguration.sdi([source1, source2]),
    });

    expect(input.inputClass).toBe('STANDARD');
    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      SdiSources: ['1234567', '7654321'],
    });
  });

  test('fails when inputNetworkLocation is not ON_PREMISES', () => {
    const source = SdiSource.fromSdiSourceAttributes(stack, 'ImportedSdi', {
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:1234567',
      sdiSourceId: '1234567',
    });

    expect(() => new Input(stack, 'MyInput', {
      inputName: 'sdi-input-bad',
      input: InputConfiguration.sdi([source]),
    })).toThrow(/ON_PREMISES/);
  });
});

describe('SMPTE 2110 receiver group input', () => {
  test('creates input with a single receiver group of SDPs', () => {
    new Input(stack, 'MyInput', {
      input: InputConfiguration.smpte2110ReceiverGroup({
        videoSdp: { sdpUrl: 'https://example.com/video.sdp', mediaIndex: 0 },
        audioSdps: [{ sdpUrl: 'https://example.com/audio.sdp' }],
        ancillarySdps: [{ sdpUrl: 'https://example.com/anc.sdp' }],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Type: 'SMPTE_2110_RECEIVER_GROUP',
      Smpte2110ReceiverGroupSettings: {
        Smpte2110ReceiverGroups: [{
          SdpSettings: {
            VideoSdp: { SdpUrl: 'https://example.com/video.sdp', MediaIndex: 0 },
            AudioSdps: [{ SdpUrl: 'https://example.com/audio.sdp' }],
            AncillarySdps: [{ SdpUrl: 'https://example.com/anc.sdp' }],
          },
        }],
      },
    });
  });

  test('fails for more than 50 audio SDPs', () => {
    const audioSdps = Array.from({ length: 51 }, (_, i) => ({ sdpUrl: `https://example.com/a${i}.sdp` }));
    expect(() => InputConfiguration.smpte2110ReceiverGroup({ audioSdps }))
      .toThrow(/at most 50 audio SDPs/);
  });
});

describe('Input tags', () => {
  test('applies tags to the input', () => {
    new Input(stack, 'MyInput', {
      inputName: 'tagged-input',
      input: InputConfiguration.urlPull([InputSource.url('https://example.com/stream.m3u8')]),
      tags: { team: 'video', env: 'prod' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Input', {
      Tags: { team: 'video', env: 'prod' },
    });
  });
});
