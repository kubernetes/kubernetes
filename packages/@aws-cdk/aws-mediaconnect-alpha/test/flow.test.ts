import { App, Bitrate, Duration, Stack } from 'aws-cdk-lib';
import { Annotations, Match, Template } from 'aws-cdk-lib/assertions';
import { Vpc, IpAddresses, SubnetType, PrivateSubnet, SecurityGroup } from 'aws-cdk-lib/aws-ec2';
import { Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Bridge, BridgeType } from '../lib/bridge';
import { AudioStreamOrderOptions, Colorimetry, EncodingProfile, FailoverConfig, Flow, FlowSize, MediaStream, MediaVideoFormat, ScanMode, Tcs, VideoRange } from '../lib/flow';
import { FlowEntitlement } from '../lib/flow-entitlement';
import { OutputConfiguration } from '../lib/flow-output';
import { FlowSource } from '../lib/flow-source';
import { Encoding, NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { EncryptionAlgorithm, FailoverMode, Framerate, MaintenanceDay, NetworkInterface, PixelAspectRatio, State, VpcInterface } from '../lib/shared';

let app: App;
let stack: Stack;
beforeEach(() => {
  app = new App();
  stack = new Stack(app, undefined, {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

test('MediaConnect flow with Entitlement creation', () => {
  const entitlement = FlowEntitlement.fromFlowEntitlementArn(stack, 'flow-entitlement',
    'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-3333cccc4444dddd-1111aaaa2222:ExampleCorp',
  );

  new Flow(stack, 'flow', {
    source: SourceConfiguration.entitlement({
      entitlement,
      decryption: ({
        role: Role.fromRoleName(stack, 'importedRole', 'importedRole'),
        secret: Secret.fromSecretNameV2(stack, 'secret', 'secret'),
        algorithm: EncryptionAlgorithm.AES128,
      }),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'flow',
    Source:
      {
        Decryption: {
          Algorithm: 'aes128',
          KeyType: 'static-key',
          RoleArn: {
            'Fn::Join': ['', [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':iam::123456789012:role/importedRole',
            ]],
          },
          SecretArn: {
            'Fn::Join': ['', [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':secretsmanager:us-east-1:123456789012:secret:secret',
            ]],
          },
        },
        EntitlementArn: 'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-3333cccc4444dddd-1111aaaa2222:ExampleCorp',
      },
  });
});

test('MediaConnect flow with SRT Caller source', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.srtCaller({
      flowSourceName: 'aaa',
      maxLatency: Duration.seconds(1),
      sourceListenerAddress: '203.0.113.10',
      sourceListenerPort: 5000,
      decryption: ({
        role: Role.fromRoleName(stack, 'importedRole', 'importedRole'),
        secret: Secret.fromSecretNameV2(stack, 'secret', 'secret'),
      }),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'flow',
    Source: {
      Decryption: {
        KeyType: 'srt-password',
        RoleArn: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':iam::123456789012:role/importedRole',
          ]],
        },
        SecretArn: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':secretsmanager:us-east-1:123456789012:secret:secret',
          ]],
        },
      },
      MaxLatency: 1000,
      Name: 'aaa',
      Protocol: 'srt-caller',
      SourceListenerAddress: '203.0.113.10',
      SourceListenerPort: 5000,
    },
  });
});

test('MediaConnect flow with rtp-fec creation', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.rtpFec({
      flowSourceName: 'my-flow',
      network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
      description: 'my test flow',
      maxBitrate: Bitrate.mbps(10),
      port: 2099,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'flow',
    Source:
      {
        Description: 'my test flow',
        MaxBitrate: 10000000,
        Name: 'my-flow',
        Protocol: 'rtp-fec',
        IngestPort: 2099,
        WhitelistCidr: '10.1.0.0/16',
      },
  });
});

test('MediaConnect flow with rtp-fec with VPC config', () => {
  const vpc = new Vpc(stack, 'testvpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    subnetConfiguration: [{
      name: 'subnet1',
      subnetType: SubnetType.PRIVATE_ISOLATED,
      cidrMask: 24,
    }],
  });
  const subnet = new PrivateSubnet(stack, 'mysubnet', {
    availabilityZone: `${stack.region}a`,
    cidrBlock: '10.0.172.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'my-role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const vpcInterface = VpcInterface.define({
    vpcInterfaceName: 'test',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', {
      vpc,
    })],
    subnet,
  });

  new Flow(stack, 'flow', {
    source: SourceConfiguration.rtpFec({
      flowSourceName: 'my-flow',
      network: NetworkConfiguration.vpc(vpcInterface),
      description: 'my test flow',
      maxBitrate: Bitrate.mbps(10),
      port: 2099,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'flow',
    Source:
      {
        Description: 'my test flow',
        MaxBitrate: 10000000,
        Name: 'my-flow',
        Protocol: 'rtp-fec',
        IngestPort: 2099,
        VpcInterfaceName: 'test',
      },
  });
});

test('MediaConnect CDI default settings', () => {
  const vpc = new Vpc(stack, 'testvpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    subnetConfiguration: [{
      name: 'subnet1',
      subnetType: SubnetType.PRIVATE_ISOLATED,
      cidrMask: 24,
    }],
  });
  const subnet = new PrivateSubnet(stack, 'mysubnet', {
    availabilityZone: `${stack.region}a`,
    cidrBlock: '10.0.172.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'my-role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    inlinePolicies: {
      default: new PolicyDocument({
        statements: [
          new PolicyStatement({
            actions: ['ec2:CreateNetworkInterface', 'ec2:CreateNetworkInterfacePermission', 'ec2:DeleteNetworkInterface', 'ec2:DescribeSubnets'],
            resources: ['*'],
            effect: Effect.ALLOW,
          }),
        ],
      }),
    },
  });

  const vpcInterface = VpcInterface.define({
    vpcInterfaceName: 'myvpcinterface',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', {
      vpc,
    })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'mystream',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      colorimetry: Colorimetry.BT709,
      exactFramerate: Framerate.FPS_59_94,
      par: PixelAspectRatio.SQUARE,
      scanMode: ScanMode.PROGRESSIVE,
      videoRange: VideoRange.NARROW,
      tcs: Tcs.SDR,
    },
  });

  new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE_4X,
    vpcInterfaces: [vpcInterface],
    mediaStreams: [
      mediaStream,
    ],
    source: SourceConfiguration.cdi({
      flowSourceName: 'my-flow',
      maxSyncBuffer: 100,
      port: 5000,
      vpcInterface: vpcInterface,
      mediaStreamSourceConfigurations: [{
        encoding: Encoding.RAW,
        mediaStream,
      }],
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'flow',
    Source:
      {
        Name: 'my-flow',
        Protocol: 'cdi',
        IngestPort: 5000,
        MaxSyncBuffer: 100,
        VpcInterfaceName: 'myvpcinterface',
      },
  });
});

test('fails - CDI default settings with invalid port', () => {
  const vpc = new Vpc(stack, 'testvpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    subnetConfiguration: [{
      name: 'subnet1',
      subnetType: SubnetType.PRIVATE_ISOLATED,
      cidrMask: 24,
    }],
  });
  const subnet = new PrivateSubnet(stack, 'mysubnet', {
    availabilityZone: `${stack.region}a`,
    cidrBlock: '10.0.172.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'my-role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    inlinePolicies: {
      default: new PolicyDocument({
        statements: [
          new PolicyStatement({
            actions: ['ec2:CreateNetworkInterface', 'ec2:CreateNetworkInterfacePermission', 'ec2:DeleteNetworkInterface', 'ec2:DescribeSubnets'],
            resources: ['*'],
            effect: Effect.ALLOW,
          }),
        ],
      }),
    },
  });

  expect(() => {
    const mediaStream = MediaStream.video({
      mediaStreamId: 1,
      mediaStreamName: 'mystream',
      videoFormat: MediaVideoFormat.HD_1080P,
      fmtp: {
        colorimetry: Colorimetry.BT709,
        exactFramerate: Framerate.FPS_59_94,
        par: PixelAspectRatio.SQUARE,
        scanMode: ScanMode.PROGRESSIVE,
        videoRange: VideoRange.NARROW,
        tcs: Tcs.SDR,
      },
    });
    const vpcInterface = VpcInterface.define({
      vpcInterfaceName: 'myvpcinterface',
      role,
      securityGroups: [new SecurityGroup(stack, 'sg', {
        vpc,
      })],
      subnet,
      networkInterfaceType: NetworkInterface.EFA,
    });
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [vpcInterface],
      mediaStreams: [
        mediaStream,
      ],
      source: SourceConfiguration.cdi({
        flowSourceName: 'my-flow',
        maxSyncBuffer: 100,
        vpcInterface,
        port: 500000,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.RAW,
          mediaStream: mediaStream,
        }],
      }),
    });
  }).toThrow('Ingest port must be between 1024 and 65535');
});

test('MediaConnect flow with gateway bridge source and VPC interface', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    subnetConfiguration: [{
      cidrMask: 24,
      name: 'public',
      subnetType: SubnetType.PUBLIC,
    }],
  });

  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    inlinePolicies: {
      policy: new PolicyDocument({
        statements: [
          new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['ec2:*'],
            resources: ['*'],
          }),
        ],
      }),
    },
  });

  const securityGroup = new SecurityGroup(stack, 'sg', {
    vpc,
  });

  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.172.0/24',
    vpcId: vpc.vpcId,
  });

  const vpcInterface = VpcInterface.define({
    vpcInterfaceName: 'bridge-interface',
    role: role,
    securityGroups: [securityGroup],
    subnet: subnet,
  });

  const bridge = Bridge.fromBridgeAttributes(stack, 'bridge', {
    bridgeArn: 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-CAtfUwcADQRTWlYH-ddb931dc29f3:test-bridge',
    bridgeType: BridgeType.INGRESS_BRIDGE,
  });

  const flow = new Flow(stack, 'flow', {
    vpcInterfaces: [vpcInterface],
    source: SourceConfiguration.gatewayBridge({
      bridge: bridge,
      vpcInterface: vpcInterface,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      GatewayBridgeSource: {
        BridgeArn: 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-CAtfUwcADQRTWlYH-ddb931dc29f3:test-bridge',
        VpcInterfaceAttachment: {
          VpcInterfaceName: 'bridge-interface',
        },
      },
    },
    VpcInterfaces: [{
      Name: 'bridge-interface',
      RoleArn: {
        'Fn::GetAtt': ['roleC7B7E775', 'Arn'],
      },
      SecurityGroupIds: [{
        'Fn::GetAtt': ['sg29196201', 'GroupId'],
      }],
      SubnetId: {
        Ref: 'subnetSubnet39D20FD5',
      },
    }],
  });

  expect(flow.flowArn).toBeDefined();
});

test('MediaConnect flow with VPC interfaces and entitlement source', () => {
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    inlinePolicies: {
      policy: new PolicyDocument({
        statements: [
          new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['ec2:CreateNetworkInterface', 'ec2:DeleteNetworkInterface', 'ec2:DescribeSubnets'],
            resources: ['*'],
          }),
        ],
      }),
    },
  });

  const entitlement = FlowEntitlement.fromFlowEntitlementArn(stack, 'imported-entitlement',
    'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-3333cccc4444dddd-1111aaaa2222:ExampleEntitlement',
  );

  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.entitlement({
      entitlement: entitlement,
      decryption: ({
        role,
        secret: Secret.fromSecretNameV2(stack, 'secret', 'my-encryption-secret'),
        algorithm: EncryptionAlgorithm.AES256,
      }),
    }),
  });

  const template = Template.fromStack(stack);

  // Check that the flow has the correct entitlement source
  template.hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      EntitlementArn: 'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-3333cccc4444dddd-1111aaaa2222:ExampleEntitlement',
      Decryption: {
        Algorithm: 'aes256',
        KeyType: 'static-key',
      },
    },
  });

  // Entitlement sources don't require VPC interfaces

  expect(flow.flowArn).toBeDefined();
  expect(flow.sourceArn).toBeDefined();
});

test('MediaConnect flow with flowoutput on helper method', () => {
  const videoMediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video-stream',
    videoFormat: MediaVideoFormat.UHD_2160P,
    clockRate: 90000,
    fmtp: {
      exactFramerate: Framerate.FPS_59_94,
      par: PixelAspectRatio.SQUARE,
      colorimetry: Colorimetry.BT2020,
      videoRange: VideoRange.FULL,
      scanMode: ScanMode.PROGRESSIVE,
      tcs: Tcs.PQ,
    },
  });
  const flow = new Flow(stack, 'Flow', {
    flowName: 'studio-ingest-flow',
    source: SourceConfiguration.rist({
      flowSourceName: 'primary-source',
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      maxBitrate: Bitrate.mbps(5),
      port: 5000,
      maxLatency: Duration.millis(2000),
    }),
    sourceFailoverConfig: FailoverConfig.merge({
      recoveryWindow: Duration.millis(15000),
    }),
    mediaStreams: [
      videoMediaStream,
    ],
  });

  flow.addOutput('monitoring-output', {
    output: OutputConfiguration.rtp({
      destination: '198.51.100.100',
      port: 5004,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      IngestPort: 5000,
      MaxBitrate: 5000000,
      MaxLatency: 2000,
      Name: 'primary-source',
      Protocol: 'rist',
      WhitelistCidr: '10.0.0.0/16',
    },
    SourceFailoverConfig: {
      FailoverMode: 'MERGE',
      RecoveryWindow: 15000,
      State: 'ENABLED',
    },
    MediaStreams: [{
      Attributes: {
        Fmtp: {
          Colorimetry: 'BT2020',
          Range: 'FULL',
          ScanMode: 'progressive',
          Tcs: 'PQ',
        },
      },
      ClockRate: 90000,
      MediaStreamId: 1,
      MediaStreamName: 'video-stream',
      MediaStreamType: 'video',
      VideoFormat: '2160p',
    }],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::FlowOutput', {
    // Description: 'output-1',
    Destination: '198.51.100.100',
    FlowArn: { 'Fn::GetAtt': ['FlowA74D6E88', 'FlowArn'] },
    Name: 'FlowmonitoringoutputF420EC7A',
    Port: 5004,
    Protocol: 'rtp',
  });

  expect(flow.flowArn).toBeDefined();
});

// Validation tests for JPEG XS
test('JPEG XS validation - throws when not exactly 2 input interfaces', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const vpcInterface = VpcInterface.define({
    vpcInterfaceName: 'interface1',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [vpcInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.jpegXs({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.JXSV,
          port: 5000,
          inputInterface: [vpcInterface], // Only 1 interface instead of 2
          mediaStream,
        }],
      }),
    });
  }).toThrow(/JPEG XS protocol requires exactly 2 input configurations/);
});

test('JPEG XS validation - throws when ports are not unique', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const efaInterface = VpcInterface.define({
    vpcInterfaceName: 'efa',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg1', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const enaInterface = VpcInterface.define({
    vpcInterfaceName: 'ena',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg2', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA,
  });

  const videoStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  const audioStream = MediaStream.audio({
    mediaStreamId: 2,
    mediaStreamName: 'audio',
    channelOrder: AudioStreamOrderOptions.SURROUND_7_1,
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [efaInterface, enaInterface],
      mediaStreams: [videoStream, audioStream],
      source: SourceConfiguration.jpegXs({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [
          {
            encoding: Encoding.JXSV,
            port: 5000,
            inputInterface: [efaInterface, enaInterface],
            mediaStream: videoStream,
          },
          {
            encoding: Encoding.PCM,
            port: 5000, // Same port as video stream
            inputInterface: [efaInterface, enaInterface],
            mediaStream: audioStream,
          },
        ],
      }),
    });
  }).toThrow(/Each media stream must use a unique port/);
});

test('JPEG XS validation - throws when more than 1 video stream', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const enaInterface1 = VpcInterface.define({
    vpcInterfaceName: 'ena1',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg1', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA,
  });

  const enaInterface2 = VpcInterface.define({
    vpcInterfaceName: 'ena2',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg2', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA,
  });

  const videoStream1 = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video1',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  const videoStream2 = MediaStream.video({
    mediaStreamId: 2,
    mediaStreamName: 'video2',
    videoFormat: MediaVideoFormat.HD_720P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [enaInterface1, enaInterface2],
      mediaStreams: [videoStream1, videoStream2],
      source: SourceConfiguration.jpegXs({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [
          {
            encoding: Encoding.JXSV,
            port: 5000,
            inputInterface: [enaInterface1, enaInterface2],
            mediaStream: videoStream1,
          },
          {
            encoding: Encoding.JXSV,
            port: 5001,
            inputInterface: [enaInterface1, enaInterface2],
            mediaStream: videoStream2,
          },
        ],
      }),
    });
  }).toThrow(/A source may reference at most one video media stream/);
});

test('JPEG XS validation - throws when flow size is not LARGE', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const efaInterface = VpcInterface.define({
    vpcInterfaceName: 'efa',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg1', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const enaInterface = VpcInterface.define({
    vpcInterfaceName: 'ena',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg2', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA,
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.MEDIUM, // Wrong size
      vpcInterfaces: [efaInterface, enaInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.jpegXs({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.JXSV,
          port: 5000,
          inputInterface: [efaInterface, enaInterface],
          mediaStream,
        }],
      }),
    });
  }).toThrow(/JPEG XS and CDI protocols require LARGE_4X flow size/);
});

test('JPEG XS validation - throws when 2 input interfaces are the same', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const enaInterface = VpcInterface.define({
    vpcInterfaceName: 'ena',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA,
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [enaInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.jpegXs({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.JXSV,
          port: 5000,
          inputInterface: [enaInterface, enaInterface], // Same interface twice
          mediaStream,
        }],
      }),
    });
  }).toThrow(/The 2 input interfaces.*must be different VPC interfaces/);
});

// Validation tests for CDI
test('CDI validation - throws when VPC interface is not EFA', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const enaInterface = VpcInterface.define({
    vpcInterfaceName: 'ena',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.ENA, // Wrong type
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      vpcInterfaces: [enaInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.cdi({
        flowSourceName: 'source',
        maxSyncBuffer: 100,
        port: 6000,
        vpcInterface: enaInterface,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.RAW,
          mediaStream,
        }],
      }),
    });
  }).toThrow(/CDI protocol requires an EFA network interface type/);
});

test('CDI validation - throws when flow size is not LARGE', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const efaInterface = VpcInterface.define({
    vpcInterfaceName: 'efa',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.MEDIUM, // Wrong size
      vpcInterfaces: [efaInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.cdi({
        maxSyncBuffer: 100,
        port: 6000,
        flowSourceName: 'source',
        vpcInterface: efaInterface,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.RAW,
          mediaStream,
        }],
      }),
    });
  }).toThrow(/JPEG XS and CDI protocols require LARGE_4X flow size/);
});
test('Flow validation - throws when more than 1 EFA interface', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
  });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const efaInterface1 = VpcInterface.define({
    vpcInterfaceName: 'efa1',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg1', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  const efaInterface2 = VpcInterface.define({
    vpcInterfaceName: 'efa2',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg2', { vpc })],
    subnet,
    networkInterfaceType: NetworkInterface.EFA,
  });

  expect(() => {
    new Flow(stack, 'flow', {
      vpcInterfaces: [efaInterface1, efaInterface2], // 2 EFA interfaces
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
        port: 5000,
      }),
    });
  }).toThrow(/A flow can have a maximum of 1 EFA VPC interface/);
});

// Validation tests
test('Flow name validation - too long', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowName: 'a'.repeat(65),
      source: SourceConfiguration.rtpFec({
        flowSourceName: 'source',
        maxBitrate: Bitrate.mbps(10),
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow('Flow name must be between 1 and 64 characters');
});

test('Flow name validation - invalid characters', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowName: 'invalid@name!',
      source: SourceConfiguration.rtpFec({
        flowSourceName: 'source',
        maxBitrate: Bitrate.mbps(10),
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow('Flow name must contain only alphanumeric characters, hyphens, and underscores');
});

test('Flow name validation - valid name', () => {
  new Flow(stack, 'flow', {
    flowName: 'Valid-Flow_Name123',
    source: SourceConfiguration.rtpFec({
      flowSourceName: 'source',
      maxBitrate: Bitrate.mbps(10),
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Name: 'Valid-Flow_Name123',
  });
});

test('Flow with maintenance window', () => {
  new Flow(stack, 'flow', {
    flowName: 'maintenance-flow',
    maintenanceConfiguration: {
      day: MaintenanceDay.MONDAY,
      time: '02:00',
    },
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Maintenance: {
      MaintenanceDay: 'Monday',
      MaintenanceStartHour: '02:00',
    },
  });
});

test('Flow maintenance time validation - invalid format', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      maintenanceConfiguration: {
        day: MaintenanceDay.TUESDAY,
        time: '02:30',
      },
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/Maintenance time must be in HH:00 format/);
});

test('Flow recovery window validation - out of range', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      sourceFailoverConfig: FailoverConfig.merge({
        recoveryWindow: Duration.millis(50),
      }),
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/Recovery window must be between 100 and 15000 ms/);
});

test('Flow recovery window validation - too large', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      sourceFailoverConfig: FailoverConfig.merge({
        recoveryWindow: Duration.millis(20000),
      }),
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/Recovery window must be between 100 and 15000 ms/);
});

test('FailoverConfig.failover defaults to ENABLED state', () => {
  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
    sourceFailoverConfig: FailoverConfig.failover({ primarySource: 'source' }),
  });

  expect(flow.isFailoverEnabled).toBe(true);
  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    SourceFailoverConfig: {
      FailoverMode: 'FAILOVER',
      State: 'ENABLED',
      SourcePriority: { PrimarySource: 'source' },
    },
  });
});

test('FailoverConfig.failover with state: DISABLED persists config without enabling', () => {
  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
    sourceFailoverConfig: FailoverConfig.failover({
      state: State.DISABLED,
      primarySource: 'source',
    }),
  });

  expect(flow.isFailoverEnabled).toBe(false);
  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    SourceFailoverConfig: {
      FailoverMode: 'FAILOVER',
      State: 'DISABLED',
    },
  });
});

test('FailoverConfig.merge with state: DISABLED persists recovery window', () => {
  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
    sourceFailoverConfig: FailoverConfig.merge({
      state: State.DISABLED,
      recoveryWindow: Duration.millis(500),
    }),
  });

  expect(flow.isFailoverEnabled).toBe(false);
  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    SourceFailoverConfig: {
      FailoverMode: 'MERGE',
      State: 'DISABLED',
      RecoveryWindow: 500,
    },
  });
});

test('Flow with source monitoring config', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
    sourceMonitoringConfig: {
      thumbnailState: State.ENABLED,
      contentQualityAnalysisState: State.ENABLED,
      silentAudio: {
        state: State.ENABLED,
        threshold: Duration.seconds(15),
      },
      blackFrames: {
        state: State.ENABLED,
        threshold: Duration.seconds(15),
      },
      frozenFrames: {
        state: State.ENABLED,
        threshold: Duration.seconds(20),
      },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    SourceMonitoringConfig: {
      ThumbnailState: 'ENABLED',
      ContentQualityAnalysisState: 'ENABLED',
      AudioMonitoringSettings: [{
        SilentAudio: {
          State: 'ENABLED',
          ThresholdSeconds: 15,
        },
      }],
      VideoMonitoringSettings: [{
        BlackFrames: {
          State: 'ENABLED',
          ThresholdSeconds: 15,
        },
        FrozenFrames: {
          State: 'ENABLED',
          ThresholdSeconds: 20,
        },
      }],
    },
  });
});

test('fails - monitoring metrics require contentQualityAnalysisState ENABLED', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
      sourceMonitoringConfig: {
        silentAudio: { state: State.ENABLED, threshold: Duration.seconds(30) },
      },
    });
  }).toThrow(/require contentQualityAnalysisState to be State\.ENABLED/);
});

test.each([
  ['silentAudio', 'Silent audio threshold'],
  ['blackFrames', 'Black frames threshold'],
  ['frozenFrames', 'Frozen frames threshold'],
] as const)('fails - %s threshold out of range', (metric, message) => {
  expect(() => {
    new Flow(stack, 'flow', {
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
      sourceMonitoringConfig: {
        contentQualityAnalysisState: State.ENABLED,
        [metric]: { state: State.ENABLED, threshold: Duration.seconds(5) },
      },
    });
  }).toThrow(new RegExp(`${message} must be between 10 and 60 seconds`));
});

test('Flow with NDI config', () => {
  new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    ndiConfig: {
      ndiState: State.ENABLED,
      machineName: 'test-machine',
    },
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    FlowSize: 'LARGE',
    NdiConfig: {
      NdiState: 'ENABLED',
      MachineName: 'test-machine',
    },
  });
});

test('Flow with NDI source', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    maxAzs: 1,
    subnetConfiguration: [{ name: 'private', subnetType: SubnetType.PRIVATE_ISOLATED }],
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });
  const sg = new SecurityGroup(stack, 'sg', { vpc });
  const vpcInterface = VpcInterface.define({
    vpcInterfaceName: 'ndi-vpc',
    role,
    securityGroups: [sg],
    subnet: vpc.isolatedSubnets[0],
  });

  new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    vpcInterfaces: [vpcInterface],
    ndiConfig: {
      ndiState: State.ENABLED,
      ndiDiscoveryServers: [{
        discoveryServerAddress: '10.0.0.10',
        vpcInterface,
      }],
    },
    encodingConfig: {
      encodingProfile: EncodingProfile.CONTRIBUTION_H264_DEFAULT,
    },
    source: SourceConfiguration.ndi({
      flowSourceName: 'ndi-source',
      sourceName: 'MACHINE (Program)',
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      Name: 'ndi-source',
      Protocol: 'ndi-speed-hq',
      NdiSourceSettings: {
        SourceName: 'MACHINE (Program)',
      },
    },
  });
});

test('fails - NDI source without NDI config enabled', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE,
      source: SourceConfiguration.ndi({
        flowSourceName: 'ndi-source',
      }),
    });
  }).toThrow(/NDI sources require ndiConfig/);
});

test('fails - NDI output without NDI config on the flow', () => {
  const flow = new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  expect(() => flow.addOutput('out', { output: OutputConfiguration.ndi() })).toThrow(/NDI outputs require ndiConfig/);
});

test('fails - NDI output when ndiState is DISABLED', () => {
  const flow = new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    ndiConfig: { ndiState: State.DISABLED },
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  expect(() => flow.addOutput('out', { output: OutputConfiguration.ndi() })).toThrow(/NDI outputs require ndiConfig/);
});

test('NDI output succeeds when ndiState is ENABLED', () => {
  const flow = new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    ndiConfig: { ndiState: State.ENABLED },
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  flow.addOutput('out', {
    output: OutputConfiguration.ndi({
      ndiProgramName: 'program-1',
      ndiSpeedHqQuality: 150,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::FlowOutput', {
    Protocol: 'ndi-speed-hq',
    NdiProgramName: 'program-1',
    NdiSpeedHqQuality: 150,
  });
});

test('NDI output on imported flow does not enforce ndiState (state unknown)', () => {
  const imported = Flow.fromFlowAttributes(stack, 'imported', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:test',
  });

  // Imported flows have unknown ndi state, so addOutput passes synth — service catches mismatch at deploy.
  expect(() => imported.addOutput('out', { output: OutputConfiguration.ndi() })).not.toThrow();
});

test('fails - second NDI output on the same flow', () => {
  const flow = new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    ndiConfig: { ndiState: State.ENABLED },
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  flow.addOutput('first', { output: OutputConfiguration.ndi() });

  expect(() => flow.addOutput('second', { output: OutputConfiguration.ndi() })).toThrow(/maximum of 1 NDI output/);
});

test('fails - NDI output when source is also NDI (passthrough not supported)', () => {
  const flow = new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    ndiConfig: { ndiState: State.ENABLED },
    encodingConfig: {},
    source: SourceConfiguration.ndi({
      flowSourceName: 'ndi-source',
    }),
  });

  expect(() => flow.addOutput('out', { output: OutputConfiguration.ndi() })).toThrow(/NDI passthrough is not supported/);
});

test('fails - NDI source without encodingConfig', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE,
      ndiConfig: { ndiState: State.ENABLED },
      source: SourceConfiguration.ndi({
        flowSourceName: 'ndi-source',
      }),
    });
  }).toThrow(/NDI sources require encodingConfig/);
});

test('fails - NDI source with wrong flow size', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.MEDIUM,
      ndiConfig: { ndiState: State.ENABLED },
      source: SourceConfiguration.ndi({
        flowSourceName: 'ndi-source',
      }),
    });
  }).toThrow(/NDI requires LARGE flow size/);
});

test('NDI validation - throws when flow size is not LARGE', () => {
  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.MEDIUM,
      ndiConfig: {
        ndiState: State.ENABLED,
      },
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/NDI requires LARGE flow size/);
});

test('NDI validation - throws when used with CDI protocol', () => {
  const vpc = new Vpc(stack, 'vpc', {
    ipAddresses: IpAddresses.cidr('10.0.0.0/16'),
    maxAzs: 1,
    subnetConfiguration: [{ name: 'private', subnetType: SubnetType.PRIVATE_ISOLATED }],
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });
  const sg = new SecurityGroup(stack, 'sg', { vpc });
  const efaInterface = VpcInterface.define({
    vpcInterfaceName: 'efa-interface',
    role,
    securityGroups: [sg],
    subnet: vpc.isolatedSubnets[0],
    networkInterfaceType: NetworkInterface.EFA,
  });
  const mediaStream = MediaStream.video({
    mediaStreamId: 1,
    mediaStreamName: 'video',
    videoFormat: MediaVideoFormat.HD_1080P,
    fmtp: {
      exactFramerate: Framerate.FPS_29_97,
      par: PixelAspectRatio.SQUARE,
    },
  });

  expect(() => {
    new Flow(stack, 'flow', {
      flowSize: FlowSize.LARGE_4X,
      ndiConfig: {
        ndiState: State.ENABLED,
      },
      vpcInterfaces: [efaInterface],
      mediaStreams: [mediaStream],
      source: SourceConfiguration.cdi({
        flowSourceName: 'cdi-source',
        vpcInterface: efaInterface,
        port: 5000,
        maxSyncBuffer: 100,
        mediaStreamSourceConfigurations: [{
          encoding: Encoding.RAW,
          mediaStream: mediaStream,
        }],
      }),
    });
  }).toThrow(/NDI cannot be used with CDI or JPEG XS protocols/);
});

test('LARGE flow size is allowed without NDI', () => {
  new Flow(stack, 'flow', {
    flowSize: FlowSize.LARGE,
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    FlowSize: 'LARGE',
  });
});

test('Flow fromFlowAttributes', () => {
  const flow = Flow.fromFlowAttributes(stack, 'imported', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-abc:my-source',
    isFailoverEnabled: true,
  });

  expect(flow.flowArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow');
  expect(flow.sourceArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:source:1-abc:my-source');
  expect(flow.isFailoverEnabled).toBe(true);
});

test('Flow fromFlowArn', () => {
  const flow = Flow.fromFlowArn(stack, 'imported', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow');

  expect(flow.flowArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow');
  expect(flow.isFailoverEnabled).toBe(false);
});

test('fails - Flow imported by ARN throws when accessing sourceArn', () => {
  const flow = Flow.fromFlowArn(stack, 'imported', 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow');
  expect(() => flow.sourceArn).toThrow(/sourceArn.*was not provided/);
});

test('Flow metrics', () => {
  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  expect(flow.metricSourceBitrate()).toBeDefined();
  expect(flow.metricSourceNotRecoveredPackets()).toBeDefined();
  expect(flow.metricSourceTotalPackets()).toBeDefined();
  expect(flow.metricSourceSelected()).toBeDefined();
  expect(flow.metricSourceConnected()).toBeDefined();
  expect(flow.metricSourceDisconnections()).toBeDefined();
  expect(flow.metricSourceDroppedPackets()).toBeDefined();
  expect(flow.metricSourcePacketLossPercent()).toBeDefined();
  expect(flow.metricSourceRoundTripTime()).toBeDefined();
  expect(flow.metricSourceJitter()).toBeDefined();
  expect(flow.metric('CustomMetric')).toBeDefined();
});

test('Flow with audio media stream', () => {
  const audioStream = MediaStream.audio({
    mediaStreamId: 2,
    mediaStreamName: 'audio',
    channelOrder: AudioStreamOrderOptions.SURROUND_5_1,
    clockRate: 48000,
    lang: 'en',
  });

  new Flow(stack, 'flow', {
    mediaStreams: [audioStream],
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    MediaStreams: [{
      Attributes: {
        Fmtp: { ChannelOrder: 'SMPTE2110.(51)' },
        Lang: 'en',
      },
      ClockRate: 48000,
      MediaStreamId: 2,
      MediaStreamName: 'audio',
      MediaStreamType: 'audio',
    }],
  });
});

test('Flow with ancillary data media stream', () => {
  const ancillaryStream = MediaStream.ancillaryData({
    mediaStreamId: 3,
    mediaStreamName: 'ancillary',
    description: 'captions',
  });

  new Flow(stack, 'flow', {
    mediaStreams: [ancillaryStream],
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    MediaStreams: [{
      Description: 'captions',
      MediaStreamId: 3,
      MediaStreamName: 'ancillary',
      MediaStreamType: 'ancillary-data',
    }],
  });
});

test('Enum-like class of() and toString()', () => {
  expect(FlowSize.of('CUSTOM').value).toBe('CUSTOM');
  expect(FlowSize.LARGE_4X.toString()).toBe('LARGE_4X');
  expect(Colorimetry.of('CUSTOM').value).toBe('CUSTOM');
  expect(VideoRange.of('CUSTOM').value).toBe('CUSTOM');
  expect(ScanMode.of('CUSTOM').value).toBe('CUSTOM');
  expect(Tcs.of('CUSTOM').value).toBe('CUSTOM');
  expect(AudioStreamOrderOptions.of('CUSTOM').value).toBe('CUSTOM');
  expect(MediaVideoFormat.of('CUSTOM').value).toBe('CUSTOM');
});

test('Shared enum-like classes of() and toString()', () => {
  expect(EncryptionAlgorithm.of('custom').value).toBe('custom');
  expect(EncryptionAlgorithm.AES128.toString()).toBe('aes128');
  expect(FailoverMode.of('custom').value).toBe('custom');
  expect(FailoverMode.MERGE.toString()).toBe('MERGE');
  expect(NetworkInterface.of('custom').value).toBe('custom');
  expect(NetworkInterface.ENA.toString()).toBe('ena');
});

test('Flow with SRT Listener source', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.srtListener({
      flowSourceName: 'srt-source',
      port: 5000,
      minLatency: Duration.millis(200),
      maxBitrate: Bitrate.mbps(50),
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      Name: 'srt-source',
      Protocol: 'srt-listener',
      IngestPort: 5000,
      MinLatency: 200,
      MaxBitrate: 50000000,
      WhitelistCidr: '10.0.0.0/16',
    },
  });
});

test('Flow with Zixi Push source', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.zixiPush({
      flowSourceName: 'zixi-source',
      maxLatency: Duration.millis(1000),
      streamId: 'test-stream',
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      Name: 'zixi-source',
      Protocol: 'zixi-push',
      IngestPort: 2088,
      MaxLatency: 1000,
      StreamId: 'test-stream',
      WhitelistCidr: '10.0.0.0/16',
    },
  });
});

test('Flow with RIST source', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.rist({
      flowSourceName: 'rist-source',
      port: 5000,
      maxLatency: Duration.millis(2000),
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      Name: 'rist-source',
      Protocol: 'rist',
      IngestPort: 5000,
      MaxLatency: 2000,
      WhitelistCidr: '10.0.0.0/16',
    },
  });
});

test('Flow with router source', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.router(),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {});
});

test('SRT Listener source validation - reserved port 2077', () => {
  expect(() => {
    SourceConfiguration.srtListener({
      flowSourceName: 'source',
      port: 2077,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    });
  }).toThrow(/Ports 2077 and 2088 are reserved and cannot be used for SRT Listener/);
});

test('SRT Caller source validation - reserved port 2088', () => {
  expect(() => {
    SourceConfiguration.srtCaller({
      flowSourceName: 'source',
      sourceListenerAddress: '203.0.113.11',
      sourceListenerPort: 2088,
    });
  }).toThrow(/Ports 2077 and 2088 are reserved and cannot be used for SRT Caller/);
});

test('RTP source validation - port out of range', () => {
  expect(() => {
    SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 500,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    });
  }).toThrow(/Ingest port must be between 1024 and 65535/);
});

test('VpcInterface.fromNetworkInterfaces creates config with existing ENIs', () => {
  const vpc = new Vpc(stack, 'vpc', { ipAddresses: IpAddresses.cidr('10.0.0.0/16') });
  const subnet = new PrivateSubnet(stack, 'subnet', {
    availabilityZone: 'us-east-1a',
    cidrBlock: '10.0.0.0/24',
    vpcId: vpc.vpcId,
  });
  const role = new Role(stack, 'role', {
    assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  });

  const config = VpcInterface.fromNetworkInterfaces({
    vpcInterfaceName: 'existing-enis',
    role,
    securityGroups: [new SecurityGroup(stack, 'sg', { vpc })],
    subnet,
    networkInterfaceIds: ['eni-1234567890abcdef0', 'eni-0987654321fedcba0'],
  });

  expect(config.name).toBe('existing-enis');
  expect(config.networkInterfaceIds).toEqual(['eni-1234567890abcdef0', 'eni-0987654321fedcba0']);
  expect(config.networkInterfaceType).toBeUndefined();
});

test('Flow with SRT Caller source with all options', () => {
  new Flow(stack, 'flow', {
    source: SourceConfiguration.srtCaller({
      flowSourceName: 'srt-caller',
      sourceListenerAddress: '203.0.113.11',
      sourceListenerPort: 5000,
      maxLatency: Duration.millis(1000),
      minLatency: Duration.millis(100),
      streamId: 'test-stream',
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      Name: 'srt-caller',
      Protocol: 'srt-caller',
      SourceListenerAddress: '203.0.113.11',
      SourceListenerPort: 5000,
      MaxLatency: 1000,
      MinLatency: 100,
      StreamId: 'test-stream',
    },
  });
});

test('Flow with gateway bridge source without VPC interface', () => {
  const bridge = Bridge.fromBridgeAttributes(stack, 'bridge', {
    bridgeArn: 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge',
    bridgeType: BridgeType.INGRESS_BRIDGE,
  });

  new Flow(stack, 'flow', {
    source: SourceConfiguration.gatewayBridge({ bridge }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Flow', {
    Source: {
      GatewayBridgeSource: {
        BridgeArn: 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge',
      },
    },
  });
});

test('Flow source name validation - too long', () => {
  expect(() => {
    new FlowSource(stack, 'source', {
      flow: Flow.fromFlowAttributes(stack, 'flow', {
        flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:f',
        sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-abc:s',
        isFailoverEnabled: true,
      }),
      source: SourceConfiguration.rtp({
        flowSourceName: 'a'.repeat(65),
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/Flow source name must be between 1 and 64 characters/);
});

test('Flow source name validation - invalid characters', () => {
  expect(() => {
    new FlowSource(stack, 'source', {
      flow: Flow.fromFlowAttributes(stack, 'flow', {
        flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:f',
        sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-abc:s',
        isFailoverEnabled: true,
      }),
      source: SourceConfiguration.rtp({
        flowSourceName: 'invalid@name',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });
  }).toThrow(/Flow source name must contain only alphanumeric characters/);
});

test('imported flow throws when accessing egressIp', () => {
  const imported = Flow.fromFlowAttributes(stack, 'ImportedFlow', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa-bbb:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-aaa-bbb:my-source',
  });
  expect(() => imported.egressIp).toThrow(/egressIp.*was not provided/);
});

test('imported flow throws when accessing sourceIngestIp', () => {
  const imported = Flow.fromFlowAttributes(stack, 'ImportedFlow2', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa-bbb:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-aaa-bbb:my-source',
  });
  expect(() => imported.sourceIngestIp).toThrow(/sourceIngestIp.*was not provided/);
});

test('imported flow throws when accessing sourceIngestPort', () => {
  const imported = Flow.fromFlowAttributes(stack, 'ImportedFlow3', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa-bbb:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-aaa-bbb:my-source',
  });
  expect(() => imported.sourceIngestPort).toThrow(/sourceIngestPort.*was not provided/);
});

test('imported flow has undefined flowAvailabilityZone', () => {
  const imported = Flow.fromFlowAttributes(stack, 'ImportedFlow4', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa-bbb:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-aaa-bbb:my-source',
  });
  expect(imported.flowAvailabilityZone).toBeUndefined();
});

test('imported flow throws when accessing sourceIngestUrl', () => {
  const imported = Flow.fromFlowAttributes(stack, 'ImportedFlow5', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-aaa-bbb:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-aaa-bbb:my-source',
  });
  expect(() => imported.sourceIngestUrl).toThrow(/sourceIngestUrl.*is not available/);
});

test('SRT caller flow throws when accessing source ingest attributes', () => {
  const flow = new Flow(stack, 'SrtCallerFlow', {
    source: SourceConfiguration.srtCaller({
      flowSourceName: 'srt-caller-source',
      maxLatency: Duration.seconds(1),
      sourceListenerAddress: '203.0.113.50',
      sourceListenerPort: 5000,
    }),
  });

  expect(() => flow.sourceIngestIp).toThrow(/sourceIngestIp.*is not available/);
  expect(() => flow.sourceIngestPort).toThrow(/sourceIngestPort.*is not available/);
  expect(() => flow.sourceIngestUrl).toThrow(/sourceIngestUrl.*is not available/);
});

test('entitlement source flow throws when accessing source ingest attributes', () => {
  const entitlement = FlowEntitlement.fromFlowEntitlementArn(
    stack,
    'EntForFlow',
    'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-3333cccc4444dddd-1111aaaa2222:ExampleCorp',
  );
  const flow = new Flow(stack, 'EntFlow', {
    source: SourceConfiguration.entitlement({
      entitlement,
    }),
  });

  expect(() => flow.sourceIngestIp).toThrow(/sourceIngestIp.*is not available/);
  expect(() => flow.sourceIngestPort).toThrow(/sourceIngestPort.*is not available/);
  expect(() => flow.sourceIngestUrl).toThrow(/sourceIngestUrl.*is not available/);
});

test('listener-style source flow exposes source ingest attributes', () => {
  const flow = new Flow(stack, 'RtpFlow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'rtp-source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
    }),
  });

  // No throw; values are tokens that resolve at deploy time.
  expect(() => flow.sourceIngestIp).not.toThrow();
  expect(() => flow.sourceIngestPort).not.toThrow();
  expect(() => flow.sourceIngestUrl).not.toThrow();
});

test('grants.start() adds correct IAM policy', () => {
  const role = new Role(stack, 'OrchestratorRole', {
    assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
  });

  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      port: 5000,
    }),
  });
  flow.grants.start(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'mediaconnect:StartFlow',
        Effect: 'Allow',
        Resource: Match.anyValue(),
      }],
    },
  });
});

test('grants.stop() adds correct IAM policy', () => {
  const role = new Role(stack, 'OrchestratorRole', {
    assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
  });

  const flow = new Flow(stack, 'flow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      port: 5000,
    }),
  });
  flow.grants.stop(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'mediaconnect:StopFlow',
        Effect: 'Allow',
        Resource: Match.anyValue(),
      }],
    },
  });
});

describe('open source CIDR allowlist warning', () => {
  test.each(['0.0.0.0/0', '::/0'])('warns when a public source allowlist is fully open (%s)', (openCidr) => {
    new Flow(stack, 'flow', {
      flowName: 'flow',
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork(openCidr),
      }),
    });

    Annotations.fromStack(stack).hasWarning(
      '*',
      Match.stringLikeRegexp(`Source CIDR allowlist '${openCidr.replace('/', '\\/')}' allows traffic from any IP`),
    );
  });

  test('does not warn for a narrow public source allowlist', () => {
    new Flow(stack, 'flow', {
      flowName: 'flow',
      source: SourceConfiguration.rtp({
        flowSourceName: 'source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
      }),
    });

    Annotations.fromStack(stack).hasNoWarning('*', Match.stringLikeRegexp('allows traffic from any IP'));
  });
});
