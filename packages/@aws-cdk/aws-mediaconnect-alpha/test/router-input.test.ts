import { App, Bitrate, Duration, Lazy, Stack } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import { SourceConfiguration, NetworkConfiguration, FlowOutput, OutputConfiguration } from '../lib';
import { Flow } from '../lib/flow';
import { ForwardErrorCorrection, PrimarySource, RouterInput, RouterInputConfiguration, RouterInputProtocol, RouterInputTier, RoutingScope, SourcePriorityConfig } from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';
import { MaintenanceDay, MediaLivePipeline } from '../lib/shared';

let app: App;
let stack: Stack;
let networkInterface: RouterNetworkInterface;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, undefined, {
    env: { account: '123456789012', region: 'us-east-1' },
  });

  networkInterface = new RouterNetworkInterface(stack, 'network', {
    routerNetworkInterfaceName: 'test-network',
    configuration: RouterNetworkConfiguration.publicNetwork({
      cidr: ['10.0.0.0/16'],
    }),
  });
});

describe('RouterInput', () => {
  test('creates router input with standard configuration', () => {
    new RouterInput(stack, 'routerInput', {
      routerInputName: 'test',
      maximumBitrate: Bitrate.mbps(5),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterInputTier.INPUT_100,
      configuration: RouterInputConfiguration.standard({
        protocol: RouterInputProtocol.rtp({
          port: 5000,
          forwardErrorCorrection: ForwardErrorCorrection.ENABLED,
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
      Name: 'test',
      MaximumBitrate: 5000000,
      RoutingScope: 'GLOBAL',
      Tier: 'INPUT_100',
      Configuration: {
        Standard: {
          NetworkInterfaceArn: {
            'Fn::GetAtt': ['network39EEAA36', 'Arn'],
          },
          Protocol: 'RTP',
          ProtocolConfiguration: {
            Rtp: {
              Port: 5000,
              ForwardErrorCorrection: 'ENABLED',
            },
          },
        },
      },
    });
  });

  test('creates router input with failover configuration', () => {
    const protocol1 = RouterInputProtocol.rtp({ port: 5000 });
    const protocol2 = RouterInputProtocol.rtp({ port: 5001 });

    new RouterInput(stack, 'routerInput', {
      routerInputName: 'test-failover',
      maximumBitrate: Bitrate.mbps(10),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterInputTier.INPUT_50,
      configuration: RouterInputConfiguration.failover({
        networkInterface,
        protocols: [protocol1, protocol2],
        sourcePriority: SourcePriorityConfig.primarySecondary(PrimarySource.FIRST_SOURCE),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
      Name: 'test-failover',
      Configuration: {
        Failover: {
          NetworkInterfaceArn: {
            'Fn::GetAtt': ['network39EEAA36', 'Arn'],
          },
          ProtocolConfigurations: [
            { Rtp: { Port: 5000 } },
            { Rtp: { Port: 5001 } },
          ],
          SourcePriorityMode: 'PRIMARY_SECONDARY',
          PrimarySourceIndex: 0,
        },
      },
    });
  });

  test('creates router input with merge configuration', () => {
    const protocol1 = RouterInputProtocol.rtp({ port: 5000 });
    const protocol2 = RouterInputProtocol.rtp({ port: 5001 });

    new RouterInput(stack, 'routerInput', {
      routerInputName: 'test-merge',
      maximumBitrate: Bitrate.mbps(15),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterInputTier.INPUT_20,
      configuration: RouterInputConfiguration.merge({
        networkInterface,
        protocols: [protocol1, protocol2],
        mergeRecoveryWindow: Duration.millis(10000),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
      Name: 'test-merge',
      Configuration: {
        Merge: {
          NetworkInterfaceArn: {
            'Fn::GetAtt': ['network39EEAA36', 'Arn'],
          },
          ProtocolConfigurations: [
            { Rtp: { Port: 5000 } },
            { Rtp: { Port: 5001 } },
          ],
          MergeRecoveryWindowMilliseconds: 10000,
        },
      },
    });
  });

  test('creates router input with mediaConnectFlow configuration', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');

    const flow = new Flow(stack, 'testFlow', {
      source: SourceConfiguration.rtp({
        port: 5000,
        flowSourceName: 'ingest',
        network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
      }),
    });
    const flowOutput = new FlowOutput(stack, 'RouterOutputFlow', {
      flow,
      output: OutputConfiguration.router({}),
    });
    new RouterInput(stack, 'routerInput', {
      routerInputName: 'test-flow',
      maximumBitrate: Bitrate.mbps(20),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterInputTier.INPUT_100,
      configuration: RouterInputConfiguration.mediaConnectFlow({
        flow,
        flowOutput,
        sourceTransitDecryption: ({
          role,
          secret,
        }),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
      Name: 'test-flow',
      Configuration: {
        MediaConnectFlow: {
          FlowArn: {
            'Fn::GetAtt': ['testFlowDFDC0C54', 'FlowArn'],
          },
          SourceTransitDecryption: {
            EncryptionKeyType: 'SECRETS_MANAGER',
            EncryptionKeyConfiguration: {
              SecretsManager: {
                RoleArn: {
                  'Fn::GetAtt': ['TestRole6C9272DF', 'Arn'],
                },
                SecretArn: {
                  Ref: 'TestSecret16AF87B1',
                },
              },
            },
          },
        },
      },
    });
  });

  test('creates router input with mediaConnectFlow configuration without encryption', () => {
    const flow = new Flow(stack, 'testFlow', {
      source: SourceConfiguration.rtp({
        port: 5000,
        flowSourceName: 'ingest',
        network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
      }),
    });
    const flowOutput = new FlowOutput(stack, 'RouterOutputFlow', {
      flow,
      output: OutputConfiguration.router({}),
    });
    new RouterInput(stack, 'routerInput', {
      routerInputName: 'test-flow-no-encryption',
      maximumBitrate: Bitrate.mbps(20),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterInputTier.INPUT_100,
      configuration: RouterInputConfiguration.mediaConnectFlow({
        flow,
        flowOutput,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
      Name: 'test-flow-no-encryption',
      Configuration: {
        MediaConnectFlow: {
          FlowArn: {
            'Fn::GetAtt': ['testFlowDFDC0C54', 'FlowArn'],
          },
          SourceTransitDecryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });
});

describe('RouterInputConfiguration validation', () => {
  test('failover throws error when protocols do not match', () => {
    const rtpProtocol = RouterInputProtocol.rtp({ port: 5000 });
    const ristProtocol = RouterInputProtocol.rist({ port: 5001, recoveryLatency: Duration.millis(100000) });

    expect(() => {
      RouterInputConfiguration.failover({
        networkInterface,
        protocols: [rtpProtocol, ristProtocol],
      });
    }).toThrow('Protocols must match');
  });

  test('merge throws error when SRT protocols are used', () => {
    const srtCaller = RouterInputProtocol.srtCaller({
      sourceAddress: '203.0.113.1',
      sourcePort: 5000,
      minimumLatency: Duration.millis(100000),
    });
    const srtListener = RouterInputProtocol.srtListener({
      port: 5001,
      minimumLatency: Duration.millis(100000),
    });

    expect(() => {
      RouterInputConfiguration.merge({
        networkInterface,
        protocols: [srtCaller, srtListener],
        mergeRecoveryWindow: Duration.millis(100000),
      });
    }).toThrow('SRT protocols are not supported in merge configuration');
  });

  test('merge throws error when protocols do not match', () => {
    const rtpProtocol = RouterInputProtocol.rtp({ port: 5000 });
    const ristProtocol = RouterInputProtocol.rist({ port: 5001, recoveryLatency: Duration.millis(100000) });

    expect(() => {
      RouterInputConfiguration.merge({
        networkInterface,
        protocols: [rtpProtocol, ristProtocol],
        mergeRecoveryWindow: Duration.millis(100000),
      });
    }).toThrow('Protocols must match');
  });

  test('merge works with matching non-SRT protocols', () => {
    const rtpProtocol1 = RouterInputProtocol.rtp({ port: 5000 });
    const rtpProtocol2 = RouterInputProtocol.rtp({ port: 5001 });

    expect(() => {
      RouterInputConfiguration.merge({
        networkInterface,
        protocols: [rtpProtocol1, rtpProtocol2],
        mergeRecoveryWindow: Duration.millis(100000),
      });
    }).not.toThrow();
  });
});

// Validation tests
test('Router input name validation - too long', () => {
  const importedNetworkInterface = RouterNetworkInterface.fromRouterNetworkInterfaceArn(
    stack,
    'network-interface',
    'arn:aws:mediaconnect:us-east-1:123456789012:router-network-interface:test-interface',
  );

  expect(() => {
    new RouterInput(stack, 'router-input', {
      routerInputName: 'a'.repeat(129),
      maximumBitrate: Bitrate.mbps(100),
      tier: RouterInputTier.INPUT_100,
      routingScope: RoutingScope.REGIONAL,
      configuration: RouterInputConfiguration.standard({
        networkInterface: importedNetworkInterface,
        protocol: RouterInputProtocol.rtp({ port: 5000 }),
      }),
    });
  }).toThrow('Router input name must be between 1 and 128');
});

test('Router input name validation - invalid characters', () => {
  const importedNetworkInterface = RouterNetworkInterface.fromRouterNetworkInterfaceArn(
    stack,
    'network-interface',
    'arn:aws:mediaconnect:us-east-1:123456789012:router-network-interface:test-interface',
  );

  expect(() => {
    new RouterInput(stack, 'router-input', {
      routerInputName: 'invalid@name!',
      maximumBitrate: Bitrate.mbps(100),
      tier: RouterInputTier.INPUT_100,
      routingScope: RoutingScope.REGIONAL,
      configuration: RouterInputConfiguration.standard({
        networkInterface: importedNetworkInterface,
        protocol: RouterInputProtocol.rtp({ port: 5000 }),
      }),
    });
  }).toThrow('Router input name must contain only alphanumeric characters, hyphens, and underscores');
});

describe('maximum bitrate validation', () => {
  test('throws when below the 1 Mbps minimum', () => {
    expect(() => {
      new RouterInput(stack, 'router-input', {
        maximumBitrate: Bitrate.bps(500000),
        routingScope: RoutingScope.REGIONAL,
        configuration: RouterInputConfiguration.standard({
          networkInterface,
          protocol: RouterInputProtocol.rtp({ port: 5000 }),
        }),
      });
    }).toThrow('Maximum bitrate must be at least 1,000,000 bits/s (1 Mbps)');
  });

  test('throws when exceeding the selected tier limit', () => {
    expect(() => {
      new RouterInput(stack, 'router-input', {
        maximumBitrate: Bitrate.mbps(30),
        routingScope: RoutingScope.REGIONAL,
        tier: RouterInputTier.INPUT_20,
        configuration: RouterInputConfiguration.standard({
          networkInterface,
          protocol: RouterInputProtocol.rtp({ port: 5000 }),
        }),
      });
    }).toThrow('exceeds the INPUT_20 tier limit of 20 Mbps');
  });

  test('skips bitrate validation for a tokenized maximum bitrate', () => {
    expect(() => {
      new RouterInput(stack, 'router-input', {
        maximumBitrate: Bitrate.bps(Lazy.number({ produce: () => 5000000 })),
        routingScope: RoutingScope.REGIONAL,
        tier: RouterInputTier.INPUT_20,
        configuration: RouterInputConfiguration.standard({
          networkInterface,
          protocol: RouterInputProtocol.rtp({ port: 5000 }),
        }),
      });
    }).not.toThrow();
  });
});

test('failover throws when not exactly 2 protocols', () => {
  expect(() => {
    RouterInputConfiguration.failover({
      networkInterface,
      protocols: [RouterInputProtocol.rtp({ port: 5000 })],
    });
  }).toThrow('Failover configuration requires exactly 2 protocols');
});

test('merge throws when not exactly 2 protocols', () => {
  expect(() => {
    RouterInputConfiguration.merge({
      networkInterface,
      protocols: [RouterInputProtocol.rtp({ port: 5000 })],
      mergeRecoveryWindow: Duration.millis(1000),
    });
  }).toThrow('Merge configuration requires exactly 2 protocols');
});

test('failover RIST duplicate port throws', () => {
  expect(() => {
    RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.rist({ port: 5000, recoveryLatency: Duration.millis(1000) }),
        RouterInputProtocol.rist({ port: 5000, recoveryLatency: Duration.millis(1000) }),
      ],
    });
  }).toThrow(/cannot use the same RIST port/);
});

test('failover RIST consecutive port throws', () => {
  expect(() => {
    RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.rist({ port: 5000, recoveryLatency: Duration.millis(1000) }),
        RouterInputProtocol.rist({ port: 5001, recoveryLatency: Duration.millis(1000) }),
      ],
    });
  }).toThrow(/cannot use consecutive RIST ports/);
});

test('failover RTP duplicate port throws', () => {
  expect(() => {
    RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.rtp({ port: 5000 }),
        RouterInputProtocol.rtp({ port: 5000 }),
      ],
    });
  }).toThrow(/cannot use the same RTP port/);
});

test('failover SRT Listener duplicate port throws', () => {
  expect(() => {
    RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(100) }),
        RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(100) }),
      ],
    });
  }).toThrow(/cannot use the same SRT Listener port/);
});

test('port validation - out of range', () => {
  expect(() => {
    RouterInputProtocol.rtp({ port: 100 });
  }).toThrow(/Port must be between 3000 and 30000/);
});

test('SRT Caller port validation - out of range', () => {
  expect(() => {
    RouterInputProtocol.srtCaller({
      sourceAddress: '203.0.113.2',
      sourcePort: 70000,
      minimumLatency: Duration.millis(100),
    });
  }).toThrow(/SRT Caller port must be between 0 and 65535/);
});

test('SRT Listener with decryption', () => {
  const role = new Role(stack, 'role', { assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com') });
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'srt-encrypted',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.srtListener({
        port: 5000,
        minimumLatency: Duration.millis(200),
        decryptionConfiguration: ({ role, secret }),
      }),
      networkInterface,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Configuration: {
      Standard: {
        Protocol: 'SRT_LISTENER',
        ProtocolConfiguration: {
          SrtListener: {
            Port: 5000,
            MinimumLatencyMilliseconds: 200,
            DecryptionConfiguration: {
              EncryptionKey: {
                RoleArn: { 'Fn::GetAtt': ['roleC7B7E775', 'Arn'] },
                SecretArn: { Ref: 'secret4DA88516' },
              },
            },
          },
        },
      },
    },
  });
});

test('SRT Listener with decryption auto-creates a role when none is provided', () => {
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'srt-auto-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.srtListener({
        port: 5000,
        minimumLatency: Duration.millis(200),
        decryptionConfiguration: { secret },
      }),
      networkInterface,
    }),
  });

  const template = Template.fromStack(stack);
  // A scoped role was auto-created on the RouterInput. The trust policy carries
  // `aws:SourceAccount` and a wildcarded `aws:SourceArn` ArnLike — the wildcard covers the
  // service-generated router-input id segment in the ARN.
  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Principal: { Service: 'mediaconnect.amazonaws.com' },
          Condition: {
            StringEquals: { 'aws:SourceAccount': { Ref: 'AWS::AccountId' } },
            ArnLike: {
              'aws:SourceArn': {
                'Fn::Join': ['', [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':mediaconnect:us-east-1:123456789012:routerInput:*',
                ]],
              },
            },
          },
        }),
      ]),
    },
  });
  template.hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Configuration: {
      Standard: {
        ProtocolConfiguration: {
          SrtListener: Match.objectLike({
            DecryptionConfiguration: { EncryptionKey: Match.anyValue() },
          }),
        },
      },
    },
  });
});

test('input-level transit encryption auto-creates a role and orders its secret-read policy before the router input', () => {
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'transit-auto-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    transitEncryption: { secret },
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(200) }),
      networkInterface,
    }),
  });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::MediaConnect::RouterInput', {
    TransitEncryption: { EncryptionKeyType: 'SECRETS_MANAGER' },
  });

  template.hasResource('AWS::MediaConnect::RouterInput', {
    DependsOn: Match.arrayWith([Match.stringLikeRegexp('TransitEncryptionRoleDefaultPolicy')]),
  });
});

test('input-level transit encryption defaults to AUTOMATIC and adds no role or dependency when no secret is given', () => {
  new RouterInput(stack, 'routerInput', {
    routerInputName: 'transit-automatic',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(200) }),
      networkInterface,
    }),
  });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::MediaConnect::RouterInput', {
    TransitEncryption: { EncryptionKeyType: 'AUTOMATIC' },
  });
  // No secret means no auto-created role and no ordering dependency on the router input.
  const roles = template.findResources('AWS::IAM::Role', {
    Properties: { Description: 'Auto-generated MediaConnect role for accessing the encryption secret' },
  });
  expect(Object.keys(roles)).toHaveLength(0);
  template.hasResource('AWS::MediaConnect::RouterInput', { DependsOn: Match.absent() });
});

test('input-level transit encryption with an explicit role does not auto-create a role', () => {
  const role = new Role(stack, 'role', { assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com') });
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'transit-explicit-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    transitEncryption: { role, secret },
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(200) }),
      networkInterface,
    }),
  });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::MediaConnect::RouterInput', {
    TransitEncryption: { EncryptionKeyType: 'SECRETS_MANAGER' },
  });
  // Only the user-supplied role exists — no auto-created MediaConnect encryption role.
  const roles = template.findResources('AWS::IAM::Role', {
    Properties: { Description: 'Auto-generated MediaConnect role for accessing the encryption secret' },
  });
  expect(Object.keys(roles)).toHaveLength(0);
});

test('mediaConnectFlowWithoutConnection', () => {
  new RouterInput(stack, 'routerInput', {
    routerInputName: 'flow-no-conn',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.mediaConnectFlowWithoutConnection({
      availabilityZone: 'us-east-1a',
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    AvailabilityZone: 'us-east-1a',
    Configuration: {
      MediaConnectFlow: {
        SourceTransitDecryption: {
          EncryptionKeyType: 'AUTOMATIC',
        },
      },
    },
  });
});

test('mediaConnectFlowWithoutConnection with encryption', () => {
  const role = new Role(stack, 'role', { assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com') });
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'flow-no-conn-enc',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.mediaConnectFlowWithoutConnection({
      availabilityZone: 'us-east-1a',
      sourceTransitDecryption: ({ role, secret }),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Configuration: {
      MediaConnectFlow: {
        SourceTransitDecryption: {
          EncryptionKeyType: 'SECRETS_MANAGER',
        },
      },
    },
  });
});

test('mediaLiveChannel configuration', () => {
  new RouterInput(stack, 'routerInput', {
    routerInputName: 'medialive-channel',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.mediaLiveChannel({
      mediaLiveChannelArn: 'arn:aws:medialive:us-east-1:123456789012:channel:1234567',
      mediaLiveChannelOutputName: 'output1',
      mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Configuration: {
      MediaLiveChannel: {
        MediaLiveChannelArn: 'arn:aws:medialive:us-east-1:123456789012:channel:1234567',
        MediaLiveChannelOutputName: 'output1',
        MediaLivePipelineId: 'PIPELINE_0',
        SourceTransitDecryption: {
          EncryptionKeyType: 'AUTOMATIC',
        },
      },
    },
  });
});

test('mediaLiveChannel configuration with encryption', () => {
  const role = new Role(stack, 'role', { assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com') });
  const secret = new Secret(stack, 'secret');

  new RouterInput(stack, 'routerInput', {
    routerInputName: 'medialive-channel-enc',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.mediaLiveChannel({
      mediaLiveChannelArn: 'arn:aws:medialive:us-east-1:123456789012:channel:1234567',
      mediaLiveChannelOutputName: 'output1',
      mediaLivePipelineId: MediaLivePipeline.PIPELINE_1,
      sourceTransitDecryption: ({ role, secret }),
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Configuration: {
      MediaLiveChannel: {
        MediaLivePipelineId: 'PIPELINE_1',
        SourceTransitDecryption: {
          EncryptionKeyType: 'SECRETS_MANAGER',
        },
      },
    },
  });
});

test('mediaLiveChannelWithoutConnection', () => {
  new RouterInput(stack, 'routerInput', {
    routerInputName: 'medialive-no-conn',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.mediaLiveChannelWithoutConnection({
      availabilityZone: 'us-east-1a',
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    AvailabilityZone: 'us-east-1a',
    Configuration: {
      MediaLiveChannel: {
        SourceTransitDecryption: {
          EncryptionKeyType: 'AUTOMATIC',
        },
      },
    },
  });
});

test('router input with tags and maintenance', () => {
  new RouterInput(stack, 'routerInput', {
    routerInputName: 'tagged-input',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    tags: { Environment: 'test' },
    maintenanceConfiguration: { day: MaintenanceDay.MONDAY, time: '03:00' },
    configuration: RouterInputConfiguration.standard({
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
      networkInterface,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterInput', {
    Tags: [{ Key: 'Environment', Value: 'test' }],
  });
});

test('fromRouterInputArn', () => {
  const imported = RouterInput.fromRouterInputArn(stack, 'imported', 'arn:aws:mediaconnect:us-east-1:123456789012:router-input:1-abc:test');
  expect(imported.routerInputArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:router-input:1-abc:test');
});

test('AZ validation skips when region is a token (no explicit env)', () => {
  const noEnvStack = new Stack(app, 'NoEnvStack'); // no env = token region
  const ni = new RouterNetworkInterface(noEnvStack, 'NI', {
    routerNetworkInterfaceName: 'test-ni',
    configuration: RouterNetworkConfiguration.publicNetwork({ cidr: ['10.0.0.0/16'] }),
  });

  expect(() => new RouterInput(noEnvStack, 'Input', {
    routerInputName: 'token-region-input',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      networkInterface: ni,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
      availabilityZone: 'us-east-1c',
    }),
  })).not.toThrow();
});

test('AZ validation throws when region and AZ mismatch (both concrete)', () => {
  expect(() => new RouterInput(stack, 'MismatchInput', {
    routerInputName: 'mismatch-input',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
      availabilityZone: 'us-west-2a',
    }),
  })).toThrow(/must be within region/);
});

test('AZ validation skips when AZ contains a token', () => {
  const noEnvStack = new Stack(app, 'TokenAzStack');
  const ni = new RouterNetworkInterface(noEnvStack, 'NI', {
    routerNetworkInterfaceName: 'test-ni-2',
    configuration: RouterNetworkConfiguration.publicNetwork({ cidr: ['10.0.0.0/16'] }),
  });

  expect(() => new RouterInput(noEnvStack, 'Input', {
    routerInputName: 'token-az-input',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterInputTier.INPUT_20,
    configuration: RouterInputConfiguration.standard({
      networkInterface: ni,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
      availabilityZone: `${noEnvStack.region}a`,
    }),
  })).not.toThrow();
});

test('RoutingScope of() and toString()', () => {
  expect(RoutingScope.of('CUSTOM').value).toBe('CUSTOM');
  expect(RoutingScope.REGIONAL.toString()).toBe('REGIONAL');
});

test('RouterInputTier of() and toString()', () => {
  expect(RouterInputTier.of('CUSTOM').value).toBe('CUSTOM');
  expect(RouterInputTier.INPUT_100.toString()).toBe('INPUT_100');
});

test('SourcePriorityConfig.none() renders NO_PRIORITY without a primary index', () => {
  const cfg = SourcePriorityConfig.none()._bind();
  expect(cfg.sourcePriorityMode).toBe('NO_PRIORITY');
  expect(cfg.primarySourceIndex).toBeUndefined();
});

test('SourcePriorityConfig.primarySecondary() renders PRIMARY_SECONDARY with the chosen index', () => {
  const first = SourcePriorityConfig.primarySecondary(PrimarySource.FIRST_SOURCE)._bind();
  expect(first.sourcePriorityMode).toBe('PRIMARY_SECONDARY');
  expect(first.primarySourceIndex).toBe(0);

  const second = SourcePriorityConfig.primarySecondary(PrimarySource.SECOND_SOURCE)._bind();
  expect(second.sourcePriorityMode).toBe('PRIMARY_SECONDARY');
  expect(second.primarySourceIndex).toBe(1);
});

test('imported router input throws when accessing routerInputId', () => {
  const imported = RouterInput.fromRouterInputArn(stack, 'ImportedInput', 'arn:aws:mediaconnect:us-east-1:123456789012:router-input:test');
  expect(() => imported.routerInputId).toThrow(/routerInputId.*is not available/);
});

test('imported router input throws when accessing ipAddress', () => {
  const imported = RouterInput.fromRouterInputArn(stack, 'ImportedInput2', 'arn:aws:mediaconnect:us-east-1:123456789012:router-input:test');
  expect(() => imported.ipAddress).toThrow(/ipAddress.*is not available/);
});

test('imported router input has undefined createdAt and updatedAt', () => {
  const imported = RouterInput.fromRouterInputArn(stack, 'ImportedInput3', 'arn:aws:mediaconnect:us-east-1:123456789012:router-input:test');
  expect(imported.createdAt).toBeUndefined();
  expect(imported.updatedAt).toBeUndefined();
});

test('standard router input has a single endpoint', () => {
  const input = new RouterInput(stack, 'UrlInput', {
    routerInputName: 'url-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
    }),
  });
  expect(input.endpoints).toHaveLength(1);
  expect(input.endpoints[0].port).toBe(5000);
});

test('failover router input has two endpoints sharing the input IP', () => {
  const input = new RouterInput(stack, 'FailoverUrlInput', {
    routerInputName: 'failover-url-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.rtp({ port: 5000 }),
        RouterInputProtocol.rtp({ port: 5001 }),
      ],
    }),
  });
  expect(input.endpoints).toHaveLength(2);
  expect(input.endpoints.map(e => e.port)).toEqual([5000, 5001]);
});

test('merge router input has two endpoints sharing the input IP', () => {
  const input = new RouterInput(stack, 'MergeUrlInput', {
    routerInputName: 'merge-url-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.merge({
      networkInterface,
      protocols: [
        RouterInputProtocol.rtp({ port: 6000 }),
        RouterInputProtocol.rtp({ port: 6001 }),
      ],
      mergeRecoveryWindow: Duration.millis(50),
    }),
  });
  expect(input.endpoints).toHaveLength(2);
  expect(input.endpoints.map(e => e.port)).toEqual([6000, 6001]);
});

test('SRT caller router input throws when accessing endpoints', () => {
  const input = new RouterInput(stack, 'SrtCallerInput', {
    routerInputName: 'srt-caller-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.srtCaller({
        sourceAddress: '203.0.113.50',
        sourcePort: 5000,
        minimumLatency: Duration.millis(1000),
      }),
    }),
  });
  expect(() => input.endpoints).toThrow(/endpoints.*is not available/);
});

test('flow-based router input throws when accessing endpoints', () => {
  const flow = new Flow(stack, 'TestFlow', {
    source: SourceConfiguration.rtp({
      flowSourceName: 'source',
      port: 5000,
      network: NetworkConfiguration.publicNetwork('10.0.0.0/8'),
    }),
  });
  const flowOutput = new FlowOutput(stack, 'TestFlowOutput', {
    flow,
    output: OutputConfiguration.router(),
  });
  const input = new RouterInput(stack, 'FlowUrlInput', {
    routerInputName: 'flow-url-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.mediaConnectFlow({
      flow,
      flowOutput,
    }),
  });
  expect(() => input.endpoints).toThrow(/endpoints.*is not available/);
});

test('imported router input throws when accessing endpoints', () => {
  const imported = RouterInput.fromRouterInputArn(stack, 'ImportedUrlInput', 'arn:aws:mediaconnect:us-east-1:123456789012:router-input:test');
  expect(() => imported.endpoints).toThrow(/endpoints.*is not available/);
});

test('MediaLive channel router input throws when accessing endpoints', () => {
  const input = new RouterInput(stack, 'MlcUrlInput', {
    routerInputName: 'mlc-url-test',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.mediaLiveChannel({
      mediaLiveChannelArn: 'arn:aws:medialive:us-east-1:123456789012:channel:1234567',
      mediaLiveChannelOutputName: 'output1',
      mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
    }),
  });
  expect(() => input.endpoints).toThrow(/endpoints.*is not available/);
});

test('SRT caller failover throws on endpoints', () => {
  const input = new RouterInput(stack, 'SrtCallerFailoverInput', {
    routerInputName: 'srt-caller-failover',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.failover({
      networkInterface,
      protocols: [
        RouterInputProtocol.srtCaller({
          sourceAddress: '203.0.113.50',
          sourcePort: 5000,
          minimumLatency: Duration.millis(1000),
        }),
        RouterInputProtocol.srtCaller({
          sourceAddress: '203.0.113.51',
          sourcePort: 5001,
          minimumLatency: Duration.millis(1000),
        }),
      ],
    }),
  });
  // SRT caller failover doesn't expose listening endpoints — both halves dial out.
  expect(() => input.endpoints).toThrow(/endpoints.*is not available/);
});

test.each([
  ['rtp', () => RouterInputProtocol.rtp({ port: 5000 }), 'rtp'],
  ['rist', () => RouterInputProtocol.rist({ port: 5000, recoveryLatency: Duration.millis(1000) }), 'rist'],
  ['srt-listener', () => RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(1000) }), 'srt'],
])('listening %s protocol exposes endpoints with the right URL scheme', (label, protocolFactory, scheme) => {
  const input = new RouterInput(stack, `Url-${label}`, {
    routerInputName: `url-${label.replace(/-/g, '')}`,
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: protocolFactory(),
    }),
  });
  expect(input.endpoints).toHaveLength(1);
  const endpoint = input.endpoints[0];
  expect(endpoint.port).toBe(5000);
  // URL is a token (Fn.join), so we synthesize and assert the scheme prefix.
  const stackForResolve = Stack.of(input);
  const resolved = JSON.stringify(stackForResolve.resolve(endpoint.url));
  expect(resolved).toContain(`${scheme}://`);
});

test('grants.start() adds correct IAM policy', () => {
  const role = new Role(stack, 'OrchestratorRole', {
    assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
  });

  const input = new RouterInput(stack, 'GrantInput', {
    routerInputName: 'grant-test',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(1000) }),
    }),
  });
  input.grants.start(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'mediaconnect:StartRouterInput',
        Effect: 'Allow',
        Resource: Match.anyValue(),
      }],
    },
  });
});

test('grants.restart() adds correct IAM policy', () => {
  const role = new Role(stack, 'OrchestratorRole', {
    assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
  });

  const input = new RouterInput(stack, 'RestartInput', {
    routerInputName: 'restart-test',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.srtListener({ port: 5000, minimumLatency: Duration.millis(1000) }),
    }),
  });
  input.grants.restart(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'mediaconnect:RestartRouterInput',
        Effect: 'Allow',
        Resource: Match.anyValue(),
      }],
    },
  });
});

test('metric() merges a caller-supplied dimensionsMap without dropping RouterInputARN', () => {
  const input = new RouterInput(stack, 'MetricInput', {
    routerInputName: 'metric-test',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
    }),
  });

  const metric = input.metric('RouterInputBitRate', { dimensionsMap: { Custom: 'value' } });

  expect(metric.dimensions).toEqual({
    RouterInputARN: input.routerInputArn,
    Custom: 'value',
  });
});

test('named metric helper keeps RouterInputARN when given a custom dimensionsMap', () => {
  const input = new RouterInput(stack, 'MetricInput2', {
    routerInputName: 'metric-test-2',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterInputConfiguration.standard({
      networkInterface,
      protocol: RouterInputProtocol.rtp({ port: 5000 }),
    }),
  });

  const metric = input.metricBitrate({ dimensionsMap: { Custom: 'value' } });

  expect(metric.dimensions).toEqual({
    RouterInputARN: input.routerInputArn,
    Custom: 'value',
  });
});
