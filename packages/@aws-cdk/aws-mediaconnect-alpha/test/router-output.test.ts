import { App, ArnFormat, Bitrate, Duration, Lazy, Stack, Token } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import * as medialive from 'aws-cdk-lib/aws-medialive';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Flow } from '../lib/flow';
import { SourceConfiguration, NetworkConfiguration } from '../lib/flow-source-configuration';
import { ForwardErrorCorrection, RoutingScope } from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';
import { RouterOutput, RouterOutputConfiguration, RouterOutputProtocol, RouterOutputTier } from '../lib/router-output';
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

describe('RouterOutput', () => {
  test('creates router output with standard RTP configuration', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-rtp',
      maximumBitrate: Bitrate.mbps(5),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_100,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.rtp({
          destinationAddress: '198.51.100.100',
          port: 5000,
          forwardErrorCorrection: ForwardErrorCorrection.ENABLED,
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-rtp',
      MaximumBitrate: 5000000,
      RoutingScope: 'GLOBAL',
      Tier: 'OUTPUT_100',
      Configuration: {
        Standard: {
          NetworkInterfaceArn: {
            'Fn::GetAtt': ['network39EEAA36', 'Arn'],
          },
          Protocol: 'RTP',
          ProtocolConfiguration: {
            Rtp: {
              DestinationAddress: '198.51.100.100',
              DestinationPort: 5000,
              ForwardErrorCorrection: 'ENABLED',
            },
          },
        },
      },
    });
  });

  test('creates router output with standard RIST configuration', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-rist',
      maximumBitrate: Bitrate.mbps(10),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.rist({
          destinationAddress: '198.51.100.101',
          port: 5001,
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-rist',
      MaximumBitrate: 10000000,
      RoutingScope: 'REGIONAL',
      Tier: 'OUTPUT_50',
      Configuration: {
        Standard: {
          NetworkInterfaceArn: {
            'Fn::GetAtt': ['network39EEAA36', 'Arn'],
          },
          Protocol: 'RIST',
          ProtocolConfiguration: {
            Rist: {
              DestinationAddress: '198.51.100.101',
              DestinationPort: 5001,
            },
          },
        },
      },
    });
  });

  test('creates router output with SRT Listener configuration', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-srt-listener',
      maximumBitrate: Bitrate.mbps(8),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_20,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.srtListener({
          port: 9001,
          minimumLatency: Duration.millis(200),
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-srt-listener',
      Configuration: {
        Standard: {
          Protocol: 'SRT_LISTENER',
          ProtocolConfiguration: {
            SrtListener: {
              Port: 9001,
              MinimumLatencyMilliseconds: 200,
            },
          },
        },
      },
    });
  });

  test('creates router output with SRT Caller configuration', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-srt-caller',
      maximumBitrate: Bitrate.mbps(12),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_100,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.srtCaller({
          destinationAddress: '198.51.100.102',
          destinationPort: 9002,
          minimumLatency: Duration.millis(150),
          streamId: 'test-stream-123',
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-srt-caller',
      Configuration: {
        Standard: {
          Protocol: 'SRT_CALLER',
          ProtocolConfiguration: {
            SrtCaller: {
              DestinationAddress: '198.51.100.102',
              DestinationPort: 9002,
              MinimumLatencyMilliseconds: 150,
              StreamId: 'test-stream-123',
            },
          },
        },
      },
    });
  });

  test('creates router output with SRT encryption configuration', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-srt-encrypted',
      maximumBitrate: Bitrate.mbps(15),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.srtListener({
          port: 9003,
          minimumLatency: Duration.millis(300),
          encryptionConfiguration: ({
            role,
            secret,
          }),
        }),
        networkInterface,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Configuration: {
        Standard: {
          ProtocolConfiguration: {
            SrtListener: {
              Port: 9003,
              MinimumLatencyMilliseconds: 300,
              EncryptionConfiguration: {
                EncryptionKey: {
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
      },
    });
  });

  test('creates router output with MediaLive input configuration (automatic encryption)', () => {
    const mlInput = new medialive.CfnInput(stack, 'MlInput', { name: 'ml-in', type: 'MEDIACONNECT_ROUTER' });
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-medialive',
      maximumBitrate: Bitrate.mbps(20),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_100,
      configuration: RouterOutputConfiguration.mediaLiveInput({
        input: mlInput,
        pipeline: MediaLivePipeline.PIPELINE_0,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-medialive',
      MaximumBitrate: 20000000,
      Configuration: {
        MediaLiveInput: {
          MediaLiveInputArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('MlInput'), 'Arn'] },
          MediaLivePipelineId: 'PIPELINE_0',
          DestinationTransitEncryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });

  test('creates router output with MediaLive input configuration (secrets manager encryption)', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');
    const mlInput = new medialive.CfnInput(stack, 'MlInput', { name: 'ml-in', type: 'MEDIACONNECT_ROUTER' });

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-medialive-encrypted',
      maximumBitrate: Bitrate.mbps(15),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_20,
      configuration: RouterOutputConfiguration.mediaLiveInput({
        input: mlInput,
        pipeline: MediaLivePipeline.PIPELINE_1,
        destinationTransitEncryption: ({ role, secret }),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Configuration: {
        MediaLiveInput: {
          MediaLiveInputArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('MlInput'), 'Arn'] },
          MediaLivePipelineId: 'PIPELINE_1',
          DestinationTransitEncryption: {
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

  test('creates router output with MediaConnect Flow configuration (automatic encryption)', () => {
    const testFlow = new Flow(stack, 'TestFlow', {
      flowName: 'test-flow',
      source: SourceConfiguration.rtp({
        flowSourceName: 'test-source',
        port: 5000,
        network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
      }),
    });

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-flow',
      maximumBitrate: Bitrate.mbps(18),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.mediaConnectFlow({
        flow: testFlow,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-flow',
      MaximumBitrate: 18000000,
      Configuration: {
        MediaConnectFlow: {
          FlowArn: {
            'Fn::GetAtt': ['TestFlow1799437A', 'FlowArn'],
          },
          DestinationTransitEncryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });

  test('creates router output with MediaConnect Flow configuration (secrets manager encryption)', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');
    const testFlow = new Flow(stack, 'TestFlowEncrypted', {
      flowName: 'test-flow-encrypted',
      source: SourceConfiguration.rtp({
        flowSourceName: 'test-encrypted-source',
        port: 5001,
        network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
      }),
    });

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-flow-encrypted',
      maximumBitrate: Bitrate.mbps(22),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_100,
      configuration: RouterOutputConfiguration.mediaConnectFlow({
        flow: testFlow,
        destinationTransitEncryption: ({ role, secret }),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Configuration: {
        MediaConnectFlow: {
          FlowArn: {
            'Fn::GetAtt': ['TestFlowEncrypted0B8361DC', 'FlowArn'],
          },
          DestinationTransitEncryption: {
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

  test('creates router output with MediaConnect Flow configuration without flow connection', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-flow-no-connection',
      maximumBitrate: Bitrate.mbps(16),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.mediaConnectFlowWithoutConnection({
        availabilityZone: 'us-east-1c',
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-flow-no-connection',
      MaximumBitrate: 16000000,
      AvailabilityZone: 'us-east-1c',
      Configuration: {
        MediaConnectFlow: {
          // No FlowArn - "Do not connect to a MediaConnect flow"
          DestinationTransitEncryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });

  test('creates router output with MediaConnect Flow no connection (with encryption)', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-flow-no-connection-encrypted',
      maximumBitrate: Bitrate.mbps(20),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_20,
      configuration: RouterOutputConfiguration.mediaConnectFlowWithoutConnection({
        availabilityZone: 'us-east-1d',
        destinationTransitEncryption: ({ role, secret }),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      AvailabilityZone: 'us-east-1d',
      Configuration: {
        MediaConnectFlow: {
          // No FlowArn
          DestinationTransitEncryption: {
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

  test('creates router output with maintenance configuration', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-maintenance',
      maximumBitrate: Bitrate.mbps(10),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_20,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.rtp({
          destinationAddress: '198.51.100.200',
          port: 5200,
        }),
        networkInterface,
      }),
      maintenanceConfiguration: {
        day: MaintenanceDay.SUNDAY,
        time: '03:00',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      MaintenanceConfiguration: {
        PreferredDayTime: {
          Day: 'SUNDAY',
          Time: '03:00',
        },
      },
    });
  });

  test('creates router output with tags', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-tags',
      maximumBitrate: Bitrate.mbps(8),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.rist({
          destinationAddress: '198.51.100.30',
          port: 5300,
        }),
        networkInterface,
      }),
      tags: {
        Environment: 'test',
        Project: 'router-output',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Tags: [
        { Key: 'Environment', Value: 'test' },
        { Key: 'Project', Value: 'router-output' },
      ],
    });
  });

  test('can import existing router output from attributes', () => {
    const imported = RouterOutput.fromRouterOutputAttributes(
      stack,
      'ImportedOutput',
      {
        routerOutputArn: 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:existing-output',
        routerOutputName: 'existing-output',
      },
    );

    expect(imported.routerOutputArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:router-output:existing-output');
    expect(imported.routerOutputName).toBe('existing-output');
  });

  test('creates router output with MediaLive configuration without input connection', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-medialive-no-input',
      maximumBitrate: Bitrate.mbps(15),
      routingScope: RoutingScope.REGIONAL,
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
        availabilityZone: 'us-east-1a',
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-medialive-no-input',
      MaximumBitrate: 15000000,
      AvailabilityZone: 'us-east-1a',
      Configuration: {
        MediaLiveInput: {
          // No MediaLiveInputArn or MediaLivePipelineId - "Do not connect to a MediaLive input"
          DestinationTransitEncryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });

  test('creates router output with MediaLive no input configuration (with encryption)', () => {
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
    });
    const secret = new Secret(stack, 'TestSecret');

    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-medialive-no-input-encrypted',
      maximumBitrate: Bitrate.mbps(18),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_20,
      configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
        availabilityZone: 'us-east-1b',
        destinationTransitEncryption: ({ role, secret }),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      AvailabilityZone: 'us-east-1b',
      Configuration: {
        MediaLiveInput: {
          // No MediaLiveInputArn or MediaLivePipelineId
          DestinationTransitEncryption: {
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

  test('generates unique name when routerOutputName is not provided', () => {
    new RouterOutput(stack, 'routerOutput', {
      maximumBitrate: Bitrate.mbps(5),
      routingScope: RoutingScope.GLOBAL,
      tier: RouterOutputTier.OUTPUT_100,
      configuration: RouterOutputConfiguration.standard({
        protocol: RouterOutputProtocol.rtp({
          destinationAddress: '198.51.100.40',
          port: 5400,
        }),
        networkInterface,
      }),
    });

    // Should create resource without explicit name (CDK will generate one)
    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      MaximumBitrate: 5000000,
      RoutingScope: 'GLOBAL',
      Tier: 'OUTPUT_100',
    });
  });

  test('creates router output without MediaLive connection - 2 azs specified', () => {
    new RouterOutput(stack, 'routerOutput', {
      routerOutputName: 'test-medialive-no-input',
      maximumBitrate: Bitrate.mbps(15),
      routingScope: RoutingScope.REGIONAL,
      regionName: 'eu-west-1',
      tier: RouterOutputTier.OUTPUT_50,
      configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
        availabilityZone: 'eu-west-1a',
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::RouterOutput', {
      Name: 'test-medialive-no-input',
      MaximumBitrate: 15000000,
      AvailabilityZone: 'eu-west-1a',
      Configuration: {
        MediaLiveInput: {
          // No MediaLiveInputArn or MediaLivePipelineId - "Do not connect to a MediaLive input"
          DestinationTransitEncryption: {
            EncryptionKeyType: 'AUTOMATIC',
            EncryptionKeyConfiguration: {
              Automatic: {},
            },
          },
        },
      },
    });
  });

  test('RouterOutput with default region from stack, and configured AZ in output - so 2 different AZs - produces error', () => {
    expect(()=>{
      new RouterOutput(stack, 'routerOutput', {
        routerOutputName: 'test-medialive-no-input',
        maximumBitrate: Bitrate.mbps(15),
        routingScope: RoutingScope.REGIONAL,
        tier: RouterOutputTier.OUTPUT_50,
        configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
          availabilityZone: 'eu-west-1a',
        }),
      });
    }).toThrow('Availability zone \'eu-west-1a\' must be within region \'us-east-1');
  });
});

describe('maximum bitrate validation', () => {
  test('throws when below the 1 Mbps minimum', () => {
    expect(() => {
      new RouterOutput(stack, 'routerOutput', {
        maximumBitrate: Bitrate.bps(500000),
        routingScope: RoutingScope.REGIONAL,
        configuration: RouterOutputConfiguration.standard({
          networkInterface,
          protocol: RouterOutputProtocol.rtp({ destinationAddress: '198.51.100.1', port: 5000 }),
        }),
      });
    }).toThrow('Maximum bitrate must be at least 1,000,000 bits/s (1 Mbps)');
  });

  test('throws when exceeding the selected tier limit', () => {
    expect(() => {
      new RouterOutput(stack, 'routerOutput', {
        maximumBitrate: Bitrate.mbps(30),
        routingScope: RoutingScope.REGIONAL,
        tier: RouterOutputTier.OUTPUT_20,
        configuration: RouterOutputConfiguration.standard({
          networkInterface,
          protocol: RouterOutputProtocol.rtp({ destinationAddress: '198.51.100.1', port: 5000 }),
        }),
      });
    }).toThrow('exceeds the OUTPUT_20 tier limit of 20 Mbps');
  });

  test('skips bitrate validation for a tokenized maximum bitrate', () => {
    expect(() => {
      new RouterOutput(stack, 'routerOutput', {
        maximumBitrate: Bitrate.bps(Lazy.number({ produce: () => 5000000 })),
        routingScope: RoutingScope.REGIONAL,
        tier: RouterOutputTier.OUTPUT_20,
        configuration: RouterOutputConfiguration.standard({
          networkInterface,
          protocol: RouterOutputProtocol.rtp({ destinationAddress: '198.51.100.1', port: 5000 }),
        }),
      });
    }).not.toThrow();
  });
});

test('imported router output has routerOutputId from ARN', () => {
  const imported = RouterOutput.fromRouterOutputArn(stack, 'ImportedOutput', 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:test');
  // For concrete ARNs, Fn.select + Fn.split evaluates at synth time to the literal ID.
  expect(stack.resolve(imported.routerOutputId)).toBe('test');
});

test('imported router output throws when accessing routerOutputName', () => {
  const imported = RouterOutput.fromRouterOutputArn(stack, 'ImportedOutputName', 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:test');
  expect(() => imported.routerOutputName).toThrow(/routerOutputName.*was not provided/);
});

test('imported router output from attributes has name', () => {
  const imported = RouterOutput.fromRouterOutputAttributes(stack, 'ImportedOutputAttrs', {
    routerOutputArn: 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:abc123',
    routerOutputName: 'my-router-output',
  });
  expect(imported.routerOutputName).toBe('my-router-output');
});

test('imported router output from attributes with explicit routerOutputId', () => {
  const imported = RouterOutput.fromRouterOutputAttributes(stack, 'ImportedOutputId', {
    routerOutputArn: 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:abc123',
    routerOutputId: 'explicit-id',
  });
  expect(imported.routerOutputId).toBe('explicit-id');
});

test('imported router output from attributes resolves literal id from ARN', () => {
  // formatArn with concrete env produces an ARN whose resource name is statically
  // known. Arn.split() can extract the literal id at synth time, so routerOutputId
  // is the bare string rather than a CFN intrinsic.
  const arn = Stack.of(stack).formatArn({
    service: 'mediaconnect',
    resource: 'router-output',
    resourceName: 'abc123',
    arnFormat: ArnFormat.COLON_RESOURCE_NAME,
  });
  const imported = RouterOutput.fromRouterOutputAttributes(stack, 'ImportedFormatArn', {
    routerOutputArn: arn,
    routerOutputName: 'my-output',
  });
  expect(imported.routerOutputId).toBe('abc123');
});

test('imported router output from attributes works with token ARN', () => {
  // Simulate an attrArn-style fully-tokenised ARN (the kind that comes from
  // referencing a CFN resource attribute). With this, Arn.split() can't extract
  // the resourceName at synth time, so routerOutputId remains a token.
  const tokenArn = Lazy.string({
    produce: () => 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:abc123',
  });
  const imported = RouterOutput.fromRouterOutputAttributes(stack, 'ImportedTokenArn', {
    routerOutputArn: tokenArn,
    routerOutputName: 'my-output',
  });
  expect(imported.routerOutputName).toBe('my-output');
  // routerOutputId should be a CFN intrinsic that resolves at deploy time
  expect(Token.isUnresolved(imported.routerOutputId)).toBe(true);
});

test('imported router output throws when accessing ipAddress', () => {
  const imported = RouterOutput.fromRouterOutputArn(stack, 'ImportedOutput2', 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:test');
  expect(() => imported.ipAddress).toThrow(/ipAddress.*is not available/);
});

test('imported router output has undefined createdAt and updatedAt', () => {
  const imported = RouterOutput.fromRouterOutputArn(stack, 'ImportedOutput3', 'arn:aws:mediaconnect:us-east-1:123456789012:router-output:test');
  expect(imported.createdAt).toBeUndefined();
  expect(imported.updatedAt).toBeUndefined();
});

test('MediaLive input transit encryption auto-creates a role and orders its policy before the router output', () => {
  const secret = new Secret(stack, 'secret');
  const mlInput = new medialive.CfnInput(stack, 'MlInput', { name: 'ml-in', type: 'MEDIACONNECT_ROUTER' });

  new RouterOutput(stack, 'routerOutput', {
    routerOutputName: 'transit-auto-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterOutputTier.OUTPUT_20,
    configuration: RouterOutputConfiguration.mediaLiveInput({
      input: mlInput,
      pipeline: MediaLivePipeline.PIPELINE_0,
      destinationTransitEncryption: { secret },
    }),
  });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::MediaConnect::RouterOutput', {
    Configuration: {
      MediaLiveInput: {
        DestinationTransitEncryption: { EncryptionKeyType: 'SECRETS_MANAGER' },
      },
    },
  });

  template.hasResource('AWS::MediaConnect::RouterOutput', {
    DependsOn: Match.arrayWith([Match.stringLikeRegexp('DestinationTransitEncryptionRoleDefaultPolicy')]),
  });
});

test('SRT output encryption auto-creates a role and orders its policy before the router output', () => {
  const secret = new Secret(stack, 'secret');

  new RouterOutput(stack, 'routerOutput', {
    routerOutputName: 'srt-auto-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterOutputTier.OUTPUT_20,
    configuration: RouterOutputConfiguration.standard({
      protocol: RouterOutputProtocol.srtCaller({
        destinationAddress: '203.0.113.10',
        destinationPort: 9001,
        minimumLatency: Duration.millis(200),
        encryptionConfiguration: { secret },
      }),
      networkInterface,
    }),
  });

  Template.fromStack(stack).hasResource('AWS::MediaConnect::RouterOutput', {
    DependsOn: Match.arrayWith([Match.stringLikeRegexp('EncryptionRoleDefaultPolicy')]),
  });
});

test('MediaLive input transit encryption defaults to AUTOMATIC and adds no role or dependency when no secret is given', () => {
  const mlInput = new medialive.CfnInput(stack, 'MlInput', { name: 'ml-in', type: 'MEDIACONNECT_ROUTER' });

  new RouterOutput(stack, 'routerOutput', {
    routerOutputName: 'transit-automatic',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterOutputTier.OUTPUT_20,
    configuration: RouterOutputConfiguration.mediaLiveInput({
      input: mlInput,
      pipeline: MediaLivePipeline.PIPELINE_0,
    }),
  });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::MediaConnect::RouterOutput', {
    Configuration: {
      MediaLiveInput: {
        DestinationTransitEncryption: { EncryptionKeyType: 'AUTOMATIC' },
      },
    },
  });
  // No secret means no auto-created role and no ordering dependency on the router output.
  const roles = template.findResources('AWS::IAM::Role', {
    Properties: { Description: 'Auto-generated MediaConnect role for accessing the encryption secret' },
  });
  expect(Object.keys(roles)).toHaveLength(0);
  template.hasResource('AWS::MediaConnect::RouterOutput', { DependsOn: Match.absent() });
});

test('MediaLive input transit encryption with an explicit role does not auto-create a role', () => {
  const role = new Role(stack, 'role', { assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com') });
  const secret = new Secret(stack, 'secret');
  const mlInput = new medialive.CfnInput(stack, 'MlInput', { name: 'ml-in', type: 'MEDIACONNECT_ROUTER' });

  new RouterOutput(stack, 'routerOutput', {
    routerOutputName: 'transit-explicit-role',
    maximumBitrate: Bitrate.mbps(5),
    routingScope: RoutingScope.REGIONAL,
    tier: RouterOutputTier.OUTPUT_20,
    configuration: RouterOutputConfiguration.mediaLiveInput({
      input: mlInput,
      pipeline: MediaLivePipeline.PIPELINE_0,
      destinationTransitEncryption: { role, secret },
    }),
  });

  const roles = Template.fromStack(stack).findResources('AWS::IAM::Role', {
    Properties: { Description: 'Auto-generated MediaConnect role for accessing the encryption secret' },
  });
  expect(Object.keys(roles)).toHaveLength(0);
});

test('metric() merges a caller-supplied dimensionsMap without dropping RouterOutputARN', () => {
  const output = new RouterOutput(stack, 'MetricOutput', {
    routerOutputName: 'metric-test',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterOutputConfiguration.standard({
      networkInterface,
      protocol: RouterOutputProtocol.rtp({ destinationAddress: '198.51.100.1', port: 5000 }),
    }),
  });

  const metric = output.metric('RouterOutputBitRate', { dimensionsMap: { Custom: 'value' } });

  expect(metric.dimensions).toEqual({
    RouterOutputARN: output.routerOutputArn,
    Custom: 'value',
  });
});

test('named metric helper keeps RouterOutputARN when given a custom dimensionsMap', () => {
  const output = new RouterOutput(stack, 'MetricOutput2', {
    routerOutputName: 'metric-test-2',
    maximumBitrate: Bitrate.mbps(10),
    routingScope: RoutingScope.REGIONAL,
    configuration: RouterOutputConfiguration.standard({
      networkInterface,
      protocol: RouterOutputProtocol.rtp({ destinationAddress: '198.51.100.1', port: 5000 }),
    }),
  });

  const metric = output.metricBitrate({ dimensionsMap: { Custom: 'value' } });

  expect(metric.dimensions).toEqual({
    RouterOutputARN: output.routerOutputArn,
    Custom: 'value',
  });
});
