import { App, Bitrate, Lazy, Stack } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { Vpc, IpAddresses, SubnetType, PrivateSubnet, SecurityGroup } from 'aws-cdk-lib/aws-ec2';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Bridge, BridgeConfiguration, BridgeFailoverConfig, BridgeOutputConfiguration, BridgeType } from '../lib/bridge';
import { Flow } from '../lib/flow';
import { Gateway } from '../lib/gateway';
import { BridgeProtocol, State, VpcInterface, GatewayNetwork } from '../lib/shared';

let app: App;
let stack: Stack;
beforeEach(() => {
  app = new App();
  stack = new Stack(app, undefined, {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

test('MediaConnect Bridge Creation', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'aaaaaa');
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
  new Bridge(stack, 'bridge', {
    config: BridgeConfiguration.egress({
      maxBitrate: Bitrate.mbps(5),
      flowSources: [{
        name: 'aaaa',
        source: {
          vpcInterface,
          flow: Flow.fromFlowAttributes(stack, 'flow', {
            flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-CQwNVVFcCVgNBg8D-3104ecf5a408:my-flow',
            sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-CQwNVVFcCVgNBg8D-3104ecf5a408:test',
          }),
        },
      }],
      networkOutputs: [{
        name: 'network',
        output: BridgeOutputConfiguration.network({
          ipAddress: '192.168.1.60',
          network: GatewayNetwork.define({ name: 'my-test-network', cidrBlock: '198.51.100.0/24' }),
          port: 5000,
          protocol: BridgeProtocol.RTP,
          ttl: 64,
        }),
      }],
    }),
    gateway,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Bridge', {
    Outputs: [{
      NetworkOutput: {
        IpAddress: '192.168.1.60',
        Name: 'network',
        NetworkName: 'my-test-network',
        Port: 5000,
        Protocol: 'rtp',
        Ttl: 64,
      },
    }],
    PlacementArn: 'aaaaaa',
    Sources: [{
      FlowSource: {
        FlowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-CQwNVVFcCVgNBg8D-3104ecf5a408:my-flow',
        FlowVpcInterfaceAttachment: {
          VpcInterfaceName: 'test',
        },
      },
    }],
  });
});

test('MediaConnect Bridge Creation - ingress bridge with network sources', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'aaaaaa');

  new Bridge(stack, 'bridge', {
    config: BridgeConfiguration.ingress({
      maxBitrate: Bitrate.mbps(5),
      maxOutputs: 2,
      networkSources: [{
        name: 'network-source',
        source: {
          multicastIp: '239.1.1.1',
          network: GatewayNetwork.define({ name: 'my-network', cidrBlock: '198.51.100.0/24' }),
          port: 5000,
          protocol: BridgeProtocol.RTP,
          multicastSourceIp: '192.168.1.1',
        },
      }],
    }),
    gateway,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Bridge', {
    IngressGatewayBridge: {
      MaxBitrate: 5000000,
      MaxOutputs: 2,
    },
    Sources: [{
      NetworkSource: {
        Name: 'network-source',
        MulticastIp: '239.1.1.1',
        NetworkName: 'my-network',
        Port: 5000,
        Protocol: 'rtp',
        MulticastSourceSettings: {
          MulticastSourceIp: '192.168.1.1',
        },
      },
    }],
  });
});

// Validation tests
test('Bridge name validation - too long', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-23456789ABCDEFGH:test-gateway');

  const flow = Flow.fromFlowAttributes(stack, 'flow', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-test:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-test:source',
  });

  expect(() => {
    new Bridge(stack, 'bridge', {
      bridgeName: 'a'.repeat(65),
      gateway,
      config: BridgeConfiguration.egress({
        maxBitrate: Bitrate.mbps(100),
        flowSources: [{ name: 'source', source: { flow } }],
        networkOutputs: [],
      }),
    });
  }).toThrow('Bridge name must be between 1 and 64 characters');
});

test('Bridge name validation - invalid characters', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway2', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-23456789ABCDEFGH:test-gateway');

  const flow = Flow.fromFlowAttributes(stack, 'flow2', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-test:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-test:source',
  });

  expect(() => {
    new Bridge(stack, 'bridge', {
      bridgeName: 'invalid@name!',
      gateway,
      config: BridgeConfiguration.egress({
        maxBitrate: Bitrate.mbps(100),
        flowSources: [{ name: 'source', source: { flow } }],
        networkOutputs: [],
      }),
    });
  }).toThrow('Bridge name must contain only alphanumeric characters, hyphens, and underscores');
});

test('Bridge name validation - valid name', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway3', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-23456789ABCDEFGH:test-gateway');

  const flow = Flow.fromFlowAttributes(stack, 'flow3', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-test:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-test:source',
  });

  new Bridge(stack, 'bridge', {
    bridgeName: 'Valid-Bridge-Name',
    gateway,
    config: BridgeConfiguration.egress({
      maxBitrate: Bitrate.mbps(100),
      flowSources: [{ name: 'source', source: { flow } }],
      networkOutputs: [],
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Bridge', {
    Name: 'Valid-Bridge-Name',
  });
});

test('Bridge TTL validation - out of range', () => {
  expect(() => {
    BridgeOutputConfiguration.network({
      ipAddress: '192.168.1.50',
      network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
      port: 5000,
      protocol: BridgeProtocol.RTP,
      ttl: 0,
    });
  }).toThrow(/TTL must be between 1 and 255/);
});

test('Bridge TTL validation - too high', () => {
  expect(() => {
    BridgeOutputConfiguration.network({
      ipAddress: '192.168.1.50',
      network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
      port: 5000,
      protocol: BridgeProtocol.RTP,
      ttl: 256,
    });
  }).toThrow(/TTL must be between 1 and 255/);
});

test('Bridge ingress bitrate validation - too low', () => {
  expect(() => {
    BridgeConfiguration.ingress({
      maxBitrate: Bitrate.kbps(500),
      maxOutputs: 1,
      networkSources: [{
        name: 'source',
        source: {
          multicastIp: '239.0.0.1',
          network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
          port: 5000,
          protocol: BridgeProtocol.RTP,
        },
      }],
    });
  }).toThrow(/Bridge ingress max bitrate/);
});

test('Bridge ingress bitrate validation - skipped for a tokenized bitrate', () => {
  expect(() => {
    BridgeConfiguration.ingress({
      maxBitrate: Bitrate.bps(Lazy.number({ produce: () => 5000000 })),
      maxOutputs: 1,
      networkSources: [{
        name: 'source',
        source: {
          multicastIp: '239.0.0.1',
          network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
          port: 5000,
          protocol: BridgeProtocol.RTP,
        },
      }],
    });
  }).not.toThrow();
});

test('Bridge addOutput on egress bridge', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-abc:gw');

  const flow = Flow.fromFlowAttributes(stack, 'flow', {
    flowArn: 'arn:aws:mediaconnect:us-east-1:123456789012:flow:1-abc:my-flow',
    sourceArn: 'arn:aws:mediaconnect:us-east-1:123456789012:source:1-abc:source',
  });

  const bridge = new Bridge(stack, 'bridge', {
    gateway,
    config: BridgeConfiguration.egress({
      maxBitrate: Bitrate.mbps(10),
      flowSources: [{ name: 'source', source: { flow } }],
      networkOutputs: [],
    }),
  });

  bridge.addOutput('extra-output', {
    name: 'extra',
    output: BridgeOutputConfiguration.network({
      ipAddress: '10.0.0.1',
      network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
      port: 6000,
      protocol: BridgeProtocol.RTP_FEC,
      ttl: 128,
    }),
  });

  Template.fromStack(stack).resourceCountIs('AWS::MediaConnect::BridgeOutput', 1);
});

test('Bridge addOutput on ingress bridge throws', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-abc:gw');

  const bridge = new Bridge(stack, 'bridge', {
    gateway,
    config: BridgeConfiguration.ingress({
      maxBitrate: Bitrate.mbps(5),
      maxOutputs: 1,
      networkSources: [{
        name: 'source',
        source: {
          multicastIp: '239.0.0.1',
          network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
          port: 5000,
          protocol: BridgeProtocol.RTP,
        },
      }],
    }),
  });

  expect(() => {
    bridge.addOutput('output', {
      name: 'out',
      output: BridgeOutputConfiguration.network({
        ipAddress: '10.0.0.1',
        network: GatewayNetwork.define({ name: 'net', cidrBlock: '198.51.100.0/24' }),
        port: 6000,
        protocol: BridgeProtocol.RTP,
        ttl: 64,
      }),
    });
  }).toThrow(/addOutput can only be called on egress bridges/);
});

test('Bridge fromBridgeAttributes', () => {
  const bridge = Bridge.fromBridgeAttributes(stack, 'imported', {
    bridgeArn: 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge',
    bridgeType: BridgeType.EGRESS_BRIDGE,
  });

  expect(bridge.bridgeArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge');
});

test('Bridge fromBridgeArn', () => {
  const bridge = Bridge.fromBridgeArn(stack, 'imported', 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge');
  expect(bridge.bridgeArn).toBe('arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge');
});

test('fails - Bridge imported by ARN throws when accessing bridgeType', () => {
  const bridge = Bridge.fromBridgeArn(stack, 'imported', 'arn:aws:mediaconnect:us-east-1:123456789012:bridge:1-abc:my-bridge');
  expect(() => bridge.bridgeType).toThrow(/bridgeType.*was not provided/);
});

test('BridgeProtocol of() and toString()', () => {
  expect(BridgeProtocol.of('custom').value).toBe('custom');
  expect(BridgeProtocol.RTP.toString()).toBe('rtp');
});

test('BridgeType toString()', () => {
  expect(BridgeType.EGRESS_BRIDGE.toString()).toBe('egress_bridge');
  expect(BridgeType.INGRESS_BRIDGE.toString()).toBe('ingress_bridge');
});

test('BridgeFailoverConfig.failover defaults to ENABLED state', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-abc:g');

  const bridge = new Bridge(stack, 'bridge', {
    bridgeName: 'b',
    config: BridgeConfiguration.ingress({
      maxBitrate: Bitrate.mbps(5),
      maxOutputs: 1,
      networkSources: [],
    }),
    sourceFailoverConfig: BridgeFailoverConfig.failover({ primarySource: 'src' }),
    gateway,
  });

  expect(bridge.isFailoverEnabled).toBe(true);
  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Bridge', {
    SourceFailoverConfig: {
      FailoverMode: 'FAILOVER',
      State: 'ENABLED',
      SourcePriority: { PrimarySource: 'src' },
    },
  });
});

test('BridgeFailoverConfig.failover with state: DISABLED persists config without enabling', () => {
  const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'arn:aws:mediaconnect:us-east-1:123456789012:gateway:1-abc:g');

  const bridge = new Bridge(stack, 'bridge', {
    bridgeName: 'b',
    config: BridgeConfiguration.ingress({
      maxBitrate: Bitrate.mbps(5),
      maxOutputs: 1,
      networkSources: [],
    }),
    sourceFailoverConfig: BridgeFailoverConfig.failover({
      state: State.DISABLED,
      primarySource: 'src',
    }),
    gateway,
  });

  expect(bridge.isFailoverEnabled).toBe(false);
  Template.fromStack(stack).hasResourceProperties('AWS::MediaConnect::Bridge', {
    SourceFailoverConfig: {
      FailoverMode: 'FAILOVER',
      State: 'DISABLED',
    },
  });
});

describe('bridge metrics preserve BridgeARN when a custom dimensionsMap is given', () => {
  function makeBridge(): Bridge {
    const gateway = Gateway.fromGatewayArn(stack, 'gateway', 'aaaaaa');
    return new Bridge(stack, 'bridge', {
      config: BridgeConfiguration.ingress({
        maxBitrate: Bitrate.mbps(5),
        maxOutputs: 2,
        networkSources: [],
      }),
      gateway,
    });
  }

  test('metric() merges the dimensionsMap', () => {
    const bridge = makeBridge();
    const metric = bridge.metric('IngressBridgeBitRate', { dimensionsMap: { Custom: 'value' } });
    expect(metric.dimensions).toEqual({
      BridgeARN: bridge.bridgeArn,
      Custom: 'value',
    });
  });

  test('metricSourceBitrate() keeps BridgeARN and BridgeSourceName', () => {
    const bridge = makeBridge();
    const metric = bridge.metricSourceBitrate('my-source', { dimensionsMap: { Custom: 'value' } });
    expect(metric.dimensions).toEqual({
      BridgeARN: bridge.bridgeArn,
      BridgeSourceName: 'my-source',
      Custom: 'value',
    });
  });
});
