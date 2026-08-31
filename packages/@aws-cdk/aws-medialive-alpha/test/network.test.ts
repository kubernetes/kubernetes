import { App, Stack } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { Network } from '../lib';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

describe('Network', () => {
  test('creates a minimal network', () => {
    new Network(stack, 'Net', {
      networkName: 'my-network',
      ipPools: ['10.0.0.0/16'],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Network', {
      Name: 'my-network',
      IpPools: [{ Cidr: '10.0.0.0/16' }],
    });
  });

  test('renders multiple IP pools and routes', () => {
    new Network(stack, 'Net', {
      networkName: 'multi-pool-network',
      ipPools: ['10.0.0.0/16', '10.1.0.0/16'],
      routes: [
        { cidr: '0.0.0.0/0', gateway: '10.0.0.1' },
        { cidr: '172.16.0.0/12', gateway: '10.1.0.1' },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Network', {
      IpPools: [{ Cidr: '10.0.0.0/16' }, { Cidr: '10.1.0.0/16' }],
      Routes: [
        { Cidr: '0.0.0.0/0', Gateway: '10.0.0.1' },
        { Cidr: '172.16.0.0/12', Gateway: '10.1.0.1' },
      ],
    });
  });

  test('renders tags', () => {
    new Network(stack, 'Net', {
      networkName: 'tagged-network',
      ipPools: ['10.0.0.0/16'],
      tags: { env: 'prod' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Network', {
      Tags: [{ Key: 'env', Value: 'prod' }],
    });
  });

  test('omits routes and tags when not provided', () => {
    new Network(stack, 'Net', {
      networkName: 'bare-network',
      ipPools: ['10.0.0.0/16'],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Network', {
      Routes: Match.absent(),
      Tags: Match.absent(),
    });
  });

  test('fromNetworkAttributes wires the provided attributes', () => {
    const imported = Network.fromNetworkAttributes(stack, 'Imported', {
      networkArn: 'arn:aws:medialive:us-east-1:123456789012:network:network-123',
      networkId: 'network-123',
    });

    expect(imported.networkArn).toBe('arn:aws:medialive:us-east-1:123456789012:network:network-123');
    expect(imported.networkId).toBe('network-123');
    expect(imported.networkRef).toEqual({
      networkId: 'network-123',
      networkArn: 'arn:aws:medialive:us-east-1:123456789012:network:network-123',
    });
  });

  test('networkRef resolves from the network construct', () => {
    const network = new Network(stack, 'Net', {
      networkName: 'ref-network',
      ipPools: ['10.0.0.0/16'],
    });

    expect(network.networkRef).toEqual({
      networkId: network.networkId,
      networkArn: network.networkArn,
    });
  });
});
