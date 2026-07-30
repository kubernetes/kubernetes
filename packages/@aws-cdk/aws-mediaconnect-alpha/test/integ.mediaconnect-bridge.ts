import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { Bridge, BridgeConfiguration, BridgeOutputConfiguration } from '../lib/bridge';
import { Flow } from '../lib/flow';
import { SourceConfiguration, NetworkConfiguration } from '../lib/flow-source-configuration';
import { Gateway } from '../lib/gateway';
import { BridgeProtocol, GatewayNetwork } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-bridge');

const network = GatewayNetwork.define({
  cidrBlock: '10.0.0.0/23',
  name: 'network-1',
});

const gateway = new Gateway(stack, 'gateway', {
  egressCidrBlocks: ['10.0.0.0/23'],
  networks: [network],
});

// Test 1: Ingress Bridge with Network Source
new Bridge(stack, 'bridge', {
  config: BridgeConfiguration.ingress({
    maxBitrate: cdk.Bitrate.mbps(5),
    maxOutputs: 1,
    networkSources: [{
      name: 'ingress-network-source',
      source: {
        multicastIp: '239.0.0.0',
        network,
        port: 5000,
        protocol: BridgeProtocol.RTP_FEC,
      },
    }],
  }),
  gateway,
});

const flow = new Flow(stack, 'test-flow', {
  source: SourceConfiguration.rtp({
    port: 5000,
    flowSourceName: 'ingest',
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});

// Test 2: Egress Bridge with Network Output using BridgeOutputConfiguration
new Bridge(stack, 'bridgeWithOutput', {
  config: BridgeConfiguration.egress({
    maxBitrate: cdk.Bitrate.mbps(10),
    flowSources: [{
      name: 'imported',
      source: { flow },
    }],
    networkOutputs: [{
      name: 'bridge-network-output',
      output: BridgeOutputConfiguration.network({
        ipAddress: '192.168.1.100',
        port: 5000,
        network,
        protocol: BridgeProtocol.RTP,
        ttl: 100,
      }),
    }],
  }),
  gateway,
});

new IntegTest(app, 'cdk-integ-emx-bridge', {
  testCases: [stack],
});

app.synth();
