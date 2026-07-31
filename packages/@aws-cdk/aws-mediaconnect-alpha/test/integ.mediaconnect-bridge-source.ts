/**
 * Integration test: adding additional `BridgeSource`s to an existing bridge.
 *
 * Bridges with failover enabled can have up to two sources. The `BridgeSource`
 * construct adds an additional source to an existing bridge after construction,
 * using `BridgeSourceConfiguration.flow()` (egress bridges) or
 * `BridgeSourceConfiguration.network()` (ingress bridges). This test deploys one
 * of each to verify both factory methods are accepted by the service.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { Bridge, BridgeConfiguration, BridgeFailoverConfig, BridgeOutputConfiguration } from '../lib/bridge';
import { BridgeSource, BridgeSourceConfiguration } from '../lib/bridge-source';
import { Flow } from '../lib/flow';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { Gateway } from '../lib/gateway';
import { BridgeProtocol, GatewayNetwork } from '../lib/shared';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-bridge-source');

const network = GatewayNetwork.define({
  cidrBlock: '10.0.0.0/23',
  name: 'network-1',
});

const gateway = new Gateway(stack, 'Gateway', {
  egressCidrBlocks: ['10.0.0.0/23'],
  networks: [network],
});

// Ingress bridge with failover, started with one network source; we add a second
// via BridgeSource below.
const ingressBridge = new Bridge(stack, 'IngressBridge', {
  bridgeName: 'bridge-source-ingress',
  config: BridgeConfiguration.ingress({
    maxBitrate: cdk.Bitrate.mbps(5),
    maxOutputs: 1,
    networkSources: [{
      name: 'primary-network-source',
      source: {
        multicastIp: '239.0.0.1',
        network,
        port: 5000,
        protocol: BridgeProtocol.RTP_FEC,
      },
    }],
  }),
  sourceFailoverConfig: BridgeFailoverConfig.failover(),
  gateway,
});

new BridgeSource(stack, 'IngressAdditionalSource', {
  bridgeSourceName: 'backup-network-source',
  bridge: ingressBridge,
  source: BridgeSourceConfiguration.network({
    multicastIp: '239.0.0.2',
    network,
    port: 5002,
    protocol: BridgeProtocol.RTP_FEC,
  }),
});

// Feeder flow for the egress bridge below.
const feederFlow = new Flow(stack, 'FeederFlow', {
  flowName: 'bridge-source-feeder',
  source: SourceConfiguration.rtp({
    flowSourceName: 'feeder-source',
    port: 5004,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});

// Egress bridge with failover enabled; starts with one flow source and we add a
// second via BridgeSource.
const egressBridge = new Bridge(stack, 'EgressBridge', {
  bridgeName: 'bridge-source-egress',
  config: BridgeConfiguration.egress({
    maxBitrate: cdk.Bitrate.mbps(5),
    flowSources: [{
      name: 'primary-flow-source',
      source: { flow: feederFlow },
    }],
    networkOutputs: [{
      name: 'egress-output',
      output: BridgeOutputConfiguration.network({
        ipAddress: '192.168.1.100',
        port: 5010,
        network,
        protocol: BridgeProtocol.RTP,
        ttl: 64,
      }),
    }],
  }),
  sourceFailoverConfig: BridgeFailoverConfig.failover(),
  gateway,
});

new BridgeSource(stack, 'EgressAdditionalSource', {
  bridgeSourceName: 'backup-flow-source',
  bridge: egressBridge,
  source: BridgeSourceConfiguration.flow({
    flow: feederFlow,
  }),
});

new IntegTest(app, 'cdk-integ-emx-bridge-source', {
  testCases: [stack],
});

app.synth();
