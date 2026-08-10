/**
 * Integration test: Router Input and Output with explicit availability zones.
 *
 * Exercises the `availabilityZone` prop on standard, failover, and merge router input
 * configurations, plus the standard router output configuration. Pinned to us-east-1
 * since AZ values are region-specific.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import {
  RouterInput,
  RouterInputConfiguration,
  RouterInputProtocol,
  RouterInputTier,
  RouterNetworkConfiguration,
  RouterNetworkInterface,
  RouterOutput,
  RouterOutputConfiguration,
  RouterOutputProtocol,
  RouterOutputTier,
  RoutingScope,
} from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-az', {
  env: { region: 'us-east-1' },
});

const networkInterface = new RouterNetworkInterface(stack, 'Network', {
  routerNetworkInterfaceName: 'az-test-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

// Outbound-only public network interface (no inbound CIDR rules)
const outboundOnly = new RouterNetworkInterface(stack, 'OutboundOnlyNetwork', {
  routerNetworkInterfaceName: 'az-outbound-only',
  configuration: RouterNetworkConfiguration.publicNetwork(),
});

void outboundOnly;

// Standard input with explicit AZ
new RouterInput(stack, 'StandardInput', {
  routerInputName: 'az-standard-input',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_20,
  configuration: RouterInputConfiguration.standard({
    networkInterface,
    protocol: RouterInputProtocol.rtp({ port: 5000 }),
    availabilityZone: 'us-east-1c',
  }),
});

// Failover input with explicit AZ
new RouterInput(stack, 'FailoverInput', {
  routerInputName: 'az-failover-input',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_20,
  configuration: RouterInputConfiguration.failover({
    networkInterface,
    protocols: [
      RouterInputProtocol.rtp({ port: 6000 }),
      RouterInputProtocol.rtp({ port: 6001 }),
    ],
    availabilityZone: 'us-east-1c',
  }),
});

// Merge input with explicit AZ
new RouterInput(stack, 'MergeInput', {
  routerInputName: 'az-merge-input',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_20,
  configuration: RouterInputConfiguration.merge({
    networkInterface,
    protocols: [
      RouterInputProtocol.rist({ port: 7000, recoveryLatency: cdk.Duration.millis(200) }),
      RouterInputProtocol.rist({ port: 7002, recoveryLatency: cdk.Duration.millis(200) }),
    ],
    mergeRecoveryWindow: cdk.Duration.millis(500),
    availabilityZone: 'us-east-1c',
  }),
});

// Standard output with explicit AZ
new RouterOutput(stack, 'StandardOutput', {
  routerOutputName: 'az-standard-output',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.standard({
    networkInterface,
    protocol: RouterOutputProtocol.rtp({
      destinationAddress: '198.51.100.10',
      port: 5000,
    }),
    availabilityZone: 'us-east-1c',
  }),
});

new IntegTest(app, 'cdk-integ-emx-router-availability-zone', {
  testCases: [stack],
  regions: ['us-east-1'],
});

app.synth();
