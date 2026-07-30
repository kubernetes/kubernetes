/**
 * Integration test: RIST source.
 *
 * Deploys a minimal flow using the RIST protocol with a public-internet source.
 * Covers the RIST source code path end-to-end against the MediaConnect service.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { Flow } from '../lib/flow';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-rist');

new Flow(stack, 'Flow', {
  flowName: 'rist-source-flow',
  source: SourceConfiguration.rist({
    flowSourceName: 'rist-source',
    port: 5000,
    maxLatency: cdk.Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('10.0.0.0/16'),
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-rist', {
  testCases: [stack],
});

app.synth();
