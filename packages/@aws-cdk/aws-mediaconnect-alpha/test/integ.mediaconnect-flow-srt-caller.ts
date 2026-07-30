/**
 * Integration test: SRT Caller source.
 *
 * Deploys a minimal flow using the SRT Caller protocol, where MediaConnect
 * initiates the SRT connection to the upstream listener. Covers the SRT Caller
 * source code path end-to-end against the MediaConnect service.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { Flow } from '../lib/flow';
import { SourceConfiguration } from '../lib/flow-source-configuration';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-srt-caller');

new Flow(stack, 'Flow', {
  flowName: 'srt-caller-source-flow',
  source: SourceConfiguration.srtCaller({
    flowSourceName: 'srt-caller-source',
    sourceListenerAddress: '198.51.100.10',
    sourceListenerPort: 5000,
    minLatency: cdk.Duration.millis(120),
    streamId: 'integ-srt-caller-stream',
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-srt-caller', {
  testCases: [stack],
});

app.synth();
