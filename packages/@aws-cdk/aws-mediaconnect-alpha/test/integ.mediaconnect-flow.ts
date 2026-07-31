import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { Flow } from '../lib/flow';
import { SourceConfiguration, NetworkConfiguration } from '../lib/flow-source-configuration';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow');

new Flow(stack, 'flow', {
  source: SourceConfiguration.rtpFec({
    flowSourceName: 'my-flow',
    network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
    description: 'my test flow',
    maxBitrate: cdk.Bitrate.mbps(10),
    port: 2099,
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow', {
  testCases: [stack],
});

app.synth();
