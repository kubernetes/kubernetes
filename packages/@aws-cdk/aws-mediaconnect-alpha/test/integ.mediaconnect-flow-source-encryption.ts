/**
 * Integration test: FlowSource-side decryption across SRT Listener, Zixi Push, and
 * SRT Caller protocols.
 *
 * The `integ.mediaconnect-encryption-auto-role` test covers router-source transit
 * decryption and output-side encryption; this test fills the gap for protocol-based
 * flow sources that accept decryption material. Each flow uses a distinct secret so
 * the generated auto-roles stay isolated and easy to eyeball in the synthesized
 * template.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Duration } from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Flow } from '../lib/flow';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';
import { EncryptionAlgorithm } from '../lib/shared';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-source-encryption');

const srtListenerSecret = new Secret(stack, 'SrtListenerSecret', {
  description: 'SRT Listener source decryption',
});
const zixiPushSecret = new Secret(stack, 'ZixiPushSecret', {
  description: 'Zixi Push source decryption',
});
const srtCallerSecret = new Secret(stack, 'SrtCallerSecret', {
  description: 'SRT Caller source decryption',
});

// SRT Listener source with auto-role decryption.
new Flow(stack, 'SrtListenerFlow', {
  flowName: 'srt-listener-decryption-flow',
  source: SourceConfiguration.srtListener({
    flowSourceName: 'srt-listener-decrypted',
    port: 9300,
    minLatency: Duration.millis(120),
    network: NetworkConfiguration.publicNetwork('198.51.100.0/24'),
    decryption: {
      secret: srtListenerSecret,
      // role omitted -> auto-created
    },
  }),
});

// Zixi Push source with auto-role static-key decryption.
new Flow(stack, 'ZixiPushFlow', {
  flowName: 'zixi-push-decryption-flow',
  source: SourceConfiguration.zixiPush({
    flowSourceName: 'zixi-push-decrypted',
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('198.51.100.0/24'),
    decryption: {
      secret: zixiPushSecret,
      algorithm: EncryptionAlgorithm.AES256,
      // role omitted -> auto-created
    },
  }),
});

// SRT Caller source with auto-role decryption. SRT Caller initiates the outbound
// connection, so no CIDR allow list is configured — a public internet target is used.
new Flow(stack, 'SrtCallerFlow', {
  flowName: 'srt-caller-decryption-flow',
  source: SourceConfiguration.srtCaller({
    flowSourceName: 'srt-caller-decrypted',
    sourceListenerAddress: '198.51.100.20',
    sourceListenerPort: 5000,
    minLatency: Duration.millis(120),
    streamId: 'integ-srt-caller-decrypted-stream',
    decryption: {
      secret: srtCallerSecret,
      // role omitted -> auto-created
    },
  }),
});

new IntegTest(app, 'cdk-integ-emx-flow-source-encryption', {
  testCases: [stack],
});

app.synth();
