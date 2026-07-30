/**
 * Integration test: empirically verifies service-side defaults for L2 props that
 * are documented as having a specific default value.
 *
 * Each flow below omits the field whose default we want to confirm. After deploy,
 * the test calls DescribeFlow and asserts the service stored the documented default.
 * If AWS ever changes a default, the snapshot diff catches it.
 *
 * Verified service defaults:
 * - SourceRist.maxLatency               → 2000 ms
 * - FailoverConfig.recoveryWindow       → 200 ms
 * - FlowEntitlementProps.entitlementStatus → ENABLED
 * - FlowOutput.outputStatus             → ENABLED
 *
 * Defaults verified by other dedicated integs:
 * - NDI source/output defaults          → integ.mediaconnect-flow-ndi.ts (LARGE flow + discovery server)
 * - CDI/JPEGXS maxSyncBuffer            → unit-tested; the L2 fills 100 ms when omitted
 *
 * Not asserted (service does not surface the value when omitted):
 * - FlowProps.flowSize                  → service default is MEDIUM (visible in console / behaviour) but DescribeFlow.FlowSize is undefined when the field was omitted on input
 * - SourceZixiPush.maxLatency           → service applies no default (console field is blank, response is undefined)
 */
import { IntegTest, ExpectedResult } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { FailoverConfig, Flow } from '../lib/flow';
import { FlowEntitlement } from '../lib/flow-entitlement';
import { FlowOutput, OutputConfiguration } from '../lib/flow-output';
import { NetworkConfiguration, SourceConfiguration } from '../lib/flow-source-configuration';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-flow-defaults');

// RIST source — omit maxLatency, expect service default 2000 ms.
const ristFlow = new Flow(stack, 'RistFlow', {
  flowName: 'defaults-rist',
  source: SourceConfiguration.rist({
    flowSourceName: 'rist-default',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});

// Failover (MERGE) — omit recoveryWindow, expect service default 200 ms.
const mergeFlow = new Flow(stack, 'MergeFlow', {
  flowName: 'defaults-merge',
  source: SourceConfiguration.rist({
    flowSourceName: 'merge-primary',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
  sourceFailoverConfig: FailoverConfig.merge(),
});

// Output and entitlement — attached to the RIST flow above. Both omit their status
// fields so we can assert the service applies ENABLED by default.
new FlowOutput(stack, 'OutputDefault', {
  flow: ristFlow,
  output: OutputConfiguration.rtp({
    destination: '203.0.113.20',
    port: 6000,
  }),
});

new FlowEntitlement(stack, 'EntitlementDefault', {
  flow: ristFlow,
  description: 'Defaults integ entitlement',
  subscribers: ['111122223333'],
});

const test = new IntegTest(app, 'cdk-integ-emx-flow-defaults', {
  testCases: [stack],
});

// RIST maxLatency default + entitlement/output status defaults.
test.assertions
  .awsApiCall('MediaConnect', 'describeFlow', { FlowArn: ristFlow.flowArn })
  .expect(ExpectedResult.objectLike({
    Flow: {
      Source: { Transport: { MaxLatency: 2000 } },
      Outputs: [{ OutputStatus: 'ENABLED' }],
      Entitlements: [{ EntitlementStatus: 'ENABLED' }],
    },
  }));

// Failover MERGE recoveryWindow default
test.assertions
  .awsApiCall('MediaConnect', 'describeFlow', { FlowArn: mergeFlow.flowArn })
  .expect(ExpectedResult.objectLike({
    Flow: { SourceFailoverConfig: { RecoveryWindow: 200 } },
  }));

app.synth();
