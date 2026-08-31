/**
 * Integration test: MediaConnect Router input with default availability zone.
 * Tests that omitting the AZ defaults to the stack's first AZ.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-router-default-az');

// Router input with default AZ (stack's first AZ)
new medialive.Input(stack, 'RouterInputDefaultAz', {
  inputName: 'integ-router-default-az',
  input: medialive.InputConfiguration.mediaConnectRouter(),
});

new IntegTest(app, 'cdk-integ-medialive-input-router-default-az', {
  testCases: [stack],
});

app.synth();
