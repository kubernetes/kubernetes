import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-mc-router');
const azs = cdk.Stack.of(stack).availabilityZones;

new medialive.Input(stack, 'McRouterInput', {
  inputName: 'integ-mc-router',
  input: medialive.InputConfiguration.mediaConnectRouter({
    availabilityZones: [azs[0], azs[1]],
  }),
});

new IntegTest(app, 'cdk-integ-medialive-input-mc-router', {
  testCases: [stack],
});

app.synth();
