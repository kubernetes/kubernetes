import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Gateway } from '../lib/gateway';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-gateway');

new Gateway(stack, 'gateway', {
  gatewayName: 'gateway',
  egressCidrBlocks: ['10.0.0.0/23'],
  networks: [{
    cidrBlock: '10.0.1.0/24',
    name: 'gateway',
  }],
});

new IntegTest(app, 'cdk-integ-emx-gateway', {
  testCases: [stack],
});

app.synth();
