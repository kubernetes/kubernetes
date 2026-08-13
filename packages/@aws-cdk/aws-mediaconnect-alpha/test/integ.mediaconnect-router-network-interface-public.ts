import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { RouterNetworkInterface, RouterNetworkConfiguration } from '../lib/router-network-interface';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-ni-public');

new RouterNetworkInterface(stack, 'network', {
  routerNetworkInterfaceName: 'test_interface',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['203.0.113.0/24'],
  }),
});

new IntegTest(app, 'cdk-integ-emx-router-network-interface-public', {
  testCases: [stack],
});

app.synth();
