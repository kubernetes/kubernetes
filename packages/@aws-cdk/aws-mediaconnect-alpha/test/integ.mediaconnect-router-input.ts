import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';

import { ForwardErrorCorrection, RouterInput, RouterInputConfiguration, RouterInputProtocol, RouterInputTier, RoutingScope } from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-input');

const networkInterface = new RouterNetworkInterface(stack, 'network', {
  routerNetworkInterfaceName: 'test',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new RouterInput(stack, 'routerInput', {
  routerInputName: 'test_input',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterInputTier.INPUT_100,
  configuration: RouterInputConfiguration.standard({
    protocol: RouterInputProtocol.rtp({
      port: 5000,
      forwardErrorCorrection: ForwardErrorCorrection.ENABLED,
    }),
    networkInterface,
  }),
});

new IntegTest(app, 'cdk-integ-emx-router-network-interface', {
  testCases: [stack],
});

app.synth();
