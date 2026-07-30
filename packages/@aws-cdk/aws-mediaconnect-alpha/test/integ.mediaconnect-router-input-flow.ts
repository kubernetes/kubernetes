import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Flow } from '../lib/flow';
import { FlowOutput, OutputConfiguration } from '../lib/flow-output';
import { SourceConfiguration, NetworkConfiguration } from '../lib/flow-source-configuration';
import { RouterInput, RouterInputConfiguration, RouterInputTier, RoutingScope } from '../lib/router-input';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-input-flow');

// Create a source Flow for the RouterInput to consume
const sourceFlow = new Flow(stack, 'SourceFlow', {
  flowName: 'router-input-source-flow',
  source: SourceConfiguration.rtp({
    flowSourceName: 'source-for-router',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('10.1.0.0/16'),
  }),
});

// Create role and secret for transit encryption
const transitRole = new Role(stack, 'TransitRole', {
  assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
});

const transitSecret = new Secret(stack, 'TransitSecret', {
  description: 'Transit encryption key for RouterInput Flow',
});

const grant = transitSecret.grantRead(transitRole);

// Encrypted flow output (matches encrypted router input)
const encryptedFlowOutput = new FlowOutput(stack, 'EncryptedRouterOutputFlow', {
  flow: sourceFlow,
  output: OutputConfiguration.router({
    encryption: {
      role: transitRole,
      secret: transitSecret,
    },
  }),
});

// Unencrypted flow output (matches simple router input)
const simpleFlowOutput = new FlowOutput(stack, 'SimpleRouterOutputFlow', {
  flow: sourceFlow,
  output: OutputConfiguration.router({}),
});

// RouterInput with MediaConnect Flow configuration (with encryption)
const encryptedInput = new RouterInput(stack, 'routerInputFlowEncrypted', {
  routerInputName: 'flow-input-encrypted',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterInputTier.INPUT_100,
  configuration: RouterInputConfiguration.mediaConnectFlow({
    flow: sourceFlow,
    flowOutput: encryptedFlowOutput,
    sourceTransitDecryption: {
      role: transitRole,
      secret: transitSecret,
    },
  }),
});

// RouterInput with MediaConnect Flow configuration (no encryption)
encryptedInput.node.addDependency(grant);

new RouterInput(stack, 'routerInputFlowSimple', {
  routerInputName: 'flow-input-simple',
  maximumBitrate: cdk.Bitrate.mbps(6),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_50,
  configuration: RouterInputConfiguration.mediaConnectFlow({
    flow: sourceFlow,
    flowOutput: simpleFlowOutput,
  }),
});

new IntegTest(app, 'cdk-integ-emx-router-input-flow', {
  testCases: [stack],
});

app.synth();
