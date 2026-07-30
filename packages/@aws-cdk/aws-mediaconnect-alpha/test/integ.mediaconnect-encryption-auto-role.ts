/**
 * Integration test: encryption without an explicit `role`.
 *
 * Every encryption struct in this module — `StaticKeyEncryption`, `SrtPasswordEncryption`,
 * `TransitEncryption`, `RouterSrtEncryption` — accepts an optional `role`. When omitted,
 * the module creates a scoped `mediaconnect.amazonaws.com` role with `aws:SourceAccount`
 * in the trust policy and `secretsmanager:GetSecretValue` (plus any KMS `Decrypt` on the
 * secret's encryption key) via `Secret.grantRead()`.
 *
 * This test exercises the auto-role path end-to-end for all four encryption structs
 * and deploys to verify the generated trust/permission policies are accepted by
 * MediaConnect at runtime.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Duration } from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Flow, FlowOutput, OutputConfiguration } from '../lib';
import { SourceConfiguration } from '../lib/flow-source-configuration';
import {
  RouterInput,
  RouterInputConfiguration,
  RouterInputProtocol,
  RouterInputTier,
  RoutingScope,
} from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';
import { RouterOutput, RouterOutputConfiguration, RouterOutputProtocol, RouterOutputTier } from '../lib/router-output';
import { EncryptionAlgorithm } from '../lib/shared';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-encryption-auto-role');

// One secret per encryption type keeps grants from overlapping and makes the generated
// role-per-encryption-point easy to eyeball in the synthesized template.
const transitSecret = new Secret(stack, 'TransitSecret', {
  description: 'Transit encryption (SourceConfiguration.router decryption)',
});
const staticKeySecret = new Secret(stack, 'StaticKeySecret', {
  description: 'Static-key encryption for Zixi Pull output',
});
const srtPasswordSecret = new Secret(stack, 'SrtPasswordSecret', {
  description: 'SRT password encryption for SRT Caller output',
});
const routerSrtSecret = new Secret(stack, 'RouterSrtSecret', {
  description: 'RouterInput SRT decryption',
});
const routerTransitSecret = new Secret(stack, 'RouterTransitSecret', {
  description: 'RouterInput input-level transit encryption',
});

// Flow with TransitEncryption auto-role on the router source.
const flow = new Flow(stack, 'Flow', {
  flowName: 'auto-role-flow',
  source: SourceConfiguration.router({
    decryption: {
      secret: transitSecret,
      // role omitted -> auto-created
    },
  }),
});

// FlowOutput with StaticKeyEncryption auto-role (Zixi Pull).
new FlowOutput(stack, 'ZixiPullOutput', {
  flow,
  output: OutputConfiguration.zixiPull({
    streamId: 'auto-role-zixi-pull',
    remoteId: 'auto-role-remote',
    maxLatency: Duration.millis(6000),
    cidrAllowList: ['198.51.100.0/24'],
    encryption: {
      secret: staticKeySecret,
      algorithm: EncryptionAlgorithm.AES256,
      // role omitted -> auto-created
    },
  }),
});

// FlowOutput with SrtPasswordEncryption auto-role (SRT Caller).
new FlowOutput(stack, 'SrtCallerOutput', {
  flow,
  output: OutputConfiguration.srtCaller({
    destination: '198.51.100.10',
    port: 9100,
    minLatency: Duration.millis(120),
    encryption: {
      secret: srtPasswordSecret,
      // role omitted -> auto-created
    },
  }),
});

// RouterInput SRT Listener with RouterSrtEncryption auto-role.
const network = new RouterNetworkInterface(stack, 'Network', {
  routerNetworkInterfaceName: 'auto-role-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

new RouterInput(stack, 'RouterInput', {
  routerInputName: 'auto-role-router-input',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_20,
  // Input-level transit encryption with an auto-created role. MediaConnect reads this
  // secret at create time, so the router input must depend on the role's secret-read
  // policy — this exercises that ordering end-to-end.
  transitEncryption: { secret: routerTransitSecret },
  configuration: RouterInputConfiguration.standard({
    networkInterface: network,
    protocol: RouterInputProtocol.srtListener({
      port: 9200,
      minimumLatency: Duration.millis(120),
      decryptionConfiguration: {
        secret: routerSrtSecret,
        // role omitted -> auto-created
      },
    }),
  }),
});

// RouterOutput SRT Caller with RouterSrtEncryption auto-role, and a MediaLive-input
// destination with TransitEncryption auto-role — both exercise the create-time secret-read
// ordering on the output side.
const outputSrtSecret = new Secret(stack, 'OutputSrtSecret', {
  description: 'RouterOutput SRT encryption',
});
const outputTransitSecret = new Secret(stack, 'OutputTransitSecret', {
  description: 'RouterOutput transit encryption',
});

new RouterOutput(stack, 'SrtOutput', {
  routerOutputName: 'auto-role-srt-output',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.standard({
    networkInterface: network,
    protocol: RouterOutputProtocol.srtCaller({
      destinationAddress: '203.0.113.20',
      destinationPort: 9300,
      minimumLatency: Duration.millis(120),
      encryptionConfiguration: {
        secret: outputSrtSecret,
        // role omitted -> auto-created
      },
    }),
  }),
});

new RouterOutput(stack, 'MediaLiveOutput', {
  routerOutputName: 'auto-role-medialive-output',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
    availabilityZone: `${stack.region}a`,
    destinationTransitEncryption: {
      secret: outputTransitSecret,
      // role omitted -> auto-created
    },
  }),
});

new IntegTest(app, 'cdk-integ-emx-encryption-auto-role', {
  testCases: [stack],
});

app.synth();
