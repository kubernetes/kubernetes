import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import * as medialive from 'aws-cdk-lib/aws-medialive';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';

import { Flow } from '../lib/flow';
import { SourceConfiguration } from '../lib/flow-source-configuration';
import { RoutingScope } from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';
import { RouterOutput, RouterOutputConfiguration, RouterOutputProtocol, RouterOutputTier } from '../lib/router-output';
import { MaintenanceDay, MediaLivePipeline } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-output-encryption');

// Create IAM role for MediaConnect
const mediaConnectRole = new Role(stack, 'MediaConnectRole', {
  assumedBy: new ServicePrincipal('mediaconnect.amazonaws.com'),
  description: 'Role for MediaConnect to access encryption secrets',
});

// Create secrets for encryption
const srtSecret = new Secret(stack, 'SrtEncryptionSecret', {
  description: 'SRT encryption key for router output',
});

const transitSecret = new Secret(stack, 'TransitEncryptionSecret', {
  description: 'Transit encryption key for router output',
});

// Grant the role access to read the secrets
const srtGrant = srtSecret.grantRead(mediaConnectRole);
const transitGrant = transitSecret.grantRead(mediaConnectRole);

// Create a MediaConnect flow for testing encrypted router output
const encryptedTestFlow = new Flow(stack, 'EncryptedTestFlow', {
  flowName: 'test-encrypted-router-output-flow',
  source: SourceConfiguration.router({
    decryption: {
      role: mediaConnectRole,
      secret: transitSecret,
    },
  }),
});

// Create a MediaLive input for testing router output with encryption (using MEDIACONNECT_ROUTER type)
// Only L1 available at the moment (update when L2 available)
const mediaLiveInput = new medialive.CfnInput(stack, 'MediaLiveInput', {
  name: 'test-router-output-input',
  type: 'MEDIACONNECT_ROUTER',
  routerSettings: {
    encryptionType: 'SECRETS_MANAGER',
    secretArn: transitSecret.secretArn,
    destinations: [{
      availabilityZoneName: `${stack.region}a`,
    }, {
      availabilityZoneName: `${stack.region}b`,
    }],
  },
});

// Create network interface
const networkInterface = new RouterNetworkInterface(stack, 'encryptionNetwork', {
  routerNetworkInterfaceName: 'test-encryption-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['172.16.0.0/16'],
  }),
});

// Test 1: SRT Caller with Encryption
const srtCallerEncrypted = new RouterOutput(stack, 'srtCallerEncrypted', {
  routerOutputName: 'test-srt-caller-encrypted',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.srtCaller({
      destinationAddress: '198.51.100.10',
      destinationPort: 9002,
      minimumLatency: cdk.Duration.millis(150),
      streamId: 'encrypted-stream-123',
      encryptionConfiguration: {
        role: mediaConnectRole,
        secret: srtSecret,
      },
    }),
    networkInterface,
  }),
  tags: {
    Environment: 'test',
    Encryption: 'SRT',
  },
});

// Test 2: MediaLive router output with Secrets Manager transit encryption
const mediaLiveEncrypted = new RouterOutput(stack, 'mediaLiveEncrypted', {
  routerOutputName: 'test-medialive-encrypted',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.mediaLiveInput({
    input: mediaLiveInput,
    pipeline: MediaLivePipeline.PIPELINE_1,
    destinationTransitEncryption: {
      role: mediaConnectRole,
      secret: transitSecret,
    },
  }),
});

// Test 3: MediaConnect Flow with Secrets Manager Encryption
const flowEncrypted = new RouterOutput(stack, 'flowEncrypted', {
  routerOutputName: 'test-flow-encrypted',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.mediaConnectFlow({
    flow: encryptedTestFlow,
    destinationTransitEncryption: {
      role: mediaConnectRole,
      secret: transitSecret,
    },
  }),
  maintenanceConfiguration: {
    day: MaintenanceDay.SATURDAY,
    time: '02:00',
  },
});

// Test 4: RIST Output (no encryption, but different protocol)
new RouterOutput(stack, 'ristOutput', {
  routerOutputName: 'test-rist-output',
  maximumBitrate: cdk.Bitrate.mbps(5),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.rist({
      destinationAddress: '198.51.100.20',
      port: 5200,
    }),
    networkInterface,
  }),
  tags: {
    Environment: 'test',
    Protocol: 'RIST',
  },
});

// Ensure IAM grants are in place before router outputs try to use the secrets
srtCallerEncrypted.node.addDependency(srtGrant);
mediaLiveEncrypted.node.addDependency(transitGrant);
flowEncrypted.node.addDependency(transitGrant);

new IntegTest(app, 'cdk-integ-emx-router-output-encryption', {
  testCases: [stack],
});

app.synth();
