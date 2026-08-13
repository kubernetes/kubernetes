import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from 'aws-cdk-lib/aws-medialive';

import { Flow } from '../lib/flow';
import { SourceConfiguration } from '../lib/flow-source-configuration';
import { ForwardErrorCorrection, RoutingScope } from '../lib/router-input';
import { RouterNetworkConfiguration, RouterNetworkInterface } from '../lib/router-network-interface';
import { RouterOutput, RouterOutputConfiguration, RouterOutputProtocol, RouterOutputTier } from '../lib/router-output';
import { MaintenanceDay, MediaLivePipeline } from '../lib/shared';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-mediaconnect-router-output');

// Create a network interface for standard outputs
const networkInterface = new RouterNetworkInterface(stack, 'network', {
  routerNetworkInterfaceName: 'test-output-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['10.0.0.0/16'],
  }),
});

// Test 1: Standard RTP Output
new RouterOutput(stack, 'rtpOutput', {
  routerOutputName: 'test_rtp_output',
  maximumBitrate: cdk.Bitrate.mbps(10),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.rtp({
      destinationAddress: '198.51.100.10',
      port: 5000,
      forwardErrorCorrection: ForwardErrorCorrection.ENABLED,
    }),
    networkInterface,
  }),
  tags: {
    Environment: 'test',
    Protocol: 'RTP',
  },
});

// Test 2: Standard SRT Listener Output
new RouterOutput(stack, 'srtOutput', {
  routerOutputName: 'test-srt-output',
  maximumBitrate: cdk.Bitrate.mbps(15),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.srtListener({
      port: 9001,
      minimumLatency: cdk.Duration.millis(200),
    }),
    networkInterface,
  }),
  maintenanceConfiguration: {
    day: MaintenanceDay.SUNDAY,
    time: '03:00',
  },
});

// Create a MediaLive input for testing router output (using MEDIACONNECT_ROUTER type)
// L1 construct as L2 doesn't yet exist
const mediaLiveInput = new medialive.CfnInput(stack, 'TestMediaLiveInput', {
  name: 'test-router-output-input',
  type: 'MEDIACONNECT_ROUTER',
  routerSettings: {
    destinations: [{
      availabilityZoneName: `${stack.region}a`,
    }],
    encryptionType: 'AUTOMATIC',
  },
});

// Test 3: MediaLive Input Output (using created MediaLive input)
new RouterOutput(stack, 'mediaLiveOutput', {
  routerOutputName: 'test-medialive-output',
  maximumBitrate: cdk.Bitrate.mbps(20),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.mediaLiveInput({
    mediaLiveInputArn: mediaLiveInput.attrArn, // needs updating when MediaLive has an L2
    mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
  }),
});

// Add dependency so MediaLive input is deleted before router output during stack deletion
// When MediaLive is there - we need to migrate this inside the class.
// mediaLiveInput.addDependency(mediaLiveOutput.node.defaultChild as cdk.CfnResource);

// Test 4: MediaLive Output without connecting to a specific input
const routerOutput = new RouterOutput(stack, 'mediaLiveNoInput', {
  routerOutputName: 'test-medialive-no-input',
  maximumBitrate: cdk.Bitrate.mbps(12),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
    availabilityZone: `${stack.region}a`,
  }),
});

// Test 5: MediaConnect Flow Output (using created flow)
const flow = new Flow(stack, 'TestFlow2', {
  flowName: 'source-flow',
  availabilityZone: `${stack.region}a`,
  source: SourceConfiguration.router({
    routerOutput,
  }),
});

new RouterOutput(stack, 'flowOutput', {
  routerOutputName: 'test-flow-output',
  maximumBitrate: cdk.Bitrate.mbps(25),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.mediaConnectFlow({
    flow,
  }),
});

// Test 6: MediaConnect Flow Output without connecting to a specific flow
new RouterOutput(stack, 'flowNoConnection', {
  routerOutputName: 'test-flow-no-connection',
  maximumBitrate: cdk.Bitrate.mbps(14),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.mediaConnectFlowWithoutConnection({
    availabilityZone: `${stack.region}b`,
  }),
});

new IntegTest(app, 'cdk-integ-emx-router-output', {
  testCases: [stack],
});

app.synth();
