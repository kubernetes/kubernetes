/**
 * Integration test: MediaLive Anywhere — Network, Cluster, ChannelPlacementGroup, SdiSource.
 */
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-anywhere');

// --- IAM Role for cluster nodes ---
const nodeRole = new iam.Role(stack, 'NodeRole', {
  assumedBy: new iam.ServicePrincipal('medialive.amazonaws.com'),
});

// --- Network ---
const network = new medialive.Network(stack, 'AnywhereNetwork', {
  networkName: 'integ-anywhere-network',
  ipPools: ['10.0.0.0/16'],
  routes: [
    { cidr: '0.0.0.0/0', gateway: '10.0.0.1' },
  ],
});

// --- Cluster ---
const cluster = new medialive.Cluster(stack, 'AnywhereCluster', {
  clusterName: 'integ-anywhere-cluster',
  clusterType: medialive.ClusterType.ON_PREMISES,
  instanceRole: nodeRole,
  networkSettings: {
    defaultRoute: 'data',
    interfaceMappings: [
      { logicalInterfaceName: 'data', networkId: network.networkId },
    ],
  },
});

// --- Channel Placement Group ---
const placementGroup = new medialive.ChannelPlacementGroup(stack, 'PlacementGroup', {
  channelPlacementGroupName: 'integ-placement-group',
  cluster,
});

// --- Input ---
const input = new medialive.Input(stack, 'SrtInput', {
  inputName: 'integ-anywhere-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '10.0.1.100',
    srtListenerPort: 9000,
  }]),
});

// --- SDI Source + SDI input (MediaLive Anywhere only) ---
const sdiSource = new medialive.SdiSource(stack, 'SdiSource', {
  sdiSourceName: 'integ-anywhere-sdi-cam',
  type: medialive.SdiType.SINGLE,
});

new medialive.Input(stack, 'SdiInput', {
  inputName: 'integ-anywhere-sdi-input',
  // SDI inputs are on-premises hardware sources — the network location must be ON_PREMISES.
  inputNetworkLocation: medialive.InputNetworkLocation.ON_PREMISES,
  input: medialive.InputConfiguration.sdi([sdiSource]),
});

// --- On-premises push input with network routing (MediaLive Anywhere) ---
// On-premises inputs do not use input security groups.
new medialive.Input(stack, 'OnPremPushInput', {
  inputName: 'integ-anywhere-rtp-push',
  inputNetworkLocation: medialive.InputNetworkLocation.ON_PREMISES,
  input: medialive.InputConfiguration.rtpPush({
    destinations: [{
      network,
      networkRoutes: [{ cidr: '10.1.0.0/24', gateway: '10.0.0.1' }],
      staticIpAddress: '10.0.2.50',
    }],
  }),
});

// --- Multicast input (MediaLive Anywhere only) ---
// A network-addressable source (multicast group + port) — no physical hardware required.
new medialive.Input(stack, 'MulticastInput', {
  inputName: 'integ-anywhere-multicast',
  input: medialive.InputConfiguration.multicast({
    sources: [{
      address: '239.1.1.1',
      port: 5000,
      protocol: medialive.MulticastProtocol.RTP,
    }],
  }),
});

// --- SMPTE 2110 receiver group input (MediaLive Anywhere only) ---
// References SDP files describing the stream — also no physical hardware required.
new medialive.Input(stack, 'Smpte2110Input', {
  inputName: 'integ-anywhere-smpte2110',
  input: medialive.InputConfiguration.smpte2110ReceiverGroup({
    videoSdp: { sdpUrl: 'http://10.0.1.50/video.sdp' },
    audioSdps: [{ sdpUrl: 'http://10.0.1.50/audio.sdp' }],
  }),
});

// --- Encode ---
const hd = medialive.EncodeConfiguration.video({
  name: 'hd-1080p',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'aac-stereo', codec: medialive.AudioCodecSettings.aac() });

// --- Channel with Anywhere settings ---
new medialive.Channel(stack, 'AnywhereChannel', {
  channelName: 'integ-anywhere-channel',
  anywhereSettings: {
    cluster,
    channelPlacementGroup: placementGroup,
  },
  inputs: [{
    input,
    // Map this input to the cluster's 'data' logical interface (defined in networkSettings above).
    logicalInterfaceNames: ['data'],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-output',
      destinations: [medialive.OutputDestination.url('s3ssl://my-bucket/anywhere-live')],
      outputs: [
        { encodes: [hd, audio], outputName: 'hd_output' },
      ],
    }),
  ],
});

new IntegTest(app, 'cdk-integ-medialive-anywhere', {
  testCases: [stack],
});

app.synth();
