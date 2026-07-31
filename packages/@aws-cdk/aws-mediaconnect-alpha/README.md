# AWS::MediaConnect Construct Library
<!--BEGIN STABILITY BANNER-->

---

![cdk-constructs: Experimental](https://img.shields.io/badge/cdk--constructs-experimental-important.svg?style=for-the-badge)

> The APIs of higher level constructs in this module are experimental and under active development.
> They are subject to non-backward compatible changes or removal in any future version. These are
> not subject to the [Semantic Versioning](https://semver.org/) model and breaking changes will be
> announced in the release notes. This means that while you may use them, you may need to update
> your source code when upgrading to a newer version of this package.

---

<!--END STABILITY BANNER-->


## AWS Elemental MediaConnect

AWS Elemental MediaConnect is a high-quality transport service for live video. It provides the reliability and security of satellite and fiber-optic combined with the flexibility, agility, and economics of IP-based networks. MediaConnect enables you to build mission-critical live video workflows in a fraction of the time and cost of satellite or fiber services.

This package contains constructs for working with AWS Elemental MediaConnect, allowing you to define Flows, Bridges, Gateways, Router Inputs, Router Outputs, and Router Network Interfaces for transporting live video streams.

For further information on AWS Elemental MediaConnect, see [the documentation](https://aws.amazon.com/mediaconnect/).

## Table of Contents

- [Router Resources](#router-resources)
  - [Router Network Interfaces](#router-network-interfaces)
  - [Router Inputs](#router-inputs)
  - [Router Outputs](#router-outputs)
- [Flows](#flows)
  - [Creating a Flow](#creating-a-flow)
  - [Flow Sources](#flow-sources)
  - [VPC Interfaces](#vpc-interfaces)
  - [Media Streams](#media-streams)
  - [Flow Sizes](#flow-sizes)
- [Gateways](#gateways)
  - [Creating a Gateway](#creating-a-gateway)
  - [Importing an Existing Gateway](#importing-an-existing-gateway)
- [Bridges](#bridges)
  - [Creating a Bridge](#creating-a-bridge)
  - [Bridge Sources](#bridge-sources)
  - [Bridge Outputs](#bridge-outputs)
- [Encryption](#encryption)
- [CloudWatch Metrics](#cloudwatch-metrics)
- [Public CIDR warnings](#public-cidr-warnings)

## Router Resources

MediaConnect routers provide high-performance, low-latency video routing capabilities for building complex live video workflows. Router resources include network interfaces, inputs, and outputs.

### How Router Resources Relate

Router resources work together in a pipeline:

- A **RouterNetworkInterface** defines the network connectivity (public internet or VPC). Required for standard protocol-based inputs and outputs, but not needed when connecting to MediaLive inputs or MediaConnect flows directly.
- A **RouterInput** is the entry point — it receives video from a source via a protocol (RTP, SRT, RIST), or from a MediaConnect flow.
- A **RouterOutput** is the exit point — it sends video to a destination via a protocol, to a MediaLive input, or to a MediaConnect flow.

A typical camera-to-cloud workflow looks like:

```text
Camera → RouterNetworkInterface → RouterInput (SRT) → [Router] → RouterOutput (MediaLive) → MediaLive Channel
```

The router sits in the middle, routing content from inputs to outputs. You can have many outputs per input — for example, one camera feed going to both a MediaLive channel and a MediaConnect flow simultaneously.

### End-to-End Example: SRT Source to MediaLive

Here's a complete example showing how to connect an SRT source through a router to MediaLive:

```ts
declare const stack: Stack;
declare const mediaLiveInput: medialive.CfnInput;

// 1. A public network interface for the SRT input
const networkInterface = new RouterNetworkInterface(stack, 'NetworkInterface', {
  routerNetworkInterfaceName: 'camera-network',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['203.0.113.0/24'],
  }),
});

// 2. A router input receiving SRT from an upstream encoder
const input = new RouterInput(stack, 'Input', {
  routerInputName: 'camera-input',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_20,
  configuration: RouterInputConfiguration.standard({
    networkInterface,
    protocol: RouterInputProtocol.srtListener({
      port: 9000,
      minimumLatency: Duration.millis(200),
    }),
  }),
});

// 3. A router output delivering to MediaLive
const output = new RouterOutput(stack, 'Output', {
  routerOutputName: 'medialive-output',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_20,
  configuration: RouterOutputConfiguration.mediaLiveInput({
    mediaLiveInputArn: mediaLiveInput.attrArn,
    mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
  }),
});
```

This gives you a complete pipeline: the encoder pushes SRT to the network interface, the router receives it as an input, and the router output delivers it to MediaLive. For routing rules — including how bitrate affects which outputs can receive which inputs — see the [documentation](https://docs.aws.amazon.com/mediaconnect/latest/ug/using-router-control-panel.html).

### Router Network Interfaces

Network interfaces define the network connectivity for router inputs and outputs:

#### Public Network Interface

```ts
declare const stack: Stack;

const publicInterface = new RouterNetworkInterface(stack, 'PublicInterface', {
  routerNetworkInterfaceName: 'public-interface',
  configuration: RouterNetworkConfiguration.publicNetwork({
    cidr: ['203.0.113.0/24'],
  }),
});
```

#### Private Network Interface

```ts
declare const stack: Stack;
declare const securityGroup: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;

const privateInterface = new RouterNetworkInterface(stack, 'PrivateInterface', {
  routerNetworkInterfaceName: 'private-interface',
  configuration: RouterNetworkConfiguration.vpc({
    securityGroups: [securityGroup],
    subnet: subnet,
  }),
});
```

### Router Inputs

Router inputs receive live video streams from various sources and make them available for routing:

#### Standard Input with RTP Protocol

```ts
declare const stack: Stack;
declare const networkInterface: RouterNetworkInterface;

const input = new RouterInput(stack, 'RtpInput', {
  routerInputName: 'rtp-input',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  // tier defaults to RouterInputTier.INPUT_20 (lowest cost)
  configuration: RouterInputConfiguration.standard({
    networkInterface: networkInterface,
    protocol: RouterInputProtocol.rtp({
      port: 5000,
    }),
  }),
});
```

#### Failover Input Configuration

```ts
declare const stack: Stack;
declare const networkInterface: RouterNetworkInterface;

const input = new RouterInput(stack, 'FailoverInput', {
  routerInputName: 'failover-input',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_50,
  configuration: RouterInputConfiguration.failover({
    networkInterface: networkInterface,
    protocols: [
      RouterInputProtocol.rist({
        port: 5000,
        recoveryLatency: Duration.millis(1000),
      }),
      RouterInputProtocol.rist({
        port: 5002, // Must not be consecutive with primary port
        recoveryLatency: Duration.millis(1000),
      }),
    ],
    sourcePriority: SourcePriorityConfig.primarySecondary(PrimarySource.FIRST_SOURCE),
  }),
});
```

#### MediaConnect Flow Input

Connect a router input to an existing MediaConnect flow:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const flowOutput: FlowOutput;

const input = new RouterInput(stack, 'FlowInput', {
  routerInputName: 'flow-input',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_50,
  configuration: RouterInputConfiguration.mediaConnectFlow({
    flow: flow,
    flowOutput: flowOutput,
  }),
});
```

Or prepare a router input for a flow connection without specifying the flow (requires explicit availability zone):

```ts
declare const stack: Stack;

const input = new RouterInput(stack, 'FlowInputNoConnection', {
  routerInputName: 'flow-input-no-connection',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterInputTier.INPUT_50,
  configuration: RouterInputConfiguration.mediaConnectFlowWithoutConnection({
    availabilityZone: 'us-east-1a',
  }),
});
```

### Router Outputs

Router outputs send video streams to various destinations including standard protocols, MediaLive inputs, and MediaConnect flows:

#### Standard Output with SRT Protocol

```ts
declare const stack: Stack;
declare const networkInterface: RouterNetworkInterface;

const output = new RouterOutput(stack, 'SrtOutput', {
  routerOutputName: 'srt-output',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  // tier defaults to RouterOutputTier.OUTPUT_20 (lowest cost)
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.srtListener({
      port: 9001,
      minimumLatency: Duration.millis(200),
    }),
    networkInterface: networkInterface,
  }),
});
```

> **Note:** The `tier` property defaults to the lowest (and cheapest) tier: `INPUT_20` for Router Inputs and `OUTPUT_20` for Router Outputs. The construct validates that `maximumBitrate` does not exceed the tier's capacity (20, 50, or 100 Mbps) at synth time. Per the [documentation](https://docs.aws.amazon.com/mediaconnect/latest/ug/using-router-control-panel.html), if an input is 20 Mbps you can't route it to an output set up for less than 20 Mbps.

#### MediaLive Output

Note (breaking change in the future): MediaLive configuration is currently passed in as `mediaLiveInputArn` but when L2 construct available, this will be updated to use the construct instead.

Connect a router output to an existing MediaLive input:

```ts
declare const stack: Stack;
declare const mediaLiveInput: medialive.CfnInput;

const output = new RouterOutput(stack, 'MediaLiveOutput', {
  routerOutputName: 'medialive-output',
  maximumBitrate: Bitrate.mbps(15),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.mediaLiveInput({
    mediaLiveInputArn: mediaLiveInput.attrArn,
    mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
  }),
});
```

Or prepare a router output for a MediaLive connection without specifying the input (requires explicit availability zone):

```ts
declare const stack: Stack;

const output = new RouterOutput(stack, 'MediaLiveOutputNoConnection', {
  routerOutputName: 'medialive-output-no-connection',
  maximumBitrate: Bitrate.mbps(15),
  routingScope: RoutingScope.GLOBAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.mediaLiveInputWithoutConnection({
    availabilityZone: 'us-east-1a',
  }),
});
```

#### MediaConnect Flow Output

Connect a router output to an existing MediaConnect flow:

```ts
declare const stack: Stack;
declare const flow: Flow;

const output = new RouterOutput(stack, 'FlowOutput', {
  routerOutputName: 'flow-output',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.mediaConnectFlow({
    flow: flow,
  }),
});
```

Or prepare a router output for a flow connection without specifying the flow (requires explicit availability zone):

```ts
declare const stack: Stack;

const output = new RouterOutput(stack, 'FlowOutputNoConnection', {
  routerOutputName: 'flow-output-no-connection',
  maximumBitrate: Bitrate.mbps(20),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_100,
  configuration: RouterOutputConfiguration.mediaConnectFlowWithoutConnection({
    availabilityZone: 'us-east-1a',
  }),
});
```

#### Output with Encryption

```ts
declare const stack: Stack;
declare const networkInterface: RouterNetworkInterface;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

const output = new RouterOutput(stack, 'EncryptedOutput', {
  routerOutputName: 'encrypted-output',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.srtCaller({
      destinationAddress: '203.0.113.100',
      destinationPort: 9001,
      minimumLatency: Duration.millis(200),
      encryptionConfiguration: { role, secret },
    }),
    networkInterface: networkInterface,
  }),
});
```

## Flows

A MediaConnect flow represents a transport stream connection between a source and one or more outputs. Flows are the primary resource for transporting live video content.

### Creating a Flow

The following example creates a basic MediaConnect flow with an RTP source:

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  flowName: 'my-live-stream',
  source: SourceConfiguration.rtp({
    flowSourceName: 'my-source',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

### Flow Sources

MediaConnect supports multiple source types for ingesting content into a flow. The examples below use `NetworkConfiguration.publicNetwork()` for simplicity, but all protocol-based sources can also use `NetworkConfiguration.vpc()` with a VPC interface for private connectivity.

> The source's `flowSourceName` and `description` are set on the `SourceConfiguration`, not on `FlowSourceProps`. This is because the same `SourceConfiguration` is used both for a flow's inline primary source (`FlowProps.source`, which has no separate props object) and for additional sources added via `FlowSource`. Keeping the name on the configuration lets both be described identically.

#### SRT Listener Source

SRT (Secure Reliable Transport) in listener mode configures MediaConnect to listen on a specific port for incoming content. The upstream device connects to MediaConnect as a caller.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.srtListener({
    flowSourceName: 'live-encoder-source',
    description: 'Live encoder feed',
    port: 5000,
    minLatency: Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

#### SRT Caller Source

SRT in caller mode configures MediaConnect to connect to a remote SRT listener. Use this when the source device is listening for incoming connections rather than pushing content.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.srtCaller({
    flowSourceName: 'remote-source',
    sourceListenerAddress: '203.0.113.50',
    sourceListenerPort: 5000,
    minLatency: Duration.millis(200),
  }),
});
```

#### RTP Source

RTP (Real-time Transport Protocol) is a standard protocol for delivering audio and video over IP networks.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.rtp({
    flowSourceName: 'rtp-source',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

#### RTP-FEC Source

RTP with Forward Error Correction adds redundancy to recover lost packets without retransmission. Use this when contributing via RTP and you need packet recovery.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.rtpFec({
    flowSourceName: 'rtp-fec-source',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

#### RIST Source

RIST (Reliable Internet Stream Transport) provides reliable video transport with packet recovery.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.rist({
    flowSourceName: 'rist-source',
    port: 5000,
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

#### Zixi Push Source

Zixi Push uses the Zixi protocol for reliable video transport. Content is pushed to MediaConnect from a Zixi-compatible component upstream.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.zixiPush({
    flowSourceName: 'zixi-source',
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
  }),
});
```

> Zixi Push ports are assigned by MediaConnect: 2088 for public sources, 2090-2099 for VPC sources. See [Source port assignments](https://docs.aws.amazon.com/mediaconnect/latest/ug/source-ports.html).

#### Router Source

Use a router source when the flow's source comes from a MediaConnect Router rather than a direct connection.

```ts
declare const stack: Stack;

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.router(),
});
```

#### VPC Source

Use a VPC-based source when you need a connection between a flow and your Amazon VPC. This enables private connectivity for receiving content from on-premises equipment via AWS Direct Connect or VPN, or from other AWS services running in your VPC.

```ts
declare const stack: Stack;
declare const securityGroup: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;
declare const role: iam.IRole;

const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'my-vpc-interface',
  role: role,
  securityGroups: [securityGroup],
  subnet: subnet,
});

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.rist({
    flowSourceName: 'vpc-source',
    description: 'VPC-based source',
    port: 5000,
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.vpc(vpcInterface),
  }),
  vpcInterfaces: [vpcInterface],
});
```

#### Entitled Source (From Another AWS Account)

Entitlements allow you to subscribe to content from another AWS account. The entitlement is created by the content originator in their AWS account, and you import it using the entitlement ARN they provide.

```ts
declare const stack: Stack;

// Import an entitlement from another AWS account
const entitlement = FlowEntitlement.fromFlowEntitlementArn(stack, 'ImportedEntitlement',
  'arn:aws:mediaconnect:us-west-2:111122223333:entitlement:1-11111111111111111111111111111111:MyEntitlement',
);

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.entitlement({
    entitlement: entitlement,
  }),
});
```

#### Gateway Bridge Source

Use a gateway bridge source when ingesting content from on-premises equipment through a MediaConnect gateway and bridge. Gateways define the network infrastructure that bridges use to transport video between on-premises and cloud environments.

```ts
declare const stack: Stack;
declare const bridge: Bridge;
declare const role: iam.IRole;
declare const securityGroup: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;

const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'bridge-interface',
  role: role,
  securityGroups: [securityGroup],
  subnet: subnet,
});

const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.gatewayBridge({
    bridge: bridge,
    vpcInterface: vpcInterface,
  }),
  vpcInterfaces: [vpcInterface],
});
```

### VPC Interfaces

VPC interfaces allow MediaConnect to send or receive content within your VPC. Create VPC interfaces using `VpcInterface.define()` and add them to the flow's `vpcInterfaces` array. The same interface can then be referenced in sources and outputs:

```ts
declare const stack: Stack;
declare const role: iam.IRole;
declare const securityGroup: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;

// Create VPC interface
const vpcInterface = VpcInterface.define({
  vpcInterfaceName: 'my-vpc-interface',
  role: role,
  securityGroups: [securityGroup],
  subnet: subnet,
  networkInterfaceType: NetworkInterface.ENA,
});

// Add to flow and reference in source
const flow = new Flow(stack, 'MyFlow', {
  vpcInterfaces: [vpcInterface],  // Declare at flow level
  source: SourceConfiguration.rist({
    flowSourceName: 'vpc-source',
    port: 5000,
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.vpc(vpcInterface),  // Reference in source
  }),
});
```

#### CDI and JPEG XS with EFA Interfaces

For high-performance CDI or JPEG XS workflows, use EFA (Elastic Fabric Adapter) interfaces. Note that flows can have a maximum of 1 EFA interface:

```ts
declare const stack: Stack;
declare const role: iam.IRole;
declare const securityGroup: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;

const efaInterface = VpcInterface.define({
  vpcInterfaceName: 'efa-interface',
  role: role,
  securityGroups: [securityGroup],
  subnet: subnet,
  networkInterfaceType: NetworkInterface.EFA,  // Required for CDI
});

const videoStream = MediaStream.video({
  mediaStreamId: 1,
  mediaStreamName: 'video',
  videoFormat: MediaVideoFormat.HD_1080P,
  fmtp: {
    exactFramerate: Framerate.FPS_29_97,
    par: PixelAspectRatio.SQUARE,
  },
});

const flow = new Flow(stack, 'MyCdiFlow', {
  flowSize: FlowSize.LARGE_4X,  // Required for CDI and JPEG XS
  vpcInterfaces: [efaInterface],
  mediaStreams: [videoStream],
  source: SourceConfiguration.cdi({
    flowSourceName: 'cdi-source',
    vpcInterface: efaInterface,
    port: 5000,
    maxSyncBuffer: 100,
    mediaStreamSourceConfigurations: [{
      encoding: Encoding.RAW,
      mediaStream: videoStream,
    }],
  }),
});
```

#### JPEG XS with Redundant Interfaces

JPEG XS requires exactly 2 input interfaces per media stream for redundancy. Typically one EFA and one ENA interface:

```ts
declare const stack: Stack;
declare const role: iam.IRole;
declare const sg1: ec2.ISecurityGroup;
declare const sg2: ec2.ISecurityGroup;
declare const subnet: ec2.ISubnet;

const efaInterface = VpcInterface.define({
  vpcInterfaceName: 'efa-interface',
  role: role,
  securityGroups: [sg1],
  subnet: subnet,
  networkInterfaceType: NetworkInterface.EFA,
});

const enaInterface = VpcInterface.define({
  vpcInterfaceName: 'ena-interface',
  role: role,
  securityGroups: [sg2],
  subnet: subnet,
  networkInterfaceType: NetworkInterface.ENA,
});

const videoStream = MediaStream.video({
  mediaStreamId: 1,
  mediaStreamName: 'video',
  videoFormat: MediaVideoFormat.UHD_2160P,
  fmtp: {
    exactFramerate: Framerate.FPS_59_94,
    par: PixelAspectRatio.SQUARE,
    colorimetry: Colorimetry.BT2020,
    videoRange: VideoRange.FULL,
    scanMode: ScanMode.PROGRESSIVE,
    tcs: Tcs.PQ,
  },
});

const flow = new Flow(stack, 'MyJpegXsFlow', {
  flowSize: FlowSize.LARGE_4X,  // Required for JPEG XS
  vpcInterfaces: [efaInterface, enaInterface],
  mediaStreams: [videoStream],
  source: SourceConfiguration.jpegXs({
    flowSourceName: 'jpegxs-source',
    maxSyncBuffer: 100,
    mediaStreamSourceConfigurations: [{
      encoding: Encoding.JXSV,
      port: 5000,
      inputInterface: [efaInterface, enaInterface],  // 2 interfaces for redundancy
      mediaStream: videoStream,
    }],
  }),
});
```

### Media Streams

Media streams represent individual components of your content (video, audio, ancillary data) for ST 2110 JPEG XS or CDI workflows. Create media streams using the static factory methods and add them to the flow's `mediaStreams` array:

```ts
declare const stack: Stack;

// Create media streams
const videoStream = MediaStream.video({
  mediaStreamId: 1,
  mediaStreamName: 'video-stream',
  videoFormat: MediaVideoFormat.HD_1080P,
  fmtp: {
    colorimetry: Colorimetry.BT709,
    exactFramerate: Framerate.FPS_29_97,
    par: PixelAspectRatio.SQUARE,
    videoRange: VideoRange.NARROW,
    scanMode: ScanMode.PROGRESSIVE,
    tcs: Tcs.SDR,
  },
});

const audioStream = MediaStream.audio({
  mediaStreamId: 2,
  mediaStreamName: 'audio-stream',
  channelOrder: AudioStreamOrderOptions.STANDARD_STEREO,
});

// Add to flow
const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.router(),
  mediaStreams: [videoStream, audioStream],  // Declare at flow level
});
```

#### Audio Channel Order

For audio media streams, use the `AudioStreamOrderOptions` enum to specify the SMPTE 2110-30 channel order:

```ts
// Available channel order options
AudioStreamOrderOptions.MONO                    // SMPTE2110.(M)
AudioStreamOrderOptions.DUAL_MONO               // SMPTE2110.(DM)
AudioStreamOrderOptions.STANDARD_STEREO         // SMPTE2110.(ST)
AudioStreamOrderOptions.LTRT_MATRIX_STEREO      // SMPTE2110.(LtRt)
AudioStreamOrderOptions.SURROUND_5_1            // SMPTE2110.(51)
AudioStreamOrderOptions.SURROUND_7_1            // SMPTE2110.(71)
AudioStreamOrderOptions.SURROUND_22_2           // SMPTE2110.(222)
AudioStreamOrderOptions.ONE_SDI_AUDIO_GROUP     // SMPTE2110.(SGRP)

// Example with 5.1 surround
const surroundAudio = MediaStream.audio({
  mediaStreamId: 3,
  mediaStreamName: 'surround-audio',
  channelOrder: AudioStreamOrderOptions.SURROUND_5_1,
});
```

Media streams can be referenced in source configurations (for CDI and JPEG XS) and output configurations.

### Flow Sizes

MediaConnect offers three flow sizes that determine feature support:

| Flow Size | Transport Streams | NDI | CDI / JPEG XS |
|-----------|------------------|-----|---------------|
| `MEDIUM` (default) | ✅ | ❌ | ❌ |
| `LARGE` | ✅ | ✅ | ❌ |
| `LARGE_4X` | ❌ | ❌ | ✅ |

This table maps each `FlowSize` to the capabilities the construct validates. For output counts, throughput limits, and other per-size details, see [Flow sizes and capabilities](https://docs.aws.amazon.com/mediaconnect/latest/ug/flow-sizes-capabilities.html).

The construct validates flow size constraints at synthesis time based on the source protocol and NDI configuration:

- `MEDIUM` supports transport stream protocols (RTP, SRT, RIST, etc.) but not NDI or CDI
- `LARGE` supports transport streams and NDI, and is required when NDI is enabled
- `LARGE_4X` is required for CDI and JPEG XS protocols, and does not support transport streams or NDI

These are mutually exclusive — CDI/JPEG XS and NDI cannot coexist on the same flow because they require different flow sizes.

```ts
declare const stack: Stack;
declare const ndiVpcInterface: VpcInterfaceConfig;
declare const efaInterface: VpcInterfaceConfig;
declare const videoStream: MediaStream;

// NDI requires LARGE, an encoding profile, and at least one discovery server
new Flow(stack, 'NdiFlow', {
  flowSize: FlowSize.LARGE,
  ndiConfig: {
    ndiState: State.ENABLED,
    ndiDiscoveryServers: [{
      discoveryServerAddress: '10.0.0.10',
      vpcInterface: ndiVpcInterface,
    }],
  },
  encodingConfig: {
    encodingProfile: EncodingProfile.CONTRIBUTION_H264_DEFAULT,
  },
  source: SourceConfiguration.ndi({
    flowSourceName: 'ndi-source',
  }),
});

// CDI and JPEG XS require LARGE_4X
new Flow(stack, 'CdiFlow', {
  flowSize: FlowSize.LARGE_4X,
  vpcInterfaces: [efaInterface],
  mediaStreams: [videoStream],
  source: SourceConfiguration.cdi({
    flowSourceName: 'cdi-source',
    vpcInterface: efaInterface,
    port: 5000,
    maxSyncBuffer: 100,
    mediaStreamSourceConfigurations: [{
      encoding: Encoding.RAW,
      mediaStream: videoStream,
    }],
  }),
});
```

For more information, see [Flow sizes and capabilities](https://docs.aws.amazon.com/mediaconnect/latest/ug/flow-sizes-capabilities.html).

## Gateways

MediaConnect Gateways enable the deployment of on-premises resources for transporting live video to and from the AWS Cloud. Gateways are required for creating bridges.

### Creating a Gateway

```ts
declare const stack: Stack;

const productionNetwork = GatewayNetwork.define({
  cidrBlock: '192.168.1.0/24',
  name: 'production-network',
});

const gateway = new Gateway(stack, 'MyGateway', {
  gatewayName: 'my-gateway',
  egressCidrBlocks: ['10.0.0.0/16'],
  networks: [productionNetwork],
});
```

### Importing an Existing Gateway

```ts
declare const stack: Stack;

const gateway = Gateway.fromGatewayArn(stack, 'ImportedGateway',
  'arn:aws:mediaconnect:us-west-2:123456789012:gateway:1-XXXXXX',
);
```

## Bridges

MediaConnect bridges enable you to interconnect on-premises equipment with cloud-based workflows. Bridges support both ingress (on-premises to cloud) and egress (cloud to on-premises) scenarios.

### Creating a Bridge

#### Ingress Bridge (On-premises to Cloud)

An ingress bridge receives content from on-premises equipment and makes it available in the cloud:

```ts
declare const stack: Stack;

const productionNetwork = GatewayNetwork.define({
  cidrBlock: '192.168.1.0/24',
  name: 'production-network',
});

const gateway = new Gateway(stack, 'MyGateway', {
  gatewayName: 'my-gateway',
  egressCidrBlocks: ['10.0.0.0/16'],
  networks: [productionNetwork],
});

const ingressBridge = new Bridge(stack, 'MyIngressBridge', {
  bridgeName: 'my-ingress-bridge',
  config: BridgeConfiguration.ingress({
    maxBitrate: Bitrate.mbps(10),
    maxOutputs: 2,
    networkSources: [{
      name: 'on-prem-source',
      source: {
        protocol: BridgeProtocol.RTP,
        network: productionNetwork,
        multicastIp: '239.1.1.1',
        port: 5000,
      },
    }],
  }),
  gateway: gateway,
});
```

#### Egress Bridge (Cloud to On-premises)

An egress bridge sends content from MediaConnect flows to on-premises equipment:

```ts
declare const stack: Stack;
declare const gateway: Gateway;
declare const flow: Flow;
declare const vpcInterface: VpcInterfaceConfig;
declare const productionNetwork: GatewayNetwork;

const egressBridge = new Bridge(stack, 'MyEgressBridge', {
  bridgeName: 'my-egress-bridge',
  config: BridgeConfiguration.egress({
    maxBitrate: Bitrate.mbps(10),
    flowSources: [{
      name: 'cloud-source',
      source: {
        flow: flow,
        vpcInterface: vpcInterface,
      },
    }],
    networkOutputs: [{
      name: 'on-prem-output',
      output: BridgeOutputConfiguration.network({
        ipAddress: '192.168.1.200',
        port: 5001,
        network: productionNetwork,
        protocol: BridgeProtocol.RTP,
        ttl: 50,
      }),
    }],
  }),
  gateway: gateway,
});
```

### Bridge Sources

For failover scenarios, you can add additional sources to an existing bridge using the `BridgeSource` construct:

```ts
declare const stack: Stack;
declare const bridge: Bridge;
declare const flow: Flow;

// Add a flow source to an egress bridge (requires failover to be enabled)
const additionalSource = new BridgeSource(stack, 'AdditionalSource', {
  bridgeSourceName: 'backup-source',
  bridge: bridge,
  source: BridgeSourceConfiguration.flow({
    flow: flow,
  }),
});
```

### Bridge Outputs

Bridge outputs are configured as part of the bridge configuration for egress bridges. They define where content exits the bridge to on-premises equipment.

```ts
declare const productionNetwork: GatewayNetwork;

const networkConfig = BridgeOutputConfiguration.network({
  ipAddress: '192.168.1.200',
  port: 5001,
  network: productionNetwork,
  protocol: BridgeProtocol.RTP,
  ttl: 50,
});

const namedOutput = { name: 'on-prem-output', output: networkConfig };
```

## Encryption

MediaConnect supports encryption for sources, outputs, and entitlements. This package provides type-safe encryption configuration structs that match the encryption requirements for different protocols.

### Encryption Types

MediaConnect supports two types of encryption:

1. **Static Key Encryption** - Used for Zixi Push/Pull protocols and entitlements
2. **SRT Password Encryption** - Used for SRT Listener and SRT Caller protocols

Note: CFN exposes only `static-key` and `srt-password` for flow output encryption today; SPEKE is not currently part of the surface.

**Auto-created IAM role.** Every encryption struct accepts an optional `role`. Omit it and the consuming construct creates a scoped role for you: trust policy for `mediaconnect.amazonaws.com` with `aws:SourceAccount` + `aws:SourceArn` conditions (confused-deputy protection), and just enough permission to read the provided secret (including `kms:Decrypt` when the secret uses a customer-managed KMS key).

**Providing your own role.** If you supply a `role`, it's used as-is — the construct does **not** grant it any permissions. You must grant it the necessary permissions yourself. Provide your own role when you need stricter control or a shared identity.

**Trust-policy scope: flows vs. routers.** When the L2 auto-creates a role, it also pins `aws:SourceArn` to the consuming resource:

- **Flows** — the trust policy pins the flow ARN (`arn:...:flow:*:<flow-name>`). The `*` wildcards the service-assigned id segment; the flow name is fixed at create time.
- **Routers** — the trust policy can only pin a wildcarded ARN (`arn:...:routerInput:*` / `arn:...:routerOutput:*`). Router I/O ARNs use a service-generated id segment that is unknown at synth time, and using the live ARN attribute would create a CloudFormation dependency cycle (role → router → role). If you need per-resource pinning, supply your own `role` with a trust condition that pins the exact ARN — you can compute it from the resource's `routerInputId` / `routerOutputId` after first deploy and apply it on a follow-up deploy.

### Static Key Encryption

Use a static-key encryption struct for Zixi protocols and entitlements. This requires an encryption algorithm (AES128, AES192, or AES256). Pass it inline where the construct asks for it:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

new FlowEntitlement(stack, 'MyEntitlement', {
  flow,
  subscribers: ['111122223333'],
  description: 'Grant partner access to live feed',
  encryption: {
    role,
    secret,
    algorithm: EncryptionAlgorithm.AES256,
  },
});
```

### SRT Password Encryption

SRT protocols take an encryption struct with just `role` and `secret` — no algorithm:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

new FlowOutput(stack, 'SrtOutput', {
  flow,
  output: OutputConfiguration.srtCaller({
    destination: '203.0.113.100',
    port: 7000,
    encryption: { role, secret },
  }),
});
```

### Using Encryption with Sources

Apply encryption when configuring flow sources:

```ts
declare const stack: Stack;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

// SRT Listener source with encryption
const flow = new Flow(stack, 'MyFlow', {
  source: SourceConfiguration.srtListener({
    flowSourceName: 'encrypted-source',
    port: 5000,
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
    decryption: { role, secret },
  }),
});

// Zixi Push source with encryption
const flow2 = new Flow(stack, 'MyFlow2', {
  source: SourceConfiguration.zixiPush({
    flowSourceName: 'encrypted-zixi-source',
    maxLatency: Duration.millis(2000),
    network: NetworkConfiguration.publicNetwork('203.0.113.0/24'),
    decryption: {
      role,
      secret,
      algorithm: EncryptionAlgorithm.AES256,
    },
  }),
});
```

### Using Encryption with Outputs

Apply encryption when configuring flow outputs:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

// SRT Caller output with encryption
const output = new FlowOutput(stack, 'EncryptedOutput', {
  flow: flow,
  description: 'Encrypted SRT output',
  output: OutputConfiguration.srtCaller({
    destination: '203.0.113.100',
    port: 7000,
    encryption: { role, secret },
  }),
});
```

### Using Encryption with Entitlements

Entitlements use static key encryption:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

const entitlement = new FlowEntitlement(stack, 'MyEntitlement', {
  flow: flow,
  description: 'Grant partner access to live feed',
  subscribers: ['111122223333'],
  encryption: {
    role,
    secret,
    algorithm: EncryptionAlgorithm.AES256,
  },
});
```

### Router Transit Encryption

When integrating flows with routers, use transit encryption to secure the connection between the flow and router:

```ts
declare const stack: Stack;
declare const flow: Flow;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;
declare const existingRouterOutput: RouterOutput;

// Flow output to router with transit encryption
const routerOutput = new FlowOutput(stack, 'RouterOutput', {
  flow: flow,
  output: OutputConfiguration.router({
    encryption: { role, secret },
  }),
});

// Flow source from router with transit encryption
const flowFromRouter = new Flow(stack, 'FlowFromRouter', {
  source: SourceConfiguration.router({
    routerOutput: existingRouterOutput,
    decryption: { role, secret },
  }),
});
```

### Router SRT Encryption

Router outputs using SRT protocols use `RouterSrtEncryption` for encryption:

```ts
declare const stack: Stack;
declare const networkInterface: RouterNetworkInterface;
declare const role: iam.IRole;
declare const secret: secretsmanager.ISecret;

const output = new RouterOutput(stack, 'EncryptedSrtOutput', {
  routerOutputName: 'encrypted-srt-output',
  maximumBitrate: Bitrate.mbps(10),
  routingScope: RoutingScope.REGIONAL,
  tier: RouterOutputTier.OUTPUT_50,
  configuration: RouterOutputConfiguration.standard({
    protocol: RouterOutputProtocol.srtCaller({
      destinationAddress: '203.0.113.100',
      destinationPort: 9001,
      minimumLatency: Duration.millis(200),
      encryptionConfiguration: { role, secret },
    }),
    networkInterface: networkInterface,
  }),
});
```

Note: `RouterSrtEncryption` is distinct from `SrtPasswordEncryption` (used on flow sources/outputs) — router outputs use a simpler CFN shape without a `keyType` discriminator.

## CloudWatch Metrics

Flows and Bridges expose CloudWatch metric helpers for monitoring. You can create alarms and dashboards using these metrics:

```ts
declare const flow: Flow;
declare const stack: Stack;

// Create a CloudWatch alarm on source bitrate
const alarm = flow.metricSourceBitrate().createAlarm(stack, 'LowBitrate', {
  threshold: 1000000,
  evaluationPeriods: 1,
});

// Monitor unrecovered packets
flow.metricSourceNotRecoveredPackets().createAlarm(stack, 'PacketLoss', {
  threshold: 100,
  evaluationPeriods: 2,
});

// Track total packets with custom options
const totalPackets = flow.metricSourceTotalPackets({
  statistic: 'sum',
  period: Duration.minutes(5),
});
```

### Flow metrics

- `metricSourceBitrate()` - Bitrate of content ingested into the flow (average)
- `metricSourceNotRecoveredPackets()` - Packets lost in transit that were not recovered by error correction (sum)
- `metricSourceTotalPackets()` - Total packets received by the flow sources (sum)
- `metricSourceSelected()` - Indicates which source is being used under Failover mode (max; 1 = active, 0 = standby)
- `metricSourceConnected()` - Source connection state for Zixi, SRT, and RIST (min; 1 = connected, 0 = disconnected)
- `metricSourceDisconnections()` - Number of times the source transitioned from connected to disconnected (sum)
- `metricSourceDroppedPackets()` - Packets lost before any error correction took place (sum)
- `metricSourcePacketLossPercent()` - Percentage of packets lost during transit, even if they were recovered (average)
- `metricSourceRoundTripTime()` - Round-trip time to the source for RIST, Zixi, and SRT (average, milliseconds)
- `metricSourceJitter()` - Current network jitter of the source (average, milliseconds)
- `metric(metricName)` - Create a custom metric by name

### Bridge metrics

Bridge metrics are dimensioned by `BridgeARN`. The underlying CloudWatch metric name is chosen automatically based on whether the bridge is ingress or egress.

- `metricSourceBitrate(bridgeSourceName)` - Bitrate of a specific bridge source (average)
- `metricSourcePacketLossPercent(bridgeSourceName)` - Percentage of packets lost on a specific bridge source (average)
- `metricFailoverSwitches()` - Total number of times the bridge switches between sources under `FAILOVER` failover mode (sum)
- `metric(metricName)` - Create a custom metric by name

### Router Input metrics

Router input metrics are dimensioned by `RouterInputARN`.

- `metricBitrate()` - Bitrate of the router input's payload (average)
- `metricNotRecoveredPackets()` - Packets lost in transit that were not recovered by error correction (sum)
- `metricTotalPackets()` - Total number of packets received by the router input (sum)
- `metricConnected()` - Connection state for SRT sources (min; 1 = connected, 0 = disconnected)
- `metricContinuityCounterErrors()` - Continuity counter errors in the transport stream (sum)
- `metricLatency()` - Recovery latency of the input stream for RIST, SRT, and RTP-FEC (average, milliseconds)
- `metricFailoverSwitches()` - Total times the router input switched sources under Failover mode (sum)
- `metric(metricName)` - Create a custom metric by name

### Router Output metrics

Router output metrics are dimensioned by `RouterOutputARN`.

- `metricBitrate()` - Bitrate of the router output's payload (average)
- `metricTotalPackets()` - Total number of packets sent by the router output (sum)
- `metricConnected()` - Connection state for SRT outputs (min; 1 = connected, 0 = disconnected)
- `metricArqRequests()` - Retransmitted packets requested through ARQ for RIST and SRT outputs (sum)
- `metric(metricName)` - Create a custom metric by name

### Gateway metrics

Gateway metrics are dimensioned by `GatewayARN`. Pass extra dimensions such as `NetworkName`, `InstanceId`, or `BridgeSourceName` via `props.dimensionsMap` to narrow to a specific network, appliance, or bridge source.

- `metricEgressBridgeTotalPackets()` - Total packets sent from egress bridges hosted on the gateway (sum)
- `metricEgressBridgeDroppedPackets()` - Packets dropped by egress bridges hosted on the gateway (sum)
- `metricIngressBridgeTotalPackets()` - Total packets received by ingress bridges hosted on the gateway (sum)
- `metricIngressBridgeDroppedPackets()` - Packets dropped by ingress bridges hosted on the gateway (sum)
- `metric(metricName)` - Create a custom metric by name (e.g. `IngressBridgeBitRate`, `EgressBridgeBitRate`, `IngressBridgeSourcePacketLossPercent`)

Pair the total + dropped helpers to build a dropped-packet percentage chart — for example, divide `metricEgressBridgeDroppedPackets()` by `metricEgressBridgeTotalPackets()` in a math expression.

All metrics support standard CloudWatch metric options for customizing period, statistic, and dimensions.

## Public CIDR warnings

Several constructs accept CIDR ranges that determine who can contribute content or pull outputs. Passing an open range (`0.0.0.0/0` or any `/0` prefix) makes the resource reachable from anywhere on the public internet, which is rarely what you want. The module emits synthesis-time warnings when it detects an open range on:

- `GatewayProps.egressCidrBlocks`
- `NetworkConfiguration.publicNetwork(cidr)` used on flow sources
- `cidrAllowList` on Zixi Push / Zixi Pull / SRT Listener flow outputs
- `RouterNetworkConfiguration.publicNetwork({ cidr })`

Restrict each range to the narrowest set of addresses that actually need access.
