# AWS::MediaLive Construct Library
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

## AWS Elemental MediaLive

AWS Elemental MediaLive is a real-time video service that lets you create live outputs for broadcast and streaming delivery.

This package contains constructs for working with AWS Elemental MediaLive, including Inputs, Input Security Groups, Channels, and MediaLive Anywhere resources (Networks, Clusters, Channel Placement Groups, SDI Sources).

For further information on AWS Elemental MediaLive, see [the documentation](https://docs.aws.amazon.com/medialive/latest/ug/what-is.html). See [supported codecs per output group](https://docs.aws.amazon.com/medialive/latest/ug/outputs-supported-codecs.html).

The following example creates an SRT caller input, encodes it to H.264 + AAC, and outputs HLS segments to an S3 bucket:

```ts
declare const stack: Stack;
declare const bucket: s3.IBucket;

const input = new medialive.Input(stack, 'SrtInput', {
  inputName: 'my-srt-input',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.10',
    srtListenerPort: 5000,
  }]),
});

const video = medialive.EncodeConfiguration.video({
  name: 'video_720p',
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_30,
  }),
  width: 1280,
  height: 720,
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'audio_aac',
  codec: medialive.AudioCodecSettings.aac({ bitrate: Bitrate.kbps(192) }),
});

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
    }),
  ],
});
```

## Input

An input represents the upstream source that feeds a MediaLive channel. Use `InputConfiguration` factory methods to create different input types.

### SRT Caller

MediaLive connects to a remote SRT listener:

```ts
declare const stack: Stack;
new medialive.Input(stack, 'SrtInput', {
  inputName: 'srt-caller',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.10',
    srtListenerPort: 5000,
  }]),
});
```

### SRT Listener

MediaLive listens for an incoming SRT connection. SRT listener inputs require an input security
group. To receive encrypted content, supply a `decryption` block referencing a Secrets Manager
secret that holds the passphrase — the secret is passed by reference, so MediaLive resolves the ARN
at synth time:

```ts
declare const stack: Stack;
declare const passphrase: secretsmanager.ISecret;

const sg = new medialive.InputSecurityGroup(stack, 'SrtSg', {
  allowlistRules: ['203.0.113.0/24'],
});

new medialive.Input(stack, 'SrtListenerInput', {
  inputName: 'srt-listener',
  input: medialive.InputConfiguration.srtListener({
    inputSecurityGroups: [sg],
    minimumLatency: Duration.millis(500),
    streamId: 'my-stream-id',
    decryption: {
      algorithm: medialive.SrtDecryptionAlgorithm.AES256,
      passphraseSecret: passphrase,
    },
  }),
});
```

### AWS Elemental MediaConnect Router

Creates a MediaConnect Router Input with automatic encryption:

```ts
declare const stack: Stack;
new medialive.Input(stack, 'RouterInput', {
  inputName: 'mc-router',
  input: medialive.InputConfiguration.mediaConnectRouter(),
});
```

An input created this way is the only kind `@aws-cdk/aws-mediaconnect-alpha`'s `RouterOutputConfiguration.mediaLiveInput()` can deliver to — pointing it at any other input type synths but fails at deploy.

### MP4 File from S3

Use `InputSource.fromBucket()` to reference an S3 object:

```ts
declare const stack: Stack;
declare const bucket: s3.IBucket;

new medialive.Input(stack, 'FileInput', {
  inputName: 'mp4-file',
  input: medialive.InputConfiguration.mp4File([
    medialive.InputSource.fromBucket(bucket, 'media/input.mp4'),
  ]),
});
```

### Importing an Existing Input

```ts
declare const stack: Stack;
const input = medialive.Input.fromInputArn(stack, 'Imported',
  'arn:aws:medialive:us-east-1:123456789012:input:1234567');
```

## Input Security Group

An input security group controls which IPv4 CIDR blocks can push content to a push-type input.

```ts
declare const stack: Stack;
const sg = new medialive.InputSecurityGroup(stack, 'SG', {
  allowlistRules: ['203.0.113.0/24'],
});
```

### Importing an Existing Input Security Group

```ts
declare const stack: Stack;
const sg = medialive.InputSecurityGroup.fromInputSecurityGroupArn(stack, 'Imported',
  'arn:aws:medialive:us-east-1:123456789012:inputSecurityGroup:1234567');
```

## Channel

A channel takes one or more inputs, encodes them, and produces output groups. If no `role` is provided, the channel auto-creates an IAM role with the `medialive.amazonaws.com` service principal.

Minimal example — single input, single HLS output:

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;

const video = medialive.EncodeConfiguration.video({
  name: 'video_720p',
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_30,
  }),
  width: 1280,
  height: 720,
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'audio_aac',
  codec: medialive.AudioCodecSettings.aac({ bitrate: Bitrate.kbps(192) }),
});

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
    }),
  ],
});
```

### STANDARD Channel with MediaPackage V2

A STANDARD channel runs two pipelines for redundancy. Each output group needs two destinations — one per pipeline.

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const mpChannel: mediapackagev2.IChannel;

const hdVideo = medialive.EncodeConfiguration.video({
  name: 'video_1080p',
  codec: medialive.VideoCodecSettings.h265({
    rateControl: medialive.H265RateControl.qvbr({
      maxBitrate: Bitrate.mbps(8),
      qvbrQualityLevel: 7,
    }),
    framerate: medialive.Framerate.FPS_30,
  }),
  width: 1920,
  height: 1080,
});

const sdVideo = medialive.EncodeConfiguration.video({
  name: 'video_480p',
  codec: medialive.VideoCodecSettings.h265({
    rateControl: medialive.H265RateControl.qvbr({
      maxBitrate: Bitrate.mbps(2),
      qvbrQualityLevel: 7,
    }),
    framerate: medialive.Framerate.FPS_30,
  }),
  width: 854,
  height: 480,
});

const audio = medialive.EncodeConfiguration.audio({
  name: 'audio_aac',
  codec: medialive.AudioCodecSettings.aac({ bitrate: Bitrate.kbps(192) }),
});

new medialive.Channel(stack, 'Channel', {
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.mediaPackageV2({
      name: 'emp',
      channel: mpChannel,
      outputs: [
        { encode: hdVideo, outputName: 'hd' },
        { encode: sdVideo, outputName: 'sd' },
        { encode: audio, outputName: 'audio' },
      ],
    }),
  ],
});
```

### Global Configuration

`globalConfiguration` sets channel-wide behaviour: how the pipelines are locked together and the output timing source. All fields are optional and fall back to MediaLive defaults.

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  timecodeConfig: {
    source: medialive.TimecodeSource.EMBEDDED,
  },
  globalConfiguration: {
    outputLocking: medialive.OutputLocking.epoch(),
    outputTimingSource: medialive.OutputTimingSource.INPUT_CLOCK,
  },
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
    }),
  ],
});
```

#### Output locking

`outputLocking` synchronises the frames emitted by a channel's two pipelines. Pick a strategy with
the `OutputLocking` factory:

- `OutputLocking.pipeline()` — synchronise each pipeline's output to the other. Choose how with
  `method`: `PipelineLockingMethod.SOURCE_TIMECODE` (default, needs reliable embedded timecodes) or
  `PipelineLockingMethod.VIDEO_ALIGNMENT` (visual content matching, no timecodes required).
- `OutputLocking.epoch()` — synchronise to the Unix epoch (optionally a `customEpoch`/`jamSyncTime`).
  Requires `outputTimingSource: OutputTimingSource.INPUT_CLOCK` (enforced at synth).
- `OutputLocking.disabled()` — no synchronisation.

```ts
// Video-aligned pipeline locking — useful when sources lack reliable timecodes
const locking = medialive.OutputLocking.pipeline({
  method: medialive.PipelineLockingMethod.VIDEO_ALIGNMENT,
});
```

#### Input-loss behavior

`inputLossBehavior` controls what MediaLive emits when the input is lost: a black period, then a
repeated frame, then either a solid colour or a slate image. Provide the slate as a
[`FileLocation`](#file-locations).

```ts
declare const slateBucket: s3.IBucket;

const inputLoss: medialive.InputLossBehavior = {
  blackFrame: Duration.seconds(1),
  repeatFrame: Duration.seconds(5),
  imageType: medialive.InputLossImageType.SLATE,
  imageSlate: medialive.FileLocation.fromBucket(slateBucket, 'slates/offline.png'),
};
```

## File locations

Several channel features reference a file MediaLive reads at runtime — an input-loss slate, an
avail-blanking image, a blackout-slate image, or a burn-in caption font. These all take a
`FileLocation`, created from an S3 bucket (which auto-grants the channel role read access) or a URL
(with optional SSM-backed credentials):

```ts
import { StringParameter } from 'aws-cdk-lib/aws-ssm';

declare const bucket: s3.IBucket;
declare const passwordParam: StringParameter;

// From an S3 bucket — the channel role is granted read access automatically
const fromS3 = medialive.FileLocation.fromBucket(bucket, 'assets/slate.png');

// From a URL with optional credentials (SSM parameter read access auto-granted)
const fromUrl = medialive.FileLocation.url('https://origin.example.com/font.ttf', {
  username: 'ingest-user',
  password: passwordParam,
});
```

## Color correction

A channel can apply one or more color-space conversions to its video, optionally using a 3D LUT
to remap colors. Each `ColorCorrection` declares the `inputColorSpace` to match and the
`outputColorSpace` to convert to. MediaLive reads the LUT from S3 at runtime, so it must be an S3
location — provide it via `Lut.fromBucket()` (which uses the secure `s3ssl://` form and auto-grants
the channel role read access) or `Lut.url()` with an `s3://`/`s3ssl://` URL:

```ts
declare const stack: Stack;
declare const bucket: s3.IBucket;
declare const input: medialive.IInput;
declare const video: medialive.EncodeConfiguration;
declare const destination: medialive.OutputDestination;

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  colorCorrections: [{
    inputColorSpace: medialive.ColorSpace.REC_601,
    outputColorSpace: medialive.ColorSpace.REC_709,
    lut: medialive.Lut.fromBucket(bucket, 'luts/rec601-to-rec709.cube'),
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [destination],
      outputs: [{ encodes: [video], outputName: 'video' }],
    }),
  ],
});
```

## Encode Configuration

Use `EncodeConfiguration.video()`, `EncodeConfiguration.audio()`, and `EncodeConfiguration.caption()` to define encodes.

### Video

```ts
// H.264
const h264 = medialive.EncodeConfiguration.video({
  name: 'h264_720p',
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: Bitrate.mbps(3) }),
    framerate: medialive.Framerate.FPS_30,
    profile: medialive.H264Profile.HIGH,
  }),
  width: 1280,
  height: 720,
});

// H.265
const h265 = medialive.EncodeConfiguration.video({
  name: 'h265_1080p',
  codec: medialive.VideoCodecSettings.h265({
    rateControl: medialive.H265RateControl.qvbr({
      maxBitrate: Bitrate.mbps(5),
      qvbrQualityLevel: 7,
    }),
    framerate: medialive.Framerate.FPS_30,
    profile: medialive.H265Profile.MAIN,
    tier: medialive.H265Tier.HIGH,
  }),
  width: 1920,
  height: 1080,
});
```

Video codecs accept optional overrides for adaptive quantization, scene-change detection, color space, and more. See the props interfaces for the full list:

```ts
const hdr = medialive.EncodeConfiguration.video({
  name: 'h265_hdr',
  codec: medialive.VideoCodecSettings.h265({
    rateControl: medialive.H265RateControl.qvbr({ maxBitrate: Bitrate.mbps(8), qvbrQualityLevel: 8 }),
    framerate: medialive.Framerate.FPS_30,
    sceneChangeDetect: medialive.H265SceneChangeDetect.ENABLED,
    colorSpaceSettings: medialive.H265ColorSpaceSettings.hlg2020(),
  }),
  width: 1920,
  height: 1080,
});
```

### Audio

```ts
// AAC stereo
const aac = medialive.EncodeConfiguration.audio({
  name: 'aac_stereo',
  codec: medialive.AudioCodecSettings.aac({
    bitrate: Bitrate.kbps(192),
    codingMode: medialive.AacCodingMode.CODING_MODE_2_0,
  }),
});

// AC3 5.1
const ac3 = medialive.EncodeConfiguration.audio({
  name: 'ac3_surround',
  codec: medialive.AudioCodecSettings.ac3({
    bitrate: Bitrate.kbps(384),
    codingMode: medialive.Ac3CodingMode.CODING_MODE_3_2_LFE,
  }),
});
```

### Caption

A caption encode converts a source caption track (referenced by `captionSelectorName`) to an
output format via the `CaptionDestination` factory. One selector can feed multiple encodes:

```ts
// Define a caption selector on the input attachment (see Input Attachment Settings below)
const captionSelector = medialive.CaptionSelector.embedded('captions');

// WebVTT captions — packaged alongside the video encode in the same output
const webvtt = medialive.EncodeConfiguration.caption({
  name: 'eng_webvtt',
  captionSelectorName: captionSelector.name,
  languageCode: 'eng',
  languageDescription: 'English',
  destination: medialive.CaptionDestination.webvtt(),
});

// Burned-in captions — rendered into the video, styled via the burn-in options
const burnIn = medialive.EncodeConfiguration.caption({
  name: 'eng_burnin',
  captionSelectorName: captionSelector.name,
  destination: medialive.CaptionDestination.burnIn({
    alignment: medialive.CaptionAlignment.CENTERED,
    fontColor: medialive.CaptionFontColor.WHITE,
    outlineColor: medialive.CaptionOutlineColor.BLACK,
    fontSize: medialive.CaptionFontSize.AUTO,
  }),
});
```

## Cross-service integrations

| Destination | MediaLive side | Other side | Package |
|---|---|---|---|
| MediaPackage V2 | `medialive.OutputGroupConfiguration.mediaPackageV2()` | `mediapackagev2.Channel` | `@aws-cdk/aws-mediapackagev2-alpha` |
| MediaConnect Router (output) | `medialive.OutputGroupConfiguration.mediaConnectRouter()` | `mediaconnect.RouterInputConfiguration.mediaLiveChannel()` | `@aws-cdk/aws-mediaconnect-alpha` |
| MediaConnect Router (input) | `medialive.InputConfiguration.mediaConnectRouter()` | `mediaconnect.RouterOutputConfiguration.mediaLiveInput()` | `@aws-cdk/aws-mediaconnect-alpha` |

### AWS Elemental MediaPackage V2

Use `mediaPackageV2()` and pass a single `channel` — MediaLive maps each pipeline to a MediaPackage ingest endpoint automatically (one for `SINGLE_PIPELINE`, both for `STANDARD`). Each output contains a single encode (one track per output).

In-band captions (burn-in, embedded) ride alongside a video encode via the `captions` prop:

```ts
declare const mpChannel: mediapackagev2.IChannel;
declare const hdVideo: medialive.EncodeConfiguration;
declare const sdVideo: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;
declare const burnIn: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.mediaPackageV2({
  name: 'emp',
  channel: mpChannel,
  outputs: [
    { encode: hdVideo, captions: [burnIn], outputName: 'hd' },
    { encode: sdVideo, outputName: 'sd' },
    { encode: audio, outputName: 'audio' },
  ],
});
```

For per-pipeline control — for example pinning pipeline 0 to a specific endpoint, or delivering each pipeline to a different (cross-region) channel — use `mediaPackageV2PerPipeline()` with explicit destinations:

```ts
declare const primary: mediapackagev2.IChannel;
declare const hdVideo: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.mediaPackageV2PerPipeline({
  name: 'emp',
  destinations: [
    // destinations[0] → Pipeline 0, destinations[1] → Pipeline 1
    medialive.MediaPackageV2Destination.channel(primary, medialive.MediaPackageV2EndpointId.ENDPOINT_2),
    medialive.MediaPackageV2Destination.channel(primary, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
  ],
  outputs: [
    { encode: hdVideo, outputName: 'hd' },
  ],
});
```

### HLS

Use `OutputDestination.url()` for HTTP origins or `OutputDestination.toBucket()` for S3:

```ts
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

// HLS to S3
medialive.OutputGroupConfiguration.hls({
  name: 'hls_s3',
  destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
  outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
});

// HLS to an HTTPS CDN origin.
medialive.OutputGroupConfiguration.hls({
  name: 'hls-http',
  destinations: [medialive.OutputDestination.url('https://203.0.113.10/ingest/stream')],
  hlsCdnSettings: medialive.HlsCdnSettings.basicPut(),
  outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
});
```

### Archive

Archive outputs write long-form recordings to S3:

```ts
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.archive({
  name: 'archive',
  destinations: [medialive.S3OutputDestination.toBucket(bucket, 'archive/recording')],
  rolloverInterval: Duration.seconds(600),
  outputs: [{ encodes: [video, audio], outputName: 'archive_out' }],
});
```

### RTMP

RTMP outputs support H.264 + AAC only. Each output takes one destination per channel pipeline (the console's "Destination A" / "Destination B") via `RtmpDestination.url()` — one for `SINGLE_PIPELINE`, two for `STANDARD`:

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.rtmp({
  name: 'social',
  outputs: [{
    encodes: [video, audio],
    outputName: 'live',
    destinations: [
      medialive.RtmpDestination.url('rtmp://rtmp.example.com/live', 'your-stream-key'),
    ],
  }],
});
```

### SRT

SRT outputs use `SrtDestination.caller()` for caller mode or `SrtDestination.listener()` for listener mode. When you already have a full SRT URL rather than a separate host and port, use `SrtDestination.callerUrl()`. SRT output is always encrypted, so every destination takes an `encryptionPassphraseSecret` (a Secrets Manager secret). Each output takes one destination per channel pipeline ("Destination A"/"Destination B") — one for `SINGLE_PIPELINE`, two for `STANDARD`:

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;
declare const passphrase: secretsmanager.ISecret;

// SRT caller to a remote listener
medialive.OutputGroupConfiguration.srt({
  name: 'srt_out',
  outputs: [{
    encodes: [video, audio],
    outputName: 'srt_caller',
    destinations: [medialive.SrtDestination.caller({
      address: '203.0.113.20',
      port: 5000,
      encryptionPassphraseSecret: passphrase,
    })],
  }],
});

// SRT listener — MediaLive waits for the downstream system to connect
medialive.OutputGroupConfiguration.srt({
  name: 'srt_listen',
  outputs: [{
    encodes: [video, audio],
    outputName: 'srt_listener',
    destinations: [medialive.SrtDestination.listener({
      listenerPort: 5000,
      encryptionPassphraseSecret: passphrase,
    })],
  }],
});
```

### AWS Elemental MediaConnect Router

`mediaConnectRouter()` delivers each channel pipeline to an AWS Elemental MediaConnect Router. Transit encryption defaults to AUTOMATIC; CDK derives one destination per pipeline from the channel class, so the common case needs no per-pipeline configuration. You must specify `availabilityZones` — exactly one for a `SINGLE_PIPELINE` channel, or two (one per pipeline) for `STANDARD`. The downstream wiring — which router input each pipeline feeds — is configured on the MediaConnect side, referencing this group's output by name and pipeline id.

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;
declare const passphrase: secretsmanager.ISecret;
declare const passphrase1: secretsmanager.ISecret;

// AUTOMATIC encryption on every pipeline (MPEG-TS container, like UDP)
medialive.OutputGroupConfiguration.mediaConnectRouter({
  name: 'router_out',
  availabilityZones: ['us-east-1a'],
  outputs: [{ encodes: [video, audio], outputName: 'router_ts' }],
});

// One shared Secrets Manager passphrase across all pipelines (SECRETS_MANAGER encryption)
medialive.OutputGroupConfiguration.mediaConnectRouter({
  name: 'router_out',
  availabilityZones: ['us-east-1a'],
  routerSettings: medialive.MediaConnectRouterSettings.shared({ encryptionSecret: passphrase }),
  outputs: [{ encodes: [video, audio], outputName: 'router_ts' }],
});

// Distinct encryption per pipeline — an omitted pipeline stays AUTOMATIC (STANDARD channels)
medialive.OutputGroupConfiguration.mediaConnectRouter({
  name: 'router_out',
  availabilityZones: ['us-east-1a', 'us-east-1b'],
  routerSettings: medialive.MediaConnectRouterSettings.perPipeline({
    pipeline1: { encryptionSecret: passphrase1 },
  }),
  outputs: [{ encodes: [video, audio], outputName: 'router_ts' }],
});
```

When a passphrase secret is supplied, the channel's IAM role is automatically granted read access to it.

### UDP

UDP outputs deliver MPEG-TS over UDP or RTP. Use `UdpOutputDestination.udp()` for plain UDP or `.rtp()` for RTP (required if using FEC):

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.udp({
  name: 'udp_out',
  destinations: [medialive.UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
  outputs: [{ encodes: [video, audio], outputName: 'ts_out' }],
});
```

### Frame Capture

Frame capture outputs write periodic JPEG snapshots to S3:

```ts
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.frameCapture({
  name: 'thumbnails',
  destinations: [medialive.S3OutputDestination.toBucket(bucket, 'thumbnails/live')],
  outputs: [{ encodes: [video], outputName: 'thumb' }],
});
```

### Microsoft Smooth Streaming

MS Smooth outputs push fragmented MP4 to an IIS Smooth Streaming endpoint:

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.msSmooth({
  name: 'smooth',
  destinations: [medialive.OutputDestination.url('https://smooth.example.com/live')],
  outputs: [{ encodes: [video, audio], outputName: 'smooth_out' }],
});
```

### Per-output HLS settings

HLS outputs accept per-output `hlsSettings` via the `HlsSettings` factory — `standard()` for a video
rendition (with optional `M3u8Settings` for the transport stream), `audioOnly()` for an audio
rendition (with optional cover art as a [`FileLocation`](#file-locations)), `fmp4()`, or
`frameCapture()`.

```ts
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.hls({
  name: 'hls',
  destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
  outputs: [
    {
      encodes: [video],
      outputName: 'video',
      hlsSettings: medialive.HlsSettings.standard({
        m3u8Settings: medialive.M3u8Settings.of({
          scte35Behavior: medialive.M3u8Scte35Behavior.PASSTHROUGH,
          programNum: 1,
        }),
      }),
    },
    {
      encodes: [audio],
      outputName: 'audio',
      hlsSettings: medialive.HlsSettings.audioOnly({
        audioGroupId: 'program',
        audioOnlyImage: medialive.FileLocation.fromBucket(bucket, 'art/cover.png'),
      }),
    },
  ],
});
```

### Forward Error Correction (UDP)

UDP outputs accept optional `fec` settings (SMPTE 2022-1) — column-only or column-and-row FEC.
FEC requires an `rtp://` destination:

```ts
declare const video: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.udp({
  name: 'udp',
  destinations: [medialive.UdpOutputDestination.rtp({ address: '203.0.113.5', port: 5000 })],
  outputs: [{
    encodes: [video],
    outputName: 'ts',
    fec: { mode: medialive.FecMode.COLUMN_AND_ROW, columnDepth: 10, rowLength: 10 },
  }],
});
```

### MPEG-TS Container Settings

The MPEG-TS output groups — `udp()`, `archive()`, `srt()`, and `mediaConnectRouter()` — accept optional per-output `m2tsSettings` via `M2tsSettings.of()`. Omit it to use MediaLive's service defaults. Bitrates use `Bitrate`, intervals use `Duration`, and closed-value fields use enums (e.g. `M2tsRateMode`, `M2tsScte35Control`); PID fields are strings that accept decimal, hexadecimal, ranges, or comma-separated lists.

```ts
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

medialive.OutputGroupConfiguration.udp({
  name: 'udp_out',
  destinations: [medialive.UdpOutputDestination.udp({ address: '203.0.113.5', port: 5000 })],
  outputs: [{
    encodes: [video, audio],
    outputName: 'ts',
    m2tsSettings: medialive.M2tsSettings.of({
      bitrate: Bitrate.mbps(8),
      rateMode: medialive.M2tsRateMode.VBR,
      programNum: 1,
      patInterval: Duration.millis(100),
      pmtInterval: Duration.millis(100),
      scte35Control: medialive.M2tsScte35Control.PASSTHROUGH,
      dvbSdtSettings: {
        outputSdt: medialive.DvbSdtOutputMode.SDT_MANUAL,
        serviceName: 'My Service',
        repInterval: Duration.millis(2000),
      },
    }),
  }],
});
```

## Destinations

Each output group type uses a specific destination class. Destinations are created via static factory methods:

| Destination class | Factory methods | Used by |
|---|---|---|
| `OutputDestination` | `url()`, `toBucket()` | HLS, MS Smooth, CMAF Ingest |
| `S3OutputDestination` | `url()`, `toBucket()` | Archive, Frame Capture |
| `UdpOutputDestination` | `udp()`, `rtp()`, `url()` | UDP |
| `MediaPackageV2Destination` | `channel()` | MediaPackage V2 |
| `RtmpDestination` | `url()` | RTMP |
| `SrtDestination` | `caller()`, `callerUrl()`, `listener()` | SRT |

`OutputDestination.toBucket()` (and `S3OutputDestination.toBucket()`) build canonical `s3ssl://` URLs and automatically grant the channel's IAM role the required S3 permissions; `InputSource.fromBucket()` does the same for input reads. `MediaPackageV2Destination.channel()` automatically grants ingest permissions on the MediaPackage V2 channel.

The MediaConnect Router output group has no destination class — its delivery is configured on the MediaConnect side. Per-pipeline transit encryption is set via the group's `routerSettings` prop using `MediaConnectRouterSettings.shared()` / `.perPipeline()` (see [MediaConnect Router](#mediaconnect-router) above).

## Additional Destinations

MediaPackage V2 and CMAF Ingest output groups support `additionalDestinations` for cross-region delivery or backup packaging. These are separate from pipeline redundancy — they fan out the same content to extra endpoints.

The region for each destination is resolved automatically from the channel's stack. For cross-region imports, pass the region explicitly:

```ts
declare const primaryChannel: mediapackagev2.IChannel;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

// Import a channel from another region — the region travels with the channel
const backupChannel = mediapackagev2.Channel.fromChannelAttributes(this, 'BackupChannel', {
  channelName: 'backup-channel',
  channelGroupName: 'backup-group',
  region: 'us-west-2',
});

medialive.OutputGroupConfiguration.mediaPackageV2({
  name: 'emp',
  channel: primaryChannel,
  additionalDestinations: [
    // Cross-region: the destination picks up us-west-2 from the imported channel
    medialive.MediaPackageV2Destination.channel(backupChannel, medialive.MediaPackageV2EndpointId.ENDPOINT_1),
  ],
  outputs: [
    { encode: video, outputName: 'video' },
    { encode: audio, outputName: 'audio' },
  ],
});
```

## Pipeline Redundancy

Channels default to `SINGLE_PIPELINE`. Set `channelClass: ChannelClass.STANDARD` for two-pipeline redundancy.

When using STANDARD:

- Each output group's `destinations` array must have two entries — `destinations[0]` maps to Pipeline 0, `destinations[1]` maps to Pipeline 1.
- For MediaPackage V2, use `ENDPOINT_1` for Pipeline 0 and `ENDPOINT_2` for Pipeline 1.
- `additionalDestinations` are separate from pipeline redundancy — they fan out to extra endpoints.

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;

new medialive.Channel(stack, 'StandardChannel', {
  channelClass: medialive.ChannelClass.STANDARD,
  inputs: [{ input }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [
        medialive.OutputDestination.toBucket(bucket, 'live/pipeline0'),
        medialive.OutputDestination.toBucket(bucket, 'live/pipeline1'),
      ],
      outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
    }),
  ],
});
```

## Input Attachment Settings

Each entry in `inputs` is an input attachment, which can carry per-input extraction and connection
settings beyond the input itself.

**Selectors** pick specific tracks out of the input. Use `AudioSelector` (`byLanguage()`, `byPid()`,
`byTrack()`, `hlsRendition()`, `default()`), `CaptionSelector` (`byLanguage()`, `embedded()`,
`ancillary()`, `dvbSub()`, `scte27()`, `teletext()`, `arib()`), and `videoSelector` (color space,
HDR10 metadata, and program/PID selection via `VideoSelection`). A caption encode then references a
caption selector by name.

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;

new medialive.Channel(stack, 'Channel', {
  inputs: [{
    input,
    audioSelectors: [
      medialive.AudioSelector.byLanguage('eng', 'eng', medialive.AudioLanguageSelectionPolicy.STRICT),
    ],
    captionSelectors: [
      medialive.CaptionSelector.embedded('embedded'),
    ],
    videoSelector: {
      colorSpace: medialive.VideoColorSpace.HDR10,
      colorSpaceUsage: medialive.VideoColorSpaceUsage.FORCE,
      selectBy: medialive.VideoSelection.byProgramId(1),
    },
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video], outputName: 'hls_out' }],
    }),
  ],
});
```

**Network input settings** apply to URL-pull and multicast inputs — HLS bandwidth/buffer/retry
behaviour, the SCTE-35 source (`HlsScte35Source.SEGMENTS` or `MANIFEST`), HTTPS server validation,
and a multicast source IP for source-specific multicast. `logicalInterfaceNames` maps the input to
network interfaces on MediaLive Anywhere nodes.

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;

new medialive.Channel(stack, 'Channel', {
  inputs: [{
    input,
    networkInputSettings: {
      serverValidation: medialive.ServerValidation.CHECK_CRYPTOGRAPHY_AND_VALIDATE_NAME,
      hlsInputSettings: {
        bandwidth: Bitrate.mbps(5),
        scte35Source: medialive.HlsScte35Source.MANIFEST,
      },
    },
    logicalInterfaceNames: ['eth0', 'eth1'],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video], outputName: 'hls_out' }],
    }),
  ],
});
```

## Automatic Input Failover

Automatic input failover gives you input-*source* redundancy: attach a secondary input, and
MediaLive switches to it without restarting the channel when the active input meets a failover
condition. This is separate from the pipeline redundancy of `ChannelClass.STANDARD` (which
duplicates a single source across two pipelines).

Provide `automaticInputFailover` on the input attachment. If you don't specify conditions, a
single input-loss condition is used:

```ts
declare const stack: Stack;
declare const primaryInput: medialive.IInput;
declare const secondaryInput: medialive.IInput;
declare const audioSelector: medialive.AudioSelector;
declare const video: medialive.EncodeConfiguration;
declare const audio: medialive.EncodeConfiguration;
declare const bucket: s3.IBucket;

new medialive.Channel(stack, 'Channel', {
  inputs: [{
    input: primaryInput,
    automaticInputFailover: {
      secondaryInput,
      inputPreference: medialive.InputPreference.PRIMARY_INPUT_PREFERRED,
      errorClearTime: Duration.seconds(3),
      failoverConditions: [
        medialive.FailoverCondition.inputLoss({ threshold: Duration.millis(1500) }),
        medialive.FailoverCondition.audioSilence({ audioSelector, threshold: Duration.seconds(2) }),
        medialive.FailoverCondition.videoBlack({ blackDetectThreshold: 0.1, threshold: Duration.seconds(1) }),
      ],
    },
  }, {
    // The secondary input must also be attached to the channel as its own input.
    input: secondaryInput,
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video, audio], outputName: 'hls_out' }],
    }),
  ],
});
```

The primary and secondary inputs must have the same input class. The channel's IAM role is
granted read access to the secondary input's sources automatically, just like the primary.

## Ad Avail Handling

MediaLive can blank content during ad avails, insert blackout slates, and signal SCTE-35 ad avails
to downstream systems. These are all channel-level props.

`availBlanking` replaces video/audio/captions with black (or an image) during an ad avail, and
`blackoutSlate` shows a slate when a SCTE-35 blackout is signalled. Both image fields take a
[`FileLocation`](#file-locations).

```ts
declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  availBlanking: {
    state: medialive.AvailBlankingState.ENABLED,
    image: medialive.FileLocation.fromBucket(bucket, 'slates/avail.png'),
  },
  blackoutSlate: {
    state: medialive.BlackoutSlateState.ENABLED,
    image: medialive.FileLocation.fromBucket(bucket, 'slates/blackout.png'),
  },
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video], outputName: 'hls_out' }],
    }),
  ],
});
```

`availSettings` selects how SCTE-35 ad avails are handled — `AvailSettings.spliceInsert()`,
`AvailSettings.timeSignalApos()`, or `AvailSettings.esam()` for Event Signaling and Management
against an external POIS endpoint. `scte35SegmentationScope` controls which output groups receive
the segmentation cues. The ESAM POIS password is supplied as an SSM parameter, and the channel role
is granted read access to it automatically.

```ts
import { StringParameter } from 'aws-cdk-lib/aws-ssm';

declare const stack: Stack;
declare const input: medialive.IInput;
declare const bucket: s3.IBucket;
declare const video: medialive.EncodeConfiguration;
declare const poisPassword: StringParameter;

new medialive.Channel(stack, 'Channel', {
  inputs: [{ input }],
  availSettings: medialive.AvailSettings.esam({
    pois: {
      url: 'https://pois.example.com/esam',
      username: 'pois-user',
      password: poisPassword,
    },
    acquisitionPointId: 'acquisition-point-1',
    adAvailOffset: Duration.millis(200),
  }),
  scte35SegmentationScope: medialive.Scte35SegmentationScope.SCTE35_ENABLED_OUTPUT_GROUPS,
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video], outputName: 'hls_out' }],
    }),
  ],
});
```

## Auto-Created Role and Grants

When no `role` is provided, the channel auto-creates an IAM role with the `medialive.amazonaws.com` service principal and grants it only the permissions your configuration actually needs. These automatic grants apply **only** to the channel-managed role; if you bring your own `role`, none are added.

**Channel role grants** — wired based on what you configure (channel-managed role only):

| Configuration | Grant | Scope |
|---|---|---|
| `OutputDestination.toBucket()` | S3 read/write | The destination bucket/prefix |
| `InputSource.fromBucket()` | S3 read | The input source bucket/prefix |
| `MediaPackageV2Destination.channel()` | `mediapackagev2:PutObject` | The MediaPackage V2 channel |
| `SrtDestination` with an encryption secret | Secrets Manager read | The secret |
| URL pull input with a password parameter | SSM parameter read | The parameter |
| Thumbnails (on by default) | `s3:PutObject` | `*` — uploads to an AWS service-owned bucket |
| Channel logging (`logLevel` set) | CloudWatch Logs write | The `ElementalMediaLive` log group in your account/region |
| VPC output (`vpc` set) | EC2 ENI create/delete + describe | Scoped to your subnets/SGs; `Describe*` requires `*` |

**Input role grants** — separate from the channel role, used at input create/delete time. Like the channel role, these are added only when the input auto-creates its role; pass a `role` to `mediaConnect()` or `cdi()` and no grants are added:

| Input type | Grant | Scope |
|---|---|---|
| `InputConfiguration.mediaConnect()` | `mediaconnect:ManagedDescribeFlow`, `ManagedAddOutput`, `ManagedRemoveOutput` | `*` — service rejects flow-scoped grants |
| `InputConfiguration.cdi()` | EC2 ENI create/delete + describe | Scoped to your subnets/SGs; `Describe*` requires `*` |

Both channel and input auto-created roles include confused-deputy prevention (`aws:SourceAccount` + `aws:SourceArn` conditions). For the full list of trusted-entity requirements, see [the documentation](https://docs.aws.amazon.com/medialive/latest/ug/trusted-entity-requirements.html).

The auto-created role is available on `channel.role` if you need to add further permissions.

### Bringing your pre-defined role

When you pass a `role`, the channel makes **no** automatic grants — you will need to add the permissions that role needs. That covers both the principal policy and any referenced resource policies: S3 output destinations and input sources, Secrets Manager and SSM reads, MediaPackage V2 ingest, CloudWatch Logs, and VPC output ENI management. See the [trusted-entity requirements](https://docs.aws.amazon.com/medialive/latest/ug/trusted-entity-requirements.html), or pass the account's `MediaLiveAccessRole` — an IAM role that MediaLive can assume.

## CloudWatch Metrics

Channels expose CloudWatch metric helpers in the `AWS/MediaLive` namespace, dimensioned by `ChannelId` and `Pipeline`. Use the named helpers below for the most common metrics, or `metric(metricName, pipeline)` to access any metric documented by the [MediaLive metrics reference](https://docs.aws.amazon.com/medialive/latest/ug/monitoring-eml-metrics.html).

MediaLive publishes metrics per pipeline. Every helper takes a `Pipeline` argument so you make an explicit decision about which pipeline you're monitoring. `STANDARD` channels run two redundant pipelines (`PIPELINE_0`, `PIPELINE_1`) — alarm on both to cover the full channel. `SINGLE_PIPELINE` channels only publish on `PIPELINE_0`; passing `PIPELINE_1` throws at synth time.

```ts
declare const channel: medialive.Channel;
declare const stack: Stack;

channel.metricDroppedFrames(medialive.Pipeline.PIPELINE_0).createAlarm(stack, 'DroppedFrames', {
  threshold: 1,
  evaluationPeriods: 2,
});

channel.metricSvqTime(medialive.Pipeline.PIPELINE_0).createAlarm(stack, 'SvqTime', {
  threshold: 0,
  evaluationPeriods: 1,
});

// Custom metric by name with sum statistic
channel.metric('Output4xxErrors', medialive.Pipeline.PIPELINE_0, { statistic: 'sum' });
```

For STANDARD channels, alarm on both pipelines:

```ts
declare const standardChannel: medialive.Channel;
declare const stack: Stack;

standardChannel.metricDroppedFrames(medialive.Pipeline.PIPELINE_0).createAlarm(stack, 'Drops0', {
  threshold: 1,
  evaluationPeriods: 2,
});
standardChannel.metricDroppedFrames(medialive.Pipeline.PIPELINE_1).createAlarm(stack, 'Drops1', {
  threshold: 1,
  evaluationPeriods: 2,
});
```

### Channel metrics

| Helper | Metric name | Default statistic | Notes |
|---|---|---|---|
| `metricActiveAlerts(pipeline)` | `ActiveAlerts` | Max | Total active alerts on the channel |
| `metricNetworkIn(pipeline)` | `NetworkIn` | Avg | Inbound traffic in Mbps |
| `metricNetworkOut(pipeline)` | `NetworkOut` | Avg | Outbound traffic in Mbps |
| `metricInputVideoFrameRate(pipeline)` | `InputVideoFrameRate` | Max | Source video frame rate |
| `metricFillMsec(pipeline)` | `FillMsec` | Max | Time filled with fill frames — non-zero indicates input loss |
| `metricInputLossSeconds(pipeline)` | `InputLossSeconds` | Sum | Seconds without packets (RTP / MediaConnect inputs) |
| `metricDroppedFrames(pipeline)` | `DroppedFrames` | Sum | Frames dropped because the encoder fell behind |
| `metricSvqTime(pipeline)` | `SvqTime` | Max | Percent of time MediaLive reduced quality to keep up |
| `metric(name, pipeline, props?)` | (custom) | (caller-provided) | Build any metric in `AWS/MediaLive` |

The defaults match the AWS-recommended statistic for each metric. Pass `props` to override statistic, period, dimensions, or any other `MetricOptions` field.

## MediaLive Anywhere

MediaLive Anywhere lets you run MediaLive channels on your own on-premises hardware.

Certain input types are only available with Anywhere channels (channels configured with `anywhereSettings`):
SDI, SMPTE 2110 Receiver Group, and Multicast. Attempting to use these input types on a cloud channel will throw a validation error at synth time.

### Network

A network defines IP address pools and routes for Anywhere resources:

```ts
declare const stack: Stack;
const network = new medialive.Network(stack, 'Network', {
  networkName: 'on-prem-network',
  ipPools: ['10.0.0.0/24'],
  routes: [{ cidr: '0.0.0.0/0', gateway: '10.0.0.1' }],
});
```

### Cluster

A cluster represents a group of on-premises hardware nodes:

```ts
declare const stack: Stack;
declare const instanceRole: iam.IRole;

const cluster = new medialive.Cluster(stack, 'Cluster', {
  clusterName: 'on-prem-cluster',
  clusterType: medialive.ClusterType.ON_PREMISES,
  instanceRole,
});
```

### Channel Placement Group

A channel placement group assigns channels to specific nodes within a cluster. Associate it with a channel via `anywhereSettings`:

```ts
declare const stack: Stack;
declare const cluster: medialive.ICluster;
declare const input: medialive.IInput;
declare const video: medialive.EncodeConfiguration;
declare const bucket: s3.IBucket;

const cpg = new medialive.ChannelPlacementGroup(stack, 'CPG', {
  channelPlacementGroupName: 'my-cpg',
  cluster,
});

new medialive.Channel(stack, 'AnywhereChannel', {
  inputs: [{ input }],
  anywhereSettings: { cluster, channelPlacementGroup: cpg },
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls',
      destinations: [medialive.OutputDestination.toBucket(bucket, 'live/stream')],
      outputs: [{ encodes: [video], outputName: 'hls_out' }],
    }),
  ],
});
```

### SDI Source

An SDI source represents a physical SDI input on Anywhere hardware:

```ts
declare const stack: Stack;
const sdi = new medialive.SdiSource(stack, 'Sdi', {
  sdiSourceName: 'camera-1',
  type: medialive.SdiType.SINGLE,
});
```

### On-premises input networking

For inputs that live in an on-premises network, set `inputNetworkLocation` to
`InputNetworkLocation.ON_PREMISES`. On-premises inputs do not use input security groups. Push
inputs (RTMP/RTP/UDP) can pin their destination to a `Network`, declare the `networkRoutes` to
reach it on the local network, and request a `staticIpAddress`:

```ts
declare const stack: Stack;

const network = new medialive.Network(stack, 'Network', {
  networkName: 'on-prem-network',
  ipPools: ['192.168.1.0/24'],
});

new medialive.Input(stack, 'OnPremInput', {
  inputName: 'on-prem-rtp',
  inputNetworkLocation: medialive.InputNetworkLocation.ON_PREMISES,
  input: medialive.InputConfiguration.rtpPush({
    destinations: [{
      network,
      networkRoutes: [{ cidr: '10.0.0.0/24', gateway: '10.0.0.1' }],
      staticIpAddress: '192.168.1.50',
    }],
  }),
});
```

SRT listener inputs accept a `streamId` that the upstream system uses when connecting:

```ts
declare const stack: Stack;
declare const sg: medialive.IInputSecurityGroup;

new medialive.Input(stack, 'SrtListener', {
  inputName: 'srt-listener',
  input: medialive.InputConfiguration.srtListener({
    inputSecurityGroups: [sg],
    streamId: 'my-stream-id',
  }),
});
```
