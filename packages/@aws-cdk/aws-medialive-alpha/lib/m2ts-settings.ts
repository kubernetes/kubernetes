import type { Bitrate, Duration } from 'aws-cdk-lib';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';

/** The output bitrate mode of the transport stream. */
export class M2tsRateMode {
  /** Constant bitrate — inserts null packets to fill the configured bitrate. */
  public static readonly CBR = new M2tsRateMode('CBR');
  /** Variable bitrate — the configured bitrate acts as the maximum. */
  public static readonly VBR = new M2tsRateMode('VBR');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsRateMode {
    return new M2tsRateMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** The buffer model used for the transport stream. */
export class M2tsBufferModel {
  /** Uses the multiplex buffer model for accurate interleaving. */
  public static readonly MULTIPLEX = new M2tsBufferModel('MULTIPLEX');
  /** Can lead to lower latency, but low-memory devices might not be able to play back the stream without interruptions. */
  public static readonly NONE = new M2tsBufferModel('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsBufferModel {
    return new M2tsBufferModel(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** The buffer model used for Dolby Digital audio. */
export class M2tsAudioBufferModel {
  /** ATSC buffer model. */
  public static readonly ATSC = new M2tsAudioBufferModel('ATSC');
  /** DVB buffer model. */
  public static readonly DVB = new M2tsAudioBufferModel('DVB');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsAudioBufferModel {
    return new M2tsAudioBufferModel(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** The stream type used for audio elementary streams. */
export class M2tsAudioStreamType {
  /** ATSC — stream type 0x81 for AC3, 0x87 for EAC3. */
  public static readonly ATSC = new M2tsAudioStreamType('ATSC');
  /** DVB — stream type 0x06. */
  public static readonly DVB = new M2tsAudioStreamType('DVB');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsAudioStreamType {
    return new M2tsAudioStreamType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Controls insertion of the Program Clock Reference (PCR). */
export class M2tsPcrControl {
  /** Insert PCR at the configured `pcrPeriod`. */
  public static readonly CONFIGURED_PCR_PERIOD = new M2tsPcrControl('CONFIGURED_PCR_PERIOD');
  /** Insert a PCR for every Packetized Elementary Stream (PES) header. */
  public static readonly PCR_EVERY_PES_PACKET = new M2tsPcrControl('PCR_EVERY_PES_PACKET');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsPcrControl {
    return new M2tsPcrControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether to include the ES Rate field in the PES header. */
export class M2tsEsRateInPes {
  /** Exclude the ES Rate field. */
  public static readonly EXCLUDE = new M2tsEsRateInPes('EXCLUDE');
  /** Include the ES Rate field. */
  public static readonly INCLUDE = new M2tsEsRateInPes('INCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsEsRateInPes {
    return new M2tsEsRateInPes(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** ARIB-compliant field muxing. */
export class M2tsArib {
  /** Disabled. */
  public static readonly DISABLED = new M2tsArib('DISABLED');
  /** Enabled — uses ARIB-compliant field muxing and removes the video descriptor. */
  public static readonly ENABLED = new M2tsArib('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsArib {
    return new M2tsArib(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** How the ARIB Captions PID is selected. */
export class M2tsAribCaptionsPidControl {
  /** Auto-select the PID from unused PIDs. */
  public static readonly AUTO = new M2tsAribCaptionsPidControl('AUTO');
  /** Use the configured `aribCaptionsPid`. */
  public static readonly USE_CONFIGURED = new M2tsAribCaptionsPidControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsAribCaptionsPidControl {
    return new M2tsAribCaptionsPidControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** KLV data passthrough behavior. */
export class M2tsKlv {
  /** Do not pass KLV data through. */
  public static readonly NONE = new M2tsKlv('NONE');
  /** Pass KLV data from the input through to the output. */
  public static readonly PASSTHROUGH = new M2tsKlv('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsKlv {
    return new M2tsKlv(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** EBIF data passthrough behavior. */
export class M2tsEbif {
  /** Do not pass EBIF data through. */
  public static readonly NONE = new M2tsEbif('NONE');
  /** Pass EBIF data from the input through to the output. */
  public static readonly PASSTHROUGH = new M2tsEbif('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsEbif {
    return new M2tsEbif(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Nielsen ID3 passthrough behavior. */
export class M2tsNielsenId3Behavior {
  /** Do not insert Nielsen ID3 tags. */
  public static readonly NO_PASSTHROUGH = new M2tsNielsenId3Behavior('NO_PASSTHROUGH');
  /** Nielsen inaudible tones for media tracking will be detected in the input audio and an equivalent ID3 tag will be inserted in the output. */
  public static readonly PASSTHROUGH = new M2tsNielsenId3Behavior('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsNielsenId3Behavior {
    return new M2tsNielsenId3Behavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether to generate the captionServiceDescriptor in the PMT. */
export class M2tsCcDescriptor {
  /** Disabled. */
  public static readonly DISABLED = new M2tsCcDescriptor('DISABLED');
  /** Enabled. */
  public static readonly ENABLED = new M2tsCcDescriptor('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsCcDescriptor {
    return new M2tsCcDescriptor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Behavior when the selected input audio stream is removed from the input. */
export class M2tsAbsentInputAudioBehavior {
  /** Remove the output audio streams from the program. */
  public static readonly DROP = new M2tsAbsentInputAudioBehavior('DROP');
  /** Output encoded silence when not connected to an active input stream. */
  public static readonly ENCODE_SILENCE = new M2tsAbsentInputAudioBehavior('ENCODE_SILENCE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsAbsentInputAudioBehavior {
    return new M2tsAbsentInputAudioBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Controls placement of audio Encoder Boundary Point (EBP) markers. */
export class M2tsEbpAudioInterval {
  /** Add audio EBP markers to partitions 3 and 4 at a fixed interval. */
  public static readonly VIDEO_AND_FIXED_INTERVALS = new M2tsEbpAudioInterval('VIDEO_AND_FIXED_INTERVALS');
  /** Follow the video EBP interval. */
  public static readonly VIDEO_INTERVAL = new M2tsEbpAudioInterval('VIDEO_INTERVAL');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsEbpAudioInterval {
    return new M2tsEbpAudioInterval(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Controls placement of EBP markers on audio PIDs. */
export class M2tsEbpPlacement {
  /** Place EBP markers on the video PID and all audio PIDs. */
  public static readonly VIDEO_AND_AUDIO_PIDS = new M2tsEbpPlacement('VIDEO_AND_AUDIO_PIDS');
  /** Place EBP markers only on the video PID. */
  public static readonly VIDEO_PID = new M2tsEbpPlacement('VIDEO_PID');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsEbpPlacement {
    return new M2tsEbpPlacement(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** SCTE-35 passthrough behavior. */
export class M2tsScte35Control {
  /** Do not pass SCTE-35 signals through. */
  public static readonly NONE = new M2tsScte35Control('NONE');
  /** Pass SCTE-35 signals from the input through to the output. */
  public static readonly PASSTHROUGH = new M2tsScte35Control('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsScte35Control {
    return new M2tsScte35Control(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** The type of segmentation markers to insert. */
export class M2tsSegmentationMarkers {
  /** No segmentation markers. */
  public static readonly NONE = new M2tsSegmentationMarkers('NONE');
  /** Set the Random Access Indicator (RAI) bit in the adaptation field. */
  public static readonly RAI_SEGSTART = new M2tsSegmentationMarkers('RAI_SEGSTART');
  /** Set the RAI bit and add the current timecode in the private data bytes. */
  public static readonly RAI_ADAPT = new M2tsSegmentationMarkers('RAI_ADAPT');
  /** Insert PAT and PMT tables at the start of segments. */
  public static readonly PSI_SEGSTART = new M2tsSegmentationMarkers('PSI_SEGSTART');
  /** Add Encoder Boundary Point information (OC-SP-EBP-I01-130118). */
  public static readonly EBP = new M2tsSegmentationMarkers('EBP');
  /** Add Encoder Boundary Point information using the legacy proprietary format. */
  public static readonly EBP_LEGACY = new M2tsSegmentationMarkers('EBP_LEGACY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsSegmentationMarkers {
    return new M2tsSegmentationMarkers(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** How segmentation markers respond to avails truncating a segment. */
export class M2tsSegmentationStyle {
  /** Do not reset the segmentation cadence after a truncated segment. */
  public static readonly MAINTAIN_CADENCE = new M2tsSegmentationStyle('MAINTAIN_CADENCE');
  /** Reset the segmentation cadence after a truncated segment. */
  public static readonly RESET_CADENCE = new M2tsSegmentationStyle('RESET_CADENCE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsSegmentationStyle {
    return new M2tsSegmentationStyle(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Timed metadata passthrough behavior. */
export class M2tsTimedMetadataBehavior {
  /** Do not pass timed metadata through. */
  public static readonly NO_PASSTHROUGH = new M2tsTimedMetadataBehavior('NO_PASSTHROUGH');
  /** Pass timed metadata from the input through to the output. */
  public static readonly PASSTHROUGH = new M2tsTimedMetadataBehavior('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): M2tsTimedMetadataBehavior {
    return new M2tsTimedMetadataBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** How DVB Service Description Table (SDT) information is inserted. */
export class DvbSdtOutputMode {
  /** Copy SDT information from the input stream to the output stream. */
  public static readonly SDT_FOLLOW = new DvbSdtOutputMode('SDT_FOLLOW');
  /** Copy SDT from the input if present, otherwise use the configured values. */
  public static readonly SDT_FOLLOW_IF_PRESENT = new DvbSdtOutputMode('SDT_FOLLOW_IF_PRESENT');
  /** Use the user-defined SDT information. */
  public static readonly SDT_MANUAL = new DvbSdtOutputMode('SDT_MANUAL');
  /** Do not include SDT information in the output. */
  public static readonly SDT_NONE = new DvbSdtOutputMode('SDT_NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): DvbSdtOutputMode {
    return new DvbSdtOutputMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Settings for inserting a DVB Network Information Table (NIT). */
export interface DvbNitSettings {
  /**
   * The numeric value placed in the Network Information Table (NIT).
   * @default - no network ID
   */
  readonly networkId?: number;
  /**
   * The network name placed in the networkNameDescriptor inside the NIT (max 256 characters).
   * @default - no network name
   */
  readonly networkName?: string;
  /**
   * The interval between instances of this table in the output transport stream.
   * @default - service default
   */
  readonly repInterval?: Duration;
}

/** Settings for inserting a DVB Service Description Table (SDT). */
export interface DvbSdtSettings {
  /**
   * The method of inserting SDT information into the output stream.
   * @default - service default
   */
  readonly outputSdt?: DvbSdtOutputMode;
  /**
   * The interval between instances of this table in the output transport stream.
   * @default - service default
   */
  readonly repInterval?: Duration;
  /**
   * The service name placed in the serviceDescriptor in the SDT (max 256 characters).
   * @default - no service name
   */
  readonly serviceName?: string;
  /**
   * The service provider name placed in the serviceDescriptor in the SDT (max 256 characters).
   * @default - no service provider name
   */
  readonly serviceProviderName?: string;
}

/** Settings for inserting a DVB Time and Date Table (TDT). */
export interface DvbTdtSettings {
  /**
   * The interval between instances of this table in the output transport stream.
   * @default - service default
   */
  readonly repInterval?: Duration;
}

/**
 * Properties for MPEG-2 transport stream (M2TS) container settings.
 *
 * Used by the UDP, Archive, SRT, and MediaConnect Router output groups. All properties are
 * optional; omit them to use MediaLive's service defaults.
 *
 * PID properties accept a decimal or hexadecimal value (and, where noted, ranges or comma-separated
 * lists). Each PID must be in the range 32 (0x20)..8182 (0x1ff6).
 *
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-medialive-channel-m2tssettings.html
 */
export interface M2tsSettingsProps {
  /** Behavior when the selected input audio stream is removed. @default - service default */
  readonly absentInputAudioBehavior?: M2tsAbsentInputAudioBehavior;
  /** ARIB-compliant field muxing. @default - service default */
  readonly arib?: M2tsArib;
  /** The PID for ARIB Captions. @default - service default */
  readonly aribCaptionsPid?: string;
  /** How the ARIB Captions PID is selected. @default - service default */
  readonly aribCaptionsPidControl?: M2tsAribCaptionsPidControl;
  /** The buffer model for Dolby Digital audio. @default - service default */
  readonly audioBufferModel?: M2tsAudioBufferModel;
  /** The number of audio frames to insert per PES packet. @default - service default */
  readonly audioFramesPerPes?: number;
  /** The PID(s) of the elementary audio streams (ranges/comma-separated allowed). @default - service default */
  readonly audioPids?: string;
  /** The stream type used for audio elementary streams. @default - service default */
  readonly audioStreamType?: M2tsAudioStreamType;
  /** The output bitrate of the transport stream. Set to 0 bps to let the muxer choose. @default - muxer chooses */
  readonly bitrate?: Bitrate;
  /** The transport stream buffer model. @default - service default */
  readonly bufferModel?: M2tsBufferModel;
  /** Whether to generate the captionServiceDescriptor in the PMT. @default - service default */
  readonly ccDescriptor?: M2tsCcDescriptor;
  /** DVB Network Information Table (NIT) settings. @default - no NIT */
  readonly dvbNitSettings?: DvbNitSettings;
  /** DVB Service Description Table (SDT) settings. @default - no SDT */
  readonly dvbSdtSettings?: DvbSdtSettings;
  /** The PID(s) for input source DVB Subtitle data (ranges/comma-separated allowed). @default - service default */
  readonly dvbSubPids?: string;
  /** DVB Time and Date Table (TDT) settings. @default - no TDT */
  readonly dvbTdtSettings?: DvbTdtSettings;
  /** The PID for input source DVB Teletext data. @default - service default */
  readonly dvbTeletextPid?: string;
  /** EBIF data passthrough behavior. @default - service default */
  readonly ebif?: M2tsEbif;
  /** Placement of audio EBP markers. @default - service default */
  readonly ebpAudioInterval?: M2tsEbpAudioInterval;
  /** The EBP lookahead interval. @default - service default */
  readonly ebpLookahead?: Duration;
  /** Placement of EBP markers on audio PIDs. @default - service default */
  readonly ebpPlacement?: M2tsEbpPlacement;
  /** Whether to include the ES Rate field in the PES header. @default - service default */
  readonly esRateInPes?: M2tsEsRateInPes;
  /** The PID for input source ETV Platform data. @default - service default */
  readonly etvPlatformPid?: string;
  /** The PID for input source ETV Signal data. @default - service default */
  readonly etvSignalPid?: string;
  /** The length of each fragment (used only with EBP markers). @default - service default */
  readonly fragmentTime?: Duration;
  /** KLV data passthrough behavior. @default - service default */
  readonly klv?: M2tsKlv;
  /** The PID(s) for input source KLV data (ranges/comma-separated allowed). @default - service default */
  readonly klvDataPids?: string;
  /** Nielsen ID3 passthrough behavior. @default - service default */
  readonly nielsenId3Behavior?: M2tsNielsenId3Behavior;
  /** The bitrate of extra null packets to insert into the transport stream. @default - no null packets */
  readonly nullPacketBitrate?: Bitrate;
  /** The interval between PAT instances (0, or 10ms..1000ms). @default - service default */
  readonly patInterval?: Duration;
  /** Controls insertion of the Program Clock Reference (PCR). @default - service default */
  readonly pcrControl?: M2tsPcrControl;
  /** The maximum interval between Program Clock References (PCRs). @default - service default */
  readonly pcrPeriod?: Duration;
  /** The PID of the Program Clock Reference. @default - same as the video PID */
  readonly pcrPid?: string;
  /** The interval between PMT instances (0, or 10ms..1000ms). @default - service default */
  readonly pmtInterval?: Duration;
  /** The PID for the Program Map Table (PMT). @default - service default */
  readonly pmtPid?: string;
  /** The value of the program number field in the PMT. @default - service default */
  readonly programNum?: number;
  /** The transport stream bitrate mode (CBR/VBR). @default - service default */
  readonly rateMode?: M2tsRateMode;
  /** The PID(s) for input source SCTE-27 data (ranges/comma-separated allowed). @default - service default */
  readonly scte27Pids?: string;
  /** SCTE-35 passthrough behavior. @default - service default */
  readonly scte35Control?: M2tsScte35Control;
  /** The PID of the SCTE-35 stream. @default - service default */
  readonly scte35Pid?: string;
  /** The SCTE-35 preroll pullup interval. @default - service default */
  readonly scte35PrerollPullup?: Duration;
  /** The type of segmentation markers to insert. @default - service default */
  readonly segmentationMarkers?: M2tsSegmentationMarkers;
  /** How segmentation markers respond to avails. @default - service default */
  readonly segmentationStyle?: M2tsSegmentationStyle;
  /** The length of each segment (required unless `segmentationMarkers` is NONE). @default - service default */
  readonly segmentationTime?: Duration;
  /** Timed metadata passthrough behavior. @default - service default */
  readonly timedMetadataBehavior?: M2tsTimedMetadataBehavior;
  /** The PID of the timed metadata stream. @default - service default */
  readonly timedMetadataPid?: string;
  /** The value of the transport stream ID field in the PMT. @default - service default */
  readonly transportStreamId?: number;
  /** The PID of the elementary video stream. @default - service default */
  readonly videoPid?: string;
}

/**
 * MPEG-2 transport stream (M2TS) container settings for an MPEG-TS output.
 *
 * Use `M2tsSettings.of()` to configure the transport stream produced by a UDP, Archive, SRT, or
 * MediaConnect Router output. Omitting it entirely uses MediaLive's service defaults.
 */
export class M2tsSettings {
  /** Create M2TS container settings. */
  public static of(props: M2tsSettingsProps): M2tsSettings {
    return new M2tsSettings(props);
  }

  private constructor(private readonly props: M2tsSettingsProps) {}

  /** @internal */
  public _bind(): CfnChannel.M2tsSettingsProperty {
    const p = this.props;
    return {
      absentInputAudioBehavior: p.absentInputAudioBehavior?.value,
      arib: p.arib?.value,
      aribCaptionsPid: p.aribCaptionsPid,
      aribCaptionsPidControl: p.aribCaptionsPidControl?.value,
      audioBufferModel: p.audioBufferModel?.value,
      audioFramesPerPes: p.audioFramesPerPes,
      audioPids: p.audioPids,
      audioStreamType: p.audioStreamType?.value,
      bitrate: p.bitrate?.toBps(),
      bufferModel: p.bufferModel?.value,
      ccDescriptor: p.ccDescriptor?.value,
      dvbNitSettings: p.dvbNitSettings ? {
        networkId: p.dvbNitSettings.networkId,
        networkName: p.dvbNitSettings.networkName,
        repInterval: p.dvbNitSettings.repInterval?.toMilliseconds(),
      } : undefined,
      dvbSdtSettings: p.dvbSdtSettings ? {
        outputSdt: p.dvbSdtSettings.outputSdt?.value,
        repInterval: p.dvbSdtSettings.repInterval?.toMilliseconds(),
        serviceName: p.dvbSdtSettings.serviceName,
        serviceProviderName: p.dvbSdtSettings.serviceProviderName,
      } : undefined,
      dvbSubPids: p.dvbSubPids,
      dvbTdtSettings: p.dvbTdtSettings ? {
        repInterval: p.dvbTdtSettings.repInterval?.toMilliseconds(),
      } : undefined,
      dvbTeletextPid: p.dvbTeletextPid,
      ebif: p.ebif?.value,
      ebpAudioInterval: p.ebpAudioInterval?.value,
      ebpLookaheadMs: p.ebpLookahead?.toMilliseconds(),
      ebpPlacement: p.ebpPlacement?.value,
      esRateInPes: p.esRateInPes?.value,
      etvPlatformPid: p.etvPlatformPid,
      etvSignalPid: p.etvSignalPid,
      fragmentTime: p.fragmentTime?.toSeconds(),
      klv: p.klv?.value,
      klvDataPids: p.klvDataPids,
      nielsenId3Behavior: p.nielsenId3Behavior?.value,
      nullPacketBitrate: p.nullPacketBitrate?.toBps(),
      patInterval: p.patInterval?.toMilliseconds(),
      pcrControl: p.pcrControl?.value,
      pcrPeriod: p.pcrPeriod?.toMilliseconds(),
      pcrPid: p.pcrPid,
      pmtInterval: p.pmtInterval?.toMilliseconds(),
      pmtPid: p.pmtPid,
      programNum: p.programNum,
      rateMode: p.rateMode?.value,
      scte27Pids: p.scte27Pids,
      scte35Control: p.scte35Control?.value,
      scte35Pid: p.scte35Pid,
      scte35PrerollPullupMilliseconds: p.scte35PrerollPullup?.toMilliseconds(),
      segmentationMarkers: p.segmentationMarkers?.value,
      segmentationStyle: p.segmentationStyle?.value,
      segmentationTime: p.segmentationTime?.toSeconds(),
      timedMetadataBehavior: p.timedMetadataBehavior?.value,
      timedMetadataPid: p.timedMetadataPid,
      transportStreamId: p.transportStreamId,
      videoPid: p.videoPid,
    };
  }
}
