/** HLS output mode. */
export class HlsMode {
  /** Live mode — older segments are removed */
  public static readonly LIVE = new HlsMode('LIVE');
  /** VOD mode — all segments are kept */
  public static readonly VOD = new HlsMode('VOD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsMode {
    return new HlsMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS input loss action. */
export class HlsInputLossAction {
  /** Emit output with slate/black frames */
  public static readonly EMIT_OUTPUT = new HlsInputLossAction('EMIT_OUTPUT');
  /** Pause the output */
  public static readonly PAUSE_OUTPUT = new HlsInputLossAction('PAUSE_OUTPUT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsInputLossAction {
    return new HlsInputLossAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS client cache control. */
export class HlsClientCache {
  /** Enable client caching */
  public static readonly ENABLED = new HlsClientCache('ENABLED');
  /** Disable client caching */
  public static readonly DISABLED = new HlsClientCache('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsClientCache {
    return new HlsClientCache(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS codec specification. */
export class HlsCodecSpecification {
  /** RFC 4281 */
  public static readonly RFC_4281 = new HlsCodecSpecification('RFC_4281');
  /** RFC 6381 */
  public static readonly RFC_6381 = new HlsCodecSpecification('RFC_6381');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsCodecSpecification {
    return new HlsCodecSpecification(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS directory structure. */
export class HlsDirectoryStructure {
  /** Single directory */
  public static readonly SINGLE_DIRECTORY = new HlsDirectoryStructure('SINGLE_DIRECTORY');
  /** Subdirectory per stream */
  public static readonly SUBDIRECTORY_PER_STREAM = new HlsDirectoryStructure('SUBDIRECTORY_PER_STREAM');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsDirectoryStructure {
    return new HlsDirectoryStructure(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS discontinuity tags. */
export class HlsDiscontinuityTags {
  /** Insert discontinuity tags */
  public static readonly INSERT = new HlsDiscontinuityTags('INSERT');
  /** Never insert discontinuity tags */
  public static readonly NEVER_INSERT = new HlsDiscontinuityTags('NEVER_INSERT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsDiscontinuityTags {
    return new HlsDiscontinuityTags(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS encryption type. */
export class HlsEncryptionType {
  /** AES-128 encryption */
  public static readonly AES128 = new HlsEncryptionType('AES128');
  /** Sample AES encryption */
  public static readonly SAMPLE_AES = new HlsEncryptionType('SAMPLE_AES');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsEncryptionType {
    return new HlsEncryptionType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS ID3 segment tagging state. */
export class HlsId3SegmentTaggingState {
  /** Disabled */
  public static readonly DISABLED = new HlsId3SegmentTaggingState('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new HlsId3SegmentTaggingState('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsId3SegmentTaggingState {
    return new HlsId3SegmentTaggingState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS I-frame only playlists. */
export class HlsIFrameOnlyPlaylists {
  /** Disabled */
  public static readonly DISABLED = new HlsIFrameOnlyPlaylists('DISABLED');
  /** Standard */
  public static readonly STANDARD = new HlsIFrameOnlyPlaylists('STANDARD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsIFrameOnlyPlaylists {
    return new HlsIFrameOnlyPlaylists(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS incomplete segment behavior. */
export class HlsIncompleteSegmentBehavior {
  /** Auto */
  public static readonly AUTO = new HlsIncompleteSegmentBehavior('AUTO');
  /** Suppress */
  public static readonly SUPPRESS = new HlsIncompleteSegmentBehavior('SUPPRESS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsIncompleteSegmentBehavior {
    return new HlsIncompleteSegmentBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS IV in manifest. */
export class HlsIvInManifest {
  /** Include IV in manifest */
  public static readonly INCLUDE = new HlsIvInManifest('INCLUDE');
  /** Exclude IV from manifest */
  public static readonly EXCLUDE = new HlsIvInManifest('EXCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsIvInManifest {
    return new HlsIvInManifest(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS IV source. */
export class HlsIvSource {
  /** IV follows segment number */
  public static readonly FOLLOWS_SEGMENT_NUMBER = new HlsIvSource('FOLLOWS_SEGMENT_NUMBER');
  /** Explicit IV */
  public static readonly EXPLICIT = new HlsIvSource('EXPLICIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsIvSource {
    return new HlsIvSource(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS manifest compression. */
export class HlsManifestCompression {
  /** No compression */
  public static readonly NONE = new HlsManifestCompression('NONE');
  /** Gzip compression */
  public static readonly GZIP = new HlsManifestCompression('GZIP');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsManifestCompression {
    return new HlsManifestCompression(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS manifest duration format. */
export class HlsManifestDurationFormat {
  /** Floating point */
  public static readonly FLOATING_POINT = new HlsManifestDurationFormat('FLOATING_POINT');
  /** Integer */
  public static readonly INTEGER = new HlsManifestDurationFormat('INTEGER');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsManifestDurationFormat {
    return new HlsManifestDurationFormat(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS output selection. */
export class HlsOutputSelection {
  /** Manifests and segments */
  public static readonly MANIFESTS_AND_SEGMENTS = new HlsOutputSelection('MANIFESTS_AND_SEGMENTS');
  /** Segments only */
  public static readonly SEGMENTS_ONLY = new HlsOutputSelection('SEGMENTS_ONLY');
  /** Variant manifests and segments */
  public static readonly VARIANT_MANIFESTS_AND_SEGMENTS = new HlsOutputSelection('VARIANT_MANIFESTS_AND_SEGMENTS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsOutputSelection {
    return new HlsOutputSelection(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS program date time. */
export class HlsProgramDateTime {
  /** Include */
  public static readonly INCLUDE = new HlsProgramDateTime('INCLUDE');
  /** Exclude */
  public static readonly EXCLUDE = new HlsProgramDateTime('EXCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsProgramDateTime {
    return new HlsProgramDateTime(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS program date time clock. */
export class HlsProgramDateTimeClock {
  /** Initialize from output timecode */
  public static readonly INITIALIZE_FROM_OUTPUT_TIMECODE = new HlsProgramDateTimeClock('INITIALIZE_FROM_OUTPUT_TIMECODE');
  /** System clock */
  public static readonly SYSTEM_CLOCK = new HlsProgramDateTimeClock('SYSTEM_CLOCK');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsProgramDateTimeClock {
    return new HlsProgramDateTimeClock(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS redundant manifest. */
export class HlsRedundantManifest {
  /** Disabled */
  public static readonly DISABLED = new HlsRedundantManifest('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new HlsRedundantManifest('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsRedundantManifest {
    return new HlsRedundantManifest(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS segmentation mode. */
export class HlsSegmentationMode {
  /** Use input segmentation */
  public static readonly USE_INPUT_SEGMENTATION = new HlsSegmentationMode('USE_INPUT_SEGMENTATION');
  /** Use segment duration */
  public static readonly USE_SEGMENT_DURATION = new HlsSegmentationMode('USE_SEGMENT_DURATION');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsSegmentationMode {
    return new HlsSegmentationMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS stream inf resolution. */
export class HlsStreamInfResolution {
  /** Include */
  public static readonly INCLUDE = new HlsStreamInfResolution('INCLUDE');
  /** Exclude */
  public static readonly EXCLUDE = new HlsStreamInfResolution('EXCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsStreamInfResolution {
    return new HlsStreamInfResolution(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS caption language setting. */
export class HlsCaptionLanguageSetting {
  /** Insert */
  public static readonly INSERT = new HlsCaptionLanguageSetting('INSERT');
  /** None */
  public static readonly NONE = new HlsCaptionLanguageSetting('NONE');
  /** Omit */
  public static readonly OMIT = new HlsCaptionLanguageSetting('OMIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsCaptionLanguageSetting {
    return new HlsCaptionLanguageSetting(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Whether MediaPackage sets a MediaPackage V2 audio rendition as default / auto-select in the HLS
 * manifest. Across all renditions: at most one may be `YES`; not all may be `NO`.
 */
export class MediaPackageV2HlsSetting {
  /** Set this rendition as default / auto-select. */
  public static readonly YES = new MediaPackageV2HlsSetting('YES');
  /** Do not set this rendition as default / auto-select. */
  public static readonly NO = new MediaPackageV2HlsSetting('NO');
  /** Let MediaPackage decide for this rendition. */
  public static readonly OMIT = new MediaPackageV2HlsSetting('OMIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MediaPackageV2HlsSetting {
    return new MediaPackageV2HlsSetting(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Ad marker type for an HLS output group. */
export class HlsAdMarkers {
  /** Adobe ad markers. */
  public static readonly ADOBE = new HlsAdMarkers('ADOBE');
  /** Elemental ad markers. */
  public static readonly ELEMENTAL = new HlsAdMarkers('ELEMENTAL');
  /** Elemental SCTE-35 ad markers. */
  public static readonly ELEMENTAL_SCTE35 = new HlsAdMarkers('ELEMENTAL_SCTE35');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsAdMarkers {
    return new HlsAdMarkers(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Ad marker type for an RTMP output group. */
export class RtmpAdMarkers {
  /** onCuePoint SCTE-35 ad markers. */
  public static readonly ON_CUE_POINT_SCTE35 = new RtmpAdMarkers('ON_CUE_POINT_SCTE35');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpAdMarkers {
    return new RtmpAdMarkers(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS timed metadata ID3 frame. */
export class HlsTimedMetadataId3Frame {
  /** None */
  public static readonly NONE = new HlsTimedMetadataId3Frame('NONE');
  /** PRIV */
  public static readonly PRIV = new HlsTimedMetadataId3Frame('PRIV');
  /** TDRL */
  public static readonly TDRL = new HlsTimedMetadataId3Frame('TDRL');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsTimedMetadataId3Frame {
    return new HlsTimedMetadataId3Frame(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether to use chunked transfer encoding for an HLS CDN connection (Akamai, WebDAV). */
export class HttpTransferMode {
  /** Use chunked transfer encoding. */
  public static readonly CHUNKED = new HttpTransferMode('CHUNKED');
  /** Do not use chunked transfer encoding. */
  public static readonly NON_CHUNKED = new HttpTransferMode('NON_CHUNKED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HttpTransferMode {
    return new HttpTransferMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** HLS TS file mode. */
export class HlsTsFileMode {
  /** Segmented files */
  public static readonly SEGMENTED_FILES = new HlsTsFileMode('SEGMENTED_FILES');
  /** Single file */
  public static readonly SINGLE_FILE = new HlsTsFileMode('SINGLE_FILE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): HlsTsFileMode {
    return new HlsTsFileMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP authentication scheme. */
export class RtmpAuthenticationScheme {
  /** Common authentication */
  public static readonly COMMON = new RtmpAuthenticationScheme('COMMON');
  /** Akamai authentication */
  public static readonly AKAMAI = new RtmpAuthenticationScheme('AKAMAI');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpAuthenticationScheme {
    return new RtmpAuthenticationScheme(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP cache full behavior. */
export class RtmpCacheFullBehavior {
  /** Disconnect immediately */
  public static readonly DISCONNECT_IMMEDIATELY = new RtmpCacheFullBehavior('DISCONNECT_IMMEDIATELY');
  /** Wait for server */
  public static readonly WAIT_FOR_SERVER = new RtmpCacheFullBehavior('WAIT_FOR_SERVER');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpCacheFullBehavior {
    return new RtmpCacheFullBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP caption data. */
export class RtmpCaptionData {
  /** All */
  public static readonly ALL = new RtmpCaptionData('ALL');
  /** Field 1 and field 2 608 */
  public static readonly FIELD1_AND_FIELD2_608 = new RtmpCaptionData('FIELD1_AND_FIELD2_608');
  /** Field 1 608 */
  public static readonly FIELD1_608 = new RtmpCaptionData('FIELD1_608');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpCaptionData {
    return new RtmpCaptionData(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP input loss action. */
export class RtmpInputLossAction {
  /** Emit output */
  public static readonly EMIT_OUTPUT = new RtmpInputLossAction('EMIT_OUTPUT');
  /** Pause output */
  public static readonly PAUSE_OUTPUT = new RtmpInputLossAction('PAUSE_OUTPUT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpInputLossAction {
    return new RtmpInputLossAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP include filler NAL units. */
export class RtmpIncludeFillerNalUnits {
  /** Auto */
  public static readonly AUTO = new RtmpIncludeFillerNalUnits('AUTO');
  /** Drop */
  public static readonly DROP = new RtmpIncludeFillerNalUnits('DROP');
  /** Include */
  public static readonly INCLUDE = new RtmpIncludeFillerNalUnits('INCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpIncludeFillerNalUnits {
    return new RtmpIncludeFillerNalUnits(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** RTMP TLS certificate verification mode. */
export class RtmpCertificateMode {
  /** Verify the TLS certificate chain */
  public static readonly VERIFY_AUTHENTICITY = new RtmpCertificateMode('VERIFY_AUTHENTICITY');
  /** Do not verify the TLS certificate */
  public static readonly SELF_SIGNED = new RtmpCertificateMode('SELF_SIGNED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RtmpCertificateMode {
    return new RtmpCertificateMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Behavior of last resort when input video is lost and no more backup inputs are available,
 * for an SRT output group.
 */
export class SrtInputLossAction {
  /** Drop the entire transport stream. */
  public static readonly DROP_TS = new SrtInputLossAction('DROP_TS');
  /** Drop the program from the transport stream (replaced with null packets to meet bitrate). */
  public static readonly DROP_PROGRAM = new SrtInputLossAction('DROP_PROGRAM');
  /** Continue emitting with repeat, black, or slate frames substituted for the absent video. */
  public static readonly EMIT_PROGRAM = new SrtInputLossAction('EMIT_PROGRAM');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SrtInputLossAction {
    return new SrtInputLossAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** SRT output encryption type. */
export class SrtEncryptionType {
  /** AES-128 encryption */
  public static readonly AES128 = new SrtEncryptionType('AES128');
  /** AES-192 encryption */
  public static readonly AES192 = new SrtEncryptionType('AES192');
  /** AES-256 encryption */
  public static readonly AES256 = new SrtEncryptionType('AES256');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SrtEncryptionType {
    return new SrtEncryptionType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** UDP input loss action. */
export class UdpInputLossAction {
  /** Drop the entire transport stream */
  public static readonly DROP_TS = new UdpInputLossAction('DROP_TS');
  /** Drop the program from the transport stream */
  public static readonly DROP_PROGRAM = new UdpInputLossAction('DROP_PROGRAM');
  /** Continue emitting with substitute frames */
  public static readonly EMIT_PROGRAM = new UdpInputLossAction('EMIT_PROGRAM');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): UdpInputLossAction {
    return new UdpInputLossAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Enables column-only or column-and-row FEC for a UDP output. */
export class FecMode {
  /** Column-only FEC. */
  public static readonly COLUMN = new FecMode('COLUMN');
  /** Column-and-row FEC (more robust). */
  public static readonly COLUMN_AND_ROW = new FecMode('COLUMN_AND_ROW');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): FecMode {
    return new FecMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** UDP timed metadata ID3 frame. */
export class UdpTimedMetadataId3Frame {
  /** None */
  public static readonly NONE = new UdpTimedMetadataId3Frame('NONE');
  /** PRIV */
  public static readonly PRIV = new UdpTimedMetadataId3Frame('PRIV');
  /** TDRL */
  public static readonly TDRL = new UdpTimedMetadataId3Frame('TDRL');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): UdpTimedMetadataId3Frame {
    return new UdpTimedMetadataId3Frame(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth audio-only timecode control. */
export class MsSmoothAudioOnlyTimecodeControl {
  /** Passthrough */
  public static readonly PASSTHROUGH = new MsSmoothAudioOnlyTimecodeControl('PASSTHROUGH');
  /** Use configured clock */
  public static readonly USE_CONFIGURED_CLOCK = new MsSmoothAudioOnlyTimecodeControl('USE_CONFIGURED_CLOCK');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothAudioOnlyTimecodeControl {
    return new MsSmoothAudioOnlyTimecodeControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth certificate mode. */
export class MsSmoothCertificateMode {
  /** Self-signed */
  public static readonly SELF_SIGNED = new MsSmoothCertificateMode('SELF_SIGNED');
  /** Verify authenticity */
  public static readonly VERIFY_AUTHENTICITY = new MsSmoothCertificateMode('VERIFY_AUTHENTICITY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothCertificateMode {
    return new MsSmoothCertificateMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth event ID mode. */
export class MsSmoothEventIdMode {
  /** No event ID */
  public static readonly NO_EVENT_ID = new MsSmoothEventIdMode('NO_EVENT_ID');
  /** Use configured */
  public static readonly USE_CONFIGURED = new MsSmoothEventIdMode('USE_CONFIGURED');
  /** Use timestamp */
  public static readonly USE_TIMESTAMP = new MsSmoothEventIdMode('USE_TIMESTAMP');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothEventIdMode {
    return new MsSmoothEventIdMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth event stop behavior. */
export class MsSmoothEventStopBehavior {
  /** None */
  public static readonly NONE = new MsSmoothEventStopBehavior('NONE');
  /** Send EOS */
  public static readonly SEND_EOS = new MsSmoothEventStopBehavior('SEND_EOS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothEventStopBehavior {
    return new MsSmoothEventStopBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth input loss action. */
export class MsSmoothInputLossAction {
  /** Emit output */
  public static readonly EMIT_OUTPUT = new MsSmoothInputLossAction('EMIT_OUTPUT');
  /** Pause output */
  public static readonly PAUSE_OUTPUT = new MsSmoothInputLossAction('PAUSE_OUTPUT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothInputLossAction {
    return new MsSmoothInputLossAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth segmentation mode. */
export class MsSmoothSegmentationMode {
  /** Use input segmentation */
  public static readonly USE_INPUT_SEGMENTATION = new MsSmoothSegmentationMode('USE_INPUT_SEGMENTATION');
  /** Use segment duration */
  public static readonly USE_SEGMENT_DURATION = new MsSmoothSegmentationMode('USE_SEGMENT_DURATION');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothSegmentationMode {
    return new MsSmoothSegmentationMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth sparse track type. */
export class MsSmoothSparseTrackType {
  /** None */
  public static readonly NONE = new MsSmoothSparseTrackType('NONE');
  /** SCTE-35 */
  public static readonly SCTE_35 = new MsSmoothSparseTrackType('SCTE_35');
  /** SCTE-35 without segmentation */
  public static readonly SCTE_35_WITHOUT_SEGMENTATION = new MsSmoothSparseTrackType('SCTE_35_WITHOUT_SEGMENTATION');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothSparseTrackType {
    return new MsSmoothSparseTrackType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth stream manifest behavior. */
export class MsSmoothStreamManifestBehavior {
  /** Do not send */
  public static readonly DO_NOT_SEND = new MsSmoothStreamManifestBehavior('DO_NOT_SEND');
  /** Send */
  public static readonly SEND = new MsSmoothStreamManifestBehavior('SEND');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothStreamManifestBehavior {
    return new MsSmoothStreamManifestBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** MS Smooth timestamp offset mode. */
export class MsSmoothTimestampOffsetMode {
  /** Use configured offset */
  public static readonly USE_CONFIGURED_OFFSET = new MsSmoothTimestampOffsetMode('USE_CONFIGURED_OFFSET');
  /** Use event start date */
  public static readonly USE_EVENT_START_DATE = new MsSmoothTimestampOffsetMode('USE_EVENT_START_DATE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MsSmoothTimestampOffsetMode {
    return new MsSmoothTimestampOffsetMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** ID3 metadata insertion behavior (CMAF Ingest and MediaPackage V2 output groups). */
export class Id3Behavior {
  /** Do not insert ID3 metadata. */
  public static readonly DISABLED = new Id3Behavior('DISABLED');
  /** Enable ID3 metadata insertion. */
  public static readonly ENABLED = new Id3Behavior('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Id3Behavior {
    return new Id3Behavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** CMAF Ingest KLV behavior. */
export class KlvBehavior {
  /** No passthrough */
  public static readonly NO_PASSTHROUGH = new KlvBehavior('NO_PASSTHROUGH');
  /** Passthrough */
  public static readonly PASSTHROUGH = new KlvBehavior('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): KlvBehavior {
    return new KlvBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** CMAF Ingest Nielsen ID3 behavior. */
export class NielsenId3Behavior {
  /** No passthrough */
  public static readonly NO_PASSTHROUGH = new NielsenId3Behavior('NO_PASSTHROUGH');
  /** Passthrough */
  public static readonly PASSTHROUGH = new NielsenId3Behavior('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NielsenId3Behavior {
    return new NielsenId3Behavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** CMAF Ingest SCTE-35 type. */
export class Scte35Type {
  /** None */
  public static readonly NONE = new Scte35Type('NONE');
  /** SCTE-35 without segmentation */
  public static readonly SCTE_35_WITHOUT_SEGMENTATION = new Scte35Type('SCTE_35_WITHOUT_SEGMENTATION');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Scte35Type {
    return new Scte35Type(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Segment length units.
 * @internal
 */
export enum SegmentLengthUnits {
  /** Milliseconds */
  MILLISECONDS = 'MILLISECONDS',
  /** Seconds */
  SECONDS = 'SECONDS',
}

/** CMAF Ingest timed metadata ID3 frame. */
export class TimedMetadataId3Frame {
  /** None */
  public static readonly NONE = new TimedMetadataId3Frame('NONE');
  /** PRIV */
  public static readonly PRIV = new TimedMetadataId3Frame('PRIV');
  /** TDRL */
  public static readonly TDRL = new TimedMetadataId3Frame('TDRL');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimedMetadataId3Frame {
    return new TimedMetadataId3Frame(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** CMAF Ingest timed metadata passthrough. */
export class TimedMetadataPassthrough {
  /** Disabled */
  public static readonly DISABLED = new TimedMetadataPassthrough('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new TimedMetadataPassthrough('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimedMetadataPassthrough {
    return new TimedMetadataPassthrough(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * S3 canned ACL for output destinations.
 */
export class S3CannedAcl {
  /** Grants the owner full control and authenticated AWS users read access. */
  public static readonly AUTHENTICATED_READ = new S3CannedAcl('AUTHENTICATED_READ');
  /** Grants the object owner and bucket owner full control. */
  public static readonly BUCKET_OWNER_FULL_CONTROL = new S3CannedAcl('BUCKET_OWNER_FULL_CONTROL');
  /** Grants the owner full control and the bucket owner read access. */
  public static readonly BUCKET_OWNER_READ = new S3CannedAcl('BUCKET_OWNER_READ');
  /** Grants the owner full control and all users read access. */
  public static readonly PUBLIC_READ = new S3CannedAcl('PUBLIC_READ');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): S3CannedAcl {
    return new S3CannedAcl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** H.265 packaging type for HLS/MS Smooth outputs. */
export class H265PackagingType {
  /** HEV1 packaging */
  public static readonly HEV1 = new H265PackagingType('HEV1');
  /** HVC1 packaging */
  public static readonly HVC1 = new H265PackagingType('HVC1');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265PackagingType {
    return new H265PackagingType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}
