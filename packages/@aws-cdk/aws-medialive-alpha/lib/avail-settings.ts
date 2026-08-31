import type { Duration } from 'aws-cdk-lib';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import type { IStringParameter } from 'aws-cdk-lib/aws-ssm';
import type { FileLocation } from './file-location';

/**
 * Avail blanking state.
 */
export class AvailBlankingState {
  /** Enable blanking during ad avails */
  public static readonly ENABLED = new AvailBlankingState('ENABLED');
  /** Disable blanking during ad avails */
  public static readonly DISABLED = new AvailBlankingState('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AvailBlankingState {
    return new AvailBlankingState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Settings for blanking video, audio, and captions during ad avails.
 */
export interface AvailBlanking {
  /**
   * Whether to blank the output during ad avails.
   * @default - ENABLED if image is provided, DISABLED otherwise
   */
  readonly state?: AvailBlankingState;
  /**
   * A blanking image to display during ad avails. Provide a `FileLocation` referencing an S3
   * bucket (`FileLocation.fromBucket`, which auto-grants read access) or a URL (`FileLocation.url`).
   * Only .bmp and .png images are supported. If not set, solid black is used.
   * @default - solid black
   */
  readonly image?: FileLocation;
}

/**
 * How to handle SCTE-35 regional blackout and web delivery flags.
 */
export class Scte35FlagBehavior {
  /** Follow the flag — trigger blackouts/slates when the flag is set */
  public static readonly FOLLOW = new Scte35FlagBehavior('FOLLOW');
  /** Ignore the flag */
  public static readonly IGNORE = new Scte35FlagBehavior('IGNORE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Scte35FlagBehavior {
    return new Scte35FlagBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * SCTE-35 splice insert avail settings.
 */
export interface Scte35SpliceInsertSettings {
  /**
   * Offset in milliseconds added to the input ad avail PTS time.
   * Applies only to embedded SCTE 104/35 messages.
   * @default - service default
   */
  readonly adAvailOffset?: number;
  /**
   * When set to `IGNORE`, segment descriptors with `noRegionalBlackoutFlag` set to 0 no longer
   * trigger blackouts or ad avail slates.
   * @default - service default
   */
  readonly noRegionalBlackoutFlag?: Scte35FlagBehavior;
  /**
   * When set to `IGNORE`, segment descriptors with `webDeliveryAllowedFlag` set to 0 no longer
   * trigger blackouts or ad avail slates.
   * @default - service default
   */
  readonly webDeliveryAllowedFlag?: Scte35FlagBehavior;
}

/**
 * SCTE-35 time signal APOS avail settings.
 */
export interface Scte35TimeSignalAposSettings {
  /**
   * Offset in milliseconds added to the input ad avail PTS time.
   * Applies only to embedded SCTE 104/35 messages.
   * @default - service default
   */
  readonly adAvailOffset?: number;
  /**
   * When set to `IGNORE`, segment descriptors with `noRegionalBlackoutFlag` set to 0 no longer
   * trigger blackouts or ad avail slates.
   * @default - service default
   */
  readonly noRegionalBlackoutFlag?: Scte35FlagBehavior;
  /**
   * When set to `IGNORE`, segment descriptors with `webDeliveryAllowedFlag` set to 0 no longer
   * trigger blackouts or ad avail slates.
   * @default - service default
   */
  readonly webDeliveryAllowedFlag?: Scte35FlagBehavior;
}

/**
 * Controls which output groups receive SCTE-35 segmentation cues.
 */
export class Scte35SegmentationScope {
  /** Insert segment breaks in all output groups. */
  public static readonly ALL_OUTPUT_GROUPS = new Scte35SegmentationScope('ALL_OUTPUT_GROUPS');
  /** Insert segment breaks only in output groups with SCTE-35 passthrough enabled (recommended). */
  public static readonly SCTE35_ENABLED_OUTPUT_GROUPS = new Scte35SegmentationScope('SCTE35_ENABLED_OUTPUT_GROUPS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Scte35SegmentationScope {
    return new Scte35SegmentationScope(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Connection details for an ESAM POIS (Placement Opportunity Information System) endpoint.
 */
export interface PoisEndpoint {
  /**
   * The POIS endpoint URL that MediaLive sends signal conditioning information to.
   */
  readonly url: string;
  /**
   * The username used to connect to the POIS endpoint.
   * @default - no credentials
   */
  readonly username?: string;
  /**
   * An SSM parameter holding the password for the POIS endpoint. The channel role is granted
   * read access to the parameter automatically.
   * @default - no credentials
   */
  readonly password?: IStringParameter;
}

/**
 * Settings for ESAM (Event Signaling and Management) ad avail handling. MediaLive signals ad avail
 * events to an external POIS (Placement Opportunity Information System) endpoint.
 */
export interface EsamSettings {
  /**
   * The POIS endpoint connection details — URL and optional credentials.
   */
  readonly pois: PoisEndpoint;
  /**
   * The acquisition point identity sent to the POIS in MCC requests.
   */
  readonly acquisitionPointId: string;
  /**
   * Offset added to the input ad avail PTS time.
   * @default - service default
   */
  readonly adAvailOffset?: Duration;
  /**
   * The ID of a zone the POIS uses to control the placement of ad avails.
   * @default - service default
   */
  readonly zoneIdentity?: string;
}

/**
 * Avail settings — how SCTE-35 ad avail markers are handled.
 * Use the static factory methods to create.
 */
export abstract class AvailSettings {
  /**
   * Use SCTE-35 splice insert mode for ad avail handling.
   */
  public static spliceInsert(props?: Scte35SpliceInsertSettings): AvailSettings {
    return new Scte35SpliceInsertAvailSettings(props ?? {});
  }

  /**
   * Use SCTE-35 time signal APOS mode for ad avail handling.
   */
  public static timeSignalApos(props?: Scte35TimeSignalAposSettings): AvailSettings {
    return new Scte35TimeSignalAposAvailSettings(props ?? {});
  }

  /**
   * Use ESAM (Event Signaling and Management) mode, signaling ad avails to an external POIS.
   */
  public static esam(props: EsamSettings): AvailSettings {
    return new EsamAvailSettings(props);
  }

  /** @internal */
  public abstract _bind(): CfnChannel.AvailSettingsProperty;

  /**
   * Grant the channel role any permissions these avail settings require (e.g. read access to
   * an ESAM password parameter).
   * @internal
   */
  public _grantPermissions(_role: IRole): void {}
}

/** @internal */
class Scte35SpliceInsertAvailSettings extends AvailSettings {
  constructor(private readonly props: Scte35SpliceInsertSettings) { super(); }
  public _bind(): CfnChannel.AvailSettingsProperty {
    return {
      scte35SpliceInsert: {
        adAvailOffset: this.props.adAvailOffset,
        noRegionalBlackoutFlag: this.props.noRegionalBlackoutFlag?.value,
        webDeliveryAllowedFlag: this.props.webDeliveryAllowedFlag?.value,
      },
    };
  }
}

/** @internal */
class Scte35TimeSignalAposAvailSettings extends AvailSettings {
  constructor(private readonly props: Scte35TimeSignalAposSettings) { super(); }
  public _bind(): CfnChannel.AvailSettingsProperty {
    return {
      scte35TimeSignalApos: {
        adAvailOffset: this.props.adAvailOffset,
        noRegionalBlackoutFlag: this.props.noRegionalBlackoutFlag?.value,
        webDeliveryAllowedFlag: this.props.webDeliveryAllowedFlag?.value,
      },
    };
  }
}

/** @internal */
class EsamAvailSettings extends AvailSettings {
  constructor(private readonly props: EsamSettings) { super(); }
  public _bind(): CfnChannel.AvailSettingsProperty {
    return {
      esam: {
        poisEndpoint: this.props.pois.url,
        acquisitionPointId: this.props.acquisitionPointId,
        adAvailOffset: this.props.adAvailOffset?.toMilliseconds(),
        // MediaLive's ESAM passwordParam expects the SSM parameter reference in `ssm://<name>`
        // form, not the bare parameter name.
        passwordParam: this.props.pois.password ? `ssm://${this.props.pois.password.parameterName}` : undefined,
        username: this.props.pois.username,
        zoneIdentity: this.props.zoneIdentity,
      },
    };
  }
  public override _grantPermissions(role: IRole): void {
    // MediaLive reads the POIS password from SSM Parameter Store at channel runtime, so the
    // channel role needs read access to the parameter (scoped to the parameter ARN).
    this.props.pois.password?.grantRead(role);
  }
}

/**
 * Blackout slate state.
 */
export class BlackoutSlateState {
  /** Enable blackout slate */
  public static readonly ENABLED = new BlackoutSlateState('ENABLED');
  /** Disable blackout slate */
  public static readonly DISABLED = new BlackoutSlateState('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): BlackoutSlateState {
    return new BlackoutSlateState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Network end blackout state.
 */
export class NetworkEndBlackout {
  /** Enable network end blackout */
  public static readonly ENABLED = new NetworkEndBlackout('ENABLED');
  /** Disable network end blackout */
  public static readonly DISABLED = new NetworkEndBlackout('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NetworkEndBlackout {
    return new NetworkEndBlackout(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Blackout slate configuration for the channel.
 */
export interface BlackoutSlate {
  /**
   * Whether to enable the blackout slate.
   * @default - ENABLED if image is provided, DISABLED otherwise
   */
  readonly state?: BlackoutSlateState;
  /**
   * The blackout slate image. Provide a `FileLocation` referencing an S3 bucket
   * (`FileLocation.fromBucket`, which auto-grants read access) or a URL (`FileLocation.url`).
   * Only .bmp and .png supported.
   * @default - solid black
   */
  readonly image?: FileLocation;
  /**
   * Whether to enable network end blackout (triggered by SCTE-35 Network End Segmentation Descriptor).
   * @default - ENABLED if networkEndBlackoutImage is provided, DISABLED otherwise
   */
  readonly networkEndBlackout?: NetworkEndBlackout;
  /**
   * The network end blackout image. Provide a `FileLocation` referencing an S3 bucket
   * (`FileLocation.fromBucket`, which auto-grants read access) or a URL (`FileLocation.url`).
   * @default - solid black
   */
  readonly networkEndBlackoutImage?: FileLocation;
  /**
   * The EIDR network ID (e.g. '10.XXXX/XXXX-XXXX-XXXX-XXXX-XXXX-C').
   * @default - no network ID
   */
  readonly networkId?: string;
}
