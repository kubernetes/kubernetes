/**
 * Feature activation state.
 */
export class FeatureActivationState {
  /** Enable the feature */
  public static readonly ENABLED = new FeatureActivationState('ENABLED');
  /** Disable the feature */
  public static readonly DISABLED = new FeatureActivationState('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): FeatureActivationState {
    return new FeatureActivationState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Feature activations for the channel.
 */
export interface FeatureActivations {
  /**
   * Enable Input Prepare schedule actions.
   * @default - DISABLED, applied by MediaLive
   */
  readonly inputPrepareScheduleActions?: FeatureActivationState;
  /**
   * Enable output static image overlay schedule actions.
   * @default - DISABLED, applied by MediaLive
   */
  readonly outputStaticImageOverlayScheduleActions?: FeatureActivationState;
}

/**
 * Motion graphics insertion state.
 */
export class MotionGraphicsInsertion {
  /** Enable motion graphics overlay */
  public static readonly ENABLED = new MotionGraphicsInsertion('ENABLED');
  /** Disable motion graphics overlay */
  public static readonly DISABLED = new MotionGraphicsInsertion('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MotionGraphicsInsertion {
    return new MotionGraphicsInsertion(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Motion graphics overlay configuration.
 */
export interface MotionGraphicsConfiguration {
  /**
   * Whether to enable the motion graphics overlay.
   * @default MotionGraphicsInsertion.DISABLED
   */
  readonly motionGraphicsInsertion?: MotionGraphicsInsertion;
}

/**
 * Whether Nielsen PCM to ID3 tagging is enabled.
 */
export class NielsenPcmToId3TaggingState {
  /** Disabled. */
  public static readonly DISABLED = new NielsenPcmToId3TaggingState('DISABLED');
  /** Enabled. */
  public static readonly ENABLED = new NielsenPcmToId3TaggingState('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NielsenPcmToId3TaggingState {
    return new NielsenPcmToId3TaggingState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Nielsen watermark configuration.
 */
export interface NielsenConfiguration {
  /**
   * The Distributor ID assigned to your organization by Nielsen.
   * @default - no distributor ID
   */
  readonly distributorId?: string;
  /**
   * Whether to enable Nielsen PCM to ID3 tagging.
   * @default - service default
   */
  readonly nielsenPcmToId3Tagging?: NielsenPcmToId3TaggingState;
}

/**
 * Thumbnail state.
 */
export class ThumbnailState {
  /** Enable thumbnail generation. */
  public static readonly AUTO = new ThumbnailState('AUTO');
  /** Disable thumbnail generation. */
  public static readonly DISABLED = new ThumbnailState('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ThumbnailState {
    return new ThumbnailState(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Thumbnail configuration for the channel.
 */
export interface ThumbnailConfiguration {
  /**
   * Whether to enable thumbnail generation.
   * @default ThumbnailState.AUTO
   */
  readonly state?: ThumbnailState;
}
