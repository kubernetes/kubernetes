import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';

/**
 * Video color space.
 */
export class VideoColorSpace {
  /** Follow the source color space */
  public static readonly FOLLOW = new VideoColorSpace('FOLLOW');
  /** Rec. 601 */
  public static readonly REC_601 = new VideoColorSpace('REC_601');
  /** Rec. 709 */
  public static readonly REC_709 = new VideoColorSpace('REC_709');
  /** HDR10 */
  public static readonly HDR10 = new VideoColorSpace('HDR10');
  /** HLG 2020 */
  public static readonly HLG_2020 = new VideoColorSpace('HLG_2020');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): VideoColorSpace {
    return new VideoColorSpace(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Controls how the `colorSpace` value is used when it is not `FOLLOW`.
 */
export class VideoColorSpaceUsage {
  /**
   * Use the input's color space data when present; fall back to `colorSpace` only when the
   * input has none.
   */
  public static readonly FALLBACK = new VideoColorSpaceUsage('FALLBACK');
  /** Always use `colorSpace`, ignoring any color space data in the input. */
  public static readonly FORCE = new VideoColorSpaceUsage('FORCE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): VideoColorSpaceUsage {
    return new VideoColorSpaceUsage(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * HDR10 color space metadata for the input video.
 */
export interface Hdr10Settings {
  /**
   * Maximum Content Light Level (MaxCLL) — the maximum light level, in nits, of any single
   * pixel in the stream.
   * @default - not set
   */
  readonly maxContentLightLevel?: number;
  /**
   * Maximum Frame Average Light Level (MaxFALL) — the maximum average light level, in nits,
   * of any single frame in the stream.
   * @default - not set
   */
  readonly maxFrameAverageLightLevel?: number;
}

/**
 * Selects the specific video to extract from the input — by PID or by program. Create with
 * the static factory methods; exactly one selection applies, enforced by the type.
 */
export abstract class VideoSelection {
  /** Extract the video with this PID. */
  public static byPid(pid: number): VideoSelection {
    return new VideoPidSelection(pid);
  }
  /**
   * Extract the video from this program within a multi-program transport stream. If the
   * program doesn't exist, MediaLive selects the first program in the stream.
   */
  public static byProgramId(programId: number): VideoSelection {
    return new VideoProgramIdSelection(programId);
  }

  /** @internal */
  public abstract _bind(): CfnChannel.VideoSelectorSettingsProperty;
}

/** @internal */
class VideoPidSelection extends VideoSelection {
  constructor(private readonly pid: number) { super(); }
  public _bind(): CfnChannel.VideoSelectorSettingsProperty {
    return { videoSelectorPid: { pid: this.pid } };
  }
}

/** @internal */
class VideoProgramIdSelection extends VideoSelection {
  constructor(private readonly programId: number) { super(); }
  public _bind(): CfnChannel.VideoSelectorSettingsProperty {
    return { videoSelectorProgramId: { programId: this.programId } };
  }
}

/**
 * Video selector settings for an input.
 */
export interface VideoSelectorSettings {
  /**
   * The color space of the input video.
   * @default - service default
   */
  readonly colorSpace?: VideoColorSpace;
  /**
   * How `colorSpace` is applied when it is not `FOLLOW`.
   * @default - MediaLive service default
   */
  readonly colorSpaceUsage?: VideoColorSpaceUsage;
  /**
   * HDR10 color space metadata for the input.
   * @default - none
   */
  readonly hdr10?: Hdr10Settings;
  /**
   * Selects the specific video to extract from the input (by PID or by program).
   * @default - MediaLive selects the video automatically
   */
  readonly selectBy?: VideoSelection;
}
