import type { Bitrate, Duration } from 'aws-cdk-lib';
import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type { Framerate } from './framerate';
import { PixelAspectRatio } from './shared';

/**
 * H.264 profile.
 */
export class H264Profile {
  /** Baseline profile */
  public static readonly BASELINE = new H264Profile('BASELINE');
  /** Main profile */
  public static readonly MAIN = new H264Profile('MAIN');
  /** High profile */
  public static readonly HIGH = new H264Profile('HIGH');
  /** High 10-bit profile */
  public static readonly HIGH_10BIT = new H264Profile('HIGH_10BIT');
  /** High 4:2:2 profile */
  public static readonly HIGH_422 = new H264Profile('HIGH_422');
  /** High 4:2:2 10-bit profile */
  public static readonly HIGH_422_10BIT = new H264Profile('HIGH_422_10BIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264Profile {
    return new H264Profile(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 level.
 */
export class H264Level {
  /** Level 1 */
  public static readonly H264_LEVEL_1 = new H264Level('H264_LEVEL_1');
  /** Level 1.1 */
  public static readonly H264_LEVEL_1_1 = new H264Level('H264_LEVEL_1_1');
  /** Level 1.2 */
  public static readonly H264_LEVEL_1_2 = new H264Level('H264_LEVEL_1_2');
  /** Level 1.3 */
  public static readonly H264_LEVEL_1_3 = new H264Level('H264_LEVEL_1_3');
  /** Level 2 */
  public static readonly H264_LEVEL_2 = new H264Level('H264_LEVEL_2');
  /** Level 2.1 */
  public static readonly H264_LEVEL_2_1 = new H264Level('H264_LEVEL_2_1');
  /** Level 2.2 */
  public static readonly H264_LEVEL_2_2 = new H264Level('H264_LEVEL_2_2');
  /** Level 3 */
  public static readonly H264_LEVEL_3 = new H264Level('H264_LEVEL_3');
  /** Level 3.1 */
  public static readonly H264_LEVEL_3_1 = new H264Level('H264_LEVEL_3_1');
  /** Level 3.2 */
  public static readonly H264_LEVEL_3_2 = new H264Level('H264_LEVEL_3_2');
  /** Level 4 */
  public static readonly H264_LEVEL_4 = new H264Level('H264_LEVEL_4');
  /** Level 4.1 */
  public static readonly H264_LEVEL_4_1 = new H264Level('H264_LEVEL_4_1');
  /** Level 4.2 */
  public static readonly H264_LEVEL_4_2 = new H264Level('H264_LEVEL_4_2');
  /** Level 5 */
  public static readonly H264_LEVEL_5 = new H264Level('H264_LEVEL_5');
  /** Level 5.1 */
  public static readonly H264_LEVEL_5_1 = new H264Level('H264_LEVEL_5_1');
  /** Level 5.2 */
  public static readonly H264_LEVEL_5_2 = new H264Level('H264_LEVEL_5_2');
  /** Auto-select the level based on the encode configuration */
  public static readonly H264_LEVEL_AUTO = new H264Level('H264_LEVEL_AUTO');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264Level {
    return new H264Level(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 level.
 */
export class H265Level {
  /** Level 1 */
  public static readonly H265_LEVEL_1 = new H265Level('H265_LEVEL_1');
  /** Level 2 */
  public static readonly H265_LEVEL_2 = new H265Level('H265_LEVEL_2');
  /** Level 2.1 */
  public static readonly H265_LEVEL_2_1 = new H265Level('H265_LEVEL_2_1');
  /** Level 3 */
  public static readonly H265_LEVEL_3 = new H265Level('H265_LEVEL_3');
  /** Level 3.1 */
  public static readonly H265_LEVEL_3_1 = new H265Level('H265_LEVEL_3_1');
  /** Level 4 */
  public static readonly H265_LEVEL_4 = new H265Level('H265_LEVEL_4');
  /** Level 4.1 */
  public static readonly H265_LEVEL_4_1 = new H265Level('H265_LEVEL_4_1');
  /** Level 5 */
  public static readonly H265_LEVEL_5 = new H265Level('H265_LEVEL_5');
  /** Level 5.1 */
  public static readonly H265_LEVEL_5_1 = new H265Level('H265_LEVEL_5_1');
  /** Level 5.2 */
  public static readonly H265_LEVEL_5_2 = new H265Level('H265_LEVEL_5_2');
  /** Level 6 */
  public static readonly H265_LEVEL_6 = new H265Level('H265_LEVEL_6');
  /** Level 6.1 */
  public static readonly H265_LEVEL_6_1 = new H265Level('H265_LEVEL_6_1');
  /** Level 6.2 */
  public static readonly H265_LEVEL_6_2 = new H265Level('H265_LEVEL_6_2');
  /** Auto-select the level based on the encode configuration */
  public static readonly H265_LEVEL_AUTO = new H265Level('H265_LEVEL_AUTO');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265Level {
    return new H265Level(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AV1 level.
 */
export class Av1Level {
  /** Level 2 */
  public static readonly AV1_LEVEL_2 = new Av1Level('AV1_LEVEL_2');
  /** Level 2.1 */
  public static readonly AV1_LEVEL_2_1 = new Av1Level('AV1_LEVEL_2_1');
  /** Level 3 */
  public static readonly AV1_LEVEL_3 = new Av1Level('AV1_LEVEL_3');
  /** Level 3.1 */
  public static readonly AV1_LEVEL_3_1 = new Av1Level('AV1_LEVEL_3_1');
  /** Level 4 */
  public static readonly AV1_LEVEL_4 = new Av1Level('AV1_LEVEL_4');
  /** Level 4.1 */
  public static readonly AV1_LEVEL_4_1 = new Av1Level('AV1_LEVEL_4_1');
  /** Level 5 */
  public static readonly AV1_LEVEL_5 = new Av1Level('AV1_LEVEL_5');
  /** Level 5.1 */
  public static readonly AV1_LEVEL_5_1 = new Av1Level('AV1_LEVEL_5_1');
  /** Level 5.2 */
  public static readonly AV1_LEVEL_5_2 = new Av1Level('AV1_LEVEL_5_2');
  /** Level 5.3 */
  public static readonly AV1_LEVEL_5_3 = new Av1Level('AV1_LEVEL_5_3');
  /** Level 6 */
  public static readonly AV1_LEVEL_6 = new Av1Level('AV1_LEVEL_6');
  /** Level 6.1 */
  public static readonly AV1_LEVEL_6_1 = new Av1Level('AV1_LEVEL_6_1');
  /** Level 6.2 */
  public static readonly AV1_LEVEL_6_2 = new Av1Level('AV1_LEVEL_6_2');
  /** Level 6.3 */
  public static readonly AV1_LEVEL_6_3 = new Av1Level('AV1_LEVEL_6_3');
  /** Auto-select the level based on the encode configuration */
  public static readonly AV1_LEVEL_AUTO = new Av1Level('AV1_LEVEL_AUTO');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1Level {
    return new Av1Level(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 adaptive quantization strength.
 */
export class H264AdaptiveQuantization {
  /** Auto */
  public static readonly AUTO = new H264AdaptiveQuantization('AUTO');
  /** High */
  public static readonly HIGH = new H264AdaptiveQuantization('HIGH');
  /** Higher */
  public static readonly HIGHER = new H264AdaptiveQuantization('HIGHER');
  /** Low */
  public static readonly LOW = new H264AdaptiveQuantization('LOW');
  /** Max */
  public static readonly MAX = new H264AdaptiveQuantization('MAX');
  /** Medium */
  public static readonly MEDIUM = new H264AdaptiveQuantization('MEDIUM');
  /** Off */
  public static readonly OFF = new H264AdaptiveQuantization('OFF');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264AdaptiveQuantization {
    return new H264AdaptiveQuantization(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * GOP size (keyframe interval). Use the static factory methods to specify in frames or seconds.
 *
 * The value must be greater than zero. When expressed in frames it must be a whole number;
 * when expressed in seconds it may be fractional.
 */
export class GopSize {
  /** GOP size in seconds. May be fractional (e.g. `1.5`). */
  public static seconds(value: number): GopSize {
    return new GopSize(value, 'SECONDS');
  }
  /** GOP size in frames. Must be a whole number. */
  public static frames(value: number): GopSize {
    return new GopSize(value, 'FRAMES');
  }

  /** @internal */
  public readonly _value: number;
  /** @internal */
  public readonly _units: string;

  private constructor(value: number, units: string) {
    // MediaLive requires gopSize > 0 for both units. When the unit is FRAMES the value
    // must be a whole number (a fractional frame is meaningless); SECONDS may be
    // fractional. See the H264/H265/AV1/MPEG-2 gopSize docs ("must be greater than zero";
    // frames are converted to a frame count at runtime).
    if (!Token.isUnresolved(value)) {
      if (value <= 0) {
        throw new UnscopedValidationError(lit`GopSize`, `GOP size must be greater than zero, got ${JSON.stringify(value)}`);
      }
      if (units === 'FRAMES' && !Number.isInteger(value)) {
        throw new UnscopedValidationError(lit`GopSizeFrames`, `GOP size in frames must be a whole number, got ${JSON.stringify(value)}`);
      }
    }
    this._value = value;
    this._units = units;
  }
}

// =============================================================================
// Shared enums (used by multiple codecs)
// =============================================================================

/**
 * AFD signaling mode.
 */
export class AfdSignaling {
  /** Auto — preserve input AFD value */
  public static readonly AUTO = new AfdSignaling('AUTO');
  /** Fixed — use the value from fixedAfd */
  public static readonly FIXED = new AfdSignaling('FIXED');
  /** None — do not write AFD */
  public static readonly NONE = new AfdSignaling('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AfdSignaling {
    return new AfdSignaling(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Color metadata inclusion.
 */
export class ColorMetadata {
  /** Ignore — do not include color metadata */
  public static readonly IGNORE = new ColorMetadata('IGNORE');
  /** Insert — include color metadata */
  public static readonly INSERT = new ColorMetadata('INSERT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ColorMetadata {
    return new ColorMetadata(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Scan type for the output video.
 */
export class ScanType {
  /** Interlaced (top field first) */
  public static readonly INTERLACED = new ScanType('INTERLACED');
  /** Progressive */
  public static readonly PROGRESSIVE = new ScanType('PROGRESSIVE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ScanType {
    return new ScanType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Flicker adaptive quantization.
 */
export class FlickerAq {
  /** Enabled */
  public static readonly ENABLED = new FlickerAq('ENABLED');
  /** Disabled */
  public static readonly DISABLED = new FlickerAq('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): FlickerAq {
    return new FlickerAq(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * GOP B-frame reference.
 */
export class GopBReference {
  /** Enabled */
  public static readonly ENABLED = new GopBReference('ENABLED');
  /** Disabled */
  public static readonly DISABLED = new GopBReference('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): GopBReference {
    return new GopBReference(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Lookahead rate control.
 */
export class LookAheadRateControl {
  /** High — better quality, more latency and memory */
  public static readonly HIGH = new LookAheadRateControl('HIGH');
  /** Low — less latency and memory */
  public static readonly LOW = new LookAheadRateControl('LOW');
  /** Medium */
  public static readonly MEDIUM = new LookAheadRateControl('MEDIUM');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): LookAheadRateControl {
    return new LookAheadRateControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Timecode insertion mode.
 *
 * @remarks This controls timecode insertion in the output elementary stream.
 * To preserve source timecodes, set `TimecodeSource.EMBEDDED` on the channel's `timecodeConfig`.
 */
export class TimecodeInsertion {
  /** Disabled — do not include timecodes */
  public static readonly DISABLED = new TimecodeInsertion('DISABLED');
  /** PIC_TIMING_SEI — pass through picture timing SEI messages */
  public static readonly PIC_TIMING_SEI = new TimecodeInsertion('PIC_TIMING_SEI');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimecodeInsertion {
    return new TimecodeInsertion(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Sub-GOP length mode.
 */
export class SubgopLength {
  /** Dynamic — let MediaLive optimize B-frames per sub-GOP */
  public static readonly DYNAMIC = new SubgopLength('DYNAMIC');
  /** Fixed — use gopNumBFrames in each sub-GOP */
  public static readonly FIXED = new SubgopLength('FIXED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SubgopLength {
    return new SubgopLength(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// H.264-specific enums
// =============================================================================

/**
 * H.264 entropy encoding mode.
 */
export class H264EntropyEncoding {
  /** CABAC (requires Main or High profile) */
  public static readonly CABAC = new H264EntropyEncoding('CABAC');
  /** CAVLC */
  public static readonly CAVLC = new H264EntropyEncoding('CAVLC');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264EntropyEncoding {
    return new H264EntropyEncoding(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 force field pictures.
 */
export class H264ForceFieldPictures {
  /** Enabled — force coding on a field basis */
  public static readonly ENABLED = new H264ForceFieldPictures('ENABLED');
  /** Disabled — let encoder decide */
  public static readonly DISABLED = new H264ForceFieldPictures('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264ForceFieldPictures {
    return new H264ForceFieldPictures(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 syntax mode.
 */
export class H264Syntax {
  /** Default */
  public static readonly DEFAULT = new H264Syntax('DEFAULT');
  /** RP-2027 compliant */
  public static readonly RP2027 = new H264Syntax('RP2027');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264Syntax {
    return new H264Syntax(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 quality level.
 */
export class H264QualityLevel {
  /** Enhanced quality (may incur additional cost) */
  public static readonly ENHANCED_QUALITY = new H264QualityLevel('ENHANCED_QUALITY');
  /** Standard quality */
  public static readonly STANDARD_QUALITY = new H264QualityLevel('STANDARD_QUALITY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264QualityLevel {
    return new H264QualityLevel(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// H.265-specific enums
// =============================================================================

/**
 * H.265 adaptive quantization.
 */
export class H265AdaptiveQuantization {
  /** Auto */
  public static readonly AUTO = new H265AdaptiveQuantization('AUTO');
  /** High */
  public static readonly HIGH = new H265AdaptiveQuantization('HIGH');
  /** Higher */
  public static readonly HIGHER = new H265AdaptiveQuantization('HIGHER');
  /** Low */
  public static readonly LOW = new H265AdaptiveQuantization('LOW');
  /** Max */
  public static readonly MAX = new H265AdaptiveQuantization('MAX');
  /** Medium */
  public static readonly MEDIUM = new H265AdaptiveQuantization('MEDIUM');
  /** Off */
  public static readonly OFF = new H265AdaptiveQuantization('OFF');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265AdaptiveQuantization {
    return new H265AdaptiveQuantization(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 alternative transfer function.
 */
export class H265AlternativeTransferFunction {
  /** Insert */
  public static readonly INSERT = new H265AlternativeTransferFunction('INSERT');
  /** Omit */
  public static readonly OMIT = new H265AlternativeTransferFunction('OMIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265AlternativeTransferFunction {
    return new H265AlternativeTransferFunction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 deblocking filter.
 */
export class H265Deblocking {
  /** Disabled */
  public static readonly DISABLED = new H265Deblocking('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H265Deblocking('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265Deblocking {
    return new H265Deblocking(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 motion vector over picture boundaries.
 */
export class H265MvOverPictureBoundaries {
  /** Disabled */
  public static readonly DISABLED = new H265MvOverPictureBoundaries('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H265MvOverPictureBoundaries('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265MvOverPictureBoundaries {
    return new H265MvOverPictureBoundaries(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 motion vector temporal predictor.
 */
export class H265MvTemporalPredictor {
  /** Disabled */
  public static readonly DISABLED = new H265MvTemporalPredictor('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H265MvTemporalPredictor('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265MvTemporalPredictor {
    return new H265MvTemporalPredictor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 tile padding.
 */
export class H265TilePadding {
  /** None */
  public static readonly NONE = new H265TilePadding('NONE');
  /** Padded */
  public static readonly PADDED = new H265TilePadding('PADDED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265TilePadding {
    return new H265TilePadding(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 treeblock size.
 */
export class H265TreeblockSize {
  /** Auto */
  public static readonly AUTO = new H265TreeblockSize('AUTO');
  /** 32x32 */
  public static readonly TREE_SIZE_32X32 = new H265TreeblockSize('TREE_SIZE_32X32');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265TreeblockSize {
    return new H265TreeblockSize(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// AV1-specific enums
// =============================================================================

/**
 * AV1 bit depth.
 */
export class Av1BitDepth {
  /** 8-bit */
  public static readonly BIT_DEPTH_8 = new Av1BitDepth('DEPTH_8');
  /** 10-bit */
  public static readonly BIT_DEPTH_10 = new Av1BitDepth('DEPTH_10');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1BitDepth {
    return new Av1BitDepth(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AV1 scene change detection.
 */
export class Av1SceneChangeDetect {
  /** Disabled */
  public static readonly DISABLED = new Av1SceneChangeDetect('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Av1SceneChangeDetect('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1SceneChangeDetect {
    return new Av1SceneChangeDetect(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AV1 spatial adaptive quantization.
 */
export class Av1SpatialAq {
  /** Disabled */
  public static readonly DISABLED = new Av1SpatialAq('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Av1SpatialAq('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1SpatialAq {
    return new Av1SpatialAq(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AV1 temporal adaptive quantization.
 */
export class Av1TemporalAq {
  /** Disabled */
  public static readonly DISABLED = new Av1TemporalAq('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Av1TemporalAq('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1TemporalAq {
    return new Av1TemporalAq(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 scene change detection.
 */
export class H264SceneChangeDetect {
  /** Disabled */
  public static readonly DISABLED = new H264SceneChangeDetect('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H264SceneChangeDetect('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264SceneChangeDetect {
    return new H264SceneChangeDetect(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 spatial adaptive quantization.
 */
export class H264SpatialAq {
  /** Disabled */
  public static readonly DISABLED = new H264SpatialAq('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H264SpatialAq('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264SpatialAq {
    return new H264SpatialAq(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.264 temporal adaptive quantization.
 */
export class H264TemporalAq {
  /** Disabled */
  public static readonly DISABLED = new H264TemporalAq('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H264TemporalAq('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H264TemporalAq {
    return new H264TemporalAq(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 scene change detection.
 */
export class H265SceneChangeDetect {
  /** Disabled */
  public static readonly DISABLED = new H265SceneChangeDetect('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new H265SceneChangeDetect('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265SceneChangeDetect {
    return new H265SceneChangeDetect(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AV1 timecode insertion.
 */
export class Av1TimecodeInsertion {
  /** Disabled — do not insert timecodes */
  public static readonly DISABLED = new Av1TimecodeInsertion('DISABLED');
  /**
   * Include timecodes as a metadata OBU (Open Bitstream Unit) of type
   * `METADATA_TYPE_TIMECODE`, based on the source specified in the channel's timecode config.
   */
  public static readonly METADATA_OBU = new Av1TimecodeInsertion('METADATA_OBU');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Av1TimecodeInsertion {
    return new Av1TimecodeInsertion(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// Timecode burn-in settings
// =============================================================================

/**
 * Font size for timecode burn-in.
 */
export class TimecodeBurninFontSize {
  /** Extra small */
  public static readonly EXTRA_SMALL_10 = new TimecodeBurninFontSize('EXTRA_SMALL_10');
  /** Large */
  public static readonly LARGE_48 = new TimecodeBurninFontSize('LARGE_48');
  /** Medium */
  public static readonly MEDIUM_32 = new TimecodeBurninFontSize('MEDIUM_32');
  /** Small */
  public static readonly SMALL_16 = new TimecodeBurninFontSize('SMALL_16');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimecodeBurninFontSize {
    return new TimecodeBurninFontSize(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Position for timecode burn-in overlay.
 */
export class TimecodeBurninPosition {
  /** Bottom center */
  public static readonly BOTTOM_CENTER = new TimecodeBurninPosition('BOTTOM_CENTER');
  /** Bottom left */
  public static readonly BOTTOM_LEFT = new TimecodeBurninPosition('BOTTOM_LEFT');
  /** Bottom right */
  public static readonly BOTTOM_RIGHT = new TimecodeBurninPosition('BOTTOM_RIGHT');
  /** Middle center */
  public static readonly MIDDLE_CENTER = new TimecodeBurninPosition('MIDDLE_CENTER');
  /** Middle left */
  public static readonly MIDDLE_LEFT = new TimecodeBurninPosition('MIDDLE_LEFT');
  /** Middle right */
  public static readonly MIDDLE_RIGHT = new TimecodeBurninPosition('MIDDLE_RIGHT');
  /** Top center */
  public static readonly TOP_CENTER = new TimecodeBurninPosition('TOP_CENTER');
  /** Top left */
  public static readonly TOP_LEFT = new TimecodeBurninPosition('TOP_LEFT');
  /** Top right */
  public static readonly TOP_RIGHT = new TimecodeBurninPosition('TOP_RIGHT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimecodeBurninPosition {
    return new TimecodeBurninPosition(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Settings for burning a timecode overlay into the video output.
 */
export interface TimecodeBurninSettings {
  /**
   * The font size of the timecode overlay.
   * @default - service default
   */
  readonly fontSize?: TimecodeBurninFontSize;
  /**
   * The position of the timecode overlay on the video.
   * @default - service default
   */
  readonly position?: TimecodeBurninPosition;
  /**
   * A string prepended to the timecode (e.g. a channel name).
   * @default - no prefix
   */
  readonly prefix?: string;
}

// =============================================================================
// Rate control classes (per codec)
// =============================================================================

/** Properties for CBR rate control. */
export interface CbrRateControlProps {
  /** The constant bitrate. */
  readonly bitrate: Bitrate;
}

/** Properties for VBR rate control. */
export interface VbrRateControlProps {
  /** The average bitrate. */
  readonly bitrate: Bitrate;
  /** The maximum bitrate. */
  readonly maxBitrate: Bitrate;
}

/** Properties for QVBR rate control. */
export interface QvbrRateControlProps {
  /** The maximum bitrate. */
  readonly maxBitrate: Bitrate;
  /**
   * The QVBR quality level (1-10). Leave unset to let MediaLive infer the target quality from the
   * output resolution and max bitrate.
   * @default - MediaLive infers the quality level from the resolution and max bitrate
   */
  readonly qvbrQualityLevel?: number;
}

/**
 * H.264 rate control. Use the static factory methods to create.
 */
export class H264RateControl {
  /** Constant bitrate. */
  public static cbr(props: CbrRateControlProps): H264RateControl {
    return new H264RateControl('CBR', props.bitrate, undefined, undefined);
  }
  /** Variable bitrate. */
  public static vbr(props: VbrRateControlProps): H264RateControl {
    return new H264RateControl('VBR', props.bitrate, props.maxBitrate, undefined);
  }
  /** Quality-defined variable bitrate. */
  public static qvbr(props: QvbrRateControlProps): H264RateControl {
    return new H264RateControl('QVBR', undefined, props.maxBitrate, props.qvbrQualityLevel);
  }

  /** @internal */
  public readonly _mode: string;
  /** @internal */
  public readonly _bitrate: Bitrate | undefined;
  /** @internal */
  public readonly _maxBitrate: Bitrate | undefined;
  /** @internal */
  public readonly _qvbrQualityLevel: number | undefined;

  private constructor(mode: string, bitrate: Bitrate | undefined, maxBitrate: Bitrate | undefined, qvbrQualityLevel: number | undefined) {
    this._mode = mode;
    this._bitrate = bitrate;
    this._maxBitrate = maxBitrate;
    this._qvbrQualityLevel = qvbrQualityLevel;
  }
}

/**
 * H.265 rate control. Use the static factory methods to create.
 */
export class H265RateControl {
  /** Constant bitrate. */
  public static cbr(props: CbrRateControlProps): H265RateControl {
    return new H265RateControl('CBR', props.bitrate, undefined, undefined);
  }
  /** Variable bitrate. */
  public static vbr(props: VbrRateControlProps): H265RateControl {
    return new H265RateControl('VBR', props.bitrate, props.maxBitrate, undefined);
  }
  /** Quality-defined variable bitrate. */
  public static qvbr(props: QvbrRateControlProps): H265RateControl {
    return new H265RateControl('QVBR', undefined, props.maxBitrate, props.qvbrQualityLevel);
  }

  /** @internal */
  public readonly _mode: string;
  /** @internal */
  public readonly _bitrate: Bitrate | undefined;
  /** @internal */
  public readonly _maxBitrate: Bitrate | undefined;
  /** @internal */
  public readonly _qvbrQualityLevel: number | undefined;

  private constructor(mode: string, bitrate: Bitrate | undefined, maxBitrate: Bitrate | undefined, qvbrQualityLevel: number | undefined) {
    this._mode = mode;
    this._bitrate = bitrate;
    this._maxBitrate = maxBitrate;
    this._qvbrQualityLevel = qvbrQualityLevel;
  }
}

/**
 * AV1 rate control. AV1 supports QVBR and CBR.
 */
export class Av1RateControl {
  /** Quality-defined variable bitrate. */
  public static qvbr(props: QvbrRateControlProps): Av1RateControl {
    return new Av1RateControl('QVBR', undefined, props.maxBitrate, props.qvbrQualityLevel);
  }
  /** Constant bitrate. */
  public static cbr(props: CbrRateControlProps): Av1RateControl {
    return new Av1RateControl('CBR', props.bitrate, undefined, undefined);
  }

  /** @internal */
  public readonly _mode: string;
  /** @internal */
  public readonly _bitrate: Bitrate | undefined;
  /** @internal */
  public readonly _maxBitrate: Bitrate | undefined;
  /** @internal */
  public readonly _qvbrQualityLevel: number | undefined;

  private constructor(mode: string, bitrate: Bitrate | undefined, maxBitrate: Bitrate | undefined, qvbrQualityLevel: number | undefined) {
    this._mode = mode;
    this._bitrate = bitrate;
    this._maxBitrate = maxBitrate;
    this._qvbrQualityLevel = qvbrQualityLevel;
  }
}

// =============================================================================
// Color space settings classes
// =============================================================================

/**
 * Color space settings for H.264 video.
 */
export class H264ColorSpaceSettings {
  /** Pass through the source color space with no conversion. */
  public static passthrough(): H264ColorSpaceSettings {
    return new H264ColorSpaceSettings({ colorSpacePassthroughSettings: {} });
  }
  /** Convert to Rec.601 color space. */
  public static rec601(): H264ColorSpaceSettings {
    return new H264ColorSpaceSettings({ rec601Settings: {} });
  }
  /** Convert to Rec.709 color space. */
  public static rec709(): H264ColorSpaceSettings {
    return new H264ColorSpaceSettings({ rec709Settings: {} });
  }

  private readonly config: CfnChannel.H264ColorSpaceSettingsProperty;
  private constructor(config: CfnChannel.H264ColorSpaceSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.H264ColorSpaceSettingsProperty { return this.config; }
}

/**
 * Properties for HDR10 color space settings.
 */
export interface Hdr10SettingsProps {
  /**
   * Maximum Content Light Level — the maximum light level of any single pixel in nits.
   * @default - service default
   */
  readonly maxCll?: number;
  /**
   * Maximum Frame Average Light Level — the maximum average light level of any single frame in nits.
   * @default - service default
   */
  readonly maxFall?: number;
}

/**
 * Color space settings for H.265 video.
 */
export class H265ColorSpaceSettings {
  /** Pass through the source color space with no conversion. */
  public static passthrough(): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ colorSpacePassthroughSettings: {} });
  }
  /** Dolby Vision 8.1 color space. */
  public static dolbyVision81(): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ dolbyVision81Settings: {} });
  }
  /** HDR10 color space. */
  public static hdr10(props?: Hdr10SettingsProps): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ hdr10Settings: props ?? {} });
  }
  /** HLG 2020 color space. */
  public static hlg2020(): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ hlg2020Settings: {} });
  }
  /** Convert to Rec.601 color space. */
  public static rec601(): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ rec601Settings: {} });
  }
  /** Convert to Rec.709 color space. */
  public static rec709(): H265ColorSpaceSettings {
    return new H265ColorSpaceSettings({ rec709Settings: {} });
  }

  private readonly config: CfnChannel.H265ColorSpaceSettingsProperty;
  private constructor(config: CfnChannel.H265ColorSpaceSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.H265ColorSpaceSettingsProperty { return this.config; }
}

/**
 * Color space settings for AV1 video.
 */
export class Av1ColorSpaceSettings {
  /** Pass through the source color space with no conversion. */
  public static passthrough(): Av1ColorSpaceSettings {
    return new Av1ColorSpaceSettings({ colorSpacePassthroughSettings: {} });
  }
  /** HDR10 color space. */
  public static hdr10(props?: Hdr10SettingsProps): Av1ColorSpaceSettings {
    return new Av1ColorSpaceSettings({ hdr10Settings: props ?? {} });
  }
  /** HLG 2020 color space. */
  public static hlg2020(): Av1ColorSpaceSettings {
    return new Av1ColorSpaceSettings({ hlg2020Settings: {} });
  }
  /** Convert to Rec.601 color space. */
  public static rec601(): Av1ColorSpaceSettings {
    return new Av1ColorSpaceSettings({ rec601Settings: {} });
  }
  /** Convert to Rec.709 color space. */
  public static rec709(): Av1ColorSpaceSettings {
    return new Av1ColorSpaceSettings({ rec709Settings: {} });
  }

  private readonly config: CfnChannel.Av1ColorSpaceSettingsProperty;
  private constructor(config: CfnChannel.Av1ColorSpaceSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.Av1ColorSpaceSettingsProperty { return this.config; }
}

// =============================================================================
// Filter settings classes
// =============================================================================

/**
 * Post-filter sharpening for temporal filter.
 */
export class TemporalFilterPostFilterSharpening {
  /** Auto */
  public static readonly AUTO = new TemporalFilterPostFilterSharpening('AUTO');
  /** Disabled */
  public static readonly DISABLED = new TemporalFilterPostFilterSharpening('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new TemporalFilterPostFilterSharpening('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TemporalFilterPostFilterSharpening {
    return new TemporalFilterPostFilterSharpening(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Temporal filter strength.
 */
export class TemporalFilterStrength {
  /** Auto */
  public static readonly AUTO = new TemporalFilterStrength('AUTO');
  /** Strength 1 (recommended) */
  public static readonly STRENGTH_1 = new TemporalFilterStrength('STRENGTH_1');
  /** Strength 2 (recommended) */
  public static readonly STRENGTH_2 = new TemporalFilterStrength('STRENGTH_2');
  /** Strength 3 */
  public static readonly STRENGTH_3 = new TemporalFilterStrength('STRENGTH_3');
  /** Strength 4 */
  public static readonly STRENGTH_4 = new TemporalFilterStrength('STRENGTH_4');
  /** Strength 5 */
  public static readonly STRENGTH_5 = new TemporalFilterStrength('STRENGTH_5');
  /** Strength 6 */
  public static readonly STRENGTH_6 = new TemporalFilterStrength('STRENGTH_6');
  /** Strength 7 */
  public static readonly STRENGTH_7 = new TemporalFilterStrength('STRENGTH_7');
  /** Strength 8 */
  public static readonly STRENGTH_8 = new TemporalFilterStrength('STRENGTH_8');
  /** Strength 9 */
  public static readonly STRENGTH_9 = new TemporalFilterStrength('STRENGTH_9');
  /** Strength 10 */
  public static readonly STRENGTH_10 = new TemporalFilterStrength('STRENGTH_10');
  /** Strength 11 */
  public static readonly STRENGTH_11 = new TemporalFilterStrength('STRENGTH_11');
  /** Strength 12 */
  public static readonly STRENGTH_12 = new TemporalFilterStrength('STRENGTH_12');
  /** Strength 13 */
  public static readonly STRENGTH_13 = new TemporalFilterStrength('STRENGTH_13');
  /** Strength 14 */
  public static readonly STRENGTH_14 = new TemporalFilterStrength('STRENGTH_14');
  /** Strength 15 */
  public static readonly STRENGTH_15 = new TemporalFilterStrength('STRENGTH_15');
  /** Strength 16 */
  public static readonly STRENGTH_16 = new TemporalFilterStrength('STRENGTH_16');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TemporalFilterStrength {
    return new TemporalFilterStrength(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Post-filter sharpening for bandwidth reduction filter.
 */
export class BandwidthReductionPostFilterSharpening {
  /** Disabled */
  public static readonly DISABLED = new BandwidthReductionPostFilterSharpening('DISABLED');
  /** Sharpening level 1 */
  public static readonly SHARPENING_1 = new BandwidthReductionPostFilterSharpening('SHARPENING_1');
  /** Sharpening level 2 */
  public static readonly SHARPENING_2 = new BandwidthReductionPostFilterSharpening('SHARPENING_2');
  /** Sharpening level 3 */
  public static readonly SHARPENING_3 = new BandwidthReductionPostFilterSharpening('SHARPENING_3');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): BandwidthReductionPostFilterSharpening {
    return new BandwidthReductionPostFilterSharpening(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Bandwidth reduction filter strength.
 */
export class BandwidthReductionStrength {
  /** Auto */
  public static readonly AUTO = new BandwidthReductionStrength('AUTO');
  /** Strength 1 */
  public static readonly STRENGTH_1 = new BandwidthReductionStrength('STRENGTH_1');
  /** Strength 2 */
  public static readonly STRENGTH_2 = new BandwidthReductionStrength('STRENGTH_2');
  /** Strength 3 */
  public static readonly STRENGTH_3 = new BandwidthReductionStrength('STRENGTH_3');
  /** Strength 4 */
  public static readonly STRENGTH_4 = new BandwidthReductionStrength('STRENGTH_4');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): BandwidthReductionStrength {
    return new BandwidthReductionStrength(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for a temporal filter.
 */
export interface TemporalFilterProps {
  /**
   * Post-filter sharpening control.
   *
   * @default - service default
   */
  readonly postFilterSharpening?: TemporalFilterPostFilterSharpening;
  /**
   * Filter strength. We recommend 1 or 2. Higher values may remove useful detail.
   *
   * @default - service default
   */
  readonly strength?: TemporalFilterStrength;
}

/**
 * Properties for a bandwidth reduction filter.
 */
export interface BandwidthReductionFilterProps {
  /**
   * Post-filter sharpening control.
   *
   * @default - service default
   */
  readonly postFilterSharpening?: BandwidthReductionPostFilterSharpening;
  /**
   * Bandwidth reduction strength.
   *
   * @default - service default
   */
  readonly strength?: BandwidthReductionStrength;
}

/**
 * Filter settings for H.264 video. Supports temporal filter and bandwidth reduction filter.
 */
export class H264FilterSettings {
  /** Apply a temporal filter. */
  public static temporalFilter(props?: TemporalFilterProps): H264FilterSettings {
    return new H264FilterSettings({
      temporalFilterSettings: props ? {
        postFilterSharpening: props.postFilterSharpening?.value,
        strength: props.strength?.value,
      } : {},
    });
  }
  /** Apply a bandwidth reduction filter. */
  public static bandwidthReductionFilter(props?: BandwidthReductionFilterProps): H264FilterSettings {
    return new H264FilterSettings({
      bandwidthReductionFilterSettings: props ? {
        postFilterSharpening: props.postFilterSharpening?.value,
        strength: props.strength?.value,
      } : {},
    });
  }

  private readonly config: CfnChannel.H264FilterSettingsProperty;
  private constructor(config: CfnChannel.H264FilterSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.H264FilterSettingsProperty { return this.config; }
}

/**
 * Filter settings for H.265 video. Supports temporal filter and bandwidth reduction filter.
 */
export class H265FilterSettings {
  /** Apply a temporal filter. */
  public static temporalFilter(props?: TemporalFilterProps): H265FilterSettings {
    return new H265FilterSettings({
      temporalFilterSettings: props ? {
        postFilterSharpening: props.postFilterSharpening?.value,
        strength: props.strength?.value,
      } : {},
    });
  }
  /** Apply a bandwidth reduction filter. */
  public static bandwidthReductionFilter(props?: BandwidthReductionFilterProps): H265FilterSettings {
    return new H265FilterSettings({
      bandwidthReductionFilterSettings: props ? {
        postFilterSharpening: props.postFilterSharpening?.value,
        strength: props.strength?.value,
      } : {},
    });
  }

  private readonly config: CfnChannel.H265FilterSettingsProperty;
  private constructor(config: CfnChannel.H265FilterSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.H265FilterSettingsProperty { return this.config; }
}

// =============================================================================
// Codec settings props
// =============================================================================

/**
 * Properties for H.264 codec settings.
 */
export interface H264SettingsProps {
  /**
   * The rate control configuration.
   * @default - CBR with no bitrate (service default)
   */
  readonly rateControl?: H264RateControl;
  /**
   * The H.264 profile.
   * @default H264Profile.MAIN
   */
  readonly profile?: H264Profile;
  /**
   * The GOP size (keyframe interval).
   * @default GopSize.seconds(1)
   */
  readonly gopSize?: GopSize;
  /**
   * The number of B-frames between reference frames.
   * @default - service default
   */
  readonly gopNumBFrames?: number;
  /**
   * The adaptive quantization. This allows intra-frame quantizers to vary to improve visual quality.
   * @default H264AdaptiveQuantization.AUTO
   */
  readonly adaptiveQuantization?: H264AdaptiveQuantization;
  /**
   * The video frame rate.
   * @default - follow source
   */
  readonly framerate?: Framerate;
  /**
   * The pixel aspect ratio (PAR) of the video.
   * @default - follow source (or square pixels when framerate is specified)
   */
  readonly pixelAspectRatio?: PixelAspectRatio;
  /**
   * Timecode burn-in settings to overlay timecode on the video.
   * @default - no timecode burn-in
   */
  readonly timecodeBurnin?: TimecodeBurninSettings;
  /**
   * Indicates that AFD values will be written into the output stream. If afdSignaling is auto, the
   * system tries to preserve the input AFD value (in cases where multiple AFD values are valid). If
   * set to fixed, the AFD value is the value configured in the fixedAfd parameter.
   * @default AfdSignaling.NONE
   */
  readonly afdSignaling?: AfdSignaling;
  /**
   * Percentage of the buffer that should initially be filled (HRD buffer model).
   * @default - service default
   */
  readonly bufFillPct?: number;
  /**
   * Size of the buffer (HRD buffer model) in bits/second.
   * @default - service default
   */
  readonly bufSize?: number;
  /**
   * Whether to include color space metadata in the output.
   * @default - service default
   */
  readonly colorMetadata?: ColorMetadata;
  /**
   * Color space settings for the video.
   *
   * @default - service default
   */
  readonly colorSpaceSettings?: H264ColorSpaceSettings;
  /**
   * The entropy encoding mode. CABAC requires Main or High profile.
   * @default - service default
   */
  readonly entropyEncoding?: H264EntropyEncoding;
  /**
   * Optional video filter settings.
   *
   * @default - service default
   */
  readonly filterSettings?: H264FilterSettings;
  /**
   * Four-bit AFD value to write on all frames. Only valid when afdSignaling is FIXED.
   *
   * Valid values: FIXED_0000, FIXED_0010, FIXED_0011, FIXED_0100, FIXED_1000,
   * FIXED_1001, FIXED_1010, FIXED_1011, FIXED_1100, FIXED_1101, FIXED_1110, FIXED_1111.
   *
   * @default - service default
   */
  readonly fixedAfd?: string;
  /**
   * If enabled, adjusts quantization within each frame to reduce flicker on I-frames.
   * @default FlickerAq.ENABLED
   */
  readonly flickerAq?: FlickerAq;
  /**
   * Controls whether coding is on a field basis or frame basis when scan type is interlaced.
   * @default - service default
   */
  readonly forceFieldPictures?: H264ForceFieldPictures;
  /**
   * If enabled, uses reference B frames for GOP structures that have B frames > 1.
   * @default - service default
   */
  readonly gopBReference?: GopBReference;
  /**
   * Frequency of closed GOPs. Set to 1 for streaming so decoders joining mid-stream get an IDR frame quickly.
   * @default - service default
   */
  readonly gopClosedCadence?: number;
  /**
   * The H.264 level.
   *
   * Valid values: H264_LEVEL_1, H264_LEVEL_1_1, H264_LEVEL_1_2, H264_LEVEL_1_3,
   * H264_LEVEL_2, H264_LEVEL_2_1, H264_LEVEL_2_2, H264_LEVEL_3, H264_LEVEL_3_1,
   * H264_LEVEL_3_2, H264_LEVEL_4, H264_LEVEL_4_1, H264_LEVEL_4_2, H264_LEVEL_5,
   * H264_LEVEL_5_1, H264_LEVEL_5_2, H264_LEVEL_AUTO.
   *
   * @default H264Level.H264_LEVEL_AUTO
   */
  readonly level?: H264Level;
  /**
   * Amount of lookahead. Low decreases latency/memory; high can produce better quality.
   * @default LookAheadRateControl.HIGH
   */
  readonly lookAheadRateControl?: LookAheadRateControl;
  /**
   * Only meaningful if sceneChangeDetect is enabled. Enforces separation between
   * repeated (cadence) I-frames and I-frames inserted by scene change detection.
   * @default - service default
   */
  readonly minIInterval?: number;
  /**
   * The number of reference frames to use. The encoder might use more if B-frames or interlaced encoding is used.
   * @default - service default
   */
  readonly numRefFrames?: number;
  /**
   * Sets the scan type of the output.
   * @default ScanType.PROGRESSIVE
   */
  readonly scanType?: ScanType;
  /**
   * Number of slices per picture. Must be <= macroblock rows (progressive) or half (interlaced).
   * @default - encoder chooses based on resolution
   */
  readonly slices?: number;
  /**
   * Softness. Selects a quantizer matrix; larger values reduce high-frequency content.
   * @default - service default
   */
  readonly softness?: number;
  /**
   * Produces a bitstream compliant with SMPTE RP-2027.
   * @default H264Syntax.DEFAULT
   */
  readonly syntax?: H264Syntax;
  /**
   * Determines how timecodes are inserted into the video elementary stream.
   * This controls insertion into the output elementary stream. The channel's `timecodeConfig` controls the
   * source of the timecode used for output.
   * @default - service default
   */
  readonly timecodeInsertion?: TimecodeInsertion;
  /**
   * Minimum QP value. Sets a floor on the quantization parameter.
   * @default - service default
   */
  readonly minQp?: number;
  /**
   * Minimum bitrate in bits/second.
   * @default - service default
   */
  readonly minBitrate?: number;
  /**
   * Quality level. ENHANCED_QUALITY produces slightly better video without increasing bitrate.
   * @default - service default
   */
  readonly qualityLevel?: H264QualityLevel;
  /**
   * Whether scene change detection inserts I-frames on scene changes.
   * @default H264SceneChangeDetect.ENABLED
   */
  readonly sceneChangeDetect?: H264SceneChangeDetect;
  /**
   * Whether spatial adaptive quantization adjusts quantization within each frame based on spatial variation.
   * @default H264SpatialAq.ENABLED
   */
  readonly spatialAq?: H264SpatialAq;
  /**
   * Whether temporal adaptive quantization adjusts quantization based on temporal variation between frames.
   * @default H264TemporalAq.ENABLED
   */
  readonly temporalAq?: H264TemporalAq;
  /**
   * Sub-GOP length mode.
   * @default - service default
   */
  readonly subgopLength?: SubgopLength;
}

/**
 * H.265 profile.
 */
export class H265Profile {
  /** Main profile */
  public static readonly MAIN = new H265Profile('MAIN');
  /** Main 10-bit profile */
  public static readonly MAIN_10BIT = new H265Profile('MAIN_10BIT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265Profile {
    return new H265Profile(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * H.265 tier.
 */
export class H265Tier {
  /** Main tier */
  public static readonly MAIN = new H265Tier('MAIN');
  /** High tier */
  public static readonly HIGH = new H265Tier('HIGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): H265Tier {
    return new H265Tier(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for H.265 codec settings.
 */
export interface H265SettingsProps {
  /**
   * The rate control configuration.
   * @default - CBR with no bitrate (service default)
   */
  readonly rateControl?: H265RateControl;
  /**
   * The H.265 profile.
   * @default H265Profile.MAIN
   */
  readonly profile?: H265Profile;
  /**
   * The H.265 tier.
   * @default H265Tier.MAIN
   */
  readonly tier?: H265Tier;
  /**
   * The GOP size (keyframe interval).
   * @default GopSize.seconds(1)
   */
  readonly gopSize?: GopSize;
  /**
   * The video frame rate. Required for H.265.
   */
  readonly framerate: Framerate;
  /**
   * The pixel aspect ratio (PAR) of the video.
   * @default - square pixels
   */
  readonly pixelAspectRatio?: PixelAspectRatio;
  /**
   * Timecode burn-in settings to overlay timecode on the video.
   * @default - no timecode burn-in
   */
  readonly timecodeBurnin?: TimecodeBurninSettings;
  /**
   * The adaptive quantization. Allows intra-frame quantizers to vary to improve visual quality.
   * @default H265AdaptiveQuantization.AUTO
   */
  readonly adaptiveQuantization?: H265AdaptiveQuantization;
  /**
   * AFD signaling mode.
   * @default AfdSignaling.NONE
   */
  readonly afdSignaling?: AfdSignaling;
  /**
   * Whether to insert an Alternative Transfer Function SEI message for backwards compatibility with non-HDR decoders.
   * @default - service default
   */
  readonly alternativeTransferFunction?: H265AlternativeTransferFunction;
  /**
   * Size of buffer (HRD buffer model) in bits.
   * @default - service default
   */
  readonly bufSize?: number;
  /**
   * Whether to include color space metadata in the output.
   * @default - service default
   */
  readonly colorMetadata?: ColorMetadata;
  /**
   * Color space settings for the video.
   *
   * @default - service default
   */
  readonly colorSpaceSettings?: H265ColorSpaceSettings;
  /**
   * Deblocking filter control.
   * @default - service default
   */
  readonly deblocking?: H265Deblocking;
  /**
   * Optional video filter settings.
   *
   * @default - service default
   */
  readonly filterSettings?: H265FilterSettings;
  /**
   * Four-bit AFD value to write on all frames. Only valid when afdSignaling is FIXED.
   *
   * Valid values: FIXED_0000, FIXED_0010, FIXED_0011, FIXED_0100, FIXED_1000,
   * FIXED_1001, FIXED_1010, FIXED_1011, FIXED_1100, FIXED_1101, FIXED_1110, FIXED_1111.
   *
   * @default - service default
   */
  readonly fixedAfd?: string;
  /**
   * If enabled, adjusts quantization within each frame to reduce flicker on I-frames.
   * @default - service default
   */
  readonly flickerAq?: FlickerAq;
  /**
   * If enabled, uses reference B frames for GOP structures that have B frames > 1.
   * @default - service default
   */
  readonly gopBReference?: GopBReference;
  /**
   * Frequency of closed GOPs. Set to 1 for streaming so decoders joining mid-stream get an IDR frame quickly.
   * @default - service default
   */
  readonly gopClosedCadence?: number;
  /**
   * Number of B-frames between reference frames.
   * @default - service default
   */
  readonly gopNumBFrames?: number;
  /**
   * The H.265 level.
   *
   * Valid values: H265_LEVEL_1, H265_LEVEL_2, H265_LEVEL_2_1, H265_LEVEL_3,
   * H265_LEVEL_3_1, H265_LEVEL_4, H265_LEVEL_4_1, H265_LEVEL_5, H265_LEVEL_5_1,
   * H265_LEVEL_5_2, H265_LEVEL_6, H265_LEVEL_6_1, H265_LEVEL_6_2, H265_LEVEL_AUTO.
   *
   * @default - service default (auto)
   */
  readonly level?: H265Level;
  /**
   * Amount of lookahead. Low decreases latency/memory; high can produce better quality.
   * @default - service default
   */
  readonly lookAheadRateControl?: LookAheadRateControl;
  /**
   * Only meaningful if sceneChangeDetect is enabled. Enforces separation between
   * repeated (cadence) I-frames and I-frames inserted by scene change detection.
   * @default - service default
   */
  readonly minIInterval?: number;
  /**
   * Sets the scan type of the output.
   * @default ScanType.PROGRESSIVE
   */
  readonly scanType?: ScanType;
  /**
   * Number of slices per picture.
   * @default - encoder chooses based on resolution
   */
  readonly slices?: number;
  /**
   * Determines how timecodes are inserted into the video elementary stream.
   * This controls insertion into the output elementary stream. The channel's `timecodeConfig` controls the
   * source of the timecode used for output.
   * @default - service default
   */
  readonly timecodeInsertion?: TimecodeInsertion;
  /**
   * Minimum QP value.
   * @default - service default
   */
  readonly minQp?: number;
  /**
   * Minimum bitrate in bits/second.
   * @default - service default
   */
  readonly minBitrate?: number;
  /**
   * Whether motion vectors can cross picture boundaries.
   * @default - service default
   */
  readonly mvOverPictureBoundaries?: H265MvOverPictureBoundaries;
  /**
   * Whether to use temporal motion vector prediction.
   * @default - service default
   */
  readonly mvTemporalPredictor?: H265MvTemporalPredictor;
  /**
   * Sub-GOP length mode.
   * @default - service default
   */
  readonly subgopLength?: SubgopLength;
  /**
   * Tile height in pixels. Must be a multiple of the CTU size.
   * @default - service default
   */
  readonly tileHeight?: number;
  /**
   * Tile padding mode.
   * @default - service default
   */
  readonly tilePadding?: H265TilePadding;
  /**
   * Tile width in pixels. Must be a multiple of the CTU size.
   * @default - service default
   */
  readonly tileWidth?: number;
  /**
   * Treeblock size for the encoder.
   * @default - service default
   */
  readonly treeblockSize?: H265TreeblockSize;
  /**
   * Whether scene change detection inserts I-frames on scene changes.
   * @default H265SceneChangeDetect.ENABLED
   */
  readonly sceneChangeDetect?: H265SceneChangeDetect;
}

/**
 * Properties for AV1 codec settings.
 */
export interface Av1SettingsProps {
  /**
   * The rate control configuration.
   * @default - service default
   */
  readonly rateControl?: Av1RateControl;
  /**
   * The GOP size (keyframe interval).
   * @default GopSize.seconds(1)
   */
  readonly gopSize?: GopSize;
  /**
   * The video frame rate.
   * @default - follow source
   */
  readonly framerate?: Framerate;
  /**
   * Timecode burn-in settings to overlay timecode on the video.
   * @default - no timecode burn-in
   */
  readonly timecodeBurnin?: TimecodeBurninSettings;
  /**
   * AFD signaling mode.
   * @default AfdSignaling.NONE
   */
  readonly afdSignaling?: AfdSignaling;
  /**
   * Bit depth for the AV1 encode.
   * @default - service default
   */
  readonly bitDepth?: Av1BitDepth;
  /**
   * Size of buffer (HRD buffer model) in bits.
   * @default - service default
   */
  readonly bufSize?: number;
  /**
   * Color space settings for the video.
   *
   * @default - service default
   */
  readonly colorSpaceSettings?: Av1ColorSpaceSettings;
  /**
   * Four-bit AFD value to write on all frames. Only valid when afdSignaling is FIXED.
   *
   * Valid values: FIXED_0000, FIXED_0010, FIXED_0011, FIXED_0100, FIXED_1000,
   * FIXED_1001, FIXED_1010, FIXED_1011, FIXED_1100, FIXED_1101, FIXED_1110, FIXED_1111.
   *
   * @default - service default
   */
  readonly fixedAfd?: string;
  /**
   * The AV1 level.
   *
   * Valid values: AV1_LEVEL_2, AV1_LEVEL_2_1, AV1_LEVEL_3, AV1_LEVEL_3_1,
   * AV1_LEVEL_4, AV1_LEVEL_4_1, AV1_LEVEL_5, AV1_LEVEL_5_1, AV1_LEVEL_5_2,
   * AV1_LEVEL_5_3, AV1_LEVEL_6, AV1_LEVEL_6_1, AV1_LEVEL_6_2, AV1_LEVEL_6_3,
   * AV1_LEVEL_AUTO.
   *
   * @default Av1Level.AV1_LEVEL_AUTO
   */
  readonly level?: Av1Level;
  /**
   * Amount of lookahead. Low decreases latency/memory; high can produce better quality.
   * @default LookAheadRateControl.HIGH
   */
  readonly lookAheadRateControl?: LookAheadRateControl;
  /**
   * Minimum bitrate in bits/second.
   * @default - service default
   */
  readonly minBitrate?: number;
  /**
   * Only meaningful if sceneChangeDetect is enabled. Enforces separation between
   * repeated (cadence) I-frames and I-frames inserted by scene change detection.
   * @default - service default
   */
  readonly minIInterval?: number;
  /**
   * The pixel aspect ratio (PAR) of the video.
   * @default - service default
   */
  readonly pixelAspectRatio?: PixelAspectRatio;
  /**
   * Scene change detection.
   * @default Av1SceneChangeDetect.ENABLED
   */
  readonly sceneChangeDetect?: Av1SceneChangeDetect;
  /**
   * Spatial adaptive quantization.
   * @default Av1SpatialAq.ENABLED
   */
  readonly spatialAq?: Av1SpatialAq;
  /**
   * Temporal adaptive quantization.
   * @default Av1TemporalAq.ENABLED
   */
  readonly temporalAq?: Av1TemporalAq;
  /**
   * Timecode insertion mode.
   * @default - service default
   */
  readonly timecodeInsertion?: Av1TimecodeInsertion;
}

/**
 * Properties for frame capture codec settings.
 */
export interface FrameCaptureSettingsProps {
  /**
   * The interval between frame captures.
   * @default - service default
   */
  readonly captureInterval?: Duration;
  /**
   * Timecode burn-in settings to overlay timecode on the video.
   * @default - no timecode burn-in
   */
  readonly timecodeBurnin?: TimecodeBurninSettings;
}

// =============================================================================
// VideoCodecSettings (abstract + subclasses)
// =============================================================================

/**
 * The type of video codec. Users select a codec via the
 * `VideoCodecSettings` factory methods, never by passing this type directly.
 * @internal
 */
export enum VideoCodecType {
  /** H.264 (AVC) */
  H264 = 'H264',
  /** H.265 (HEVC) */
  H265 = 'H265',
  /** AV1 */
  AV1 = 'AV1',
  /** Frame Capture (JPEG) */
  FRAME_CAPTURE = 'FRAME_CAPTURE',
}

/**
 * Video codec settings. Use the static factory methods to create.
 */
export abstract class VideoCodecSettings {
  /** Create H.264 (AVC) codec settings. */
  public static h264(props?: H264SettingsProps): VideoCodecSettings {
    return new H264VideoCodecSettings(props ?? {});
  }
  /** Create H.265 (HEVC) codec settings. Framerate is required for H.265. */
  public static h265(props: H265SettingsProps): VideoCodecSettings {
    return new H265VideoCodecSettings(props);
  }
  /** Create AV1 codec settings. */
  public static av1(props?: Av1SettingsProps): VideoCodecSettings {
    return new Av1VideoCodecSettings(props ?? {});
  }
  /** Create frame capture codec settings. */
  public static frameCapture(props?: FrameCaptureSettingsProps): VideoCodecSettings {
    return new FrameCaptureVideoCodecSettings(props ?? {});
  }

  /** @internal */
  public abstract readonly _codecType: VideoCodecType;
  /** @internal */
  public abstract _bind(): CfnChannel.VideoCodecSettingsProperty;
  /** @internal */
  public abstract _hasExplicitFramerate(): boolean;
}

/** @internal */
class H264VideoCodecSettings extends VideoCodecSettings {
  public readonly _codecType = VideoCodecType.H264;
  constructor(private readonly props: H264SettingsProps) { super(); }

  public _hasExplicitFramerate(): boolean {
    return this.props.framerate != null;
  }

  public _bind(): CfnChannel.VideoCodecSettingsProperty {
    const p = this.props;
    const rc = p.rateControl;
    return {
      h264Settings: {
        bitrate: rc?._bitrate?.toBps(),
        maxBitrate: rc?._maxBitrate?.toBps(),
        rateControlMode: rc?._mode,
        qvbrQualityLevel: rc?._qvbrQualityLevel,
        profile: (p.profile ?? H264Profile.MAIN).value,
        gopSize: p.gopSize?._value ?? 1,
        gopSizeUnits: p.gopSize?._units ?? 'SECONDS',
        gopNumBFrames: p.gopNumBFrames,
        adaptiveQuantization: (p.adaptiveQuantization ?? H264AdaptiveQuantization.AUTO).value,
        framerateControl: p.framerate ? 'SPECIFIED' : 'INITIALIZE_FROM_SOURCE',
        framerateNumerator: p.framerate?._numerator(),
        framerateDenominator: p.framerate?._denominator(),
        parControl: (p.pixelAspectRatio || p.framerate) ? 'SPECIFIED' : 'INITIALIZE_FROM_SOURCE',
        parNumerator: p.pixelAspectRatio?._numerator() ?? (p.framerate ? PixelAspectRatio.SQUARE._numerator() : undefined),
        parDenominator: p.pixelAspectRatio?._denominator() ?? (p.framerate ? PixelAspectRatio.SQUARE._denominator() : undefined),
        sceneChangeDetect: (p.sceneChangeDetect ?? H264SceneChangeDetect.ENABLED).value,
        spatialAq: (p.spatialAq ?? H264SpatialAq.ENABLED).value,
        temporalAq: (p.temporalAq ?? H264TemporalAq.ENABLED).value,
        timecodeBurninSettings: p.timecodeBurnin ? {
          fontSize: p.timecodeBurnin.fontSize?.value,
          position: p.timecodeBurnin.position?.value,
          prefix: p.timecodeBurnin.prefix,
        } : undefined,
        afdSignaling: (p.afdSignaling ?? AfdSignaling.NONE).value,
        bufFillPct: p.bufFillPct,
        bufSize: p.bufSize,
        colorMetadata: p.colorMetadata?.value,
        colorSpaceSettings: p.colorSpaceSettings?._bind(),
        entropyEncoding: p.entropyEncoding?.value,
        filterSettings: p.filterSettings?._bind(),
        fixedAfd: p.fixedAfd,
        flickerAq: (p.flickerAq ?? FlickerAq.ENABLED).value,
        forceFieldPictures: p.forceFieldPictures?.value,
        gopBReference: p.gopBReference?.value,
        gopClosedCadence: p.gopClosedCadence,
        level: (p.level ?? H264Level.H264_LEVEL_AUTO).value,
        lookAheadRateControl: (p.lookAheadRateControl ?? LookAheadRateControl.HIGH).value,
        minIInterval: p.minIInterval,
        numRefFrames: p.numRefFrames,
        scanType: (p.scanType ?? ScanType.PROGRESSIVE).value,
        slices: p.slices,
        softness: p.softness,
        syntax: (p.syntax ?? H264Syntax.DEFAULT).value,
        timecodeInsertion: p.timecodeInsertion?.value,
        minQp: p.minQp,
        minBitrate: p.minBitrate,
        qualityLevel: p.qualityLevel?.value,
        subgopLength: p.subgopLength?.value,
      },
    };
  }
}

/** @internal */
class H265VideoCodecSettings extends VideoCodecSettings {
  public readonly _codecType = VideoCodecType.H265;
  constructor(private readonly props: H265SettingsProps) { super(); }

  public _hasExplicitFramerate(): boolean {
    return this.props.framerate != null;
  }

  public _bind(): CfnChannel.VideoCodecSettingsProperty {
    const p = this.props;
    const rc = p.rateControl;
    return {
      h265Settings: {
        bitrate: rc?._bitrate?.toBps(),
        maxBitrate: rc?._maxBitrate?.toBps(),
        rateControlMode: rc?._mode,
        qvbrQualityLevel: rc?._qvbrQualityLevel,
        profile: (p.profile ?? H265Profile.MAIN).value,
        tier: (p.tier ?? H265Tier.MAIN).value,
        level: p.level?.value,
        gopSize: p.gopSize?._value ?? 1,
        gopSizeUnits: p.gopSize?._units ?? 'SECONDS',
        framerateNumerator: p.framerate._numerator(),
        framerateDenominator: p.framerate._denominator(),
        parNumerator: p.pixelAspectRatio?._numerator() ?? PixelAspectRatio.SQUARE._numerator(),
        parDenominator: p.pixelAspectRatio?._denominator() ?? PixelAspectRatio.SQUARE._denominator(),
        sceneChangeDetect: (p.sceneChangeDetect ?? H265SceneChangeDetect.ENABLED).value,
        timecodeBurninSettings: p.timecodeBurnin ? {
          fontSize: p.timecodeBurnin.fontSize?.value,
          position: p.timecodeBurnin.position?.value,
          prefix: p.timecodeBurnin.prefix,
        } : undefined,
        adaptiveQuantization: (p.adaptiveQuantization ?? H265AdaptiveQuantization.AUTO).value,
        afdSignaling: (p.afdSignaling ?? AfdSignaling.NONE).value,
        alternativeTransferFunction: p.alternativeTransferFunction?.value,
        bufSize: p.bufSize,
        colorMetadata: p.colorMetadata?.value,
        colorSpaceSettings: p.colorSpaceSettings?._bind(),
        deblocking: p.deblocking?.value,
        filterSettings: p.filterSettings?._bind(),
        fixedAfd: p.fixedAfd,
        flickerAq: p.flickerAq?.value,
        gopBReference: p.gopBReference?.value,
        gopClosedCadence: p.gopClosedCadence,
        gopNumBFrames: p.gopNumBFrames,
        lookAheadRateControl: p.lookAheadRateControl?.value,
        minIInterval: p.minIInterval,
        scanType: (p.scanType ?? ScanType.PROGRESSIVE).value,
        slices: p.slices,
        timecodeInsertion: p.timecodeInsertion?.value,
        minQp: p.minQp,
        minBitrate: p.minBitrate,
        mvOverPictureBoundaries: p.mvOverPictureBoundaries?.value,
        mvTemporalPredictor: p.mvTemporalPredictor?.value,
        subgopLength: p.subgopLength?.value,
        tileHeight: p.tileHeight,
        tilePadding: p.tilePadding?.value,
        tileWidth: p.tileWidth,
        treeblockSize: p.treeblockSize?.value,
      },
    };
  }
}

/** @internal */
class Av1VideoCodecSettings extends VideoCodecSettings {
  public readonly _codecType = VideoCodecType.AV1;
  constructor(private readonly props: Av1SettingsProps) { super(); }

  public _hasExplicitFramerate(): boolean {
    return this.props.framerate != null;
  }

  public _bind(): CfnChannel.VideoCodecSettingsProperty {
    const p = this.props;
    const rc = p.rateControl;
    return {
      av1Settings: {
        bitrate: rc?._bitrate?.toBps(),
        maxBitrate: rc?._maxBitrate?.toBps(),
        rateControlMode: rc?._mode,
        qvbrQualityLevel: rc?._qvbrQualityLevel,
        gopSize: p.gopSize?._value ?? 1,
        gopSizeUnits: p.gopSize?._units ?? 'SECONDS',
        framerateNumerator: p.framerate?._numerator(),
        framerateDenominator: p.framerate?._denominator(),
        timecodeBurninSettings: p.timecodeBurnin ? {
          fontSize: p.timecodeBurnin.fontSize?.value,
          position: p.timecodeBurnin.position?.value,
          prefix: p.timecodeBurnin.prefix,
        } : undefined,
        afdSignaling: (p.afdSignaling ?? AfdSignaling.NONE).value,
        bitDepth: p.bitDepth?.value,
        bufSize: p.bufSize,
        colorSpaceSettings: p.colorSpaceSettings?._bind(),
        fixedAfd: p.fixedAfd,
        level: (p.level ?? Av1Level.AV1_LEVEL_AUTO).value,
        minBitrate: p.minBitrate,
        minIInterval: p.minIInterval,
        parNumerator: p.pixelAspectRatio?._numerator(),
        parDenominator: p.pixelAspectRatio?._denominator(),
        sceneChangeDetect: (p.sceneChangeDetect ?? Av1SceneChangeDetect.ENABLED).value,
        spatialAq: (p.spatialAq ?? Av1SpatialAq.ENABLED).value,
        temporalAq: (p.temporalAq ?? Av1TemporalAq.ENABLED).value,
        lookAheadRateControl: (p.lookAheadRateControl ?? LookAheadRateControl.HIGH).value,
        timecodeInsertion: p.timecodeInsertion?.value,
      },
    };
  }
}

/** @internal */
class FrameCaptureVideoCodecSettings extends VideoCodecSettings {
  public readonly _codecType = VideoCodecType.FRAME_CAPTURE;
  constructor(private readonly props: FrameCaptureSettingsProps) { super(); }

  public _hasExplicitFramerate(): boolean { return true; }

  public _bind(): CfnChannel.VideoCodecSettingsProperty {
    const p = this.props;
    // Only emit an interval when set. Whole-second Durations render as SECONDS, sub-second as MILLISECONDS.
    const intervalMs = p.captureInterval?.toMilliseconds();
    const wholeSeconds = intervalMs !== undefined && intervalMs % 1000 === 0;
    return {
      frameCaptureSettings: {
        captureInterval: intervalMs === undefined ? undefined : (wholeSeconds ? intervalMs / 1000 : intervalMs),
        captureIntervalUnits: intervalMs === undefined ? undefined : (wholeSeconds ? 'SECONDS' : 'MILLISECONDS'),
        timecodeBurninSettings: p.timecodeBurnin ? {
          fontSize: p.timecodeBurnin.fontSize?.value,
          position: p.timecodeBurnin.position?.value,
          prefix: p.timecodeBurnin.prefix,
        } : undefined,
      },
    };
  }
}

