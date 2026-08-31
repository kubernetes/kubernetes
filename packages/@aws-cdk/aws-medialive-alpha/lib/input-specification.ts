import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';

/**
 * The codec for the input specification.
 */
export class InputCodec {
  /** AVC (H.264) */
  public static readonly AVC = new InputCodec('AVC');
  /** HEVC (H.265) */
  public static readonly HEVC = new InputCodec('HEVC');
  /** MPEG2 */
  public static readonly MPEG2 = new InputCodec('MPEG2');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputCodec {
    return new InputCodec(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * The maximum input bitrate for the input specification.
 */
export class InputMaximumBitrate {
  /** Max 10 Mbps */
  public static readonly MAX_10_MBPS = new InputMaximumBitrate('MAX_10_MBPS');
  /** Max 20 Mbps */
  public static readonly MAX_20_MBPS = new InputMaximumBitrate('MAX_20_MBPS');
  /** Max 50 Mbps */
  public static readonly MAX_50_MBPS = new InputMaximumBitrate('MAX_50_MBPS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputMaximumBitrate {
    return new InputMaximumBitrate(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * The resolution for the input specification.
 */
export class InputResolution {
  /** SD */
  public static readonly SD = new InputResolution('SD');
  /** HD */
  public static readonly HD = new InputResolution('HD');
  /** UHD */
  public static readonly UHD = new InputResolution('UHD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputResolution {
    return new InputResolution(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Maximum CDI input resolution.
 */
export class CdiInputResolution {
  /** SD resolution */
  public static readonly SD = new CdiInputResolution('SD');
  /** HD resolution */
  public static readonly HD = new CdiInputResolution('HD');
  /** Full HD resolution */
  public static readonly FHD = new CdiInputResolution('FHD');
  /** UHD resolution */
  public static readonly UHD = new CdiInputResolution('UHD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CdiInputResolution {
    return new CdiInputResolution(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties shared by all input specifications.
 */
export interface StandardInputSpecificationProps {
  /**
   * The codec of the input.
   * This should match the codec of your source content, not the output codec.
   * @default InputCodec.AVC
   */
  readonly codec?: InputCodec;
  /**
   * The maximum bitrate of the input.
   * @default InputMaximumBitrate.MAX_20_MBPS
   */
  readonly maximumBitrate?: InputMaximumBitrate;
  /**
   * The resolution of the input.
   * @default InputResolution.HD
   */
  readonly resolution?: InputResolution;
}

/**
 * Properties for a CDI input specification.
 */
export interface CdiInputSpecificationProps extends StandardInputSpecificationProps {
  /**
   * The maximum resolution of the most demanding CDI input.
   * @default CdiInputResolution.HD
   */
  readonly cdiResolution?: CdiInputResolution;
}

/**
 * The input specification for a channel.
 *
 * Use the static factory methods to select the input type — mirroring the console's
 * "Other" / "CDI" / "Elemental Link" choice.
 */
export abstract class InputSpecification {
  /** Standard inputs ("Other" in the console) — the most common case. */
  public static standard(props: StandardInputSpecificationProps = {}): InputSpecification {
    return new StandardInputSpecification(props);
  }

  /** CDI (uncompressed) inputs. Adds the maximum CDI input resolution. */
  public static cdi(props: CdiInputSpecificationProps = {}): InputSpecification {
    return new CdiInputSpecification(props);
  }

  /** Elemental Link inputs. No additional specification is required. */
  public static elementalLink(): InputSpecification {
    return new ElementalLinkInputSpecification();
  }

  /** @internal */
  public abstract _bindInputSpecification(): CfnChannel.InputSpecificationProperty | undefined;
  /** @internal */
  public abstract _bindCdiInputSpecification(): CfnChannel.CdiInputSpecificationProperty | undefined;
}

/** @internal */
function bindStandardSpec(props: StandardInputSpecificationProps): CfnChannel.InputSpecificationProperty {
  return {
    codec: (props.codec ?? InputCodec.AVC).value,
    maximumBitrate: (props.maximumBitrate ?? InputMaximumBitrate.MAX_20_MBPS).value,
    resolution: (props.resolution ?? InputResolution.HD).value,
  };
}

/** @internal */
class StandardInputSpecification extends InputSpecification {
  constructor(private readonly props: StandardInputSpecificationProps) { super(); }
  public _bindInputSpecification(): CfnChannel.InputSpecificationProperty {
    return bindStandardSpec(this.props);
  }
  public _bindCdiInputSpecification(): undefined {
    return undefined;
  }
}

/** @internal */
class CdiInputSpecification extends InputSpecification {
  constructor(private readonly props: CdiInputSpecificationProps) { super(); }
  public _bindInputSpecification(): CfnChannel.InputSpecificationProperty {
    return bindStandardSpec(this.props);
  }
  public _bindCdiInputSpecification(): CfnChannel.CdiInputSpecificationProperty {
    return { resolution: (this.props.cdiResolution ?? CdiInputResolution.HD).value };
  }
}

/** @internal */
class ElementalLinkInputSpecification extends InputSpecification {
  public _bindInputSpecification(): undefined {
    return undefined;
  }
  public _bindCdiInputSpecification(): undefined {
    return undefined;
  }
}
