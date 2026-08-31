import type { IResource } from 'aws-cdk-lib';
import { Resource, Lazy, Names, ValidationError } from 'aws-cdk-lib';
import type { ISdiSourceRef, SdiSourceReference } from 'aws-cdk-lib/aws-medialive';
import { CfnSdiSource } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';

/**
 * Represents a MediaLive SDI Source.
 */
export interface ISdiSource extends IResource, ISdiSourceRef {
  /**
   * The SDI Source ARN.
   * @attribute
   */
  readonly sdiSourceArn: string;

  /**
   * The SDI Source ID.
   * @attribute
   */
  readonly sdiSourceId: string;

  /**
   * The list of inputs currently using this SDI source.
   * @attribute
   */
  readonly sdiSourceInputs?: string[];

  /**
   * The current state of the SDI source.
   * @attribute
   */
  readonly sdiSourceState?: string;
}

/**
 * Properties for creating an SDI Source.
 */
export interface SdiSourceProps {
  /**
   * The name of the SDI source.
   * @default - auto-generated name
   */
  readonly sdiSourceName?: string;

  /**
   * Type of SDI input.
   */
  readonly type: SdiType;

  /**
   * SDI Mode, only applicable for QUAD type.
   * @default - no mode
   */
  readonly mode?: SdiMode;

  /**
   * Tags to add to the SDI source.
   * @default - no tagging
   */
  readonly tags?: { [key: string]: string };
}

/**
 * The type of SDI input.
 */
export class SdiType {
  /** Single SDI input */
  public static readonly SINGLE = new SdiType('SINGLE');
  /** Quad SDI input */
  public static readonly QUAD = new SdiType('QUAD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SdiType {
    return new SdiType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Mode when quad SDI input is selected.
 */
export class SdiMode {
  /** Interleave mode */
  public static readonly INTERLEAVE = new SdiMode('INTERLEAVE');
  /** Quadrant mode */
  public static readonly QUADRANT = new SdiMode('QUADRANT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SdiMode {
    return new SdiMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Attributes for importing an existing SDI Source.
 */
export interface SdiSourceAttributes {
  /**
   * The SDI Source ARN.
   * @attribute
   */
  readonly sdiSourceArn: string;

  /**
   * The SDI Source ID.
   * @attribute
   */
  readonly sdiSourceId: string;
}

/**
 * Defines an AWS Elemental MediaLive SDI Source.
 */
@propertyInjectable
export class SdiSource extends Resource implements ISdiSource {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.SdiSource';

  /** Import an existing SDI source by its attributes. */
  public static fromSdiSourceAttributes(scope: Construct, id: string, attrs: SdiSourceAttributes): ISdiSource {
    class Import extends Resource implements ISdiSource {
      public readonly sdiSourceArn = attrs.sdiSourceArn;
      public readonly sdiSourceId = attrs.sdiSourceId;
      public readonly sdiSourceInputs = undefined;
      public readonly sdiSourceState = undefined;
      public get sdiSourceRef(): SdiSourceReference {
        return { sdiSourceId: this.sdiSourceId, sdiSourceArn: this.sdiSourceArn };
      }
    }
    return new Import(scope, id);
  }

  public readonly sdiSourceArn: string;

  /**
   * The SDI Source ID.
   * @attribute
   */
  public readonly sdiSourceId: string;

  /**
   * The list of inputs currently using this SDI source.
   */
  public readonly sdiSourceInputs?: string[];

  /**
   * The current state of the SDI source.
   */
  public readonly sdiSourceState?: string;

  /** A reference to this SDI Source resource. */
  public get sdiSourceRef(): SdiSourceReference {
    return { sdiSourceId: this.sdiSourceId, sdiSourceArn: this.sdiSourceArn };
  }

  constructor(scope: Construct, id: string, props: SdiSourceProps) {
    super(scope, id, {
      physicalName: props.sdiSourceName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    if (props.mode && props.type.value !== SdiType.QUAD.value) {
      throw new ValidationError(lit`SdiModeOnlyForQuad`, 'mode is only valid when type is QUAD', this);
    }

    addConstructMetadata(this, props);

    const resource = new CfnSdiSource(this, 'Resource', {
      name: this.physicalName,
      type: props.type?.value,
      mode: props.mode?.value,
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
    });

    this.sdiSourceArn = resource.attrArn;
    this.sdiSourceId = resource.ref;
    this.sdiSourceInputs = resource.attrInputs;
    this.sdiSourceState = resource.attrState;
  }
}
