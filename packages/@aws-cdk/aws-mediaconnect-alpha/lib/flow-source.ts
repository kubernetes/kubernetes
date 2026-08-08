import type { IResource } from 'aws-cdk-lib';
import { Resource, Lazy, Names, Token, ValidationError } from 'aws-cdk-lib';
import { CfnFlowSource } from 'aws-cdk-lib/aws-mediaconnect';
import type { IFlowSourceRef, FlowSourceReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IFlow } from './flow';
import type { SourceConfiguration } from './flow-source-configuration';
import { SourceProtocol } from './flow-source-configuration';

/**
 * Interface for Flow Source
 */
export interface IFlowSource extends IResource, IFlowSourceRef {
  /**
   * The Amazon Resource Name (ARN) of the flow source.
   *
   * @attribute
   */
  readonly flowSourceArn: string;

  /**
   * The name of the flow source.
   *
   * @attribute
   */
  readonly flowSourceName: string;

  /**
   * The IP address that the flow will be listening on for incoming content.
   *
   * @attribute
   */
  readonly ingestIp: string;

  /**
   * The port that the flow will be listening on for incoming content.
   *
   * @attribute
   */
  readonly sourceIngestPort: string;
}

/**
 * Properties for the flow source
 */
export interface FlowSourceProps {
  /**
   * Flow Source Configuration.
   */
  readonly source: SourceConfiguration;

  /**
   * The flow this source is connected to. The flow must have Failover enabled to add an additional source.
   */
  readonly flow: IFlow;
}

/**
 * Attributes for importing an existing Flow Source.
 */
export interface FlowSourceAttributes {
  /**
   * The Amazon Resource Name (ARN) of the flow source.
   */
  readonly flowSourceArn: string;

  /**
   * The name of the flow source.
   *
   * @default - accessing `flowSourceName` on the imported source throws; only provide when available.
   */
  readonly flowSourceName?: string;

  /**
   * The IP address that the flow will be listening on for incoming content.
   *
   * @default - accessing `ingestIp` on the imported source throws; only provide when available.
   */
  readonly ingestIp?: string;

  /**
   * The port that the flow will be listening on for incoming content.
   *
   * @default - accessing `sourceIngestPort` on the imported source throws; only provide when available.
   */
  readonly sourceIngestPort?: string;
}

/**
 * Shared base for both real and imported flow sources.
 */
abstract class FlowSourceBase extends Resource implements IFlowSource {
  public abstract readonly flowSourceArn: string;
  public abstract readonly flowSourceName: string;
  public abstract readonly ingestIp: string;
  public abstract readonly sourceIngestPort: string;

  public get flowSourceRef(): FlowSourceReference {
    return { sourceArn: this.flowSourceArn };
  }
}

/**
 * Adds source to an existing flow.
 *
 * Adding an additional source requires Failover to be enabled. When you enable Failover,
 * the additional source must use the same protocol as the existing source. A source is
 * the external video content that includes configuration information (encryption and source type)
 * and a network address. Each flow has at least one source. A standard source comes from a source
 * other than another AWS Elemental MediaConnect flow, such as an on-premises encoder.
 */
@propertyInjectable
export class FlowSource extends FlowSourceBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.FlowSource';

  /**
   * Import an existing Flow Source from its ARN.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param flowSourceArn The ARN of the Flow Source
   * @returns A Flow Source construct
   */
  public static fromFlowSourceArn(scope: Construct, id: string, flowSourceArn: string): IFlowSource {
    return FlowSource.fromFlowSourceAttributes(scope, id, { flowSourceArn });
  }

  /**
   * Import an existing Flow Source from its attributes.
   *
   * Provide `ingestIp` and/or `sourceIngestPort` when importing a source that was deployed
   * externally — otherwise accessing those properties on the imported construct will throw.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param attrs The Flow Source attributes
   * @returns A Flow Source construct
   */
  public static fromFlowSourceAttributes(scope: Construct, id: string, attrs: FlowSourceAttributes): IFlowSource {
    class Import extends FlowSourceBase {
      public readonly flowSourceArn = attrs.flowSourceArn;

      public get flowSourceName(): string {
        if (attrs.flowSourceName) return attrs.flowSourceName;
        throw new ValidationError(
          lit`FlowSourceNameNotProvided`,
          `'flowSourceName' is not available on imported FlowSource ${this.node.path}; pass it via fromFlowSourceAttributes`,
          this,
        );
      }

      public get ingestIp(): string {
        if (attrs.ingestIp) return attrs.ingestIp;
        throw new ValidationError(
          lit`FlowSourceIngestIpNotProvided`,
          `'ingestIp' is not available on imported FlowSource ${this.node.path}; pass it via fromFlowSourceAttributes`,
          this,
        );
      }

      public get sourceIngestPort(): string {
        if (attrs.sourceIngestPort) return attrs.sourceIngestPort;
        throw new ValidationError(
          lit`FlowSourceIngestPortNotProvided`,
          `'sourceIngestPort' is not available on imported FlowSource ${this.node.path}; pass it via fromFlowSourceAttributes`,
          this,
        );
      }
    }
    return new Import(scope, id);
  }

  readonly flowSourceArn: string;
  readonly flowSourceName: string;
  readonly ingestIp: string;
  readonly sourceIngestPort: string;

  constructor(scope: Construct, id: string, props: FlowSourceProps) {
    super(scope, id, {
      // The source name comes from the `source` configuration; fall back to a
      // construct-derived name when it sets none.
      physicalName: props.source.flowSourceName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 64 }) }),
    });

    // Validate the source name when one is set on the configuration.
    if (props.source.flowSourceName != null && props.source.flowSourceName !== '' && !Token.isUnresolved(props.source.flowSourceName)) {
      if (props.source.flowSourceName.length > 64) {
        throw new ValidationError(lit`FlowSourceNameLength`, `Flow source name must be between 1 and 64 characters, got ${props.source.flowSourceName.length}`, this);
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(props.source.flowSourceName)) {
        throw new ValidationError(lit`FlowSourceNameFormat`, `Flow source name must contain only alphanumeric characters, hyphens, and underscores, got '${props.source.flowSourceName}'`, this);
      }
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const sourceConfig = props.source._bind(this);

    if (sourceConfig.protocol === SourceProtocol.CDI.value || sourceConfig.protocol === SourceProtocol.JPEGXS.value) {
      throw new ValidationError(lit`FlowSourceProtocolNotSupported`, 'CDI or JPEGXS can only be configured on the main Flow construct', this);
    }

    if (!props.flow.isFailoverEnabled) {
      throw new ValidationError(
        lit`FlowSourceFailoverRequired`,
        'Failover configuration needs to be configured and enabled to add an additional Flow source. ' +
        'If the Flow is imported via fromFlowAttributes(), set isFailoverEnabled: true',
        this,
      );
    }

    const { name: _sourceName, ...restSourceConfig } = sourceConfig;

    const resource = new CfnFlowSource(this, 'Resource', {
      ...restSourceConfig,
      flowArn: props.flow.flowArn,
      name: this.physicalName,
      description: sourceConfig.description ?? this.physicalName,
    });

    this.flowSourceArn = resource.flowSourceRef.sourceArn;
    this.flowSourceName = this.physicalName;
    this.ingestIp = resource.attrIngestIp;
    this.sourceIngestPort = resource.attrSourceIngestPort;
  }
}
