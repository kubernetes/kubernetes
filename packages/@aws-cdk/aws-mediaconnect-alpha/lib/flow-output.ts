import type { Duration, IResource } from 'aws-cdk-lib';
import { Annotations, Lazy, Names, Resource, Token, UnscopedValidationError, ValidationError } from 'aws-cdk-lib';
import { CfnFlowOutput } from 'aws-cdk-lib/aws-mediaconnect';
import type { IFlowOutputRef, FlowOutputReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IFlow } from './flow';
import type { SrtPasswordEncryption, StaticKeyEncryption, TransitEncryption } from './shared';
import { State, isOpenCidr, renderSrtPasswordEncryption, renderStaticKeyEncryption, renderTransitEncryption } from './shared';

/**
 * Interface for Flow Output
 */
export interface IFlowOutput extends IResource, IFlowOutputRef {
  /**
   * The Amazon Resource Name (ARN) of the flow output.
   *
   * @attribute
   */
  readonly flowOutputArn: string;
}

/**
 * Properties for flow output
 */
export interface FlowOutputProps {
  /**
   * The name of the output.
   *
   * @default - auto-generated
   */
  readonly flowOutputName?: string;

  /**
   * A description of the output.
   *
   * This description appears only on the MediaConnect console and will not be seen by the end user.
   *
   * @default - no description
   */
  readonly description?: string;

  /**
   * An indication of whether the output should transmit data or not.
   *
   * @default - undefined; when omitted, MediaConnect enables the output (ENABLED) at deploy time
   */
  readonly outputStatus?: State;

  /**
   * The flow this output is attached to.
   */
  readonly flow: IFlow;

  /**
   * Output configuration
   */
  readonly output: OutputConfiguration;
}

/**
 * Flow output protocol options.
 */
enum FlowOutputProtocol {
  /**
   * ZIXI_PUSH option
   */
  ZIXI_PUSH='zixi-push',
  /**
   * RTP_FEC option
   */
  RTP_FEC='rtp-fec',
  /**
   * RTP option
   */
  RTP='rtp',
  /**
   * ZIXI_PULL option
   */
  ZIXI_PULL='zixi-pull',
  /**
   * RIST option
   */
  RIST='rist',
  /**
   * SRT_LISTENER option
   */
  SRT_LISTENER='srt-listener',
  /**
   * SRT_CALLER option
   */
  SRT_CALLER='srt-caller',
  /**
   * NDI option
   */
  NDI='ndi-speed-hq',
}

/**
 * Configuration options for RTP outputs.
 */
export interface RtpOutputConfig {
  /**
   * The smoothing latency for RIST, RTP, and RTP-FEC streams.
   *
   * @default - no smoothing latency
   */
  readonly smoothingLatency?: Duration;
  /**
   * The IP address where you want to send the output.
   */
  readonly destination: string;
  /**
   * The port to use when content is distributed to this output.
   */
  readonly port: number;
  /**
   * The name of the VPC interface attachment to use for this output.
   *
   * @default - no VPC interface attachment
   */
  readonly vpcInterfaceAttachmentName?: string;
}

/**
 * Configuration options for RTP-FEC outputs.
 */
export interface RtpFecOutputConfig extends RtpOutputConfig { }
/**
 * Configuration options for RIST outputs.
 */
export interface RistOutputConfig extends RtpOutputConfig { }

/**
 * Configuration options for Zixi Pull outputs.
 */
export interface ZixiPullOutputConfig {
  /**
   * The stream ID that you want to use for this transport.
   */
  readonly streamId: string;
  /**
   * The remote ID for the Zixi-pull stream.
   */
  readonly remoteId: string;
  /**
   * The maximum latency for Zixi-based streams.
   */
  readonly maxLatency: Duration;
  /**
   * The range of IP addresses that should be allowed to initiate output requests to this flow.
   * These IP addresses should be in the form of a Classless Inter-Domain Routing (CIDR) block; for example, 10.0.0.0/16.
   *
   * Required by the MediaConnect service for Zixi Pull outputs.
   */
  readonly cidrAllowList: string[];
  /**
   * Static key encryption for this output.
   *
   * @default - no encryption
   */
  readonly encryption?: StaticKeyEncryption;
  /**
   * The name of the VPC interface attachment to use for this output.
   *
   * @default - no VPC interface attachment
   */
  readonly vpcInterfaceAttachmentName?: string;
}

/**
 * Configuration options for Zixi Push outputs.
 */
export interface ZixiPushOutputConfig {
  /**
   * The stream ID that you want to use for this transport.
   */
  readonly streamId: string;
  /**
   * The maximum latency for Zixi-based streams.
   */
  readonly maxLatency: Duration;
  /**
   * The range of IP addresses that should be allowed to initiate output requests to this flow.
   * These IP addresses should be in the form of a Classless Inter-Domain Routing (CIDR) block; for example, 10.0.0.0/16.
   *
   * @default - no CIDR allow list
   */
  readonly cidrAllowList?: string[];
  /**
   * Static key encryption for this output.
   *
   * @default - no encryption
   */
  readonly encryption?: StaticKeyEncryption;
  /**
   * The IP address where you want to send the output.
   */
  readonly destination: string;
  /**
   * The port to use when content is distributed to this output.
   */
  readonly port: number;
  /**
   * The name of the VPC interface attachment to use for this output.
   *
   * @default - no VPC interface attachment
   */
  readonly vpcInterfaceAttachmentName?: string;
}

/**
 * Configuration options for SRT Caller outputs.
 */
export interface SrtCallerOutputConfig {
  /**
   * The minimum latency in milliseconds for SRT-based streams.
   *
   * In streams that use the SRT protocol, this value that you set on your MediaConnect source or output represents
   * the minimal potential latency of that connection. The latency of the stream is set to the highest number between
   * the sender’s minimum latency and the receiver’s minimum latency.
   *
   * @default - no minimum latency
   */
  readonly minLatency?: Duration;
  /**
   * The stream ID that you want to use for this transport.
   *
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * The IP address where you want to send the output.
   */
  readonly destination: string;
  /**
   * The port to use when content is distributed to this output.
   */
  readonly port: number;
  /**
   * SRT password encryption for this output.
   *
   * @default - no encryption
   */
  readonly encryption?: SrtPasswordEncryption;
  /**
   * The name of the VPC interface attachment to use for this output.
   *
   * @default - no VPC interface attachment
   */
  readonly vpcInterfaceAttachmentName?: string;
}

/**
 * Configuration options for SRT Listener outputs.
 */
export interface SrtListenerOutputConfig {
  /**
   * The minimum latency in milliseconds for SRT-based streams.
   *
   * In streams that use the SRT protocol, this value that you set on your MediaConnect source or output represents
   * the minimal potential latency of that connection. The latency of the stream is set to the highest number between
   * the sender’s minimum latency and the receiver’s minimum latency.
   *
   * @default - no minimum latency
   */
  readonly minLatency?: Duration;
  /**
   * The port to use when content is distributed to this output.
   */
  readonly port: number;
  /**
   * The range of IP addresses that should be allowed to initiate output requests to this flow.
   * These IP addresses should be in the form of a Classless Inter-Domain Routing (CIDR) block; for example, 10.0.0.0/16.
   *
   * @default - no CIDR allow list
   */
  readonly cidrAllowList?: string[];
  /**
   * SRT password encryption for this output.
   *
   * @default - no encryption
   */
  readonly encryption?: SrtPasswordEncryption;
  /**
   * The name of the VPC interface attachment to use for this output.
   *
   * @default - no VPC interface attachment
   */
  readonly vpcInterfaceAttachmentName?: string;
}

/**
 * Configuration options for NDI outputs.
 */
export interface NdiOutputConfig {
  /**
   * A suffix for the names of the NDI sources that the flow creates.
   *
   * @default - the output name is used
   */
  readonly ndiProgramName?: string;
  /**
   * A quality setting for the NDI Speed HQ encoder, expressed as a percentage.
   *
   * Valid range: 100-200. Higher values produce higher quality and larger bitrate.
   *
   * @see https://aws.amazon.com/about-aws/whats-new/2025/03/aws-elemental-mediaconnect-support-ndi-outputs/
   * @default - the MediaConnect service default
   */
  readonly ndiSpeedHqQuality?: number;
}

/**
 * Internal flow output configuration. CFN-bound fields use L1 property types; encryption
 * fields are kept as L2 classes (resolved to CFN shape in the FlowOutput constructor).
 * @internal
 */
interface FlowOutputConfig {
  readonly cidrAllowList?: string[];
  readonly destination?: string;
  /** Static-key encryption for Zixi Pull outputs. Set at most one of these two fields. */
  readonly staticKeyEncryption?: StaticKeyEncryption;
  /** SRT-password encryption for SRT outputs. Set at most one of these two fields. */
  readonly srtPasswordEncryption?: SrtPasswordEncryption;
  readonly maxLatency?: number;
  readonly minLatency?: number;
  readonly ndiProgramName?: string;
  readonly ndiSpeedHqQuality?: number;
  readonly port?: number;
  readonly protocol?: string;
  readonly remoteId?: string;
  readonly smoothingLatency?: number;
  readonly streamId?: string;
  readonly vpcInterfaceAttachment?: CfnFlowOutput.VpcInterfaceAttachmentProperty;
  readonly routerIntegrationState?: State;
  readonly routerIntegrationTransitEncryption?: TransitEncryption;
}

/**
 * Output configuration to Router
 */
export interface RouterTransitConfig {
  /**
   * Specifies whether encryption is to be used.
   *
   * @default no encryption
   */
  readonly encryption?: TransitEncryption;
}

/**
 * Configuration options to define a FlowOutput by protocol.
 */
export class OutputConfiguration {
  /**
   * The source configuration for flows receiving a stream from router.
   */
  public static router(input?: RouterTransitConfig): OutputConfiguration {
    return new OutputConfiguration({
      routerIntegrationState: State.ENABLED,
      routerIntegrationTransitEncryption: input?.encryption,
    });
  }

  /**
   * Option for RTP configuration
   */
  public static rtp(input: RtpOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.RTP,
      destination: input.destination,
      port: input.port,
      smoothingLatency: input.smoothingLatency?.toMilliseconds(),
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for RTP-FEC configuration
   */
  public static rtpFec(input: RtpFecOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.RTP_FEC,
      destination: input.destination,
      port: input.port,
      smoothingLatency: input.smoothingLatency?.toMilliseconds(),
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for RIST configuration
   */
  public static rist(input: RistOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.RIST,
      destination: input.destination,
      port: input.port,
      smoothingLatency: input.smoothingLatency?.toMilliseconds(),
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for Zixi Push configuration
   */
  public static zixiPush(input: ZixiPushOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.ZIXI_PUSH,
      destination: input.destination,
      port: input.port,
      streamId: input.streamId,
      maxLatency: input.maxLatency.toMilliseconds(),
      cidrAllowList: input.cidrAllowList,
      staticKeyEncryption: input.encryption,
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for Zixi Pull configuration
   */
  public static zixiPull(input: ZixiPullOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.ZIXI_PULL,
      streamId: input.streamId,
      remoteId: input.remoteId,
      maxLatency: input.maxLatency.toMilliseconds(),
      cidrAllowList: input.cidrAllowList,
      staticKeyEncryption: input.encryption,
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for SRT Caller configuration
   */
  public static srtCaller(input: SrtCallerOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.SRT_CALLER,
      destination: input.destination,
      port: input.port,
      minLatency: input.minLatency?.toMilliseconds(),
      streamId: input.streamId,
      srtPasswordEncryption: input.encryption,
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for SRT Listener configuration
   */
  public static srtListener(input: SrtListenerOutputConfig): OutputConfiguration {
    return new OutputConfiguration({
      protocol: FlowOutputProtocol.SRT_LISTENER,
      port: input.port,
      minLatency: input.minLatency?.toMilliseconds(),
      cidrAllowList: input.cidrAllowList,
      srtPasswordEncryption: input.encryption,
      vpcInterfaceAttachment: toVpcInterfaceAttachment(input.vpcInterfaceAttachmentName),
    });
  }

  /**
   * Option for NDI configuration
   */
  public static ndi(input: NdiOutputConfig = {}): OutputConfiguration {
    if (input.ndiSpeedHqQuality !== undefined
      && !Token.isUnresolved(input.ndiSpeedHqQuality)
      && (input.ndiSpeedHqQuality < 100 || input.ndiSpeedHqQuality > 200)) {
      throw new UnscopedValidationError(
        lit`NdiSpeedHqQualityRange`,
        `NDI Speed HQ quality must be between 100 and 200, got ${input.ndiSpeedHqQuality}`,
      );
    }

    return new OutputConfiguration({
      protocol: FlowOutputProtocol.NDI,
      ndiProgramName: input.ndiProgramName,
      ndiSpeedHqQuality: input.ndiSpeedHqQuality,
    });
  }

  /**
   * @param output - Configuration for outputs in output
   */
  private constructor(private readonly _output: FlowOutputConfig) { }

  /**
   * Called when the output configuration is bound to a FlowOutput.
   * @internal
   */
  public _bind(): FlowOutputConfig {
    return { ...this._output };
  }
}

/**
 * Shared base for both real and imported flow outputs.
 */
abstract class FlowOutputBase extends Resource implements IFlowOutput {
  public abstract readonly flowOutputArn: string;

  public get flowOutputRef(): FlowOutputReference {
    return { outputArn: this.flowOutputArn };
  }
}

/**
 * Resource defines the destination address, protocol, and port that
 * AWS Elemental MediaConnect sends the ingested video to.
 */
@propertyInjectable
export class FlowOutput extends FlowOutputBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.FlowOutput';

  /**
   * Creates a Flow Ouput construct that represents an external (imported) Flow Output.
   */
  public static fromFlowOutputArn(scope: Construct, id: string, flowOutputArn: string): IFlowOutput {
    class Import extends FlowOutputBase {
      public readonly flowOutputArn = flowOutputArn;
    }

    return new Import(scope, id);
  }

  public readonly flowOutputArn: string;

  constructor(scope: Construct, id: string, props: FlowOutputProps) {
    super(scope, id, {
      physicalName: props?.flowOutputName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 128 }) }),
    });

    // Validate flow output name if provided
    if (props.flowOutputName != null && props.flowOutputName !== '' && !Token.isUnresolved(props.flowOutputName)) {
      if (props.flowOutputName.length > 128) {
        throw new ValidationError(lit`FlowOutputNameLength`, `Flow output name must be between 1 and 128 characters, got ${props.flowOutputName.length}`, this);
      }
      if (!/^[a-zA-Z0-9-]+$/.test(props.flowOutputName)) {
        throw new ValidationError(lit`FlowOutputNameFormat`, `Flow output name must contain only alphanumeric characters and hyphens, got '${props.flowOutputName}'`, this);
      }
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const outputConfig = props.output._bind();

    // Warn on fully-open CIDR allow lists.
    if (outputConfig.cidrAllowList !== undefined && !Token.isUnresolved(outputConfig.cidrAllowList)) {
      for (const cidr of outputConfig.cidrAllowList) {
        if (!Token.isUnresolved(cidr) && isOpenCidr(cidr)) {
          Annotations.of(this).addWarningV2(
            '@aws-cdk/aws-mediaconnect-alpha:openOutputCidr',
            `Output CIDR allow list '${cidr}' allows pull requests from any IP. Restrict to the narrowest range your consumers need.`,
          );
        }
      }
    }
    const { staticKeyEncryption, srtPasswordEncryption, routerIntegrationTransitEncryption, ...cfnOutputConfig } = outputConfig;

    const resource = new CfnFlowOutput(this, 'Resource', {
      ...cfnOutputConfig,
      encryption: renderOutputEncryption(this, staticKeyEncryption, srtPasswordEncryption, props.flow.flowArn),
      name: this.physicalName,
      flowArn: props.flow.flowArn,
      description: props.description,
      outputStatus: props.outputStatus,
      routerIntegrationTransitEncryption: outputConfig.routerIntegrationState === State.ENABLED
        ? renderTransitEncryption(this, 'RouterTransitEncryptionRole', routerIntegrationTransitEncryption, props.flow.flowArn)
        : undefined,
    });

    this.flowOutputArn = resource.attrOutputArn;
  }
}

/**
 * Convert an optional VPC interface name to the CFN attachment property shape.
 */
function toVpcInterfaceAttachment(name?: string): CfnFlowOutput.VpcInterfaceAttachmentProperty | undefined {
  return name ? { vpcInterfaceName: name } : undefined;
}

/**
 * Resolve the output's key-based encryption (either static-key or SRT-password) to its
 * CFN shape. At most one of the two encryption fields is set on a given output.
 */
function renderOutputEncryption(
  scope: Construct,
  staticKey: StaticKeyEncryption | undefined,
  srtPassword: SrtPasswordEncryption | undefined,
  sourceArn?: string,
): CfnFlowOutput.EncryptionProperty | undefined {
  if (staticKey) return renderStaticKeyEncryption(scope, staticKey, sourceArn);
  if (srtPassword) return renderSrtPasswordEncryption(scope, srtPassword, sourceArn);
  return undefined;
}
