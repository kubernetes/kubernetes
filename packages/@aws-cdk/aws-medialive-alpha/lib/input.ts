import type { Duration, IResource } from 'aws-cdk-lib';
import { Resource, Lazy, Names, Stack, Aws, ArnFormat, UnscopedValidationError, ValidationError } from 'aws-cdk-lib';
import type { ISecurityGroupRef, ISubnetRef } from 'aws-cdk-lib/aws-ec2';
import type { IRole, IRoleRef } from 'aws-cdk-lib/aws-iam';
import { Grant, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import type { IFlowRef } from 'aws-cdk-lib/aws-mediaconnect';
import type { IInputRef, InputReference, IInputSecurityGroupRef, INetworkRef } from 'aws-cdk-lib/aws-medialive';
import { CfnInput } from 'aws-cdk-lib/aws-medialive';
import type { IBucket } from 'aws-cdk-lib/aws-s3';
import type { ISecret } from 'aws-cdk-lib/aws-secretsmanager';
import type { IStringParameter } from 'aws-cdk-lib/aws-ssm';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { ChannelClass } from './channel';
import type { NetworkRoute } from './network';
import type { ISdiSource } from './sdi-source';
import { extractResourceId } from './shared';

/**
 * Represents a MediaLive Input.
 */
export interface IInput extends IResource, IInputRef {
  /**
   * The Amazon Resource Name (ARN) of the input.
   * @attribute
   */
  readonly inputArn: string;
  /**
   * The ID of the input.
   * @attribute
   */
  readonly inputId: string;
  /**
   * The input class (STANDARD or SINGLE_PIPELINE).
   * Only available for input types where the pipeline count is known at construct time
   * (e.g. mediaConnectRouter). Undefined for imported inputs and other types.
   *
   * @attribute
   */
  readonly inputClass?: string;
  /**
   * The input type (e.g. SRT_CALLER, MP4_FILE, URL_PULL).
   * Undefined for imported inputs.
   *
   * @attribute
   */
  readonly inputType?: string;
  /**
   * For push inputs, the destination URLs where the upstream system sends content.
   * @attribute
   */
  readonly inputDestinations?: string[];
  /**
   * For pull inputs, the source URLs where MediaLive pulls content from.
   * @attribute
   */
  readonly inputSources?: string[];

  /**
   * Grant the channel role read access to this input's sources.
   *
   * Owned inputs grant access to the resources backing their sources (e.g. S3 buckets
   * for file inputs); imported inputs are a no-op, since their sources aren't owned by
   * this stack.
   *
   * @internal
   */
  _grantPermissions(role: IRoleRef): void;
}

/**
 * The network location of a MediaLive input — the AWS cloud, or an on-premises network for
 * MediaLive Anywhere.
 */
export class InputNetworkLocation {
  /** The input exists in the AWS cloud (the default). */
  public static readonly AWS = new InputNetworkLocation('AWS');
  /** The input exists in an on-premises network (MediaLive Anywhere). */
  public static readonly ON_PREMISES = new InputNetworkLocation('ON_PREMISES');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputNetworkLocation {
    return new InputNetworkLocation(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for creating a MediaLive Input.
 */
export interface InputProps {
  /**
   * Input name.
   * @default - auto-generated
   */
  readonly inputName?: string;
  /**
   * Input configuration that defines the input type and source settings.
   */
  readonly input: InputConfiguration;
  /**
   * The network location of the input — AWS cloud or on-premises (MediaLive Anywhere).
   * @default - AWS, applied by MediaLive
   */
  readonly inputNetworkLocation?: InputNetworkLocation;
  /**
   * Tags to add to the input.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };
}

/**
 * The type of MediaLive input. Users select an input type via
 * the `InputConfiguration` factory methods, never by passing this type directly.
 * @internal
 */
export enum InputType {
  /** Push a UDP transport stream to MediaLive */
  UDP_PUSH = 'UDP_PUSH',
  /** Push an RTP transport stream to MediaLive */
  RTP_PUSH = 'RTP_PUSH',
  /** Push an RTMP stream to MediaLive */
  RTMP_PUSH = 'RTMP_PUSH',
  /** Pull an RTMP stream from a server */
  RTMP_PULL = 'RTMP_PULL',
  /** Pull HLS or TS content from a URL */
  URL_PULL = 'URL_PULL',
  /** Pull an MP4 file from a URL */
  MP4_FILE = 'MP4_FILE',
  /** Receive content from AWS Elemental MediaConnect */
  MEDIACONNECT = 'MEDIACONNECT',
  /** Receive content from an Elemental Link device */
  INPUT_DEVICE = 'INPUT_DEVICE',
  /** Receive uncompressed content via AWS CDI */
  AWS_CDI = 'AWS_CDI',
  /** Pull a TS file from a URL */
  TS_FILE = 'TS_FILE',
  /** Pull content from an SRT source (caller mode) */
  SRT_CALLER = 'SRT_CALLER',
  /** Receive content pushed via SRT (listener mode) */
  SRT_LISTENER = 'SRT_LISTENER',
  /** Receive multicast content. Requires `anywhereSettings` on the MediaLive Channel. */
  MULTICAST = 'MULTICAST',
  /** Receive content via SMPTE 2110. Requires `anywhereSettings` on the MediaLive Channel. */
  SMPTE_2110_RECEIVER_GROUP = 'SMPTE_2110_RECEIVER_GROUP',
  /** Receive content from an SDI source. Requires `anywhereSettings` on the MediaLive Channel. */
  SDI = 'SDI',
  /** Receive content from a MediaConnect router */
  MEDIACONNECT_ROUTER = 'MEDIACONNECT_ROUTER',
}

/**
 * Options for a URL-based input source.
 */
export interface InputSourceOptions {
  /**
   * The username for accessing the upstream system.
   * @default - no username
   */
  readonly username?: string;
  /**
   * The SSM parameter that holds the password for accessing the upstream system.
   * @default - no password
   */
  readonly password?: IStringParameter;
}

/**
 * A source for a pull-type input. Use the static factory methods to create.
 */
export abstract class InputSource {
  /**
   * Create a source from a URL.
   * @param url The URL where MediaLive pulls the source content from.
   * @param options Optional credentials.
   */
  public static url(url: string, options?: InputSourceOptions): InputSource {
    return new UrlInputSource(url, options);
  }

  /**
   * Create a source from an S3 bucket. Automatically grants read access
   * to the channel's role.
   *
   * @param bucket The S3 bucket containing the source file.
   * @param key The object key within the bucket (e.g. 'videos/intro.mp4').
   */
  public static fromBucket(bucket: IBucket, key: string): InputSource {
    return new S3InputSource(bucket, key);
  }

  /** @internal */
  public abstract _bind(): { url: string; username?: string; passwordParam?: string };

  /** @internal */
  public _grantRead(_role: IRole): void {}
}

/** @internal */
class UrlInputSource extends InputSource {
  constructor(private readonly url: string, private readonly options?: InputSourceOptions) { super(); }
  public _bind() {
    return { url: this.url, username: this.options?.username, passwordParam: this.options?.password?.parameterName };
  }
  public override _grantRead(role: IRole): void {
    // MediaLive reads the password from SSM Parameter Store at channel runtime, so the
    // channel role needs read access to the parameter (scoped to the parameter ARN).
    this.options?.password?.grantRead(role);
  }
}

/** @internal */
class S3InputSource extends InputSource {
  public readonly url: string;
  constructor(private readonly bucket: IBucket, key: string) {
    super();
    this.url = `s3ssl://${bucket.bucketName}/${key}`;
  }
  public _bind() {
    return { url: this.url };
  }
  public override _grantRead(role: IRole): void {
    this.bucket.grantRead(role);
  }
}

/** Properties for a MediaConnect router input. */
export interface MediaConnectRouterInputProps {
  /**
   * The availability zones for the router input pipelines.
   *
   * Provide one AZ for a single pipeline, or two for redundant pipelines.
   * If omitted, defaults to the stack's first availability zone (single pipeline).
   *
   * @default - single pipeline using the stack's first availability zone
   */
  readonly availabilityZones?: string[];
  /**
   * The Secrets Manager secret for custom encryption.
   * If not provided, automatic encryption is used.
   * @default - automatic encryption
   */
  readonly encryptionSecret?: ISecret;
}

/** Properties for a MediaConnect input. */
export interface MediaConnectInputProps {
  /**
   * The MediaConnect flows to use as sources (one, or two for pipeline redundancy).
   */
  readonly flows: IFlowRef[];
  /**
   * The IAM role MediaLive uses to manage the output it adds to the flow for this input.
   *
   * When omitted, a role is created and granted the required permissions.
   * When you provide a role, no permissions are added — you own all the permissions it needs.
   *
   * @default - a role is created with the medialive.amazonaws.com service principal
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly role?: IRole;
}

/** Properties for an SRT caller input. */
export interface SrtCallerSourceProps {
  /** The address of the SRT listener to connect to. */
  readonly srtListenerAddress: string;
  /** The port of the SRT listener. */
  readonly srtListenerPort: number;
  /**
   * The minimum latency.
   * @default - service default
   */
  readonly minimumLatency?: Duration;
  /**
   * The stream ID for the SRT connection.
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * Decryption settings for the SRT connection.
   * @default - no decryption
   */
  readonly decryption?: SrtDecryptionProps;
}

/** The encryption algorithm for SRT decryption. */
export class SrtDecryptionAlgorithm {
  /** AES-128. */
  public static readonly AES128 = new SrtDecryptionAlgorithm('AES128');
  /** AES-192. */
  public static readonly AES192 = new SrtDecryptionAlgorithm('AES192');
  /** AES-256. */
  public static readonly AES256 = new SrtDecryptionAlgorithm('AES256');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): SrtDecryptionAlgorithm {
    return new SrtDecryptionAlgorithm(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Properties for SRT decryption. */
export interface SrtDecryptionProps {
  /**
   * The encryption algorithm.
   * @default - no algorithm
   */
  readonly algorithm?: SrtDecryptionAlgorithm;
  /**
   * The Secrets Manager secret containing the passphrase used to decrypt the content.
   * @default - no passphrase
   */
  readonly passphraseSecret?: ISecret;
}

/** Properties for an SRT listener input. */
export interface SrtListenerInputProps {
  /**
   * Decryption settings for the SRT connection.
   * @default - no decryption
   */
  readonly decryption?: SrtDecryptionProps;
  /**
   * The minimum latency.
   * @default - service default
   */
  readonly minimumLatency?: Duration;
  /**
   * The stream ID that the upstream system uses when connecting to this listener.
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * The input security groups. Required for SRT listener inputs.
   */
  readonly inputSecurityGroups: IInputSecurityGroupRef[];
  /**
   * Whether this is a STANDARD (two-pipeline) or SINGLE_PIPELINE input. A STANDARD input creates
   * two listener endpoints for pipeline redundancy.
   * @default ChannelClass.SINGLE_PIPELINE
   */
  readonly inputClass?: ChannelClass;
}

/** Properties for an Elemental Link input device input. */
export interface InputDeviceInputProps {
  /** The IDs of one or two registered Elemental Link devices. Two provides pipeline redundancy. */
  readonly deviceIds: string[];
}

/** The transport protocol for a multicast source. */
export class MulticastProtocol {
  /** UDP transport. */
  public static readonly UDP = new MulticastProtocol('udp');
  /** RTP transport. */
  public static readonly RTP = new MulticastProtocol('rtp');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MulticastProtocol {
    return new MulticastProtocol(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** A source for a multicast input. */
export interface MulticastInputSource {
  /** The multicast group address. */
  readonly address: string;
  /** The multicast port. */
  readonly port: number;
  /**
   * The transport protocol.
   * @default MulticastProtocol.UDP
   */
  readonly protocol?: MulticastProtocol;
  /**
   * Filter to a specific source IP (source-specific multicast).
   * @default - accept any source
   */
  readonly sourceIp?: string;
}

/** Properties for a multicast input. Requires `anywhereSettings` on the channel. */
export interface MulticastInputProps {
  /** The multicast sources. Provide two for a STANDARD (redundant-pipeline) channel. */
  readonly sources: MulticastInputSource[];
}

/** A reference to an SDP file that describes a SMPTE 2110 stream to ingest. */
export interface Smpte2110SdpLocation {
  /** The URL of the SDP file. */
  readonly sdpUrl: string;
  /**
   * The index of the media stream within the SDP to ingest. Use when the SDP describes
   * more than one stream of that media type.
   * @default - service default
   */
  readonly mediaIndex?: number;
}

/**
 * Properties for a SMPTE 2110 receiver group input. Requires `anywhereSettings` on the channel.
 *
 * You specify the SDP files that describe the streams to ingest — one video SDP, and any
 * number of audio and ancillary SDPs.
 */
export interface Smpte2110InputProps {
  /**
   * The SDP describing the video stream.
   * @default - no video stream
   */
  readonly videoSdp?: Smpte2110SdpLocation;
  /**
   * The SDPs describing the audio streams. Up to 50.
   * @default - no audio streams
   */
  readonly audioSdps?: Smpte2110SdpLocation[];
  /**
   * The SDPs describing the ancillary data streams (SCTE-35 or captions). Up to 50.
   * @default - no ancillary data streams
   */
  readonly ancillarySdps?: Smpte2110SdpLocation[];
}

/** Properties for a CDI (uncompressed) input. */
export interface CdiInputProps {
  /**
   * Two VPC subnets, in two different availability zones, for the CDI input network interfaces.
   */
  readonly subnets: ISubnetRef[];
  /**
   * Security groups to attach to the CDI input network interfaces.
   * @default - VPC default security group
   */
  readonly securityGroups?: ISecurityGroupRef[];
  /**
   * The IAM role MediaLive assumes to create network interfaces in the VPC.
   *
   * When omitted, a role is created and granted the required permissions.
   * When you provide a role, no permissions are added — you own all the permissions it needs.
   *
   * @default - a role is created with the medialive.amazonaws.com service principal
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly role?: IRole;
}

/**
 * Properties for push-type inputs (RTMP_PUSH, RTP_PUSH, UDP_PUSH).
 */
export interface PushInputProps {
  /**
   * The input security groups that control which CIDR blocks can push to this input. Required for
   * cloud inputs; must be omitted for on-premises inputs (`InputNetworkLocation.ON_PREMISES`),
   * which do not support security groups.
   * @default - none (only valid for on-premises inputs)
   */
  readonly inputSecurityGroups?: IInputSecurityGroupRef[];
  /**
   * The destinations for push inputs. For RTMP push, each destination can have a stream name.
   * @default - MediaLive auto-generates destinations
   */
  readonly destinations?: PushInputDestination[];
}

/**
 * A destination for a push-type input.
 */
export interface PushInputDestination {
  /**
   * The stream name for RTMP push destinations (application name/instance).
   * @default - auto-generated
   */
  readonly streamName?: string;
  /**
   * The MediaLive Anywhere network this push destination lives on. Required when the input's
   * `inputNetworkLocation` is `ON_PREMISES`.
   * @default - AWS-managed network
   */
  readonly network?: INetworkRef;
  /**
   * The routes to the push destination on the local network. Required for
   * on-premises (`ON_PREMISES`) push inputs.
   * @default - no routes
   */
  readonly networkRoutes?: NetworkRoute[];
  /**
   * A static IP address to assign to the push destination on the local network.
   * @default - MediaLive Anywhere uses one from the IP pool specified on the selected network (service default)
   */
  readonly staticIpAddress?: string;
}

/**
 * Bind push-input destinations to their CFN shape.
 * @internal
 */
function bindPushDestinations(destinations?: PushInputDestination[]): CfnInput.InputDestinationRequestProperty[] | undefined {
  return destinations?.map(d => ({
    streamName: d.streamName,
    network: d.network?.networkRef.networkId,
    networkRoutes: d.networkRoutes?.map(r => ({ cidr: r.cidr, gateway: r.gateway })),
    staticIpAddress: d.staticIpAddress,
  }));
}

/**
 * Internal bind result for InputConfiguration.
 * @internal
 */
interface InputBindConfig {
  readonly type: InputType;
  readonly inputClass?: string;
  readonly inputSources?: InputSource[];
  readonly sources?: CfnInput.InputSourceRequestProperty[];
  readonly destinations?: CfnInput.InputDestinationRequestProperty[];
  readonly mediaConnectFlows?: CfnInput.MediaConnectFlowRequestProperty[];
  readonly inputDevices?: CfnInput.InputDeviceSettingsProperty[];
  readonly multicastSettings?: CfnInput.MulticastSettingsCreateRequestProperty;
  readonly smpte2110ReceiverGroupSettings?: CfnInput.Smpte2110ReceiverGroupSettingsProperty;
  readonly routerSettings?: CfnInput.RouterSettingsProperty;
  readonly srtSettings?: CfnInput.SrtSettingsRequestProperty;
  readonly sdiSources?: string[];
  readonly vpc?: CfnInput.InputVpcRequestProperty;
  readonly roleArn?: string;
  readonly inputSecurityGroups?: string[];
  /**
   * The role (and the flows it must manage) for a MediaConnect flow input. MediaLive
   * assumes this role at input create/delete time to add/remove outputs on the flows,
   * so it's granted the `mediaconnect:Managed*` flow-management actions.
   */
  readonly mediaConnectFlowGrant?: MediaConnectFlowGrant;
  /**
   * The role and VPC scope for a CDI/VPC input. MediaLive assumes this role to create the
   * input's network interfaces, so it's granted the EC2 ENI actions scoped to the VPC.
   */
  readonly vpcRoleGrant?: VpcInputRoleGrant;
}

/**
 * The input role and VPC scope for a CDI/VPC input's EC2 ENI auto-grant.
 * @internal
 */
interface VpcInputRoleGrant {
  readonly role: IRole;
  readonly subnetIds: string[];
  readonly securityGroupIds: string[];
}

/**
 * The input role for a MediaConnect flow input's auto-grant.
 * @internal
 */
interface MediaConnectFlowGrant {
  readonly role: IRole;
}

/**
 * Bind a SMPTE 2110 SDP location to its CFN shape.
 * @internal
 */
function bindSdpLocation(loc: Smpte2110SdpLocation): CfnInput.InputSdpLocationProperty {
  return { sdpUrl: loc.sdpUrl, mediaIndex: loc.mediaIndex };
}

/**
 * Defines the input configuration for a MediaLive Input.
 *
 * Use the static factory methods to create the appropriate configuration for your input type.
 */
export class InputConfiguration {
  /**
   * Create a URL pull input for HLS or TS streams.
   *
   * @example
   *
   *    InputConfiguration.urlPull([
   *      InputSource.url('https://example.com/stream.m3u8'),
   *    ]);
   *
   */
  public static urlPull(sources: InputSource[]): InputConfiguration {
    return InputConfiguration.createPullInput(InputType.URL_PULL, sources);
  }

  /** Create an RTMP pull input. */
  public static rtmpPull(sources: InputSource[]): InputConfiguration {
    return InputConfiguration.createPullInput(InputType.RTMP_PULL, sources);
  }

  /** Create an MP4 file pull input. */
  public static mp4File(sources: InputSource[]): InputConfiguration {
    return InputConfiguration.createPullInput(InputType.MP4_FILE, sources);
  }

  /** Create a TS file pull input. */
  public static tsFile(sources: InputSource[]): InputConfiguration {
    return InputConfiguration.createPullInput(InputType.TS_FILE, sources);
  }

  /**
   * Create an RTMP push input.
   *
   * @example
   *
   *    declare const securityGroup: InputSecurityGroup;
   *
   *    InputConfiguration.rtmpPush({
   *      inputSecurityGroups: [securityGroup],
   *    });
   *
   */
  public static rtmpPush(props: PushInputProps): InputConfiguration {
    return new InputConfiguration(() => ({
      type: InputType.RTMP_PUSH,
      inputClass: props.destinations?.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      inputSecurityGroups: props.inputSecurityGroups?.map(sg => sg.inputSecurityGroupRef.inputSecurityGroupId),
      destinations: bindPushDestinations(props.destinations),
    }));
  }

  /** Create an RTP push input. */
  public static rtpPush(props: PushInputProps): InputConfiguration {
    return new InputConfiguration(() => ({
      type: InputType.RTP_PUSH,
      inputClass: props.destinations?.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      inputSecurityGroups: props.inputSecurityGroups?.map(sg => sg.inputSecurityGroupRef.inputSecurityGroupId),
      destinations: bindPushDestinations(props.destinations),
    }));
  }

  /** Create a UDP push input. */
  public static udpPush(props: PushInputProps): InputConfiguration {
    return new InputConfiguration(() => ({
      type: InputType.UDP_PUSH,
      inputClass: props.destinations?.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      inputSecurityGroups: props.inputSecurityGroups?.map(sg => sg.inputSecurityGroupRef.inputSecurityGroupId),
      destinations: bindPushDestinations(props.destinations),
    }));
  }

  /**
   * Create a MediaConnect router input configuration.
   *
   * @example
   *
   *    // Single pipeline with default AZ
   *    InputConfiguration.mediaConnectRouter();
   *
   * @example
   *
   *    // Redundant pipelines with explicit AZs
   *    InputConfiguration.mediaConnectRouter({
   *      availabilityZones: ['us-east-1a', 'us-east-1b'],
   *    });
   *
   */
  public static mediaConnectRouter(props?: MediaConnectRouterInputProps): InputConfiguration {
    return new InputConfiguration((scope) => {
      const stackAzs = scope ? Stack.of(scope).availabilityZones : [];
      const azs = props?.availabilityZones ?? [stackAzs[0]];
      if (azs.length < 1 || azs.length > 2) {
        throw new UnscopedValidationError(lit`RouterAzCount`, 'MediaConnect Router input requires 1 or 2 availability zones.');
      }
      const destinations: CfnInput.RouterDestinationSettingsProperty[] = azs.map(az => ({
        availabilityZoneName: az,
      }));
      return {
        type: InputType.MEDIACONNECT_ROUTER,
        inputClass: azs.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
        routerSettings: {
          destinations,
          encryptionType: props?.encryptionSecret ? 'CUSTOM' : 'AUTOMATIC',
          secretArn: props?.encryptionSecret?.secretArn,
        },
      };
    });
  }

  /**
   * Create a MediaConnect flow input configuration.
   *
   * @example
   *
   *    declare const role: iam.IRole;
   *    declare const flow: mediaconnect.IFlowRef;
   *
   *    InputConfiguration.mediaConnect({
   *      flows: [flow],
   *      role,
   *    });
   *
   */
  public static mediaConnect(props: MediaConnectInputProps): InputConfiguration {
    if (props.flows.length < 1 || props.flows.length > 2) {
      throw new UnscopedValidationError(lit`MediaConnectFlowsLimit`, 'You must specify 1 or 2 MediaConnect flows.');
    }
    return new InputConfiguration((scope) => {
      const role = props.role ?? createInputRole(scope, 'MediaConnectRole');
      return {
        type: InputType.MEDIACONNECT,
        inputClass: props.flows.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
        mediaConnectFlows: props.flows.map(f => ({ flowArn: f.flowRef.flowArn })),
        roleArn: role.roleArn,
        // Only auto-grant actions with auto-created role. A user-provided role
        // is left alone — the caller adds its permissions.
        mediaConnectFlowGrant: props.role === undefined ? { role } : undefined,
      };
    });
  }

  /** Create an SDI input configuration. */
  public static sdi(sources: ISdiSource[]): InputConfiguration {
    return new InputConfiguration(() => ({
      type: InputType.SDI,
      inputClass: sources.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      // MediaLive's SdiSources property expects the numeric SDI source ID, not the ARN.
      sdiSources: sources.map(s => s.sdiSourceId),
    }));
  }

  /** Create an Elemental Link input from one or two registered Link device IDs. */
  public static inputDevice(props: InputDeviceInputProps): InputConfiguration {
    if (props.deviceIds.length < 1 || props.deviceIds.length > 2) {
      throw new UnscopedValidationError(lit`InputDeviceCount`, 'You must specify 1 or 2 input devices.');
    }
    return new InputConfiguration(() => ({
      type: InputType.INPUT_DEVICE,
      inputClass: props.deviceIds.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      inputDevices: props.deviceIds.map(id => ({ id })),
    }));
  }

  /**
   * Create a CDI (uncompressed) input delivered into a VPC.
   *
   * MediaLive creates network interfaces in the supplied subnets and hands back the CDI push
   * endpoints. The required EC2 permissions are granted to the role automatically.
   */
  public static cdi(props: CdiInputProps): InputConfiguration {
    if (props.subnets.length !== 2) {
      throw new UnscopedValidationError(lit`CdiSubnetCount`, 'CDI inputs require exactly 2 subnets in two different availability zones.');
    }
    const subnetIds = props.subnets.map(s => s.subnetRef.subnetId);
    const securityGroupIds = props.securityGroups?.map(sg => sg.securityGroupRef.securityGroupId);
    return new InputConfiguration((scope) => {
      const role = props.role ?? createInputRole(scope, 'CdiRole');
      return {
        type: InputType.AWS_CDI,
        inputClass: 'STANDARD',
        roleArn: role.roleArn,
        vpc: { subnetIds, securityGroupIds },
        // Only auto-grant actions with auto-created role. A user-provided role
        // is left alone — the caller adds its permissions.
        vpcRoleGrant: props.role === undefined ? { role, subnetIds, securityGroupIds: securityGroupIds ?? [] } : undefined,
      };
    });
  }

  /** Create a multicast input. Requires `anywhereSettings` on the channel. */
  public static multicast(props: MulticastInputProps): InputConfiguration {
    if (props.sources.length < 1 || props.sources.length > 2) {
      throw new UnscopedValidationError(lit`MulticastSourceCount`, 'You must specify 1 or 2 multicast sources.');
    }
    return new InputConfiguration(() => ({
      type: InputType.MULTICAST,
      inputClass: props.sources.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      multicastSettings: {
        sources: props.sources.map(s => ({
          url: `${(s.protocol ?? MulticastProtocol.UDP).value}://${s.address}:${s.port}`,
          sourceIp: s.sourceIp,
        })),
      },
    }));
  }

  /** Create a SMPTE 2110 receiver group input. Requires `anywhereSettings` on the channel. */
  public static smpte2110ReceiverGroup(props: Smpte2110InputProps): InputConfiguration {
    if ((props.audioSdps?.length ?? 0) > 50) {
      throw new UnscopedValidationError(lit`Smpte2110AudioSdpCount`, 'A SMPTE 2110 input supports at most 50 audio SDPs.');
    }
    if ((props.ancillarySdps?.length ?? 0) > 50) {
      throw new UnscopedValidationError(lit`Smpte2110AncillarySdpCount`, 'A SMPTE 2110 input supports at most 50 ancillary SDPs.');
    }
    return new InputConfiguration(() => ({
      type: InputType.SMPTE_2110_RECEIVER_GROUP,
      smpte2110ReceiverGroupSettings: {
        smpte2110ReceiverGroups: [{
          sdpSettings: {
            videoSdp: props.videoSdp ? bindSdpLocation(props.videoSdp) : undefined,
            audioSdps: props.audioSdps?.map(bindSdpLocation),
            ancillarySdps: props.ancillarySdps?.map(bindSdpLocation),
          },
        }],
      },
    }));
  }

  /**
   * Create an SRT caller input configuration.
   *
   * @example
   *
   *    InputConfiguration.srtCaller([{
   *      srtListenerAddress: '10.0.0.1',
   *      srtListenerPort: 5000,
   *    }]);
   *
   */
  public static srtCaller(sources: SrtCallerSourceProps[]): InputConfiguration {
    if (sources.length < 1 || sources.length > 2) {
      throw new UnscopedValidationError(lit`SrtCallerSourcesLimit`, 'You must specify 1 or 2 SRT caller sources.');
    }
    return new InputConfiguration(() => ({
      type: InputType.SRT_CALLER,
      inputClass: sources.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
      srtSettings: {
        srtCallerSources: sources.map(s => ({
          srtListenerAddress: s.srtListenerAddress,
          srtListenerPort: s.srtListenerPort.toString(),
          minimumLatency: s.minimumLatency?.toMilliseconds(),
          streamId: s.streamId,
          decryption: s.decryption ? {
            algorithm: s.decryption.algorithm?.value,
            passphraseSecretArn: s.decryption.passphraseSecret?.secretArn,
          } : undefined,
        })),
      },
    }));
  }

  /** Create an SRT listener input configuration. */
  public static srtListener(props: SrtListenerInputProps): InputConfiguration {
    return new InputConfiguration(() => ({
      type: InputType.SRT_LISTENER,
      inputClass: props.inputClass?.value ?? 'SINGLE_PIPELINE',
      inputSecurityGroups: props.inputSecurityGroups.map(sg => sg.inputSecurityGroupRef.inputSecurityGroupId),
      srtSettings: {
        srtListenerSettings: {
          minimumLatency: props.minimumLatency?.toMilliseconds(),
          streamId: props.streamId,
          decryption: props.decryption ? {
            algorithm: props.decryption.algorithm?.value,
            passphraseSecretArn: props.decryption.passphraseSecret?.secretArn,
          } : undefined,
        },
      },
    }));
  }

  private static createPullInput(type: InputType, sources: InputSource[]): InputConfiguration {
    if (sources.length < 1) {
      throw new UnscopedValidationError(lit`PullInputSourcesMinimum`, 'You must specify at least 1 input source.');
    }
    if (sources.length > 2) {
      throw new UnscopedValidationError(lit`PullInputSourcesLimit`, 'You cannot specify more than 2 input sources.');
    }
    return new InputConfiguration(() => {
      return {
        type,
        inputClass: sources.length === 2 ? 'STANDARD' : 'SINGLE_PIPELINE',
        inputSources: sources,
        sources: sources.map(s => s._bind()),
      };
    });
  }

  private constructor(private readonly _bindFn: (scope?: Construct) => InputBindConfig) {}

  /** @internal */
  public _bind(scope?: Construct): InputBindConfig {
    return this._bindFn(scope);
  }
}

/**
 * Create the default IAM role for an input that MediaLive assumes (MediaConnect flow or CDI/VPC).
 * The caller wires the specific managed/EC2 actions onto the role.
 */
function createInputRole(scope: Construct | undefined, id: string): IRole {
  if (!scope) {
    throw new UnscopedValidationError(
      lit`InputRoleScope`,
      'Cannot auto-create an input role without a construct scope; provide an explicit `role`.',
    );
  }
  return new Role(scope, id, {
    assumedBy: new ServicePrincipal('medialive.amazonaws.com', {
      conditions: {
        StringEquals: { 'aws:SourceAccount': Aws.ACCOUNT_ID },
        ArnLike: {
          'aws:SourceArn': Stack.of(scope).formatArn({
            service: 'medialive',
            resource: 'input',
            resourceName: '*',
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        },
      },
    }),
  });
}

/**
 * Defines an AWS Elemental MediaLive Input.
 */
@propertyInjectable
export class Input extends Resource implements IInput {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.Input';

  /** Import an existing input by its ARN. The id is parsed out of the ARN. */
  public static fromInputArn(scope: Construct, id: string, inputArn: string): IInput {
    const inputId = extractResourceId(inputArn, 'Input');

    class Import extends Resource implements IInput {
      public readonly inputArn = inputArn;
      public readonly inputId = inputId;
      public readonly inputClass = undefined;
      public readonly inputType = undefined;
      public readonly inputDestinations = undefined;
      public readonly inputSources = undefined;
      public get inputRef(): InputReference {
        return { inputId: this.inputId, inputArn: this.inputArn };
      }
      /** @internal Imported inputs own no sources, so granting is a no-op. */
      public _grantPermissions(_role: IRole): void {}
    }
    return new Import(scope, id);
  }

  public readonly inputArn: string;
  public readonly inputId: string;
  public readonly inputClass?: string;
  public readonly inputType?: string;
  public readonly inputDestinations?: string[];
  public readonly inputSources?: string[];

  /** A reference to this Input resource. */
  public get inputRef(): InputReference {
    return { inputId: this.inputId, inputArn: this.inputArn };
  }

  private readonly inputSourceRefs: InputSource[];

  constructor(scope: Construct, id: string, props: InputProps) {
    super(scope, id, {
      physicalName: props.inputName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    addConstructMetadata(this, props);

    const config = props.input._bind(this);

    // Security-group rules: on-premises inputs don't support security groups, while cloud push
    // inputs require at least one. Both always fail at deploy otherwise, so validate at synth.
    const isOnPremises = props.inputNetworkLocation?.value === InputNetworkLocation.ON_PREMISES.value;
    const hasSecurityGroups = (config.inputSecurityGroups?.length ?? 0) > 0;
    const isPushInput = config.type === InputType.RTMP_PUSH
      || config.type === InputType.RTP_PUSH
      || config.type === InputType.UDP_PUSH;
    if (isOnPremises && hasSecurityGroups) {
      throw new ValidationError(
        lit`OnPremisesInputSecurityGroups`,
        'on-premises inputs do not support input security groups; remove inputSecurityGroups for an ON_PREMISES input',
        this,
      );
    }
    if (!isOnPremises && isPushInput && !hasSecurityGroups) {
      throw new ValidationError(
        lit`PushInputSecurityGroupsRequired`,
        'cloud push inputs require at least one input security group',
        this,
      );
    }
    // SDI inputs are on-premises hardware sources — MediaLive always rejects an SDI input
    // whose network location isn't ON_PREMISES, so fail fast at synth.
    if (config.type === InputType.SDI && !isOnPremises) {
      throw new ValidationError(
        lit`SdiInputRequiresOnPremises`,
        "an SDI input's inputNetworkLocation must be InputNetworkLocation.ON_PREMISES",
        this,
      );
    }

    const resource = new CfnInput(this, 'Resource', {
      name: this.physicalName,
      type: config.type,
      roleArn: config.roleArn,
      sources: config.sources,
      destinations: config.destinations,
      mediaConnectFlows: config.mediaConnectFlows,
      inputDevices: config.inputDevices,
      multicastSettings: config.multicastSettings,
      smpte2110ReceiverGroupSettings: config.smpte2110ReceiverGroupSettings,
      routerSettings: config.routerSettings,
      srtSettings: config.srtSettings,
      sdiSources: config.sdiSources,
      vpc: config.vpc,
      inputSecurityGroups: config.inputSecurityGroups,
      inputNetworkLocation: props.inputNetworkLocation?.value,
      tags: props.tags,
    });

    this.inputArn = resource.inputRef.inputArn;
    this.inputId = resource.inputRef.inputId;
    this.inputClass = config.inputClass;
    this.inputType = config.type;
    this.inputDestinations = resource.attrDestinations;
    this.inputSources = resource.attrSources;
    this.inputSourceRefs = config.inputSources ?? [];

    // MediaConnect flow inputs: grant managed flow-management actions.
    // Resource is `*` — service rejects flow-ARN-scoped grants.
    if (config.mediaConnectFlowGrant) {
      Grant.addToPrincipal({
        grantee: config.mediaConnectFlowGrant.role,
        actions: [
          'mediaconnect:ManagedDescribeFlow',
          'mediaconnect:ManagedAddOutput',
          'mediaconnect:ManagedRemoveOutput',
        ],
        resourceArns: ['*'],
      }).applyBefore(resource);
    }

    // CDI/VPC inputs: Create/Delete scoped to ENI + subnets + SGs; Describe* needs `*`.
    if (config.vpcRoleGrant) {
      const stack = Stack.of(this);
      const grant = config.vpcRoleGrant;
      const networkInterfaceArn = stack.formatArn({ service: 'ec2', resource: 'network-interface', resourceName: '*' });
      const subnetArns = grant.subnetIds.map(s => stack.formatArn({ service: 'ec2', resource: 'subnet', resourceName: s }));
      const securityGroupArns = grant.securityGroupIds.map(sg => stack.formatArn({ service: 'ec2', resource: 'security-group', resourceName: sg }));

      Grant.addToPrincipal({
        grantee: grant.role,
        actions: [
          'ec2:CreateNetworkInterface',
          'ec2:CreateNetworkInterfacePermission',
          'ec2:DeleteNetworkInterface',
        ],
        resourceArns: [networkInterfaceArn, ...subnetArns, ...securityGroupArns],
      }).applyBefore(resource);
      Grant.addToPrincipal({
        grantee: grant.role,
        actions: [
          'ec2:DescribeNetworkInterfaces',
          'ec2:DescribeSubnets',
          'ec2:DescribeSecurityGroups',
        ],
        resourceArns: ['*'], // Describe* actions don't support resource-level permissions
      }).applyBefore(resource);
    }
  }

  /** @internal */
  public _grantPermissions(role: IRole): void {
    this.inputSourceRefs.forEach(s => s._grantRead(role));
  }
}
