/**
 *  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 *  with the License. A copy of the License is located at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES
 *  OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions
 *  and limitations under the License.
 */

import type { IConstruct, Construct } from 'constructs';
import type { IMemoryStrategy } from './memory-strategy';
import { MemoryPerms } from './perms';
import type { CfnMemoryProps, IMemoryRef, MemoryReference } from '../../../aws-bedrockagentcore';
import { CfnMemory } from '../../../aws-bedrockagentcore';
import type {
  MetricOptions,
  MetricProps,
} from '../../../aws-cloudwatch';
import {
  Metric,
  Stats,
} from '../../../aws-cloudwatch';
import * as iam from '../../../aws-iam';
import type * as kinesis from '../../../aws-kinesis';
import * as kms from '../../../aws-kms';
import { Arn, ArnFormat, Duration, Lazy, Resource, Stack, Token, Names } from '../../../core';
import type { IResource, ResourceProps } from '../../../core';
import { ValidationError } from '../../../core/lib/errors';
import { lit } from '../../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../../core/lib/metadata-resource';
import { propertyInjectable } from '../../../core/lib/prop-injectable';
import { validateFieldPattern, validateStringFieldLength, throwIfInvalid } from '../common/validation-helpers';
// Internal Libs

/******************************************************************************
 *                              CONSTANTS
 *****************************************************************************/
/**
 * Minimum length for browser name
 * @internal
 */
const MEMORY_NAME_MIN_LENGTH = 1;

/**
 * Maximum length for browser name
 * @internal
 */
const MEMORY_NAME_MAX_LENGTH = 48;

/**
 * Minimum length for browser tags
 * @internal
 */
const MEMORY_TAG_MIN_LENGTH = 1;

/**
 * Maximum length for browser tags
 * @internal
 */
const MEMORY_TAG_MAX_LENGTH = 256;

/**
 * Minimum length for memory expiration days
 * @internal
 */
const MEMORY_EXPIRATION_DAYS_MIN = 7;
/**
 * Maximum length for memory expiration days
 * @internal
 */
const MEMORY_EXPIRATION_DAYS_MAX = 365;

/******************************************************************************
 *                         Stream Delivery Types
 *****************************************************************************/
/**
 * Content type for stream delivery.
 * Defines what kind of memory content is delivered to the Kinesis stream.
 */
export enum StreamDeliveryContentType {
  /** Deliver memory record lifecycle events (created, updated, deleted) */
  MEMORY_RECORDS = 'MEMORY_RECORDS',
}

/**
 * Content detail level for stream delivery.
 * Controls how much detail is included in each delivered record.
 */
export enum StreamDeliveryContentLevel {
  /** Deliver only metadata (record ID, timestamps, event type) */
  METADATA_ONLY = 'METADATA_ONLY',
  /** Deliver full content including the memory record body */
  FULL_CONTENT = 'FULL_CONTENT',
}

/**
 * Content configuration for a stream delivery resource.
 * Defines what content type and detail level to deliver.
 */
export interface StreamDeliveryContentConfiguration {
  /**
   * The type of content to deliver.
   */
  readonly type: StreamDeliveryContentType;
  /**
   * The level of content detail to deliver.
   *
   * There is no default: the level must be chosen explicitly because
   * `FULL_CONTENT` delivers complete memory record bodies, which can contain
   * personally identifiable information and other sensitive conversation
   * content. Use `METADATA_ONLY` unless the record body is required downstream.
   */
  readonly level: StreamDeliveryContentLevel;
}

/**
 * Options for delivering memory record events to a Kinesis Data Stream.
 */
export interface KinesisStreamDeliveryOptions {
  /**
   * Content configurations defining what to deliver to the stream.
   *
   * Currently exactly one configuration is supported.
   */
  readonly contentConfigurations: StreamDeliveryContentConfiguration[];
}

/**
 * A delivery target for real-time streaming of memory record lifecycle events.
 *
 * Instances are created through the static factory methods, one per delivery
 * target type, for example `StreamDeliveryResource.kinesis()`.
 *
 * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-record-streaming.html
 */
export class StreamDeliveryResource {
  /**
   * Deliver memory record lifecycle events to an Amazon Kinesis Data Stream.
   *
   * The memory execution role is automatically granted write permissions to
   * the stream.
   *
   * @param stream The Kinesis Data Stream to deliver memory record events to
   * @param options What content to deliver to the stream
   */
  public static kinesis(stream: kinesis.IStream, options: KinesisStreamDeliveryOptions): StreamDeliveryResource {
    return new StreamDeliveryResource(stream, options.contentConfigurations);
  }

  /**
   * The Kinesis Data Stream that memory record events are delivered to.
   */
  public readonly stream: kinesis.IStream;

  /**
   * Content configurations defining what is delivered to the stream.
   */
  public readonly contentConfigurations: StreamDeliveryContentConfiguration[];

  private constructor(stream: kinesis.IStream, contentConfigurations: StreamDeliveryContentConfiguration[]) {
    this.stream = stream;
    this.contentConfigurations = contentConfigurations;
  }
}

/******************************************************************************
 *                                Interface
 *****************************************************************************/
/**
 * Interface for Memory resources
 */
export interface IMemory extends IResource, iam.IGrantable, IMemoryRef {
  /**
   * The ARN of the memory resource
   * @attribute
   */
  readonly memoryArn: string;
  /**
   * The id of the memory
   * @attribute
   */
  readonly memoryId: string;
  /**
   * The IAM role that provides permissions for the memory to access AWS services.
   */
  readonly executionRole?: iam.IRole;
  /**
   * Custom KMS key for encryption (if provided)
   */
  readonly kmsKey?: kms.IKey;
  /**
   * The status of the memory
   * @attribute
   */
  readonly status?: string;
  /**
   * Timestamp when the memory was last updated
   * @attribute
   */
  readonly updatedAt?: string;
  /**
   * Timestamp when the memory was created
   * @attribute
   */
  readonly createdAt?: string;
  /**
   * Grant the given principal identity permissions to perform actions on this memory.
   */
  grant(grantee: iam.IGrantable, ...actions: string[]): iam.Grant;
  /**
   * Grant the given principal identity permissions to write content to this memory.
   */
  grantWrite(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to read the contents of this memory.
   * Both Short-Term Memory (STM) and Long-Term Memory (LTM).
   */
  grantRead(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to read the Short-Term Memory (STM) contents of this memory.
   */
  grantReadShortTermMemory(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to read the Long-Term Memory (LTM) contents of this memory.
   */
  grantReadLongTermMemory(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to delete content on this memory.
   */
  grantDelete(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to delete Short-Term Memory (STM) content on this memory.
   */
  grantDeleteShortTermMemory(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to delete Long-Term Memory (LTM) content on this memory.
   */
  grantDeleteLongTermMemory(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to manage the control plane of this memory.
   */
  grantAdmin(grantee: iam.IGrantable): iam.Grant;
  /**
   * Grant the given principal identity permissions to do every action on this memory.
   */
  grantFullAccess(grantee: iam.IGrantable): iam.Grant;

  // ------------------------------------------------------
  // Metrics
  // ------------------------------------------------------
  /**
   * Return the given named metric for this memory.
   */
  metric(metricName: string, props?: MetricOptions): Metric;
  /**
   * Return the given named metric related to the API operation performed on this memory.
   */
  metricForApiOperation(metricName: string, operation: string, props?: MetricOptions): Metric;
  /**
   * Return a metric measuring the latency of a specific API operation performed on this memory.
   */
  metricLatencyForApiOperation(operation: string, props?: MetricOptions): Metric;
  /**
   * Return a metric containing the total number of API requests made for a specific memory operation.
   */
  metricInvocationsForApiOperation(operation: string, props?: MetricOptions): Metric;
  /**
   * Return a metric containing the number of errors for a specific API operation performed on this memory.
   */
  metricErrorsForApiOperation(operation: string, props?: MetricOptions): Metric;
  /**
   * Returns the metric containing the number of created memory events and memory records.
   */
  metricEventCreationCount(props?: MetricOptions): Metric;
}

/******************************************************************************
 *                        ABSTRACT BASE CLASS
 *****************************************************************************/
/**
 * Abstract base class for a Memory.
 * Contains methods and attributes valid for Memories either created with CDK or imported.
 */
export abstract class MemoryBase extends Resource implements IMemory {
  public abstract readonly memoryArn: string;
  public abstract readonly memoryId: string;
  public abstract readonly status?: string;
  public abstract readonly updatedAt?: string;
  public abstract readonly createdAt?: string;
  public abstract readonly executionRole?: iam.IRole;
  public abstract readonly kmsKey?: kms.IKey;
  /**
   * The principal to grant permissions to
   */
  public abstract readonly grantPrincipal: iam.IPrincipal;

  /**
   * A reference to a Memory resource.
   */
  public get memoryRef(): MemoryReference {
    return { memoryArn: this.memoryArn };
  }

  constructor(scope: Construct, id: string, props: ResourceProps = {}) {
    super(scope, id, props);
  }
  /**
   * Grants IAM actions to the IAM Principal
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant permissions to
   * @param actions - The actions to grant
   * @returns An IAM Grant object representing the granted permissions
   */
  grant(grantee: iam.IGrantable, ...actions: string[]): iam.Grant {
    return iam.Grant.addToPrincipal({
      grantee,
      actions,
      resourceArns: [this.memoryRef.memoryArn],
    });
  }
  /**
   * Grant the given principal identity permissions to write content to short-term memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant read permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:CreateEvent'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantWrite(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.STM.WRITE_PERMS);
  }
  /**
   * Grant the given principal identity permissions to read the contents of this memory.
   * Both Short-Term Memory (STM) and Long-Term Memory (LTM).
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant read permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:GetMemoryRecord',
      'bedrock-agentcore:RetrieveMemoryRecords',
      'bedrock-agentcore:ListMemoryRecords',
      'bedrock-agentcore:ListActors',
      'bedrock-agentcore:ListSessions] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantRead(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.READ_PERMS);
  }
  /**
   * Grant the given principal identity permissions to read the Short-Term Memory (STM) contents of this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant read permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:GetEvent',
      'bedrock-agentcore:ListEvents',
      'bedrock-agentcore:ListActors',
      'bedrock-agentcore:ListSessions',] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantReadShortTermMemory(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.STM.READ_PERMS);
  }
  /**
   * Grant the given principal identity permissions to read the Long-Term Memory (LTM) contents of this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant read permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:GetMemoryRecord',
      'bedrock-agentcore:RetrieveMemoryRecords',
      'bedrock-agentcore:ListMemoryRecords',
      'bedrock-agentcore:ListActors',
      'bedrock-agentcore:ListSessions',] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantReadLongTermMemory(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.LTM.READ_PERMS);
  }
  /**
   * Grant the given principal identity permissions to delete content on this memory.
   *
   * Both Short-Term Memory (STM) and Long-Term Memory (LTM).
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant delete permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:DeleteEvent',
      'bedrock-agentcore:DeleteMemoryRecord'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantDelete(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.DELETE_PERMS);
  }
  /**
   * Grant the given principal identity permissions to delete Short-Term Memory (STM) content on this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant delete permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:DeleteEvent'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantDeleteShortTermMemory(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.STM.DELETE_PERMS);
  }
  /**
   * Grant the given principal identity permissions to delete Long-Term Memory (LTM) content on this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant delete permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:DeleteMemoryRecord'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantDeleteLongTermMemory(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.LTM.DELETE_PERMS);
  }
  /**
   * Grant the given principal identity permissions to manage the control plane of this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant admin permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:CreateMemory',
      'bedrock-agentcore:GetMemory',
      'bedrock-agentcore:DeleteMemory',
      'bedrock-agentcore:UpdateMemory'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantAdmin(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.ADMIN_PERMS);
  }
  /**
   * Grant the given principal identity permissions to do every action on this memory.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee - The IAM principal to grant full access permissions to
   * @default - Default grant configuration:
   * - actions: ['bedrock-agentcore:CreateEvent',
      'bedrock-agentcore:GetEvent',
      'bedrock-agentcore:DeleteEvent',
      'bedrock-agentcore:GetMemoryRecord',
      'bedrock-agentcore:RetrieveMemoryRecords',
      'bedrock-agentcore:ListMemoryRecords',
      'bedrock-agentcore:ListActors',
      'bedrock-agentcore:ListSessions',
      'bedrock-agentcore:CreateMemory',
      'bedrock-agentcore:GetMemory',
      'bedrock-agentcore:DeleteMemory',
      'bedrock-agentcore:UpdateMemory'] on this.memoryArn
   * @returns An IAM Grant object representing the granted permissions
   */
  grantFullAccess(grantee: iam.IGrantable): iam.Grant {
    return this.grant(grantee, ...MemoryPerms.FULL_ACCESS_PERMS);
  }

  // ------------------------------------------------------
  // Metrics
  // ------------------------------------------------------
  /**
   * Return the given named metric for this memory.
   *
   * By default, the metric will be calculated as a sum over a period of 5 minutes.
   * You can customize this by using the `statistic` and `period` properties.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    const metricProps: MetricProps = {
      namespace: 'AWS/Bedrock-AgentCore',
      metricName,
      ...props,
      dimensionsMap: { Resource: this.memoryRef.memoryArn, ...props?.dimensionsMap },
    };
    return this.configureMetric(metricProps);
  }
  /**
   * Return the given named metric related to the API operation performed on this memory.
   */
  public metricForApiOperation(
    metricName: string,
    operation: string,
    props?: MetricOptions,
  ): Metric {
    return this.metric(metricName, { dimensionsMap: { Operation: operation }, ...props });
  }
  /**
   * Return a metric measuring the latency of a specific API operation performed on this memory.
   *
   * The latency metric represents the total time elapsed between receiving the request and sending
   * the final response token, measuring complete end-to-end processing time.
   *
   * For memory creation events specifically, this measures the time from the last CreateEvent
   * that met strategy criteria until memory storage is completed.
   *
   */
  public metricLatencyForApiOperation(operation: string, props?: MetricOptions): Metric {
    return this.metricForApiOperation('Latency', operation, { statistic: Stats.AVERAGE, ...props });
  }
  /**
   * Return a metric containing the total number of API requests made for a specific memory operation like
   * `CreateEvent`, `ListEvents`, `RetrieveMemoryRecords` ...
   */
  public metricInvocationsForApiOperation(operation: string, props?: MetricOptions): Metric {
    return this.metricForApiOperation('Invocations', operation, {
      statistic: Stats.SUM,
      ...props,
    });
  }
  /**
   * Return a metric containing the number of errors for a specific API operation performed on this memory.
   */
  public metricErrorsForApiOperation(operation: string, props?: MetricOptions): Metric {
    return this.metricForApiOperation('Errors', operation, { statistic: Stats.SUM, ...props });
  }
  /**
   * Returns the metric containing the number of short-term memory events.
   */
  public metricEventCreationCount(props?: MetricOptions): Metric {
    return this.metric('CreationCount', { dimensionsMap: { ItemType: 'Event' }, statistic: Stats.SUM, ...props });
  }
  /**
   * Returns the metric containing the number of long-term memory records
   * created by the long-term extraction strategies.
   */
  public metricMemoryRecordCreationCount(props?: MetricOptions): Metric {
    return this.metric(
      'CreationCount',
      { dimensionsMap: { ItemType: 'MemoryRecordsExtracted' }, statistic: Stats.SUM, ...props },
    );
  }
  /**
   * Internal method to create a metric.
   */
  private configureMetric(props: MetricProps) {
    return new Metric({
      ...props,
      region: props?.region ?? this.stack.region,
      account: props?.account ?? this.stack.account,
    });
  }
}

/******************************************************************************
 *                        PROPS FOR NEW CONSTRUCT
 *****************************************************************************/
/**
 * Properties for creating a Memory resource
 */
export interface MemoryProps {
  /**
   * The name of the memory
   * Valid characters are a-z, A-Z, 0-9, _ (underscore)
   * The name must start with a letter and can be up to 48 characters long
   * Pattern: [a-zA-Z][a-zA-Z0-9_]{0,47}
   * @default - auto generate
   */
  readonly memoryName?: string;
  /**
   * Short-term memory expiration in days (between 7 and 365).
   * Sets the short-term (raw event) memory retention.
   * Events older than the specified duration will expire and no longer be stored.
   * @default - 90 days
   */
  readonly expirationDuration?: Duration;
  /**
   * Optional description for the memory
   * Valid characters are a-z, A-Z, 0-9, _ (underscore), - (hyphen) and spaces
   * The description can have up to 200 characters
   * @default - No description
   */
  readonly description?: string;
  /**
   * Custom KMS key to use for encryption.
   * @default - Your data is encrypted with a key that AWS owns and manages for you
   */
  readonly kmsKey?: kms.IKey;
  /**
   * If you need long-term memory for context recall across sessions,
   * you can setup memory extraction strategies to extract the relevant memory from the raw events.
   * @default - No extraction strategies (short term memory only)
   */
  readonly memoryStrategies?: IMemoryStrategy[];
  /**
   * The IAM role that provides permissions for the memory to access AWS services
   * when using custom strategies.
   *
   * @default - A new role will be created.
   */
  readonly executionRole?: iam.IRole;
  /**
   * Tags (optional)
   * A list of key:value pairs of tags to apply to this memory resource
   *
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };

  /**
   * Stream delivery resources for real-time push-based streaming of memory
   * record lifecycle events (created, updated, deleted) to Amazon Kinesis Data Streams.
   *
   * The memory execution role will automatically be granted write permissions to each stream.
   *
   * Only one stream delivery resource is currently supported (CloudFormation maximum);
   * providing more than one fails at synth with `TooManyStreamDeliveryResources`:
   *
   * ```ts
   * declare const stream: kinesis.IStream;
   * new agentcore.Memory(this, 'Memory', {
   *   streamDeliveryResources: [
   *     agentcore.StreamDeliveryResource.kinesis(stream, {
   *       contentConfigurations: [{
   *         type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
   *         level: agentcore.StreamDeliveryContentLevel.METADATA_ONLY,
   *       }],
   *     }),
   *   ],
   * });
   * ```
   *
   * @default - No stream delivery (events are not pushed to Kinesis)
   * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-record-streaming.html
   */
  readonly streamDeliveryResources?: StreamDeliveryResource[];
}

/******************************************************************************
 *                      ATTRS FOR IMPORTED CONSTRUCT
 *****************************************************************************/
/**
 * Attributes for specifying an imported Memory.
 */
export interface MemoryAttributes {
  /**
   * The ARN of the memory.
   * @attribute
   */
  readonly memoryArn: string;
  /**
   * The ARN of the IAM role associated to the memory.
   * @attribute
   */
  readonly roleArn: string;
  /**
   * When this memory was last updated.
   * @default undefined - No last updated timestamp is provided
   */
  readonly updatedAt?: string;
  /**
   * Optional KMS encryption key associated with this memory
   * @default undefined - An AWS managed key is used
   */
  readonly kmsKeyArn?: string;
  /**
   * The status of the memory.
   * @default undefined - No status is provided
   */
  readonly status?: string;
  /**
   * The created timestamp of the memory.
   * @default undefined - No created timestamp is provided
   */
  readonly createdAt?: string;
}

/******************************************************************************
 *                                Class
 *****************************************************************************/
/**
 * Long-term memory store for extracted insights like user preferences, semantic facts and summaries.
 * Enables knowledge retention across sessions by storing user preferences (e.g. coding style),
 * semantic facts (e.g. learned info) and interaction summaries for context optimization.
 *
 * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html
 * @resource AWS::BedrockAgentCore::Memory
 */
@propertyInjectable
export class Memory extends MemoryBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-bedrockagentcore.Memory';

  /**
   * Static Method for importing an existing Bedrock AgentCore Memory.
   */
  /**
   * Creates an Memory reference from an existing memory's attributes.
   *
   * @param scope - The construct scope
   * @param id - Identifier of the construct
   * @param attrs - Attributes of the existing browser custom
   * @returns An IBrowserCustom reference to the existing browser
   */
  public static fromMemoryAttributes(scope: Construct, id: string, attrs: MemoryAttributes): IMemory {
    class Import extends MemoryBase {
      public readonly memoryArn = attrs.memoryArn;
      public readonly memoryId = Arn.split(attrs.memoryArn, ArnFormat.SLASH_RESOURCE_NAME).resourceName!;
      public readonly executionRole = iam.Role.fromRoleArn(scope, `${id}Role`, attrs.roleArn);
      public readonly kmsKey = attrs.kmsKeyArn ? kms.Key.fromKeyArn(scope, `${id}Key`, attrs.kmsKeyArn) : undefined;
      public readonly updatedAt = attrs.updatedAt;
      public readonly grantPrincipal = this.executionRole;
      public readonly status = attrs.status;
      public readonly createdAt = attrs.createdAt;

      constructor(s: Construct, i: string) {
        super(s, i);

        this.grantPrincipal = this.executionRole || new iam.UnknownPrincipal({ resource: this });
      }
    }

    // Return new Memory
    return new Import(scope, id);
  }

  // ------------------------------------------------------
  // Attributes
  // ------------------------------------------------------
  /**
   * The ARN of the memory resource.
   * @attribute
   */
  public readonly memoryArn: string;
  /**
   * The name of the memory.
   * @attribute
   */
  public readonly memoryName: string;
  /**
   * The id of the memory.
   * @attribute
   */
  public readonly memoryId: string;
  /**
   * The expiration days of the memory.
   */
  public readonly expirationDuration?: Duration;
  /**
   * The failure reason of the browser
   * @attribute
   */
  public readonly failureReason?: string;
  /**
   * The description of the memory.
   */
  public readonly description?: string;
  /**
   * The execution role of the memory.
   */
  public readonly executionRole?: iam.IRole;
  /**
   * The status of the memory.
   */
  public readonly status?: string;
  /**
   * The created timestamp of the memory.
   */
  public readonly createdAt?: string;
  /**
   * The updated at timestamp of the memory.
   */
  public readonly updatedAt?: string;
  /**
   * Tags applied to this browser resource
   * A map of key-value pairs for resource tagging
   * @default - No tags applied
   */
  public readonly tags?: { [key: string]: string };
  /**
   * The principal to grant permissions to
   */
  public readonly grantPrincipal: iam.IPrincipal;
  /**
   * The KMS key used to encrypt the memory.
   */
  public readonly kmsKey?: kms.IKey;
  /**
   * The memory strategies used by the memory.
   * @attribute
   */
  public readonly memoryStrategies: IMemoryStrategy[] = [];
  /**
   * The stream delivery resources configured for this memory.
   */
  public readonly streamDeliveryResources: StreamDeliveryResource[] = [];
  // ------------------------------------------------------
  // Internal Only
  // ------------------------------------------------------
  private readonly __resource: CfnMemory;

  // ------------------------------------------------------
  // CONSTRUCTOR
  // ------------------------------------------------------
  constructor(scope: Construct, id: string, props: MemoryProps = {}) {
    super(scope, id, {
      // Maximum name length of 48 characters
      // @see https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-bedrockagentcore-memory.html#cfn-bedrockagentcore-memory-name
      physicalName: props?.memoryName ??
        Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 48 }) }),
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    // ------------------------------------------------------
    // Set properties and defaults
    // ------------------------------------------------------
    this.memoryName = this.physicalName;
    this.expirationDuration = props.expirationDuration ?? Duration.days(90);
    this.description = props.description;
    this.kmsKey = props.kmsKey;
    this.executionRole = props.executionRole ?? this._createMemoryRole();
    this.grantPrincipal = this.executionRole;
    this.tags = props.tags;

    // ------------------------------------------------------
    // Permissions
    // ------------------------------------------------------
    // For KMS permissions see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/storage-encryption.html
    if (this.kmsKey) {
      this.kmsKey.grant(this.executionRole,
        'kms:CreateGrant',
        'kms:Decrypt',
        'kms:DescribeKey',
        'kms:GenerateDataKey',
        'kms:GenerateDataKeyWithoutPlaintext',
        'kms:ReEncrypt*',
      );
    }

    // ------------------------------------------------------
    // Validations
    // ------------------------------------------------------

    // Validate memory name
    throwIfInvalid(this._validateMemoryName, this.memoryName, this);

    // Validate expiration duration
    throwIfInvalid(this._validateMemoryExpirationDays, this.expirationDuration.toDays());

    // Validate memory tags
    throwIfInvalid(this._validateMemoryTags, this.tags, this);

    // Memory strategies are already validated when building them, so no need to validate them here

    // ------------------------------------------------------
    // CFN Props - With Lazy support
    // ------------------------------------------------------
    const cfnProps: CfnMemoryProps = {
      name: this.memoryName,
      description: this.description,
      eventExpiryDuration: this.expirationDuration.toDays(),
      encryptionKeyArn: this.kmsKey?.keyArn,
      memoryExecutionRoleArn: this.executionRole?.roleArn,
      memoryStrategies: Lazy.any({ produce: () => this._renderMemoryStrategies() }, { omitEmptyArray: true }),
      streamDeliveryResources: Lazy.any(
        { produce: () => this._renderStreamDeliveryResources() },
        { omitEmptyArray: true },
      ),
      tags: this.tags,
    };

    // ------------------------------------------------------
    // CFN Resource
    // ------------------------------------------------------
    this.__resource = new CfnMemory(this, 'Memory', cfnProps);

    this.memoryId = this.__resource.attrMemoryId;
    this.memoryArn = this.__resource.attrMemoryArn;
    this.status = this.__resource.attrStatus;
    this.updatedAt = this.__resource.attrUpdatedAt;
    this.createdAt = this.__resource.attrCreatedAt;
    this.failureReason = this.__resource.attrFailureReason;

    // Add memory strategies to the memory
    for (const strategy of props?.memoryStrategies ?? []) {this.addMemoryStrategy(strategy);}

    // Add stream delivery resources to the memory
    for (const resource of props?.streamDeliveryResources ?? []) {this.addStreamDeliveryResource(resource);}
  }

  // ------------------------------------------------------
  // HELPER METHODS - addX()
  // ------------------------------------------------------
  /**
   * Add memory strategy to the memory.
   * @default - No memory strategies.
   */
  @MethodMetadata()
  public addMemoryStrategy(memoryStrategy: IMemoryStrategy) {
    // Add the memory strategy to the memory
    this.memoryStrategies.push(memoryStrategy);

    // Grant necessary permissions to the execution role
    const grant = memoryStrategy.grant(this.executionRole as iam.IRole);
    grant?.applyBefore(this.__resource);
  }

  /**
   * Add a stream delivery resource to the memory.
   * Grants Kinesis write permissions to the execution role automatically.
   *
   * @param resource - The stream delivery resource configuration
   * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-record-streaming.html
   */
  @MethodMetadata()
  public addStreamDeliveryResource(resource: StreamDeliveryResource): void {
    // Validate current limit: at most 1 stream delivery resource
    if (this.streamDeliveryResources.length >= 1) {
      throw new ValidationError(
        lit`TooManyStreamDeliveryResources`,
        'Memory currently supports at most one stream delivery resource',
        this,
      );
    }

    // Validate content configurations
    throwIfInvalid(this._validateStreamDeliveryResource, resource);

    // Add to internal array
    this.streamDeliveryResources.push(resource);

    // Grant Kinesis write permissions to the execution role
    // stream.grantWrite() grants: kinesis:ListShards, kinesis:PutRecord, kinesis:PutRecords
    // stream.grantWrite() also grants KMS permissions when the stream's key is known
    // (owned streams, or streams imported via fromStreamAttributes with an encryptionKey).
    // Streams imported via fromStreamArn carry no key, so grant KMS manually in that case.
    const grant = resource.stream.grantWrite(this.executionRole as iam.IRole);

    // AgentCore also requires kinesis:DescribeStream which is not included in grantWrite().
    // Route it through the stream's own grant so that, like the write grant, a
    // cross-account stream also gets the matching resource policy statement.
    const describeGrant = resource.stream.grant(this.executionRole as iam.IRole, 'kinesis:DescribeStream');

    grant.applyBefore(this.__resource);
    describeGrant.applyBefore(this.__resource);
  }

  /**
   * Creates execution role needed for the memory to access AWS services
   * @returns The created role
   * @internal This is an internal core function and should not be called directly.
   */
  private _createMemoryRole(): iam.IRole {
    const role = new iam.Role(this, 'ServiceRole', {
      // The service appends a random suffix to the resource name at creation time (e.g. MyMemory-a1b2c3d4e5),
      // so we use ArnLike with a wildcard to match.
      // @see https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-bedrockagentcore-memory.md
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com', {
        conditions: {
          StringEquals: { 'aws:SourceAccount': Stack.of(this).account },
          ArnLike: {
            'aws:SourceArn': Arn.format({ service: 'bedrock-agentcore', resource: 'memory', resourceName: `${this.memoryName}*` }, Stack.of(this)),
          },
        },
      }),
    });

    return role;
  }

  // ------------------------------------------------------
  // VALIDATORS
  // ------------------------------------------------------
  /**
   * Validates the memory tags format
   * @param tags The tags object to validate
   * @returns Array of validation error messages, empty if valid
   */
  private _validateMemoryTags = (tags?: { [key: string]: string }, scope?: IConstruct): string[] => {
    let errors: string[] = [];
    if (!tags) {
      return errors; // Tags are optional
    }

    // Validate each tag key and value
    for (const [key, value] of Object.entries(tags)) {
      // Validate aws: prefix restriction
      if (key.toLowerCase().startsWith('aws:')) {
        errors.push(`Tag key "${key}" cannot start with "aws:" as this prefix is reserved by AWS`);
      }

      errors.push(...validateStringFieldLength({
        value: key,
        fieldName: 'Tag key',
        minLength: MEMORY_TAG_MIN_LENGTH,
        maxLength: MEMORY_TAG_MAX_LENGTH,
      }, scope));

      // Validate tag key pattern: ^[a-zA-Z0-9\s._:/=+@-]*$
      const validKeyPattern = /^[a-zA-Z0-9\s._:/=+@-]*$/;
      errors.push(...validateFieldPattern(key, 'Tag key', validKeyPattern, undefined, scope));

      // Validate tag value
      errors.push(...validateStringFieldLength({
        value: value,
        fieldName: 'Tag value',
        minLength: MEMORY_TAG_MIN_LENGTH,
        maxLength: MEMORY_TAG_MAX_LENGTH,
      }, scope));

      // Validate tag value pattern: ^[a-zA-Z0-9\s._:/=+@-]*$
      const validValuePattern = /^[a-zA-Z0-9\s._:/=+@-]*$/;
      errors.push(...validateFieldPattern(value, 'Tag value', validValuePattern, undefined, scope));
    }

    return errors;
  };

  /**
   * Validates the memory name format
   * @param name The memory name to validate
   * @returns Array of validation error messages, empty if valid
   */
  private _validateMemoryName = (name: string, scope?: IConstruct): string[] => {
    let errors: string[] = [];

    errors.push(...validateStringFieldLength({
      value: name,
      fieldName: 'Memory name',
      minLength: MEMORY_NAME_MIN_LENGTH,
      maxLength: MEMORY_NAME_MAX_LENGTH,
    }, scope));

    // Check if name matches the AWS API pattern: [a-zA-Z][a-zA-Z0-9_]{0,47}
    // Must start with a letter, followed by up to 47 letters, numbers, or underscores
    const validNamePattern = /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/;
    errors.push(...validateFieldPattern(name, 'Memory name', validNamePattern, undefined, scope));

    return errors;
  };

  /**
   * Validates the memory expiration days
   * @param expirationDays The memory expiration days to validate
   * @returns Array of validation error messages, empty if valid
   */
  private _validateMemoryExpirationDays = (expirationDays: number): string[] => {
    let errors: string[] = [];

    if (Token.isUnresolved(expirationDays)) {
      return errors;
    }

    if (expirationDays < MEMORY_EXPIRATION_DAYS_MIN || expirationDays > MEMORY_EXPIRATION_DAYS_MAX) {
      errors.push(`Memory expiration days must be between ${MEMORY_EXPIRATION_DAYS_MIN} and ${MEMORY_EXPIRATION_DAYS_MAX}`);
    }

    return errors;
  };

  /**
   * Validates a stream delivery resource configuration.
   * Currently CloudFormation limits contentConfigurations to exactly 1 entry
   * and streamDeliveryResources to at most 1 entry.
   */
  private _validateStreamDeliveryResource = (resource: StreamDeliveryResource): string[] => {
    const errors: string[] = [];

    if (!resource.contentConfigurations || resource.contentConfigurations.length === 0) {
      errors.push('contentConfigurations must be specified: choose METADATA_ONLY for event metadata or FULL_CONTENT to stream complete records (may contain sensitive data)');
    }

    if (resource.contentConfigurations && resource.contentConfigurations.length > 1) {
      errors.push('Stream delivery resource currently supports at most one content configuration');
    }

    return errors;
  };

  // ------------------------------------------------------
  // RENDERERS
  // ------------------------------------------------------
  /**
   * Render the memory strategies.
   *
   * @returns Array of MemoryStrategyProperty objects in CloudFormation format, or undefined if no strategies are defined
   * @default - undefined if no strategies are defined or array is empty
   * @internal This is an internal core function and should not be called directly.
   */
  private _renderMemoryStrategies(): CfnMemory.MemoryStrategyProperty[] | undefined {
    if (!this.memoryStrategies || this.memoryStrategies.length === 0) {
      return undefined;
    }

    return this.memoryStrategies.map(ms => ms.render());
  }

  /**
   * Render the stream delivery resources into CloudFormation format.
   *
   * @returns StreamDeliveryResourcesProperty or undefined if no resources configured
   * @internal This is an internal core function and should not be called directly.
   */
  private _renderStreamDeliveryResources(): CfnMemory.StreamDeliveryResourcesProperty | undefined {
    if (!this.streamDeliveryResources || this.streamDeliveryResources.length === 0) {
      return undefined;
    }

    return {
      resources: this.streamDeliveryResources.map(resource => ({
        kinesis: {
          dataStreamArn: resource.stream.streamArn,
          contentConfigurations: resource.contentConfigurations.map(config => ({
            type: config.type,
            level: config.level,
          })),
        },
      })),
    };
  }
}
