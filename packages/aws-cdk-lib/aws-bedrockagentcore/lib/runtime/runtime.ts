
import type { Construct } from 'constructs';
import type { RuntimeAuthorizerConfiguration } from './inbound-auth/runtime-authorizer-configuration';
import type { LoggingConfig } from './observability';
import { configureTracingDelivery, configureLoggingDelivery } from './observability';
import {
  RUNTIME_LOGS_GROUP_ACTIONS,
  RUNTIME_LOGS_DESCRIBE_ACTIONS,
  RUNTIME_LOGS_STREAM_ACTIONS,
  RUNTIME_XRAY_ACTIONS,
  RUNTIME_CLOUDWATCH_METRICS_ACTIONS,
  RUNTIME_CLOUDWATCH_NAMESPACE,
  RUNTIME_WORKLOAD_IDENTITY_ACTIONS,
} from './perms';
import type { AgentRuntimeArtifact } from './runtime-artifact';

import type { IBedrockAgentRuntime, AgentRuntimeAttributes } from './runtime-base';
import { RuntimeBase } from './runtime-base';
import { RuntimeEndpoint } from './runtime-endpoint';
import type { LifecycleConfiguration, RequestHeaderConfiguration } from './types';
import { ProtocolType } from './types';
import * as bedrockagentcore from '../../../aws-bedrockagentcore';
import * as ec2 from '../../../aws-ec2';
import * as iam from '../../../aws-iam';
import { Duration, Stack, Annotations, Token, Arn, ArnFormat, Lazy, Names } from '../../../core';
import { ValidationError } from '../../../core/lib/errors';
import { lit } from '../../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../../core/lib/metadata-resource';
import { propertyInjectable } from '../../../core/lib/prop-injectable';
import { validateStringFieldLength, validateFieldPattern } from '../common/validation-helpers';
import { RuntimeNetworkConfiguration } from '../network/network-configuration';

/******************************************************************************
 *                                Constants
 *****************************************************************************/
/**
 * Minimum timeout for idle runtime sessions
 * @internal
 */
const LIFECYCLE_MIN_TIMEOUT = Duration.seconds(60);

/**
 * Maximum lifetime for the instance
 * @internal
 */
const LIFECYCLE_MAX_LIFETIME = Duration.seconds(28800);

/******************************************************************************
 *                                Props
 *****************************************************************************/

/**
 * Properties for creating a Bedrock Agent Core Runtime resource
 */
export interface RuntimeProps {
  /**
   * The name of the agent runtime
   * Valid characters are a-z, A-Z, 0-9, _ (underscore)
   * Must start with a letter and can be up to 48 characters long
   * Pattern: ^[a-zA-Z][a-zA-Z0-9_]{0,47}$
   * @default - auto generate
   */
  readonly runtimeName?: string;

  /**
   * The artifact configuration for the agent runtime
   * Contains the container configuration with ECR URI
   */
  readonly agentRuntimeArtifact: AgentRuntimeArtifact;

  /**
   * The IAM role that provides permissions for the agent runtime
   * If not provided, a role will be created automatically
   * @default - A new role will be created
   */
  readonly executionRole?: iam.IRole;

  /**
   * Network configuration for the agent runtime
   * @default - RuntimeNetworkConfiguration.usingPublicNetwork()
   */
  readonly networkConfiguration?: RuntimeNetworkConfiguration;

  /**
   * Optional description for the agent runtime
   * @default - No description
   * Length Minimum: 1 , Maximum: 1200
   */
  readonly description?: string;

  /**
   * Protocol configuration for the agent runtime
   * @default - ProtocolType.HTTP
   */
  readonly protocolConfiguration?: ProtocolType;

  /**
   * Environment variables for the agent runtime
   * - Maximum 50 environment variables
   * - Key: Must be 1-100 characters, start with letter or underscore, contain only letters, numbers, and underscores
   * - Value: Must be 0-2048 characters (per CloudFormation specification)
   * @default - No environment variables
   */
  readonly environmentVariables?: { [key: string]: string };

  /**
   * Authorizer configuration for the agent runtime
   * Use RuntimeAuthorizerConfiguration static methods to create the configuration
   * @default - RuntimeAuthorizerConfiguration.iam() (IAM authentication)
   */
  readonly authorizerConfiguration?: RuntimeAuthorizerConfiguration;

  /**
   * Tags for the agent runtime
   * A list of key:value pairs of tags to apply to this Runtime resource
   * @default {} - no tags
   */
  readonly tags?: { [key: string]: string };

  /**
   * Configuration for HTTP request headers that will be passed through to the runtime.
   * @default - No request headers configured
   */
  readonly requestHeaderConfiguration?: RequestHeaderConfiguration;

  /**
   * The life cycle configuration for the AgentCore Runtime.
   * @default - No lifecycle configuration
   */
  readonly lifecycleConfiguration?: LifecycleConfiguration;

  /**
   * Whether to enable X-Ray tracing for this runtime.
   * When enabled, traces will be delivered to AWS X-Ray.
   *
   * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
   * @default false
   */
  readonly tracingEnabled?: boolean;

  /**
   * Logging configuration for the runtime.
   * Allows sending APPLICATION_LOGS and USAGE_LOGS to CloudWatch Logs, S3, or Kinesis Data Firehose.
   *
   * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
   * @default - No logging configured
   */
  readonly loggingConfigs?: LoggingConfig[];

  /**
   * Whether to create resource policies for log/trace delivery.
   *
   * When `false`, the `AWS::Logs::ResourcePolicy` and `AWS::XRay::ResourcePolicy`
   * are not created. This is useful when deploying many runtimes per account/Region,
   * as each resource policy consumes an account-level quota slot (CloudWatch Logs: 10,
   * X-Ray: lower).
   *
   * Setting `false` means you are responsible for ensuring delivery permissions exist.
   * There are two safe ways to use this:
   * - Same-account delivery to a `/aws/vendedlogs/` log group, where the log-delivery
   *   service-linked role grants write access implicitly.
   * - Attaching the delivery resource policy yourself.
   *
   * Otherwise delivery silently fails: synthesis and deploy succeed, but nothing is delivered.
   * Per the [vended-logs delivery docs](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-CloudWatchLogs.html),
   * a resource policy is required for CloudWatch Logs delivery outside the `/aws/vendedlogs/` same-account case.
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/cloudwatch_limits_cwl.html
   * @default true
   */
  readonly manageDeliveryResourcePolicy?: boolean;
}

/**
 * Options for adding an endpoint to the runtime
 */
export interface AddEndpointOptions {
  /**
   * Description for the endpoint, Must be between 1 and 1200 characters
   * @default - No description
   */
  readonly description?: string;
  /**
   * Override the runtime version for this endpoint
   * @default  1
   */
  readonly version?: string;
}

/******************************************************************************
 *                                Class
 *****************************************************************************/

/**
 * Bedrock Agent Core Runtime
 * Enables running containerized agents with specific network configurations,
 * security settings, and runtime artifacts.
 *
 * @resource AWS::BedrockAgentCore::Runtime
 * @see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html
 */
@propertyInjectable
export class Runtime extends RuntimeBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-bedrockagentcore.Runtime';

  /**
   * Import an existing Agent Runtime using attributes
   * This allows you to reference an Agent Runtime that was created outside of CDK
   *
   * @param scope The construct scope
   * @param id The construct id
   * @param attrs The attributes of the existing Agent Runtime
   * @returns An IBedrockAgentRuntime instance representing the imported runtime
   */
  public static fromAgentRuntimeAttributes(
    scope: Construct,
    id: string,
    attrs: AgentRuntimeAttributes,
  ): IBedrockAgentRuntime {
    class ImportedBedrockAgentRuntime extends RuntimeBase {
      public readonly agentRuntimeArn = attrs.agentRuntimeArn;
      public readonly agentRuntimeId = attrs.agentRuntimeId;
      public readonly agentRuntimeName = attrs.agentRuntimeName;
      public readonly agentRuntimeVersion = attrs.agentRuntimeVersion;
      public readonly agentStatus = attrs.agentStatus;
      public readonly description = attrs.description;
      public readonly createdAt = attrs.createdAt;
      public readonly lastUpdatedAt = attrs.lastUpdatedAt;

      public readonly role: iam.IRole;
      public readonly grantPrincipal: iam.IPrincipal;

      constructor(s: Construct, i: string) {
        super(s, i);
        this.role = iam.Role.fromRoleArn(this, 'Role', attrs.roleArn);
        this.grantPrincipal = this.role;
        if (attrs.securityGroups) {
          this._connections = new ec2.Connections({ securityGroups: attrs.securityGroups });
        }
      }
    }

    return new ImportedBedrockAgentRuntime(scope, id);
  }

  /**
   * The ARN of the agent runtime
   * @attribute
   * @returns a token representing the ARN of this agent runtime
   */
  public readonly agentRuntimeArn: string;
  /**
   * The unique identifier of the agent runtime
   * @attribute
   * @returns a token representing the ID of this agent runtime
   */
  public readonly agentRuntimeId: string;
  /**
   * The name of the agent runtime
   * @attribute
   * @returns a token representing the name of this agent runtime
   */
  public readonly agentRuntimeName: string;
  public readonly role: iam.IRole;
  /**
   * The version of the agent runtime
   * @attribute
   * @returns a token representing the version of this agent runtime
   */
  public readonly agentRuntimeVersion?: string;
  /**
   * The status of the agent runtime
   * @attribute
   * @returns a token representing the status of this agent runtime
   */
  public readonly agentStatus?: string;
  /**
   * Optional description for the agent runtime
   */
  public readonly description?: string;
  /**
   * The timestamp when the agent runtime was created
   * @attribute
   * @returns a token representing the creation timestamp of this agent runtime
   */
  public readonly createdAt?: string;
  /**
   * The timestamp when the agent runtime was last updated
   * @attribute
   * @returns a token representing the last update timestamp of this agent runtime
   */
  public readonly lastUpdatedAt?: string;
  public readonly grantPrincipal: iam.IPrincipal;
  private readonly runtimeResource: bedrockagentcore.CfnRuntime;
  /**
   * The artifact configuration for the agent runtime
   */
  public readonly agentRuntimeArtifact: AgentRuntimeArtifact;
  private readonly networkConfiguration: RuntimeNetworkConfiguration ;
  private readonly protocolConfiguration: ProtocolType;
  private readonly authorizerConfiguration?: RuntimeAuthorizerConfiguration;
  private readonly lifecycleConfiguration?: LifecycleConfiguration;

  constructor(scope: Construct, id: string, props: RuntimeProps) {
    super(scope, id, {
      // Maximum name length of 48 characters
      // @see https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-bedrockagentcore-runtime.html#cfn-bedrockagentcore-runtime-agentruntimename
      physicalName: props.runtimeName ??
        Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 48 }) }),
    });

    // CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.agentRuntimeName = this.physicalName;
    this.validateRuntimeName();

    this.description = props.description;
    if (this.description) {
      this.validateDescription();
    }

    if (props.environmentVariables) {
      this.validateEnvironmentVariables(props.environmentVariables);
    }

    if (props.tags) {
      this.validateTags(props.tags);
    }

    if (props.requestHeaderConfiguration) {
      this.validateRequestHeaderConfiguration(props.requestHeaderConfiguration);
    }

    this.lifecycleConfiguration = props.lifecycleConfiguration ? {
      idleRuntimeSessionTimeout: props.lifecycleConfiguration?.idleRuntimeSessionTimeout,
      maxLifetime: props.lifecycleConfiguration?.maxLifetime,
    } : undefined;

    if (this.lifecycleConfiguration) {
      this.validateLifecycleConfiguration(this.lifecycleConfiguration);
    }

    if (props.executionRole) {
      this.role = props.executionRole;
      if (!Token.isUnresolved(props.executionRole.roleArn)) {
        this.validateRoleArn(props.executionRole.roleArn);
      }
    } else {
      this.role = new iam.Role(this, 'ExecutionRole', {
        // The service appends a random suffix to the resource name at creation time (e.g. MyRuntime-a1b2c3d4e5),
        // so we use ArnLike with a wildcard to match.
        // @see https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-bedrockagentcore-runtime.md
        assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com', {
          conditions: {
            StringEquals: { 'aws:SourceAccount': Stack.of(this).account },
            ArnLike: {
              'aws:SourceArn': Arn.format({ service: 'bedrock-agentcore', resource: 'runtime', resourceName: `${this.agentRuntimeName}*` }, Stack.of(this)),
            },
          },
        }),
        description: 'Execution role for Bedrock Agent Core Runtime',
        maxSessionDuration: Duration.hours(8),
      });
    }
    this.addExecutionRolePermissions();

    this.grantPrincipal = this.role;
    this.agentRuntimeArtifact = props.agentRuntimeArtifact;
    // Set up network configuration with VPC support
    this.networkConfiguration = props.networkConfiguration ?? RuntimeNetworkConfiguration.usingPublicNetwork();
    // Set connections - create a shared connections object
    if (this.networkConfiguration.connections) {
      // Use the network configuration's connections as the shared object
      this._connections = this.networkConfiguration.connections;
    }
    this.protocolConfiguration = props.protocolConfiguration ?? ProtocolType.HTTP;
    this.authorizerConfiguration = props.authorizerConfiguration;

    const cfnProps: any = {
      agentRuntimeName: this.agentRuntimeName,
      roleArn: this.role.roleArn,
      agentRuntimeArtifact: Lazy.any({ produce: () => this.renderAgentRuntimeArtifact() }),
      networkConfiguration: Lazy.any({ produce: () => this.networkConfiguration._render(this._connections) }),
      protocolConfiguration: Lazy.string({ produce: () => this.protocolConfiguration.value }),
      description: props.description,
      environmentVariables: Lazy.any({ produce: () => this.renderEnvironmentVariables(props.environmentVariables) }),
      tags: props.tags ?? {},
      lifecycleConfiguration: this.renderLifecycleConfiguration(),
    };
    if (props.requestHeaderConfiguration) {
      cfnProps.requestHeaderConfiguration = this.renderRequestHeaderConfiguration(props.requestHeaderConfiguration);
    }
    if (props.authorizerConfiguration) {
      cfnProps.authorizerConfiguration = Lazy.any({ produce: () => this.authorizerConfiguration!._render() });
    }

    this.runtimeResource = new bedrockagentcore.CfnRuntime(this, 'Resource', cfnProps as bedrockagentcore.CfnRuntimeProps);

    // Add dependency on the role (for both custom and auto-created roles)
    // This ensures the Runtime waits for the role and all its policies (including ECR permissions) to be created
    this.runtimeResource.node.addDependency(this.role);

    this.agentRuntimeId = this.runtimeResource.attrAgentRuntimeId;
    this.agentRuntimeArn = this.runtimeResource.attrAgentRuntimeArn;
    this.agentStatus = this.runtimeResource.attrStatus;
    this.agentRuntimeVersion = this.runtimeResource.attrAgentRuntimeVersion;
    this.createdAt = this.runtimeResource.attrCreatedAt;
    this.lastUpdatedAt = this.runtimeResource.attrLastUpdatedAt;

    // Configure observability (tracing and logging)
    const observabilityOptions = {
      manageDeliveryResourcePolicy: props.manageDeliveryResourcePolicy,
    };

    if (props.tracingEnabled) {
      configureTracingDelivery(this, this.agentRuntimeArn, observabilityOptions);
    }

    if (props.loggingConfigs && props.loggingConfigs.length > 0) {
      configureLoggingDelivery(this, this.agentRuntimeArn, props.loggingConfigs, observabilityOptions);
    }
  }

  /**
   * Renders the environment variables for CloudFormation
   * @internal
   */
  private renderEnvironmentVariables(envVars?: { [key: string]: string }): any {
    if (!envVars || Object.keys(envVars).length === 0) {
      return undefined;
    }
    return envVars;
  }

  /**
   * Adds proper permissions to the execution role for the agent runtime
   * Based on: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html
   */
  private addExecutionRolePermissions() {
    const region = Stack.of(this).region;
    const account = Stack.of(this).account;

    // CloudWatch Logs - Log Group operations
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'LogGroupAccess',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_LOGS_GROUP_ACTIONS,
      resources: [`arn:${Stack.of(this).partition}:logs:${region}:${account}:log-group:/aws/bedrock-agentcore/runtimes/*`],
    }));

    // CloudWatch Logs - Describe all log groups
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'DescribeLogGroups',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_LOGS_DESCRIBE_ACTIONS,
      resources: [`arn:${Stack.of(this).partition}:logs:${region}:${account}:log-group:*`],
    }));

    // CloudWatch Logs - Log Stream operations
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'LogStreamAccess',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_LOGS_STREAM_ACTIONS,
      resources: [`arn:${Stack.of(this).partition}:logs:${region}:${account}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*`],
    }));

    // X-Ray Tracing - must be * for tracing
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'XRayAccess',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_XRAY_ACTIONS,
      resources: ['*'],
    }));

    // CloudWatch Metrics - scoped to bedrock-agentcore namespace
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'CloudWatchMetrics',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_CLOUDWATCH_METRICS_ACTIONS,
      resources: ['*'],
      conditions: {
        StringEquals: { 'cloudwatch:namespace': RUNTIME_CLOUDWATCH_NAMESPACE },
      },
    }));

    // Bedrock AgentCore Workload Identity Access
    // Note: The agent name will be determined at runtime, so we use a wildcard pattern
    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: 'GetAgentAccessToken',
      effect: iam.Effect.ALLOW,
      actions: RUNTIME_WORKLOAD_IDENTITY_ACTIONS,
      resources: [
        `arn:${Stack.of(this).partition}:bedrock-agentcore:${region}:${account}:workload-identity-directory/default`,
        `arn:${Stack.of(this).partition}:bedrock-agentcore:${region}:${account}:workload-identity-directory/default/workload-identity/*`,
      ],
    }));
  }

  /**
   * Renders the artifact configuration for CloudFormation
   * @internal
   */
  private renderAgentRuntimeArtifact(): any {
    // set permission with bind
    this.agentRuntimeArtifact.bind(this, this);
    const config = this.agentRuntimeArtifact._render();
    if ((config as any).code) { // S3Image
      return {
        codeConfiguration: {
          code: (config as any).code,
          runtime: (config as any).runtime,
          entryPoint: (config as any).entryPoint,
        },
      };
    } else {
      // EcrImage or AssetImage
      const containerUri = (config as any).containerUri;
      if (containerUri) {
        this.validateContainerUri(containerUri);
      }
      return {
        containerConfiguration: { containerUri: containerUri },
      };
    }
  }

  /**
   * Renders the request header configuration for CloudFormation
   * @internal
   */
  private renderRequestHeaderConfiguration(requestHeaderConfiguration?: RequestHeaderConfiguration): any {
    if (!requestHeaderConfiguration?.allowlistedHeaders) {
      return undefined;
    }
    return { requestHeaderAllowlist: requestHeaderConfiguration.allowlistedHeaders };
  }

  /**
   * Renders the lifecycle configuration for CloudFormation
   * @internal
   */
  private renderLifecycleConfiguration(): any {
    return {
      idleRuntimeSessionTimeout: this.lifecycleConfiguration?.idleRuntimeSessionTimeout?.toSeconds(),
      maxLifetime: this.lifecycleConfiguration?.maxLifetime?.toSeconds(),
    };
  }

  /**
   * Validates the request header configuration
   * @throws Error if validation fails
   */
  private validateRequestHeaderConfiguration(requestHeaderConfiguration: RequestHeaderConfiguration): void {
    const allErrors: string[] = [];
    if (requestHeaderConfiguration.allowlistedHeaders) {
      if (requestHeaderConfiguration.allowlistedHeaders.length < 1 || requestHeaderConfiguration.allowlistedHeaders.length > 20) {
        allErrors.push('Request header allow list contain between 1 and 20 headers');
      }

      for (const header of requestHeaderConfiguration.allowlistedHeaders) {
        // Validate length
        const lengthErrors = validateStringFieldLength({
          value: header,
          fieldName: 'Request header',
          minLength: 1,
          maxLength: 256,
        });
        allErrors.push(...lengthErrors);

        const patternErrors = validateFieldPattern(
          header,
          'Request header',
          /^[A-Za-z][A-Za-z0-9_-]{0,255}$/,
          'Request header must start with a letter and contain only letters, numbers, underscores, and hyphens (max 256 characters)',
        );
        allErrors.push(...patternErrors);
      }
    }
    if (allErrors.length > 0) {
      throw new ValidationError(lit`InvalidRequestHeaderConfiguration`, allErrors.join('\n'), this);
    }
  }

  /**
   * Validates the lifecycle configuration
   * @throws Error if validation fails
   */
  private validateLifecycleConfiguration(lifecycleConfiguration: LifecycleConfiguration): void {
    if (lifecycleConfiguration.idleRuntimeSessionTimeout && !lifecycleConfiguration.idleRuntimeSessionTimeout.isUnresolved()) {
      if (lifecycleConfiguration.idleRuntimeSessionTimeout.toSeconds() < LIFECYCLE_MIN_TIMEOUT.toSeconds()
        || lifecycleConfiguration.idleRuntimeSessionTimeout.toSeconds() > LIFECYCLE_MAX_LIFETIME.toSeconds()) {
        throw new ValidationError(lit`InvalidIdleRuntimeSessionTimeout`, `Idle runtime session timeout must be between ${LIFECYCLE_MIN_TIMEOUT.toSeconds()} seconds and ${LIFECYCLE_MAX_LIFETIME.toSeconds()} seconds`, this);
      }
    }
    if (lifecycleConfiguration.maxLifetime && !lifecycleConfiguration.maxLifetime.isUnresolved()) {
      if (lifecycleConfiguration.maxLifetime.toSeconds() < LIFECYCLE_MIN_TIMEOUT.toSeconds()
        || lifecycleConfiguration.maxLifetime.toSeconds() > LIFECYCLE_MAX_LIFETIME.toSeconds()) {
        throw new ValidationError(lit`InvalidMaxLifetime`, `Maximum lifetime must be between ${LIFECYCLE_MIN_TIMEOUT.toSeconds()} seconds and ${LIFECYCLE_MAX_LIFETIME.toSeconds()} seconds`, this);
      }
    }
  }

  /**
   * Validates the runtime name format
   * Pattern: ^[a-zA-Z][a-zA-Z0-9_]{0,47}$
   * @throws Error if validation fails
   */
  private validateRuntimeName(): void {
    // Skip validation if the name contains CDK tokens (unresolved values)
    if (Token.isUnresolved(this.agentRuntimeName)) {
      return;
    }

    // Validate length
    const lengthErrors = validateStringFieldLength({
      value: this.agentRuntimeName,
      fieldName: 'Runtime name',
      minLength: 1,
      maxLength: 48,
    });

    // Validate pattern
    const patternErrors = validateFieldPattern(
      this.agentRuntimeName,
      'Runtime name',
      /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/,
      'Runtime name must start with a letter and contain only letters, numbers, and underscores',
    );

    // Combine and throw if any errors
    const allErrors = [...lengthErrors, ...patternErrors];
    if (allErrors.length > 0) {
      throw new ValidationError(lit`InvalidRuntimeName`, allErrors.join('\n'), this);
    }
  }

  /**
   * Validates the description format
   * Must be between 1 and 1200 characters (per CloudFormation specification)
   * @throws Error if validation fails
   */
  private validateDescription(): void {
    // Skip validation if the description contains CDK tokens (unresolved values)
    if (Token.isUnresolved(this.description)) {
      return;
    }

    if (this.description) {
      const errors = validateStringFieldLength({
        value: this.description,
        fieldName: 'Description',
        minLength: 1,
        maxLength: 1200,
      });

      if (errors.length > 0) {
        throw new ValidationError(lit`InvalidDescription`, errors.join('\n'), this);
      }
    }
  }

  /**
   * Validates environment variables
   * - Maximum 50 entries
   * - Key: 1-100 characters
   * - Value: 0-2048 characters (per CloudFormation specification)
   * @throws Error if validation fails
   */
  private validateEnvironmentVariables(envVars: { [key: string]: string }): void {
    const entries = Object.entries(envVars);

    // Validate number of entries (0-50)
    if (entries.length > 50) {
      throw new ValidationError(
        lit`TooManyEnvironmentVariables`,
        `Too many environment variables: ${entries.length}. Maximum allowed is 50`,
        this,
      );
    }

    for (const [key, value] of entries) {
      // Skip validation if key or value contains CDK tokens
      if (Token.isUnresolved(key) || Token.isUnresolved(value)) {
        continue;
      }

      // Validate key length
      const lengthErrors = validateStringFieldLength({
        value: key,
        fieldName: `Environment variable key '${key}'`,
        minLength: 1,
        maxLength: 100,
      });

      // Validate key pattern
      const patternErrors = validateFieldPattern(
        key,
        `Environment variable key '${key}'`,
        /^[a-zA-Z_][a-zA-Z0-9_]*$/,
        `Environment variable key '${key}' must start with a letter or underscore and contain only letters, numbers, and underscores`,
      );

      // Combine and throw if any errors
      const allErrors = [...lengthErrors, ...patternErrors];
      if (allErrors.length > 0) {
        throw new ValidationError(lit`InvalidEnvironmentVariableKey`, allErrors.join('\n'), this);
      }

      // Validate value length (0-2048 characters per CloudFormation)
      if (value.length > 2048) {
        throw new ValidationError(
          lit`InvalidEnvironmentVariableValue`,
          `Invalid environment variable value length for key '${key}': ${value.length} characters. ` +
          'Values must not exceed 2048 characters',
          this,
        );
      }
    }
  }

  /**
   * Validates the tags format
   * @param tags The tags object to validate
   * @throws Error if validation fails
   */
  private validateTags(tags: { [key: string]: string }): void {
    // Validate each tag key and value
    for (const [key, value] of Object.entries(tags)) {
      // Skip validation if key or value contains CDK tokens
      if (Token.isUnresolved(key) || Token.isUnresolved(value)) {
        continue;
      }

      // Validate tag key length
      const keyLengthErrors = validateStringFieldLength({
        value: key,
        fieldName: `Tag key "${key}"`,
        minLength: 1,
        maxLength: 256,
      });

      // Validate aws: prefix restriction
      if (key.toLowerCase().startsWith('aws:')) {
        throw new ValidationError(lit`InvalidTagKey`, `Tag key "${key}" cannot start with "aws:" as this prefix is reserved by AWS`, this);
      }

      // Validate tag key pattern
      const keyPatternErrors = validateFieldPattern(
        key,
        `Tag key "${key}"`,
        /^[a-zA-Z0-9\s._:/=+@-]*$/,
        `Tag key "${key}" can only contain letters (a-z, A-Z), numbers (0-9), spaces, and special characters (._:/=+@-)`,
      );

      // Combine key errors and throw if any
      const keyErrors = [...keyLengthErrors, ...keyPatternErrors];
      if (keyErrors.length > 0) {
        throw new ValidationError(lit`InvalidTagKey`, keyErrors.join('\n'), this);
      }

      if (value === undefined || value === null) {
        throw new ValidationError(lit`NullTagValue`, `Tag value for key "${key}" cannot be null or undefined`, this);
      }

      // Validate tag value length
      const valueLengthErrors = validateStringFieldLength({
        value: value,
        fieldName: `Tag value for key "${key}"`,
        minLength: 0,
        maxLength: 256,
      });

      // Validate tag value pattern
      const valuePatternErrors = validateFieldPattern(
        value,
        `Tag value for key "${key}"`,
        /^[a-zA-Z0-9\s._:/=+@-]*$/,
        `Tag value for key "${key}" can only contain letters (a-z, A-Z), numbers (0-9), spaces, and special characters (._:/=+@-)`,
      );

      // Combine value errors and throw if any
      const valueErrors = [...valueLengthErrors, ...valuePatternErrors];
      if (valueErrors.length > 0) {
        throw new ValidationError(lit`InvalidTagValue`, valueErrors.join('\n'), this);
      }
    }
  }

  /**
   * Validates the container URI format
   */
  private validateContainerUri(uri: string): void {
    // Skip validation if the URI contains CDK tokens (unresolved values)
    if (Token.isUnresolved(uri)) {
      // Add a warning that validation will be skipped for token-based URIs
      Annotations.of(this).addInfo(
        'Container URI validation skipped as it contains unresolved CDK tokens. ' +
        'The URI will be validated at deployment time.',
      );
      return;
    }

    // Only validate if the URI is a concrete string (not a token)
    const pattern = /^\d{12}\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com\/((?:[a-z0-9]+(?:[._-][a-z0-9]+)*\/)*[a-z0-9]+(?:[._-][a-z0-9]+)*)([:@]\S+)$/;
    if (!pattern.test(uri)) {
      throw new ValidationError(
        lit`InvalidContainerUri`,
        `Invalid container URI format: ${uri}. Must be a valid ECR URI (e.g., 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest)`,
        this,
      );
    }
  }

  /**
   * Validates the IAM role ARN format and structure
   * @throws Error if validation fails
   */
  private validateRoleArn(roleArn: string): void {
    // Validate basic ARN format for IAM roles
    const arnPattern = /^arn:[a-z\-]+:iam::\d{12}:role\/[a-zA-Z0-9+=,.@\-_\/]+$/;
    if (!arnPattern.test(roleArn)) {
      throw new ValidationError(
        lit`InvalidRoleArnFormat`,
        `Invalid IAM role ARN format: ${roleArn}. ` +
        'Expected format: arn:<partition>:iam::<account-id>:role/<role-name> or arn:<partition>:iam::<account-id>:role/<path>/<role-name>',
        this,
      );
    }

    // Parse the ARN components using CDK's Arn.split()
    // Use SLASH_RESOURCE_NAME format for IAM roles (which use format: role/role-name or role/path/to/role-name)
    const arnComponents = Arn.split(roleArn, ArnFormat.SLASH_RESOURCE_NAME);

    if (arnComponents.service !== 'iam') {
      throw new ValidationError(lit`InvalidRoleArnService`, `Invalid service in ARN: ${arnComponents.service}. Expected 'iam' for IAM role ARN.`, this);
    }

    const accountId = arnComponents.account;
    if (!accountId || !/^\d{12}$/.test(accountId)) {
      throw new ValidationError(lit`InvalidRoleArnAccountId`, `Invalid AWS account ID in role ARN: ${accountId}. Must be a 12-digit number.`, this);
    }

    // Extract role name from resource
    // For IAM roles, resource will be "role" and resourceName will be the role name (with optional path)
    const resource = arnComponents.resource;
    const resourceName = arnComponents.resourceName;

    if (resource !== 'role') {
      throw new ValidationError(lit`InvalidRoleArnResourceType`, `Invalid resource type in ARN: ${resource}. Expected 'role' for IAM role ARN.`, this);
    }

    if (!resourceName) {
      throw new ValidationError(lit`MissingRoleName`, 'Role name is missing in the ARN', this);
    } else {
      const rolePathParts = resourceName.split('/');
      const roleName = rolePathParts[rolePathParts.length - 1];

      if (roleName.length > 64) {
        throw new ValidationError(lit`RoleNameTooLong`, `Role name exceeds maximum length of 64 characters: ${roleName}`, this);
      }
    }

    const stackAccount = Stack.of(this).account;
    if (!Token.isUnresolved(stackAccount) && accountId !== stackAccount) {
      Annotations.of(this).addWarning(
        `IAM role is from a different account (${accountId}) than the stack account (${stackAccount}). ` +
        'Ensure cross-account permissions are properly configured.',
      );
    }

    // Validate the region (IAM is global, so region should be empty)
    const region = arnComponents.region;
    const stackRegion = Stack.of(this).region;
    if (region && region !== '' && region !== stackRegion && !Token.isUnresolved(stackRegion)) {
      Annotations.of(this).addWarning(
        `IAM role ARN contains a region (${region}) that doesn't match the stack region (${stackRegion}). ` +
        'IAM is a global service, so this might be intentional.',
      );
    }
  }

  /**
   * Add an endpoint to this runtime
   * This is a convenience method that creates a RuntimeEndpoint associated with this runtime
   *
   * @param endpointName The name of the endpoint
   * @param options Optional configuration for the endpoint
   * @returns The created RuntimeEndpoint
   */
  @MethodMetadata()
  public addEndpoint(
    endpointName: string,
    options?: AddEndpointOptions,
  ): RuntimeEndpoint {
    // Create the endpoint (security configuration is no longer supported)
    const endpoint = new RuntimeEndpoint(this, `Endpoint${endpointName}`, {
      endpointName: endpointName,
      agentRuntimeId: this.agentRuntimeId,
      agentRuntimeVersion: options?.version ?? this.agentRuntimeVersion ?? '1',
      description: options?.description,
    });

    // Add dependency: endpoint must wait for runtime to be created
    endpoint.node.addDependency(this.runtimeResource);

    return endpoint;
  }
}
