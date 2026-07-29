import type { Construct } from 'constructs';
import type { Architecture } from './architecture';
import type { SystemLogLevel } from './function';
import type { IFunction } from './function-base';
import type { CfnFunction } from './lambda.generated';
import { CfnCapacityProvider } from './lambda.generated';
import type * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import type * as kms from '../../aws-kms';
import type * as logs from '../../aws-logs';
import type { IResource } from '../../core';
import { Annotations, Arn, ArnFormat, Resource, Stack, Token, ValidationError } from '../../core';
import { memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { CapacityProviderReference, ICapacityProviderRef } from '../../interfaces/generated/aws-lambda-interfaces.generated';

/**
 * Represents a Lambda capacity provider.
 */
export interface ICapacityProvider extends IResource, ICapacityProviderRef {
  /**
   * The ARN of the capacity provider.
   *
   * @attribute
   */
  readonly capacityProviderArn: string;

  /**
   * The name of the capacity provider.
   *
   * @attribute
   */
  readonly capacityProviderName: string;
}

/**
 * Properties for creating a Lambda capacity provider.
 */
export interface CapacityProviderProps {
  /**
   * The name of the capacity provider. The name must be unique within the AWS account and region.
   *
   * @default - AWS CloudFormation generates a unique physical ID and uses that
   * ID for the capacity provider's name.
   */
  readonly capacityProviderName?: string;

  /**
   * A list of security group IDs to associate with EC2 instances launched by the capacity provider.
   * Up to 5 security groups can be specified.
   */
  readonly securityGroups: ec2.ISecurityGroup[];

  /**
   * A list of subnets where the capacity provider can launch EC2 instances.
   * At least one subnet must be specified, and up to 16 subnets are supported.
   */
  readonly subnets: ec2.ISubnet[];

  /**
   * The IAM role that the Lambda service assumes to manage the capacity provider.
   *
   * @default - A role will be generated containing the AWSLambdaManagedEC2ResourceOperator managed policy
   */
  readonly operatorRole?: iam.IRole;

  /**
   * The instruction set architecture required for compute instances.
   * Only one architecture can be specified per capacity provider.
   *
   * @default - No architecture constraints specified
   */
  readonly architectures?: Architecture[];

  /**
   * Configuration for filtering instance types that the capacity provider can use.
   *
   * @default - No instance type filtering applied
   */
  readonly instanceTypeFilter?: InstanceTypeFilter;

  /**
   * The maximum number of vCPUs that the capacity provider can scale up to.
   *
   * @default - No maximum limit specified, service default is 400
   */
  readonly maxVCpuCount?: number;

  /**
   * The options for scaling a capacity provider, including scaling policies.
   *
   * @default - The `Auto` option is applied by default
   */
  readonly scalingOptions?: ScalingOptions;

  /**
   * The AWS Key Management Service (KMS) key used to encrypt data associated with the capacity provider.
   *
   * @default - No KMS key specified, uses an AWS-managed key instead
   */
  readonly kmsKey?: kms.IKey;

  /**
   * The CloudWatch log group for capacity provider system logs.
   *
   * @default - Service creates a default log group at /aws/lambda/capacity-provider/<name>
   */
  readonly logGroup?: logs.ILogGroupRef;

  /**
   * The level of detail for capacity provider system logs.
   *
   * @default - Service default applies (INFO)
   */
  readonly systemLogLevel?: SystemLogLevel;
}

/**
 * Configuration for filtering instance types that a capacity provider can use. Instances types can either be allowed or excluded, not both.
 */
export class InstanceTypeFilter {
  /**
   * Creates an instance type filter that allows only the specified instance types.
   *
   * @param instanceTypes A list of instance types that the capacity provider is allowed to use.
   */
  public static allow(instanceTypes: ec2.InstanceType[]) {
    return new InstanceTypeFilter({ instanceTypes, isAllow: true });
  }

  /**
   * Creates an instance type filter that excludes the specified instance types.
   *
   * @param instanceTypes A list of instance types that the capacity provider should not use.
   */
  public static exclude(instanceTypes: ec2.InstanceType[]) {
    return new InstanceTypeFilter({ instanceTypes, isAllow: false });
  }

  /**
   * A list of instance types that the capacity provider is allowed to use.
   */
  public readonly allowedInstanceTypes?: ec2.InstanceType[];

  /**
   * A list of instance types that the capacity provider should not use.
   */
  public readonly excludedInstanceTypes?: ec2.InstanceType[];

  /**
   * Creates a new InstanceTypeFilter.
   *
   * @param instanceTypes The instance type configuration
   */
  private constructor(options: { instanceTypes: ec2.InstanceType[]; isAllow: boolean }) {
    if (options.isAllow) {
      this.allowedInstanceTypes = options.instanceTypes;
    } else {
      this.excludedInstanceTypes = options.instanceTypes;
    }
  }
}

/**
 * Configuration options for scaling a capacity provider, including scaling mode and policies.
 */
export class ScalingOptions {
  /**
   * Creates scaling options where the capacity provider manages scaling automatically.
   */
  public static auto(): ScalingOptions {
    return new ScalingOptions('Auto');
  }

  /**
   * Creates manual scaling options with custom target tracking scaling policies. At least one policy is required.
   *
   * @param scalingPolicies The target tracking scaling policies to use for manual scaling.
   */
  public static manual(scalingPolicies: TargetTrackingScalingPolicy[]): ScalingOptions {
    return new ScalingOptions('Manual', scalingPolicies);
  }

  /**
   * The scaling mode for the capacity provider.
   */
  public readonly scalingMode: string;

  /**
   * The target tracking scaling policies used when scaling mode is 'Manual'.
   */
  public readonly scalingPolicies?: TargetTrackingScalingPolicy[];

  /**
   * Creates a new ScalingOptions.
   *
   * @param scalingMode The scaling mode for the capacity provider
   * @param scalingPolicies The target tracking scaling policies for manual scaling
   */
  private constructor(
    scalingMode: string,
    scalingPolicies?: TargetTrackingScalingPolicy[],
  ) {
    this.scalingMode = scalingMode;
    this.scalingPolicies = scalingPolicies;
  }
}

/**
 * A target tracking scaling policy that automatically adjusts the capacity provider's compute resources
 * to maintain a specified target value by tracking the required CloudWatch metric.
 */
export class TargetTrackingScalingPolicy {
  /**
   * Creates a target tracking scaling policy for CPU utilization.
   *
   * @param targetCpuUtilization The target value for CPU utilization. The capacity provider will scale resources to maintain this target value.
   */
  public static cpuUtilization(targetCpuUtilization: number): TargetTrackingScalingPolicy {
    return new TargetTrackingScalingPolicy('LambdaCapacityProviderAverageCPUUtilization', targetCpuUtilization);
  }

  /**
   * The predefined metric type for this scaling policy.
   */
  readonly predefinedMetricType: string;

  /**
   * The target value for the specified metric as a percentage. The capacity provider will scale resources to maintain this target value.
   */
  readonly targetValue: number;

  /**
   * Creates a new TargetTrackingScalingPolicy.
   *
   * @param metricType The predefined metric type
   * @param value The target value for the metric
   */
  private constructor(
    public readonly metricType: string,
    public readonly value: number) {
    this.predefinedMetricType = metricType;
    this.targetValue = value;
  }
}

/**
 * Base class for a Lambda capacity provider.
 */
abstract class CapacityProviderBase extends Resource implements ICapacityProvider {
  /**
   * The name of the capacity provider.
   */
  public abstract readonly capacityProviderName: string;

  /**
   * The Amazon Resource Name (ARN) of the capacity provider.
   */
  public abstract readonly capacityProviderArn: string;

  public get capacityProviderRef(): CapacityProviderReference {
    return {
      capacityProviderName: this.capacityProviderName,
      capacityProviderArn: this.capacityProviderArn,
    };
  }
}

/**
 * Attributes for importing an existing Lambda capacity provider.
 */
export interface CapacityProviderAttributes {
  /**
   * The Amazon Resource Name (ARN) of the capacity provider.
   *
   * Format: arn:<partition>:lambda:<region>:<account-id>:capacity-provider:<capacity-provider-name>
   */
  readonly capacityProviderArn: string;
}

/**
 * Options for creating a function associated with a capacity provider.
 */
export interface CapacityProviderFunctionOptions {
  /**
   * Specifies the maximum number of concurrent invokes a single execution environment can handle.
   *
   * @default Maximum is set to 10
   */
  readonly perExecutionEnvironmentMaxConcurrency?: number;

  /**
   * Specifies the execution environment memory per VCPU, in GiB.
   *
   * @default 2.0
   */
  readonly executionEnvironmentMemoryGiBPerVCpu?: number;

  /**
   * A boolean determining whether or not to automatically publish to the $LATEST.PUBLISHED version.
   *
   * @default - True
   */
  readonly publishToLatestPublished?: boolean;

  /**
   * The scaling options that are applied to the $LATEST.PUBLISHED version.
   *
   * @default - No scaling limitations are applied to the $LATEST.PUBLISHED version.
   */
  readonly latestPublishedScalingConfig?: LatestPublishedScalingConfig;
}

/**
 * The scaling configuration that will be applied to the $LATEST.PUBLISHED version.
 */
export interface LatestPublishedScalingConfig {
  /**
   * The minimum number of execution environments to maintain for the $LATEST.PUBLISHED version
   * when published into a capacity provider.
   *
   * This setting ensures that at least this many execution environments are always
   * available to handle function invocations for this specific version, reducing cold start latency.
   *
   * @default - 3 execution environments are set to be the minimum
   */
  readonly minExecutionEnvironments?: number;

  /**
   * The maximum number of execution environments allowed for the $LATEST.PUBLISHED version
   * when published into a capacity provider.
   *
   * This setting limits the total number of execution environments that can be created
   * to handle concurrent invocations of this specific version.
   *
   * @default - No maximum specified
   */
  readonly maxExecutionEnvironments?: number;
}

/**
 * A Lambda capacity provider that manages compute resources for Lambda functions.
 */
@propertyInjectable
export class CapacityProvider extends CapacityProviderBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-lambda.CapacityProvider';

  /**
   * Import an existing capacity provider by name.
   *
   * @param scope The parent construct
   * @param id The construct ID
   * @param capacityProviderName The name of the capacity provider to import
   */
  public static fromCapacityProviderName(
    scope: Construct,
    id: string,
    capacityProviderName: string,
  ): ICapacityProvider {
    const capacityProviderArn = Stack.of(scope).formatArn({
      service: 'lambda',
      resource: 'capacity-provider',
      resourceName: capacityProviderName,
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    });
    return CapacityProvider.fromCapacityProviderAttributes(scope, id, { capacityProviderArn });
  }

  /**
   * Import an existing capacity provider by ARN.
   *
   * @param scope The parent construct
   * @param id The construct ID
   * @param capacityProviderArn The ARN of the capacity provider to import
   */
  public static fromCapacityProviderArn(
    scope: Construct,
    id: string,
    capacityProviderArn: string,
  ): ICapacityProvider {
    return CapacityProvider.fromCapacityProviderAttributes(scope, id, { capacityProviderArn });
  }

  /**
   * Import an existing capacity provider using its attributes.
   *
   * @param scope The parent construct
   * @param id The construct ID
   * @param attrs The capacity provider attributes
   */
  public static fromCapacityProviderAttributes(scope: Construct, id: string, attrs: CapacityProviderAttributes): ICapacityProvider {
    class Import extends CapacityProviderBase {
      public readonly capacityProviderName: string = Arn.split(attrs.capacityProviderArn, ArnFormat.COLON_RESOURCE_NAME).resourceName!;
      public readonly capacityProviderArn: string = attrs.capacityProviderArn;

      constructor(s: Construct, i: string) {
        super(s, i);
      }
    }
    return new Import(scope, id);
  }

  private readonly capacityProvider: CfnCapacityProvider;

  /**
   * The name of the capacity provider.
   */
  @memoizedGetter
  public get capacityProviderName(): string {
    return this.getResourceNameAttribute(this.capacityProvider.ref);
  }

  /**
   * The Amazon Resource Name (ARN) of the capacity provider.
   */
  @memoizedGetter
  public get capacityProviderArn(): string {
    return this.getResourceArnAttribute(this.capacityProvider.attrArn, {
      service: 'lambda',
      resource: 'capacity-provider',
      resourceName: this.physicalName,
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    });
  }

  /**
   * Creates a new Lambda capacity provider.
   *
   * @param scope The parent construct
   * @param id The construct ID
   * @param props The capacity provider properties
   */
  constructor(scope: Construct, id: string, props: CapacityProviderProps) {
    super(scope, id, { physicalName: props.capacityProviderName });
    addConstructMetadata(this, props);

    this.validateCapacityProviderProps(props);

    const operatorRole = props.operatorRole || new iam.Role(this, 'OperatorRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLambdaManagedEC2ResourceOperator')],
    });

    const instanceRequirements = props.instanceTypeFilter || props.architectures ? {
      allowedInstanceTypes: props.instanceTypeFilter?.allowedInstanceTypes?.map((instanceType) => instanceType.toString()),
      excludedInstanceTypes: props.instanceTypeFilter?.excludedInstanceTypes?.map((instanceType) => instanceType.toString()),
      architectures: props.architectures?.map((architecture) => architecture.toString()),
    } : undefined;

    const capacityProviderScalingConfig = props.maxVCpuCount || props.scalingOptions ? {
      maxVCpuCount: props.maxVCpuCount,
      scalingMode: props.scalingOptions?.scalingMode,
      scalingPolicies: props.scalingOptions?.scalingPolicies?.map((scalingPolicy) => ({
        predefinedMetricType: scalingPolicy.predefinedMetricType,
        targetValue: scalingPolicy.targetValue,
      })),
    } : undefined;

    this.capacityProvider = new CfnCapacityProvider(this, 'Resource', {
      capacityProviderName: this.physicalName,
      vpcConfig: {
        securityGroupIds: props.securityGroups.map((sg) => sg.securityGroupId),
        subnetIds: props.subnets.map((sn) => sn.subnetId),
      },
      permissionsConfig: {
        capacityProviderOperatorRoleArn: operatorRole.roleArn,
      },
      instanceRequirements,
      capacityProviderScalingConfig,
      kmsKeyArn: props.kmsKey?.keyArn,
      telemetryConfig: props.logGroup || props.systemLogLevel ? {
        loggingConfig: {
          logGroup: props.logGroup?.logGroupRef.logGroupName,
          systemLogLevel: props.systemLogLevel,
        },
      } : undefined,
    });
  }

  private validateCapacityProviderProps(props: CapacityProviderProps) {
    const validationErrorCPName = props.capacityProviderName || 'your capacity provider';

    if (props.maxVCpuCount !== undefined && !Token.isUnresolved(props.maxVCpuCount) && (props.maxVCpuCount < 12 || props.maxVCpuCount > 15000)) {
      throw new ValidationError(lit`MaxCpuCount`, `maxVCpuCount must be between 12 and 15000, but ${validationErrorCPName} has ${props.maxVCpuCount}.`, this);
    }

    if (!Token.isUnresolved(props.subnets) && (props.subnets.length < 1 || props.subnets.length > 16)) {
      throw new ValidationError(lit`SubnetsContainItems`, `subnets must contain between 1 and 16 items but ${validationErrorCPName} has ${props.subnets.length} items.`, this);
    }

    if (!Token.isUnresolved(props.securityGroups) && (props.securityGroups.length < 1 || props.securityGroups.length > 5)) {
      throw new ValidationError(lit`SecurityGroupsContainItems`, `securityGroups must contain between 1 and 5 items but ${validationErrorCPName} has ${props.securityGroups.length} items.`, this);
    }

    if (props.capacityProviderName) {
      this.validateCapacityProviderName(props.capacityProviderName);
    }

    if (props.instanceTypeFilter) {
      this.validateInstanceTypeFilter(props.instanceTypeFilter, validationErrorCPName);
    }

    if (props.scalingOptions) {
      this.validateScalingPolicies(props.scalingOptions, validationErrorCPName);
    }
  }

  private validateCapacityProviderName(name: string) {
    if (Token.isUnresolved(name)) {
      return;
    }
    if (!/^([a-zA-Z0-9-_]+|arn:aws[a-zA-Z-]*:lambda:capacity-provider:[a-zA-Z0-9-_]+)$/.test(name)) {
      throw new ValidationError(lit`CapacityProviderNameArnAlphanumeric`, `capacityProviderName must be an arn or have only alphanumeric characters, but did not: ${name}`, this);
    }
    if (name.length > 140) {
      throw new ValidationError(lit`CapacityProviderNameLongerCharacters`, `Capacity provider name can not be longer than 140 characters but ${name} has ${name.length} characters.`, this);
    }
  }

  private validateScalingPolicies(scalingOptions: ScalingOptions, validationErrorCPName: string) {
    if (scalingOptions?.scalingPolicies && !Token.isUnresolved(scalingOptions.scalingPolicies)) {
      if (scalingOptions.scalingPolicies.length < 1) {
        throw new ValidationError(lit`ScalingoptionsLeastPolicyScalingmode`, `scalingOptions must have at least one policy when scalingMode is 'Manual', but ${validationErrorCPName} has ${scalingOptions.scalingPolicies.length} items.`, this);
      }

      if (scalingOptions.scalingPolicies.length > 10) {
        throw new ValidationError(lit`ScalingoptionsMostPoliciesScalingmode`, `scalingOptions can have at most ten policies when scalingMode is 'Manual', but ${validationErrorCPName} has ${scalingOptions.scalingPolicies.length} items.`, this);
      }
    }
  }

  private validateInstanceTypeFilter(instanceTypeFilter: InstanceTypeFilter, validationErrorCPName: string) {
    if (instanceTypeFilter?.allowedInstanceTypes && instanceTypeFilter.allowedInstanceTypes.length < 1) {
      throw new ValidationError(lit`InstanceTypeFilterLeastOne`, `instanceTypeFilter must have at least one instanceType when configured, but ${validationErrorCPName} has ${instanceTypeFilter.allowedInstanceTypes.length} items.`, this);
    }

    if (instanceTypeFilter?.excludedInstanceTypes && instanceTypeFilter.excludedInstanceTypes.length < 1) {
      throw new ValidationError(lit`InstanceTypeFilterLeastOne`, `instanceTypeFilter must have at least one instanceType when configured, but ${validationErrorCPName} has ${instanceTypeFilter.excludedInstanceTypes.length} items.`, this);
    }
  }

  // Methods related to attaching a Lambda function to the capacity provider

  /**
   * Configures a Lambda function to use this capacity provider.
   *
   * @param func The Lambda function to configure
   * @param options Optional configuration for the function's capacity provider settings
   */
  @MethodMetadata()
  public addFunction(func: IFunction, options?: CapacityProviderFunctionOptions): void {
    Annotations.of(func).addInfoV2('aws-lambda:Function.CapacityProviderConfiguration', 'Capacity provider configuration cannot be removed from a function, only modified.');

    const cfnFunction = func.node.defaultChild as CfnFunction;

    cfnFunction.publishToLatestPublished = options?.publishToLatestPublished;

    cfnFunction.capacityProviderConfig = {
      lambdaManagedInstancesCapacityProviderConfig: {
        capacityProviderArn: this.capacityProviderArn,
        perExecutionEnvironmentMaxConcurrency: options?.perExecutionEnvironmentMaxConcurrency,
        executionEnvironmentMemoryGiBPerVCpu: options?.executionEnvironmentMemoryGiBPerVCpu,
      },
    };

    const minExecutionEnvironments = options?.latestPublishedScalingConfig?.minExecutionEnvironments;
    const maxExecutionEnvironments = options?.latestPublishedScalingConfig?.maxExecutionEnvironments;

    if (minExecutionEnvironments !== undefined || maxExecutionEnvironments !== undefined) {
      this.validateFunctionScalingConfig(minExecutionEnvironments, maxExecutionEnvironments);
      cfnFunction.functionScalingConfig = {
        minExecutionEnvironments,
        maxExecutionEnvironments,
      };
    }
  }

  private validateFunctionScalingConfig(minExecutionEnvironments?: number, maxExecutionEnvironments?: number) {
    const minDefined = minExecutionEnvironments !== undefined && !Token.isUnresolved(minExecutionEnvironments);
    const maxDefined = maxExecutionEnvironments !== undefined && !Token.isUnresolved(maxExecutionEnvironments);

    if (minDefined && (minExecutionEnvironments < 0 || minExecutionEnvironments > 15000)) {
      throw new ValidationError(lit`MinExecutionEnvironments`, `minExecutionEnvironments must be between 0 and 15000, but was ${minExecutionEnvironments}.`, this);
    }

    if (maxDefined && (maxExecutionEnvironments < 0 || maxExecutionEnvironments > 15000)) {
      throw new ValidationError(lit`MaxExecutionEnvironments`, `maxExecutionEnvironments must be between 0 and 15000, but was ${maxExecutionEnvironments}.`, this);
    }

    if (minDefined && maxDefined && minExecutionEnvironments > maxExecutionEnvironments) {
      throw new ValidationError(lit`MinExecutionEnvironmentsLessEqual`, 'minExecutionEnvironments must be less than or equal to maxExecutionEnvironments.', this);
    }
  }
}
