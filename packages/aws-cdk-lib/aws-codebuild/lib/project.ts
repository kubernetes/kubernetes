import type { Construct, IConstruct } from 'constructs';
import type { IArtifacts } from './artifacts';
import { BuildSpec } from './build-spec';
import { Cache } from './cache';
import { CodeBuildMetrics } from './codebuild-canned-metrics.generated';
import { CfnProject } from './codebuild.generated';
import { CodePipelineArtifacts } from './codepipeline-artifacts';
import type { DockerServerComputeType } from './compute-type';
import { ComputeType } from './compute-type';
import { EnvironmentType } from './environment-type';
import type { IFileSystemLocation } from './file-location';
import type { IFleet } from './fleet';
import { ImagePullPrincipalType } from './image-pull-principal-type';
import { isLambdaComputeType } from './is-lambda-compute-type';
import { LinuxArmLambdaBuildImage } from './linux-arm-lambda-build-image';
import { LinuxLambdaBuildImage } from './linux-lambda-build-image';
import { NoArtifacts } from './no-artifacts';
import { NoSource } from './no-source';
import { runScriptLinuxBuildSpec, S3_BUCKET_ENV, S3_KEY_ENV } from './private/run-script-linux-build-spec';
import type { LoggingOptions } from './project-logs';
import { renderReportGroupArn } from './report-group-utils';
import type { ISource } from './source';
import { CODEPIPELINE_SOURCE_ARTIFACTS_TYPE, NO_SOURCE_TYPE } from './source-types';
import * as cloudwatch from '../../aws-cloudwatch';
import * as notifications from '../../aws-codestarnotifications';
import * as ec2 from '../../aws-ec2';
import type * as ecr from '../../aws-ecr';
import type { DockerImageAssetProps } from '../../aws-ecr-assets';
import { DockerImageAsset } from '../../aws-ecr-assets';
import * as events from '../../aws-events';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import type * as s3 from '../../aws-s3';
import type * as secretsmanager from '../../aws-secretsmanager';
import type { Duration, IResource } from '../../core';
import { Annotations, ArnFormat, Aws, Lazy, Names, PhysicalName, Reference, Resource, SecretValue, Stack, Token, TokenComparison, Tokenization, UnscopedValidationError, ValidationError } from '../../core';
import type { IArrayBox, IBox } from '../../core/lib/helpers-internal';
import { Box, memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { IProjectRef, ProjectReference } from '../../interfaces/generated/aws-codebuild-interfaces.generated';

const VPC_POLICY_SYM = Symbol.for('@aws-cdk/aws-codebuild.roleVpcPolicy');

/**
 * The type returned from `IProject#enableBatchBuilds`.
 */
export interface BatchBuildConfig {
  /** The IAM batch service Role of this Project. */
  readonly role: iam.IRole;
}

/**
 * Location of a PEM certificate on S3
 */
export interface BuildEnvironmentCertificate {
  /**
   * The bucket where the certificate is
   */
  readonly bucket: s3.IBucket;

  /**
   * The full path and name of the key file
   */
  readonly objectKey: string;
}

/**
 * Additional options to pass to the notification rule.
 */
export interface ProjectNotifyOnOptions extends notifications.NotificationRuleOptions {
  /**
   * A list of event types associated with this notification rule for CodeBuild Project.
   * For a complete list of event types and IDs, see Notification concepts in the Developer Tools Console User Guide.
   * @see https://docs.aws.amazon.com/dtconsole/latest/userguide/concepts.html#concepts-api
   */
  readonly events: ProjectNotificationEvents[];
}

export interface IProject extends IResource, iam.IGrantable, ec2.IConnectable, notifications.INotificationRuleSource, IProjectRef {
  /**
   * The ARN of this Project.
   * @attribute
   */
  readonly projectArn: string;

  /**
   * The human-visible name of this Project.
   * @attribute
   */
  readonly projectName: string;

  /** The IAM service Role of this Project. Undefined for imported Projects. */
  readonly role?: iam.IRole;

  /**
   * Enable batch builds.
   *
   * Returns an object contining the batch service role if batch builds
   * could be enabled.
   */
  enableBatchBuilds(): BatchBuildConfig | undefined;

  addToRolePolicy(policyStatement: iam.PolicyStatement): void;

  /**
   * Defines a CloudWatch event rule triggered when something happens with this project.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  onEvent(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines a CloudWatch event rule triggered when the build project state
   * changes. You can filter specific build status events using an event
   * pattern filter on the `build-status` detail field:
   *
   *    const rule = project.onStateChange('OnBuildStarted', { target });
   *    rule.addEventPattern({
   *      detail: {
   *        'build-status': [
   *          "IN_PROGRESS",
   *          "SUCCEEDED",
   *          "FAILED",
   *          "STOPPED"
   *        ]
   *      }
   *    });
   *
   * You can also use the methods `onBuildFailed` and `onBuildSucceeded` to define rules for
   * these specific state changes.
   *
   * To access fields from the event in the event target input,
   * use the static fields on the `StateChangeEvent` class.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  onStateChange(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines a CloudWatch event rule that triggers upon phase change of this
   * build project.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  onPhaseChange(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines an event rule which triggers when a build starts.
   */
  onBuildStarted(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines an event rule which triggers when a build fails.
   */
  onBuildFailed(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines an event rule which triggers when a build completes successfully.
   */
  onBuildSucceeded(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * @returns a CloudWatch metric associated with this build project.
   * @param metricName The name of the metric
   * @param props Customization properties
   */
  metric(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Measures the number of builds triggered.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  metricBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Measures the duration of all builds over time.
   *
   * Units: Seconds
   *
   * Valid CloudWatch statistics: Average (recommended), Maximum, Minimum
   *
   * @default average over 5 minutes
   */
  metricDuration(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Measures the number of successful builds.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  metricSucceededBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Measures the number of builds that failed because of client error or
   * because of a timeout.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  metricFailedBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Defines a CodeStar Notification rule triggered when the project
   * events emitted by you specified, it very similar to `onEvent` API.
   *
   * You can also use the methods `notifyOnBuildSucceeded` and
   * `notifyOnBuildFailed` to define rules for these specific event emitted.
   *
   * @param id The logical identifier of the CodeStar Notifications rule that will be created
   * @param target The target to register for the CodeStar Notifications destination.
   * @param options Customization options for CodeStar Notifications rule
   * @returns CodeStar Notifications rule associated with this build project.
   */
  notifyOn(
    id: string,
    target: notifications.INotificationRuleTarget,
    options: ProjectNotifyOnOptions,
  ): notifications.INotificationRule;

  /**
   * Defines a CodeStar notification rule which triggers when a build completes successfully.
   */
  notifyOnBuildSucceeded(
    id: string,
    target: notifications.INotificationRuleTarget,
    options?: notifications.NotificationRuleOptions,
  ): notifications.INotificationRule;

  /**
   * Defines a CodeStar notification rule which triggers when a build fails.
   */
  notifyOnBuildFailed(
    id: string,
    target: notifications.INotificationRuleTarget,
    options?: notifications.NotificationRuleOptions,
  ): notifications.INotificationRule;
}

/**
 * Represents a reference to a CodeBuild Project.
 *
 * If you're managing the Project alongside the rest of your CDK resources,
 * use the `Project` class.
 *
 * If you want to reference an already existing Project
 * (or one defined in a different CDK Stack),
 * use the `import` method.
 */
abstract class ProjectBase extends Resource implements IProject {
  public abstract readonly grantPrincipal: iam.IPrincipal;

  /** The ARN of this Project. */
  public abstract readonly projectArn: string;

  /** The human-visible name of this Project. */
  public abstract readonly projectName: string;

  /** The IAM service Role of this Project. */
  public abstract readonly role?: iam.IRole;

  public get projectRef(): ProjectReference {
    return {
      projectName: this.projectName,
      projectArn: this.projectArn,
    };
  }

  /**
   * Actual connections object for this Project.
   * May be unset, in which case this Project is not configured to use a VPC.
   * @internal
   */
  protected _connections: ec2.Connections | undefined;

  /**
   * Access the Connections object.
   * Will fail if this Project does not have a VPC set.
   */
  public get connections(): ec2.Connections {
    if (!this._connections) {
      throw new ValidationError(lit`OnlyVpcAssociatedProjectsSecurity`, 'Only VPC-associated Projects have security groups to manage. Supply the "vpc" parameter when creating the Project', this);
    }
    return this._connections;
  }

  public enableBatchBuilds(): BatchBuildConfig | undefined {
    return undefined;
  }

  /**
   * Add a permission only if there's a policy attached.
   * @param statement The permissions statement to add
   */
  public addToRolePolicy(statement: iam.PolicyStatement) {
    if (this.role) {
      this.role.addToPrincipalPolicy(statement);
    }
  }

  /**
   * Defines a CloudWatch event rule triggered when something happens with this project.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  public onEvent(id: string, options: events.OnEventOptions = {}): events.Rule {
    const rule = new events.Rule(this, id, options);
    rule.addTarget(options.target);
    rule.addEventPattern({
      source: ['aws.codebuild'],
      detail: {
        'project-name': [this.projectName],
      },
    });
    return rule;
  }

  /**
   * Defines a CloudWatch event rule triggered when the build project state
   * changes. You can filter specific build status events using an event
   * pattern filter on the `build-status` detail field:
   *
   *    const rule = project.onStateChange('OnBuildStarted', { target });
   *    rule.addEventPattern({
   *      detail: {
   *        'build-status': [
   *          "IN_PROGRESS",
   *          "SUCCEEDED",
   *          "FAILED",
   *          "STOPPED"
   *        ]
   *      }
   *    });
   *
   * You can also use the methods `onBuildFailed` and `onBuildSucceeded` to define rules for
   * these specific state changes.
   *
   * To access fields from the event in the event target input,
   * use the static fields on the `StateChangeEvent` class.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  public onStateChange(id: string, options: events.OnEventOptions = {}) {
    const rule = this.onEvent(id, options);
    rule.addEventPattern({
      detailType: ['CodeBuild Build State Change'],
    });
    return rule;
  }

  /**
   * Defines a CloudWatch event rule that triggers upon phase change of this
   * build project.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-build-notifications.html
   */
  public onPhaseChange(id: string, options: events.OnEventOptions = {}) {
    const rule = this.onEvent(id, options);
    rule.addEventPattern({
      detailType: ['CodeBuild Build Phase Change'],
    });
    return rule;
  }

  /**
   * Defines an event rule which triggers when a build starts.
   *
   * To access fields from the event in the event target input,
   * use the static fields on the `StateChangeEvent` class.
   */
  public onBuildStarted(id: string, options: events.OnEventOptions = {}) {
    const rule = this.onStateChange(id, options);
    rule.addEventPattern({
      detail: {
        'build-status': ['IN_PROGRESS'],
      },
    });
    return rule;
  }

  /**
   * Defines an event rule which triggers when a build fails.
   *
   * To access fields from the event in the event target input,
   * use the static fields on the `StateChangeEvent` class.
   */
  public onBuildFailed(id: string, options: events.OnEventOptions = {}) {
    const rule = this.onStateChange(id, options);
    rule.addEventPattern({
      detail: {
        'build-status': ['FAILED'],
      },
    });
    return rule;
  }

  /**
   * Defines an event rule which triggers when a build completes successfully.
   *
   * To access fields from the event in the event target input,
   * use the static fields on the `StateChangeEvent` class.
   */
  public onBuildSucceeded(id: string, options: events.OnEventOptions = {}) {
    const rule = this.onStateChange(id, options);
    rule.addEventPattern({
      detail: {
        'build-status': ['SUCCEEDED'],
      },
    });
    return rule;
  }

  /**
   * @returns a CloudWatch metric associated with this build project.
   * @param metricName The name of the metric
   * @param props Customization properties
   */
  public metric(metricName: string, props?: cloudwatch.MetricOptions) {
    return new cloudwatch.Metric({
      namespace: 'AWS/CodeBuild',
      metricName,
      dimensionsMap: { ProjectName: this.projectName },
      ...props,
    }).attachTo(this);
  }

  /**
   * Measures the number of builds triggered.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  public metricBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.cannedMetric(CodeBuildMetrics.buildsSum, props);
  }

  /**
   * Measures the duration of all builds over time.
   *
   * Units: Seconds
   *
   * Valid CloudWatch statistics: Average (recommended), Maximum, Minimum
   *
   * @default average over 5 minutes
   */
  public metricDuration(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.cannedMetric(CodeBuildMetrics.durationAverage, props);
  }

  /**
   * Measures the number of successful builds.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  public metricSucceededBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.cannedMetric(CodeBuildMetrics.succeededBuildsSum, props);
  }

  /**
   * Measures the number of builds that failed because of client error or
   * because of a timeout.
   *
   * Units: Count
   *
   * Valid CloudWatch statistics: Sum
   *
   * @default sum over 5 minutes
   */
  public metricFailedBuilds(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.cannedMetric(CodeBuildMetrics.failedBuildsSum, props);
  }

  public notifyOn(
    id: string,
    target: notifications.INotificationRuleTarget,
    options: ProjectNotifyOnOptions,
  ): notifications.INotificationRule {
    return new notifications.NotificationRule(this, id, {
      ...options,
      source: this,
      targets: [target],
    });
  }

  public notifyOnBuildSucceeded(
    id: string,
    target: notifications.INotificationRuleTarget,
    options?: notifications.NotificationRuleOptions,
  ): notifications.INotificationRule {
    return this.notifyOn(id, target, {
      ...options,
      events: [ProjectNotificationEvents.BUILD_SUCCEEDED],
    });
  }

  public notifyOnBuildFailed(
    id: string,
    target: notifications.INotificationRuleTarget,
    options?: notifications.NotificationRuleOptions,
  ): notifications.INotificationRule {
    return this.notifyOn(id, target, {
      ...options,
      events: [ProjectNotificationEvents.BUILD_FAILED],
    });
  }

  public bindAsNotificationRuleSource(_scope: Construct): notifications.NotificationRuleSourceConfig {
    return {
      sourceArn: this.projectArn,
    };
  }

  private cannedMetric(
    fn: (dims: { ProjectName: string }) => cloudwatch.MetricProps,
    props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return new cloudwatch.Metric({
      ...fn({ ProjectName: this.projectName }),
      ...props,
    }).attachTo(this);
  }
}

export interface CommonProjectProps {
  /**
   * A description of the project. Use the description to identify the purpose
   * of the project.
   *
   * @default - No description.
   */
  readonly description?: string;

  /**
   * Filename or contents of buildspec in JSON format.
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html#build-spec-ref-example
   *
   * @default - Empty buildspec.
   */
  readonly buildSpec?: BuildSpec;

  /**
   * Service Role to assume while running the build.
   *
   * @default - A role will be created.
   */
  readonly role?: iam.IRole;

  /**
   * Encryption key to use to read and write artifacts.
   *
   * @default - The AWS-managed CMK for Amazon Simple Storage Service (Amazon S3) is used.
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * Caching strategy to use.
   *
   * @default Cache.none
   */
  readonly cache?: Cache;

  /**
   * Build environment to use for the build.
   *
   * @default BuildEnvironment.LinuxBuildImage.STANDARD_7_0
   */
  readonly environment?: BuildEnvironment;

  /**
   * Indicates whether AWS CodeBuild generates a publicly accessible URL for
   * your project's build badge. For more information, see Build Badges Sample
   * in the AWS CodeBuild User Guide.
   *
   * @default false
   */
  readonly badge?: boolean;

  /**
   * The number of minutes after which AWS CodeBuild stops the build if it's
   * not complete. For valid values, see the timeoutInMinutes field in the AWS
   * CodeBuild User Guide.
   *
   * @default Duration.hours(1)
   */
  readonly timeout?: Duration;

  /**
   * Additional environment variables to add to the build environment.
   *
   * @default - No additional environment variables are specified.
   */
  readonly environmentVariables?: { [name: string]: BuildEnvironmentVariable };

  /**
   * Whether to check for the presence of any secrets in the environment variables of the default type, BuildEnvironmentVariableType.PLAINTEXT.
   * Since using a secret for the value of that kind of variable would result in it being displayed in plain text in the AWS Console,
   * the construct will throw an exception if it detects a secret was passed there.
   * Pass this property as false if you want to skip this validation,
   * and keep using a secret in a plain text environment variable.
   *
   * @default true
   */
  readonly checkSecretsInPlainTextEnvVariables?: boolean;

  /**
   * The physical, human-readable name of the CodeBuild Project.
   *
   * @default - Name is automatically generated.
   */
  readonly projectName?: string;

  /**
   * VPC network to place codebuild network interfaces
   *
   * Specify this if the codebuild project needs to access resources in a VPC.
   *
   * @default - No VPC is specified.
   */
  readonly vpc?: ec2.IVpc;

  /**
   * Where to place the network interfaces within the VPC.
   *
   * To access AWS services, your CodeBuild project needs to be in one of the following types of subnets:
   *
   * 1. Subnets with access to the internet (of type PRIVATE_WITH_EGRESS).
   * 2. Private subnets unconnected to the internet, but with [VPC endpoints](https://docs.aws.amazon.com/codebuild/latest/userguide/use-vpc-endpoints-with-codebuild.html) for the necessary services.
   *
   * If you don't specify a subnet selection, the default behavior is to use PRIVATE_WITH_EGRESS subnets first if they exist,
   * then PRIVATE_WITHOUT_EGRESS, and finally PUBLIC subnets. If your VPC doesn't have PRIVATE_WITH_EGRESS subnets but you need
   * AWS service access, add VPC Endpoints to your private subnets.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/vpc-support.html
   *
   * @default - private subnets if available else public subnets
   */
  readonly subnetSelection?: ec2.SubnetSelection;

  /**
   * What security group to associate with the codebuild project's network interfaces.
   * If no security group is identified, one will be created automatically.
   *
   * Only used if 'vpc' is supplied.
   *
   * @default - Security group will be automatically created.
   *
   */
  readonly securityGroups?: ec2.ISecurityGroup[];

  /**
   * Whether to allow the CodeBuild to send all network traffic
   *
   * If set to false, you must individually add traffic rules to allow the
   * CodeBuild project to connect to network targets.
   *
   * Only used if 'vpc' is supplied.
   *
   * @default true
   */
  readonly allowAllOutbound?: boolean;

  /**
   * An  ProjectFileSystemLocation objects for a CodeBuild build project.
   *
   * A ProjectFileSystemLocation object specifies the identifier, location, mountOptions, mountPoint,
   * and type of a file system created using Amazon Elastic File System.
   *
   * @default - no file system locations
   */
  readonly fileSystemLocations?: IFileSystemLocation[];

  /**
   * Add permissions to this project's role to create and use test report groups with name starting with the name of this project.
   *
   * That is the standard report group that gets created when a simple name
   * (in contrast to an ARN)
   * is used in the 'reports' section of the buildspec of this project.
   * This is usually harmless, but you can turn these off if you don't plan on using test
   * reports in this project.
   *
   * @default true
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/test-report-group-naming.html
   */
  readonly grantReportGroupPermissions?: boolean;

  /**
   * Information about logs for the build project. A project can create logs in Amazon CloudWatch Logs, an S3 bucket, or both.
   *
   * @default - no log configuration is set
   */
  readonly logging?: LoggingOptions;

  /**
   * The number of minutes after which AWS CodeBuild stops the build if it's
   * still in queue. For valid values, see the timeoutInMinutes field in the AWS
   * CodeBuild User Guide.
   *
   * @default - no queue timeout is set
   */
  readonly queuedTimeout?: Duration;

  /**
   * Maximum number of concurrent builds. Minimum value is 1 and maximum is account build limit.
   *
   * @default - no explicit limit is set
   */
  readonly concurrentBuildLimit?: number;

  /**
   * Add the permissions necessary for debugging builds with SSM Session Manager
   *
   * If the following prerequisites have been met:
   *
   * - The necessary permissions have been added by setting this flag to true.
   * - The build image has the SSM agent installed (true for default CodeBuild images).
   * - The build is started with [debugSessionEnabled](https://docs.aws.amazon.com/codebuild/latest/APIReference/API_StartBuild.html#CodeBuild-StartBuild-request-debugSessionEnabled) set to true.
   *
   * Then the build container can be paused and inspected using Session Manager
   * by invoking the `codebuild-breakpoint` command somewhere during the build.
   *
   * `codebuild-breakpoint` commands will be ignored if the build is not started
   * with `debugSessionEnabled=true`.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/session-manager.html
   * @default false
   */
  readonly ssmSessionPermissions?: boolean;

  /**
   * Specifies the visibility of the project's builds.
   *
   * @default - no visibility is set
   */
  readonly visibility?: ProjectVisibility;

  /**
   * CodeBuild will automatically call retry build using the project's service role up to the auto-retry limit.
   * `autoRetryLimit` must be between 0 and 10.
   *
   * @default - no retry
   */
  readonly autoRetryLimit?: number;
}

export interface ProjectProps extends CommonProjectProps {
  /**
   * The source of the build.
   * *Note*: if `NoSource` is given as the source,
   * then you need to provide an explicit `buildSpec`.
   *
   * @default - NoSource
   */
  readonly source?: ISource;

  /**
   * Defines where build artifacts will be stored.
   * Could be: PipelineBuildArtifacts, NoArtifacts and S3Artifacts.
   *
   * @default NoArtifacts
   */
  readonly artifacts?: IArtifacts;

  /**
   * The secondary sources for the Project.
   * Can be also added after the Project has been created by using the `Project#addSecondarySource` method.
   *
   * @default - No secondary sources.
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-multi-in-out.html
   */
  readonly secondarySources?: ISource[];

  /**
   * The secondary artifacts for the Project.
   * Can also be added after the Project has been created by using the `Project#addSecondaryArtifact` method.
   *
   * @default - No secondary artifacts.
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-multi-in-out.html
   */
  readonly secondaryArtifacts?: IArtifacts[];
}

/**
 * The extra options passed to the `IProject.bindToCodePipeline` method.
 */
export interface BindToCodePipelineOptions {
  /**
   * The artifact bucket that will be used by the action that invokes this project.
   */
  readonly artifactBucket: s3.IBucket;
}

/**
 * A representation of a CodeBuild Project.
 */
@propertyInjectable
@noBoxStackTraces
export class Project extends ProjectBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-codebuild.Project';

  public static fromProjectArn(scope: Construct, id: string, projectArn: string): IProject {
    const parsedArn = Stack.of(scope).splitArn(projectArn, ArnFormat.SLASH_RESOURCE_NAME);

    class Import extends ProjectBase {
      public readonly grantPrincipal: iam.IPrincipal;
      public readonly projectArn = projectArn;
      public readonly projectName = parsedArn.resourceName!;
      public readonly role?: iam.Role = undefined;

      constructor(s: Construct, i: string) {
        super(s, i, {
          account: parsedArn.account,
          region: parsedArn.region,
        });
        this.grantPrincipal = new iam.UnknownPrincipal({ resource: this });
      }
    }

    return new Import(scope, id);
  }

  /**
   * Import a Project defined either outside the CDK,
   * or in a different CDK Stack
   * (and exported using the `export` method).
   *
   * @note if you're importing a CodeBuild Project for use
   *   in a CodePipeline, make sure the existing Project
   *   has permissions to access the S3 Bucket of that Pipeline -
   *   otherwise, builds in that Pipeline will always fail.
   *
   * @param scope the parent Construct for this Construct
   * @param id the logical name of this Construct
   * @param projectName the name of the project to import
   * @returns a reference to the existing Project
   */
  public static fromProjectName(scope: Construct, id: string, projectName: string): IProject {
    class Import extends ProjectBase {
      public readonly grantPrincipal: iam.IPrincipal;
      public readonly projectArn: string;
      public readonly projectName: string;
      public readonly role?: iam.Role = undefined;

      constructor(s: Construct, i: string) {
        super(s, i);

        this.projectArn = Stack.of(this).formatArn({
          service: 'codebuild',
          resource: 'project',
          resourceName: projectName,
        });

        this.grantPrincipal = new iam.UnknownPrincipal({ resource: this });
        this.projectName = projectName;
      }
    }

    return new Import(scope, id);
  }

  /**
   * Convert the environment variables map of string to `BuildEnvironmentVariable`,
   * which is the customer-facing type, to a list of `CfnProject.EnvironmentVariableProperty`,
   * which is the representation of environment variables in CloudFormation.
   *
   * @param environmentVariables the map of string to environment variables
   * @param validateNoPlainTextSecrets whether to throw an exception
   *   if any of the plain text environment variables contain secrets, defaults to 'false'
   * @returns an array of `CfnProject.EnvironmentVariableProperty` instances
   */
  public static serializeEnvVariables(environmentVariables: { [name: string]: BuildEnvironmentVariable },
    validateNoPlainTextSecrets: boolean = false, principal?: iam.IGrantable): CfnProject.EnvironmentVariableProperty[] {
    const ret = new Array<CfnProject.EnvironmentVariableProperty>();
    const ssmIamResources = new Array<string>();
    const secretsManagerIamResources = new Set<string>();
    const kmsIamResources = new Set<string>();

    for (const [name, envVariable] of Object.entries(environmentVariables)) {
      const envVariableValue = envVariable.value?.toString();
      const cfnEnvVariable: CfnProject.EnvironmentVariableProperty = {
        name,
        type: envVariable.type || BuildEnvironmentVariableType.PLAINTEXT,
        value: envVariableValue,
      };
      ret.push(cfnEnvVariable);

      // validate that the plain-text environment variables don't contain any secrets in them
      if (validateNoPlainTextSecrets && cfnEnvVariable.type === BuildEnvironmentVariableType.PLAINTEXT) {
        const fragments = Tokenization.reverseString(cfnEnvVariable.value);
        for (const token of fragments.tokens) {
          if (token instanceof SecretValue) {
            throw new UnscopedValidationError(lit`PlaintextEnvironmentVariableContainsSecret`, `Plaintext environment variable '${name}' contains a secret value! ` +
              'This means the value of this variable will be visible in plain text in the AWS Console. ' +
              "Please consider using CodeBuild's SecretsManager environment variables feature instead. " +
              "If you'd like to continue with having this secret in the plaintext environment variables, " +
              'please set the checkSecretsInPlainTextEnvVariables property to false');
          }
        }
      }

      if (principal) {
        const stack = Stack.of(principal as unknown as IConstruct);

        // save the SSM env variables
        if (envVariable.type === BuildEnvironmentVariableType.PARAMETER_STORE) {
          ssmIamResources.push(stack.formatArn({
            service: 'ssm',
            resource: 'parameter',
            // If the parameter name starts with / the resource name is not separated with a double '/'
            // arn:aws:ssm:region:1111111111:parameter/PARAM_NAME
            resourceName: envVariableValue.startsWith('/')
              ? envVariableValue.slice(1)
              : envVariableValue,
          }));
        }

        // save SecretsManager env variables
        if (envVariable.type === BuildEnvironmentVariableType.SECRETS_MANAGER) {
          // We have 3 basic cases here of what envVariableValue can be:
          // 1. A string that starts with 'arn:' (and might contain Token fragments).
          // 2. A Token.
          // 3. A simple value, like 'secret-id'.
          if (envVariableValue.startsWith('arn:')) {
            const parsedArn = stack.splitArn(envVariableValue, ArnFormat.COLON_RESOURCE_NAME);
            if (!parsedArn.resourceName) {
              throw new UnscopedValidationError(lit`SecretsManagerArnMissingSecretName`, 'SecretManager ARN is missing the name of the secret: ' + envVariableValue);
            }

            // the value of the property can be a complex string, separated by ':';
            // see https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html#build-spec.env.secrets-manager
            const secretName = parsedArn.resourceName.split(':')[0];
            secretsManagerIamResources.add(stack.formatArn({
              service: 'secretsmanager',
              resource: 'secret',
              // since we don't know whether the ARN was full, or partial
              // (CodeBuild supports both),
              // stick a "*" at the end, which makes it work for both
              resourceName: `${secretName}*`,
              arnFormat: ArnFormat.COLON_RESOURCE_NAME,
              partition: parsedArn.partition,
              account: parsedArn.account,
              region: parsedArn.region,
            }));
            // if secret comes from another account, SecretsManager will need to access
            // KMS on the other account as well to be able to get the secret
            if (parsedArn.account && Token.compareStrings(parsedArn.account, stack.account) === TokenComparison.DIFFERENT) {
              kmsIamResources.add(stack.formatArn({
                service: 'kms',
                resource: 'key',
                // We do not know the ID of the key, but since this is a cross-account access,
                // the key policies have to allow this access, so a wildcard is safe here
                resourceName: '*',
                arnFormat: ArnFormat.SLASH_RESOURCE_NAME,
                partition: parsedArn.partition,
                account: parsedArn.account,
                region: parsedArn.region,
              }));
            }
          } else if (Token.isUnresolved(envVariableValue)) {
            // the value of the property can be a complex string, separated by ':';
            // see https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html#build-spec.env.secrets-manager
            let secretArn = envVariableValue.split(':')[0];

            // parse the Token, and see if it represents a single resource
            // (we will assume it's a Secret from SecretsManager)
            const fragments = Tokenization.reverseString(envVariableValue);
            if (fragments.tokens.length === 1) {
              const resolvable = fragments.tokens[0];
              if (Reference.isReference(resolvable)) {
                // check the Stack the resource owning the reference belongs to
                const resourceStack = Stack.of(resolvable.target);
                if (Token.compareStrings(stack.account, resourceStack.account) === TokenComparison.DIFFERENT) {
                  // since this is a cross-account access,
                  // add the appropriate KMS permissions
                  kmsIamResources.add(stack.formatArn({
                    service: 'kms',
                    resource: 'key',
                    // We do not know the ID of the key, but since this is a cross-account access,
                    // the key policies have to allow this access, so a wildcard is safe here
                    resourceName: '*',
                    arnFormat: ArnFormat.SLASH_RESOURCE_NAME,
                    partition: resourceStack.partition,
                    account: resourceStack.account,
                    region: resourceStack.region,
                  }));

                  // Work around a bug in SecretsManager -
                  // when the access is cross-environment,
                  // Secret.secretArn returns a partial ARN!
                  // So add a "*" at the end, so that the permissions work
                  secretArn = `${secretArn}-??????`;
                }
              }
            }

            // if we are passed a Token, we should assume it's the ARN of the Secret
            // (as the name would not work anyway, because it would be the full name, which CodeBuild does not support)
            secretsManagerIamResources.add(secretArn);
          } else {
            // the value of the property can be a complex string, separated by ':';
            // see https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html#build-spec.env.secrets-manager
            const secretName = envVariableValue.split(':')[0];
            secretsManagerIamResources.add(stack.formatArn({
              service: 'secretsmanager',
              resource: 'secret',
              resourceName: `${secretName}-??????`,
              arnFormat: ArnFormat.COLON_RESOURCE_NAME,
            }));
          }
        }
      }
    }

    if (ssmIamResources.length !== 0) {
      principal?.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ['ssm:GetParameters'],
        resources: ssmIamResources,
      }));
    }
    if (secretsManagerIamResources.size !== 0) {
      principal?.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ['secretsmanager:GetSecretValue'],
        resources: Array.from(secretsManagerIamResources),
      }));
    }
    if (kmsIamResources.size !== 0) {
      principal?.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ['kms:Decrypt'],
        resources: Array.from(kmsIamResources),
      }));
    }

    return ret;
  }

  public readonly grantPrincipal: iam.IPrincipal;

  /**
   * The IAM role for this project.
   */
  public readonly role?: iam.IRole;

  /**
   * The ARN of the project.
   */
  @memoizedGetter
  get projectArn(): string {
    return this.getResourceArnAttribute(this.resource.attrArn, {
      service: 'codebuild',
      resource: 'project',
      resourceName: this.physicalName,
    });
  }

  /**
   * The name of the project.
   */
  @memoizedGetter
  get projectName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  private readonly source: ISource;
  private readonly buildImage: IBuildImage;
  private readonly _secondarySources: IArrayBox<CfnProject.SourceProperty>;
  private readonly _secondarySourceVersions: IArrayBox<CfnProject.ProjectSourceVersionProperty>;
  private readonly resource: CfnProject;
  private readonly _secondaryArtifacts: IArrayBox<CfnProject.ArtifactsProperty>;
  private readonly _encryptionKey: IBox<kms.IKey | undefined> = Box.fromValue<kms.IKey | undefined>(undefined);
  private readonly _fileSystemLocations: IArrayBox<CfnProject.ProjectFileSystemLocationProperty>;
  private readonly _batchServiceRole: IBox<iam.Role | undefined> = Box.fromValue<iam.Role | undefined>(undefined);

  constructor(scope: Construct, id: string, props: ProjectProps) {
    super(scope, id, {
      physicalName: props.projectName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.role = props.role || new iam.Role(this, 'Role', {
      roleName: PhysicalName.GENERATE_IF_NEEDED,
      assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
    });
    this.grantPrincipal = this.role;

    this.buildImage = (props.environment && props.environment.buildImage) || LinuxBuildImage.STANDARD_7_0;

    // let source "bind" to the project. this usually involves granting permissions
    // for the code build role to interact with the source.
    this.source = props.source || new NoSource();
    const sourceConfig = this.source.bind(this, this);
    if (props.badge && !this.source.badgeSupported) {
      throw new ValidationError(lit`BadgeSupportedSourceType`, `Badge is not supported for source type ${this.source.type}`, this);
    }

    const artifacts = props.artifacts
      ? props.artifacts
      : (this.source.type === CODEPIPELINE_SOURCE_ARTIFACTS_TYPE
        ? new CodePipelineArtifacts()
        : new NoArtifacts());
    const artifactsConfig = artifacts.bind(this, this);

    const cache = props.cache || Cache.none();

    // give the caching strategy the option to grant permissions to any required resources
    cache._bind(this);

    // Inject download commands for asset if requested
    const environmentVariables = props.environmentVariables || {};
    const buildSpec = props.buildSpec;
    if (this.source.type === NO_SOURCE_TYPE && (buildSpec === undefined || !buildSpec.isImmediate)) {
      throw new ValidationError(lit`ProjectSSourceNosource`, "If the Project's source is NoSource, you need to provide a concrete buildSpec", this);
    }

    this._secondarySources = Box.fromArray();
    this._secondarySourceVersions = Box.fromArray();
    this._fileSystemLocations = Box.fromArray();
    for (const secondarySource of props.secondarySources || []) {
      this.addSecondarySource(secondarySource);
    }

    this._secondaryArtifacts = Box.fromArray();
    for (const secondaryArtifact of props.secondaryArtifacts || []) {
      this.addSecondaryArtifact(secondaryArtifact);
    }

    this.validateCodePipelineSettings(artifacts);

    for (const fileSystemLocation of props.fileSystemLocations || []) {
      this.addFileSystemLocation(fileSystemLocation);
    }

    if (!Token.isUnresolved(props.autoRetryLimit) && (props.autoRetryLimit !== undefined)) {
      if (props.autoRetryLimit < 0 || props.autoRetryLimit > 10) {
        throw new ValidationError(lit`AutoRetryLimitValue`, `autoRetryLimit must be a value between 0 and 10, got ${props.autoRetryLimit}.`, this);
      }
    }

    const resource = new CfnProject(this, 'Resource', {
      description: props.description,
      source: {
        ...sourceConfig.sourceProperty,
        buildSpec: buildSpec && buildSpec.toBuildSpec(this),
      },
      artifacts: artifactsConfig.artifactsProperty,
      serviceRole: this.role.roleArn,
      environment: this.renderEnvironment(props, environmentVariables),
      fileSystemLocations: Token.asAny(this._fileSystemLocations.derive(arr => arr.length === 0 ? undefined : arr)),
      // deferred, because we have a setter for it in setEncryptionKey
      // The 'alias/aws/s3' default is necessary because leaving the `encryptionKey` field
      // empty will not remove existing encryptionKeys during an update (ref. t/D17810523)
      // eslint-disable-next-line @cdklabs/no-unconditional-token-allocation
      encryptionKey: Token.asString(Box.combine({ key: this._encryptionKey }, ({ key }) => key ? key.keyArn : 'alias/aws/s3')),
      badgeEnabled: props.badge,
      cache: cache._toCloudFormation(),
      name: this.physicalName,
      timeoutInMinutes: props.timeout && props.timeout.toMinutes(),
      queuedTimeoutInMinutes: props.queuedTimeout && props.queuedTimeout.toMinutes(),
      concurrentBuildLimit: props.concurrentBuildLimit,
      secondarySources: Token.asAny(this._secondarySources.derive(arr => arr.length === 0 ? undefined : arr)),
      secondarySourceVersions: Token.asAny(this._secondarySourceVersions.derive(arr => arr.length === 0 ? undefined : arr)),
      secondaryArtifacts: Token.asAny(this._secondaryArtifacts.derive(arr => arr.length === 0 ? undefined : arr)),
      triggers: sourceConfig.buildTriggers,
      sourceVersion: sourceConfig.sourceVersion,
      vpcConfig: this.configureVpc(props),
      visibility: props.visibility,
      logsConfig: this.renderLoggingConfiguration(props.logging),
      buildBatchConfig: Token.asAny(this._batchServiceRole.derive(role => role ? { serviceRole: role.roleArn } : undefined)),
      autoRetryLimit: props.autoRetryLimit,
    });

    this.addVpcRequiredPermissions(props, resource);

    this.resource = resource;

    this.addToRolePolicy(this.createLoggingPermission());
    // add permissions to create and use test report groups
    // with names starting with the project's name,
    // unless the customer explicitly opts out of it
    if (props.grantReportGroupPermissions !== false) {
      this.addToRolePolicy(new iam.PolicyStatement({
        actions: [
          'codebuild:CreateReportGroup',
          'codebuild:CreateReport',
          'codebuild:UpdateReport',
          'codebuild:BatchPutTestCases',
          'codebuild:BatchPutCodeCoverages',
        ],
        resources: [renderReportGroupArn(this, `${this.projectName}-*`)],
      }));
    }

    // https://docs.aws.amazon.com/codebuild/latest/userguide/session-manager.html
    if (props.ssmSessionPermissions) {
      this.addToRolePolicy(new iam.PolicyStatement({
        actions: [
          // For the SSM channel
          'ssmmessages:CreateControlChannel',
          'ssmmessages:CreateDataChannel',
          'ssmmessages:OpenControlChannel',
          'ssmmessages:OpenDataChannel',
          // In case the SSM session is set up to log commands to CloudWatch
          'logs:DescribeLogGroups',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          // In case the SSM session is set up to log commands to S3.
          's3:GetEncryptionConfiguration',
          's3:PutObject',
        ],
        resources: ['*'],
      }));
    }

    if (props.encryptionKey) {
      this.encryptionKey = props.encryptionKey;
    }

    // bind
    if (isBindableBuildImage(this.buildImage)) {
      this.buildImage.bind(this, this, {});
    }

    this.node.addValidation({ validate: () => this.validateProject() });
  }

  @MethodMetadata()
  public enableBatchBuilds(): BatchBuildConfig | undefined {
    if (!this._batchServiceRole.get()) {
      const batchRole = new iam.Role(this, 'BatchServiceRole', {
        assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
      });
      batchRole.addToPrincipalPolicy(new iam.PolicyStatement({
        resources: [Lazy.string({
          produce: () => this.projectArn,
        })],
        actions: [
          'codebuild:StartBuild',
          'codebuild:StopBuild',
          'codebuild:RetryBuild',
        ],
      }));
      this._batchServiceRole.set(batchRole);
    }
    return {
      role: this._batchServiceRole.get()!,
    };
  }

  /**
   * Adds a secondary source to the Project.
   *
   * @param secondarySource the source to add as a secondary source
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-multi-in-out.html
   */
  @MethodMetadata()
  public addSecondarySource(secondarySource: ISource): void {
    if (!secondarySource.identifier) {
      throw new ValidationError(lit`IdentifierAttributeMandatorySecondarySources`, 'The identifier attribute is mandatory for secondary sources', this);
    }
    const secondarySourceConfig = secondarySource.bind(this, this);
    this._secondarySources.push(secondarySourceConfig.sourceProperty);
    if (secondarySourceConfig.sourceVersion) {
      this._secondarySourceVersions.push({
        sourceIdentifier: secondarySource.identifier,
        sourceVersion: secondarySourceConfig.sourceVersion,
      });
    }
  }

  /**
   * Adds a fileSystemLocation to the Project.
   *
   * @param fileSystemLocation the fileSystemLocation to add
   */
  @MethodMetadata()
  public addFileSystemLocation(fileSystemLocation: IFileSystemLocation): void {
    const fileSystemConfig = fileSystemLocation.bind(this, this);
    this._fileSystemLocations.push(fileSystemConfig.location);
  }

  /**
   * Adds a secondary artifact to the Project.
   *
   * @param secondaryArtifact the artifact to add as a secondary artifact
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-multi-in-out.html
   */
  @MethodMetadata()
  public addSecondaryArtifact(secondaryArtifact: IArtifacts): void {
    if (!secondaryArtifact.identifier) {
      throw new ValidationError(lit`IdentifierAttributeMandatorySecondaryArtifacts`, 'The identifier attribute is mandatory for secondary artifacts', this);
    }
    this._secondaryArtifacts.push(secondaryArtifact.bind(this, this).artifactsProperty);
  }

  /**
   * A callback invoked when the given project is added to a CodePipeline.
   *
   * @param _scope the construct the binding is taking place in
   * @param options additional options for the binding
   */
  @MethodMetadata()
  public bindToCodePipeline(_scope: Construct, options: BindToCodePipelineOptions): void {
    // work around a bug in CodeBuild: it ignores the KMS key set on the pipeline,
    // and always uses its own, project-level key
    if (options.artifactBucket.encryptionKey && !this._encryptionKey.get()) {
      // we cannot safely do this assignment if the key is of type kms.Key,
      // and belongs to a stack in a different account or region than the project
      // (that would cause an illegal reference, as KMS keys don't have physical names)
      const keyStack = Stack.of(options.artifactBucket.encryptionKey);
      const projectStack = Stack.of(this);
      if (!(options.artifactBucket.encryptionKey instanceof kms.Key &&
        (keyStack.account !== projectStack.account || keyStack.region !== projectStack.region))) {
        this.encryptionKey = options.artifactBucket.encryptionKey;
      }
    }
  }

  private validateProject(): string[] {
    const ret = new Array<string>();
    if (this.source.type === CODEPIPELINE_SOURCE_ARTIFACTS_TYPE) {
      if (this._secondarySources.length > 0) {
        ret.push('A Project with a CodePipeline Source cannot have secondary sources. ' +
          "Use the CodeBuild Pipeline Actions' `extraInputs` property instead");
      }
      if (this._secondaryArtifacts.length > 0) {
        ret.push('A Project with a CodePipeline Source cannot have secondary artifacts. ' +
          "Use the CodeBuild Pipeline Actions' `outputs` property instead");
      }
    }
    return ret;
  }

  private set encryptionKey(encryptionKey: kms.IKey) {
    this._encryptionKey.set(encryptionKey);
    encryptionKey.grantEncryptDecrypt(this);
  }

  private createLoggingPermission() {
    const logGroupArn = Stack.of(this).formatArn({
      service: 'logs',
      resource: 'log-group',
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
      resourceName: `/aws/codebuild/${this.projectName}`,
    });

    const logGroupStarArn = `${logGroupArn}:*`;

    return new iam.PolicyStatement({
      resources: [logGroupArn, logGroupStarArn],
      actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
    });
  }

  private renderEnvironment(
    props: ProjectProps,
    projectVars: { [name: string]: BuildEnvironmentVariable } = {}): CfnProject.EnvironmentProperty {
    const env = props.environment ?? {};
    const vars: { [name: string]: BuildEnvironmentVariable } = {};
    const containerVars = env.environmentVariables || {};

    // first apply environment variables from the container definition
    for (const name of Object.keys(containerVars)) {
      vars[name] = containerVars[name];
    }

    // now apply project-level vars
    for (const name of Object.keys(projectVars)) {
      vars[name] = projectVars[name];
    }

    const hasEnvironmentVars = Object.keys(vars).length > 0;

    const errors = this.buildImage.validate(env);

    errors.push(...this.validateLambdaBuildImage(this.buildImage, props));

    if (errors.length > 0) {
      throw new ValidationError(lit`InvalidCodeBuildEnvironment`, 'Invalid CodeBuild environment: ' + errors.join('\n'), this);
    }

    const imagePullPrincipalType = this.isLambdaBuildImage(this.buildImage) ? undefined :
      this.buildImage.imagePullPrincipalType === ImagePullPrincipalType.CODEBUILD
        ? ImagePullPrincipalType.CODEBUILD
        : ImagePullPrincipalType.SERVICE_ROLE;
    if (this.buildImage.repository) {
      if (imagePullPrincipalType === ImagePullPrincipalType.SERVICE_ROLE) {
        this.buildImage.repository.grantPull(this);
      } else {
        const statement = new iam.PolicyStatement({
          principals: [new iam.ServicePrincipal('codebuild.amazonaws.com')],
          actions: ['ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage', 'ecr:BatchCheckLayerAvailability'],
        });
        statement.sid = 'CodeBuild';
        this.buildImage.repository.addToResourcePolicy(statement);
      }
    }
    if (imagePullPrincipalType === ImagePullPrincipalType.SERVICE_ROLE) {
      this.buildImage.secretsManagerCredentials?.grantRead(this);
    }

    const secret = this.buildImage.secretsManagerCredentials;
    return {
      type: this.buildImage.type,
      image: this.buildImage.imageId,
      imagePullCredentialsType: imagePullPrincipalType,
      registryCredential: secret
        ? {
          credentialProvider: 'SECRETS_MANAGER',
          // Secrets must be referenced by either the full ARN (with SecretsManager suffix), or by name.
          // "Partial" ARNs (without the suffix) will fail a validation regex at deploy-time.
          credential: secret.secretFullArn ?? secret.secretName,
        }
        : undefined,
      certificate: env.certificate?.bucket.arnForObjects(env.certificate.objectKey),
      privilegedMode: env.privileged || false,
      fleet: this.configureFleet(env),
      computeType: env.computeType || this.buildImage.defaultComputeType,
      environmentVariables: hasEnvironmentVars
        ? Project.serializeEnvVariables(vars, props.checkSecretsInPlainTextEnvVariables ?? true, this)
        : undefined,
      dockerServer: env.dockerServer ? {
        computeType: env.dockerServer.computeType,
        securityGroupIds: env.dockerServer.securityGroups?.map(e => e.securityGroupId),
      } : undefined,
    };
  }

  private configureFleet({ fleet }: BuildEnvironment): { fleetArn: string } | undefined {
    if (!fleet) {
      return undefined;
    }

    // If the fleetArn is resolved, the fleet is imported and we cannot validate the environment type
    if (Token.isUnresolved(fleet.fleetArn) && this.buildImage.type !== fleet.environmentType) {
      throw new ValidationError(lit`EnvironmentTypeFleet`, `The environment type of the fleet (${fleet.environmentType}) must match the environment type of the build image (${this.buildImage.type})`, this);
    }

    return { fleetArn: fleet.fleetArn };
  }

  /**
   * If configured, set up the VPC-related properties
   *
   * Returns the VpcConfig that should be added to the
   * codebuild creation properties.
   */
  private configureVpc(props: ProjectProps): CfnProject.VpcConfigProperty | undefined {
    if ((props.securityGroups || props.allowAllOutbound !== undefined) && !props.vpc) {
      throw new ValidationError(lit`CannotConfigureSecurityGroupAllow`, 'Cannot configure \'securityGroup\' or \'allowAllOutbound\' without configuring a VPC', this);
    }

    if (!props.vpc) { return undefined; }

    if (props.environment?.fleet) {
      // Should throw a ValidationError, but we are only warning, to preserve
      // backward compatibility.
      Annotations.of(this).addWarningV2(
        '@aws-cdk/aws-codebuild:noUselessProjectVpc',
        'Project \'vpc\' does nothing when using a Fleet. Configure the VPC on the fleet instead.',
      );
    }

    if ((props.securityGroups && props.securityGroups.length > 0) && props.allowAllOutbound !== undefined) {
      throw new ValidationError(lit`ConfigureAllowOutboundDirectlySupplied`, 'Configure \'allowAllOutbound\' directly on the supplied SecurityGroup.', this);
    }

    let securityGroups: ec2.ISecurityGroup[];
    if (props.securityGroups && props.securityGroups.length > 0) {
      securityGroups = props.securityGroups;
    } else {
      const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
        vpc: props.vpc,
        description: 'Automatic generated security group for CodeBuild ' + Names.uniqueId(this),
        allowAllOutbound: props.allowAllOutbound,
      });
      securityGroups = [securityGroup];
    }
    this._connections = new ec2.Connections({ securityGroups });

    return {
      vpcId: props.vpc.vpcId,
      subnets: props.vpc.selectSubnets(props.subnetSelection).subnetIds,
      securityGroupIds: this.connections.securityGroups.map(s => s.securityGroupId),
    };
  }

  private renderLoggingConfiguration(props: LoggingOptions | undefined): CfnProject.LogsConfigProperty | undefined {
    if (props === undefined) {
      return undefined;
    }

    let s3Config: CfnProject.S3LogsConfigProperty | undefined = undefined;
    let cloudwatchConfig: CfnProject.CloudWatchLogsConfigProperty | undefined = undefined;

    if (props.s3) {
      const s3Logs = props.s3;
      s3Config = {
        status: (s3Logs.enabled ?? true) ? 'ENABLED' : 'DISABLED',
        location: `${s3Logs.bucket.bucketName}` + (s3Logs.prefix ? `/${s3Logs.prefix}` : ''),
        encryptionDisabled: s3Logs.encrypted !== undefined ? !s3Logs.encrypted : undefined,
      };
      s3Logs.bucket?.grantWrite(this);
    }

    if (props.cloudWatch) {
      const cloudWatchLogs = props.cloudWatch;
      const status = (cloudWatchLogs.enabled ?? true) ? 'ENABLED' : 'DISABLED';

      if (status === 'ENABLED' && !(cloudWatchLogs.logGroup)) {
        throw new ValidationError(lit`SpecifyingLogGroupRequiredCloud`, 'Specifying a LogGroup is required if CloudWatch logging for CodeBuild is enabled', this);
      }
      cloudWatchLogs.logGroup?.grantWrite(this);

      cloudwatchConfig = {
        status,
        groupName: cloudWatchLogs.logGroup?.logGroupRef.logGroupName,
        streamName: cloudWatchLogs.prefix,
      };
    }

    return {
      s3Logs: s3Config,
      cloudWatchLogs: cloudwatchConfig,
    };
  }

  private addVpcRequiredPermissions(props: ProjectProps, project: CfnProject): void {
    if (!props.vpc || !this.role) {
      return;
    }

    this.role.addToPrincipalPolicy(new iam.PolicyStatement({
      resources: [`arn:${Aws.PARTITION}:ec2:${Aws.REGION}:${Aws.ACCOUNT_ID}:network-interface/*`],
      actions: ['ec2:CreateNetworkInterfacePermission'],
      conditions: {
        StringEquals: {
          'ec2:Subnet': props.vpc
            .selectSubnets(props.subnetSelection).subnetIds
            .map(si => `arn:${Aws.PARTITION}:ec2:${Aws.REGION}:${Aws.ACCOUNT_ID}:subnet/${si}`),
          'ec2:AuthorizedService': 'codebuild.amazonaws.com',
        },
      },
    }));

    // If the same Role is used for multiple Projects, always creating a new `iam.Policy`
    // will attach the same policy multiple times, probably exceeding the maximum size of the
    // Role policy. Make sure we only do it once for the same role.
    //
    // This deduplication could be a feature of the Role itself, but that feels risky and
    // is hard to implement (what with Tokens and all). Safer to fix it locally for now.
    let policy: iam.Policy | undefined = (this.role as any)[VPC_POLICY_SYM];
    if (!policy) {
      policy = new iam.Policy(this, 'PolicyDocument', {
        statements: [
          new iam.PolicyStatement({
            resources: ['*'],
            actions: [
              'ec2:CreateNetworkInterface',
              'ec2:DescribeNetworkInterfaces',
              'ec2:DeleteNetworkInterface',
              'ec2:DescribeSubnets',
              'ec2:DescribeSecurityGroups',
              'ec2:DescribeDhcpOptions',
              'ec2:DescribeVpcs',
            ],
          }),
        ],
      });
      this.role.attachInlinePolicy(policy);
      (this.role as any)[VPC_POLICY_SYM] = policy;
    }

    // add an explicit dependency between the EC2 Policy and this Project -
    // otherwise, creating the Project fails, as it requires these permissions
    // to be already attached to the Project's Role
    project.node.addDependency(policy);
  }

  private validateCodePipelineSettings(artifacts: IArtifacts) {
    const sourceType = this.source.type;
    const artifactsType = artifacts.type;

    if ((sourceType === CODEPIPELINE_SOURCE_ARTIFACTS_TYPE ||
        artifactsType === CODEPIPELINE_SOURCE_ARTIFACTS_TYPE) &&
        (sourceType !== artifactsType)) {
      throw new ValidationError(lit`SourceArtifactsSetCodePipeline`, 'Both source and artifacts must be set to CodePipeline', this);
    }
  }

  private isLambdaBuildImage(buildImage: IBuildImage): boolean {
    return buildImage instanceof LinuxLambdaBuildImage || buildImage instanceof LinuxArmLambdaBuildImage;
  }

  /**
   * Validates a Lambda build image given the project properties.
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/lambda.html#lambda.limitations
   */
  private validateLambdaBuildImage(buildImage: IBuildImage, props: ProjectProps): string[] {
    if (!this.isLambdaBuildImage(buildImage)) return [];
    const errors = [];
    if (props.timeout) {
      errors.push('Cannot specify timeout for Lambda compute');
    }
    if (props.queuedTimeout) {
      errors.push('Cannot specify queuedTimeout for Lambda compute');
    }
    if (props.cache) {
      errors.push('Cannot specify cache for Lambda compute');
    }
    if (props.badge) {
      errors.push('Cannot enable badge for Lambda compute');
    }
    return errors;
  }
}

export interface BuildEnvironment {
  /**
   * The image used for the builds.
   *
   * @default LinuxBuildImage.STANDARD_7_0
   */
  readonly buildImage?: IBuildImage;

  /**
   * The type of compute to use for this build.
   * See the `ComputeType` enum for the possible values.
   *
   * @default taken from `#buildImage#defaultComputeType`
   */
  readonly computeType?: ComputeType;

  /**
   * The Docker server configuration CodeBuild use to build your Docker image.
   *
   * @note Security groups configured for Docker servers should allow ingress network traffic
   * from the VPC configured in the project. They should allow ingress on port 9876.
   * @default - Doesn't use remote docker server
   */
  readonly dockerServer?: DockerServerOptions;

  /**
   * Fleet resource for a reserved capacity CodeBuild project.
   *
   * Fleets allow for process builds or tests to run immediately and reduces build durations,
   * by reserving compute resources for your projects.
   *
   * You will be charged for the resources in the fleet, even if they are idle.
   *
   * @default - No fleet will be attached to the project, which will remain on-demand.
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/fleets.html
   */
  readonly fleet?: IFleet;

  /**
   * Indicates how the project builds Docker images. Specify true to enable
   * running the Docker daemon inside a Docker container. This value must be
   * set to true only if this build project will be used to build Docker
   * images, and the specified build environment image is not one provided by
   * AWS CodeBuild with Docker support. Otherwise, all associated builds that
   * attempt to interact with the Docker daemon will fail.
   *
   * @default false
   */
  readonly privileged?: boolean;

  /**
   * The location of the PEM-encoded certificate for the build project
   *
   * @default - No external certificate is added to the project
   */
  readonly certificate?: BuildEnvironmentCertificate;

  /**
   * The environment variables that your builds can use.
   */
  readonly environmentVariables?: { [name: string]: BuildEnvironmentVariable };
}

/**
 * The Docker server configuration CodeBuild use to build your Docker image.
 */
export interface DockerServerOptions {
  /**
   * The type of compute to use for the docker server.
   * See the `DockerServerComputeType` enum for the possible values.
   */
  readonly computeType: DockerServerComputeType;

  /**
   * A list of maximum 5 security groups.
   *
   * @note Security groups configured for Docker servers should allow ingress network traffic
   * from the VPC configured in the project. They should allow ingress on port 9876.
   * @default - no security group
   */
  readonly securityGroups?: ec2.ISecurityGroup[];
}

/**
 * Represents a Docker image used for the CodeBuild Project builds.
 * Use the concrete subclasses, either:
 * `LinuxBuildImage` or `WindowsBuildImage`.
 */
export interface IBuildImage {
  /**
   * The type of build environment.
   */
  readonly type: string;

  /**
   * The Docker image identifier that the build environment uses.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
   */
  readonly imageId: string;

  /**
   * The default `ComputeType` to use with this image,
   * if one was not specified in `BuildEnvironment#computeType` explicitly.
   */
  readonly defaultComputeType: ComputeType;

  /**
   * The type of principal that CodeBuild will use to pull this build Docker image.
   *
   * @default ImagePullPrincipalType.SERVICE_ROLE
   */
  readonly imagePullPrincipalType?: ImagePullPrincipalType;

  /**
   * The secretsManagerCredentials for access to a private registry.
   *
   * @default no credentials will be used
   */
  readonly secretsManagerCredentials?: secretsmanager.ISecret;

  /**
   * An optional ECR repository that the image is hosted in.
   *
   * @default no repository
   */
  readonly repository?: ecr.IRepository;

  /**
   * Allows the image a chance to validate whether the passed configuration is correct.
   *
   * @param buildEnvironment the current build environment
   */
  validate(buildEnvironment: BuildEnvironment): string[];

  /**
   * Make a buildspec to run the indicated script
   */
  runScriptBuildspec(entrypoint: string): BuildSpec;
}

/** Optional arguments to `IBuildImage.binder` - currently empty. */
export interface BuildImageBindOptions { }

/** The return type from `IBuildImage.binder` - currently empty. */
export interface BuildImageConfig { }

// @deprecated(not in tsdoc on purpose): add bind() to IBuildImage
// and get rid of IBindableBuildImage

/** A variant of `IBuildImage` that allows binding to the project. */
export interface IBindableBuildImage extends IBuildImage {
  /** Function that allows the build image access to the construct tree. */
  bind(scope: Construct, project: IProject, options: BuildImageBindOptions): BuildImageConfig;
}

/**
 * The options when creating a CodeBuild Docker build image
 * using `LinuxBuildImage.fromDockerRegistry`,
 * `WindowsBuildImage.fromDockerRegistry`,
 * or `MacBuildImage.fromDockerRegistry`
 */
export interface DockerImageOptions {
  /**
   * The credentials, stored in Secrets Manager,
   * used for accessing the repository holding the image,
   * if the repository is private.
   *
   * @default no credentials will be used (we assume the repository is public)
   */
  readonly secretsManagerCredentials?: secretsmanager.ISecret;
}

/**
 * Construction properties of `LinuxBuildImage`.
 * Module-private, as the constructor of `LinuxBuildImage` is private.
 */
interface LinuxBuildImageProps {
  readonly imageId: string;
  readonly imagePullPrincipalType?: ImagePullPrincipalType;
  readonly secretsManagerCredentials?: secretsmanager.ISecret;
  readonly repository?: ecr.IRepository;
}

// Keep around to resolve a circular dependency until removing deprecated ARM image constants from LinuxBuildImage
// eslint-disable-next-line import/order
import { LinuxArmBuildImage } from './linux-arm-build-image';

/**
 * A CodeBuild image running x86-64 Linux.
 *
 * This class has a bunch of public constants that represent the most popular images.
 *
 * You can also specify a custom image using one of the static methods:
 *
 * - LinuxBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }])
 * - LinuxBuildImage.fromEcrRepository(repo[, tag])
 * - LinuxBuildImage.fromAsset(parent, id, props)
 *
 *
 * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
 */
export class LinuxBuildImage implements IBuildImage {
  /** @deprecated Use {@link LinuxBuildImage.STANDARD_7_0} instead. */
  public static readonly STANDARD_1_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:1.0');
  /** @deprecated Use {@link LinuxBuildImage.STANDARD_7_0} instead. */
  public static readonly STANDARD_2_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:2.0');
  /** @deprecated Use {@link LinuxBuildImage.STANDARD_7_0} instead. */
  public static readonly STANDARD_3_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:3.0');
  /**
   * The `aws/codebuild/standard:4.0` build image.
   * @deprecated Use {@link LinuxBuildImage.STANDARD_7_0} instead.
   * */
  public static readonly STANDARD_4_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:4.0');
  /** The `aws/codebuild/standard:5.0` build image. */
  public static readonly STANDARD_5_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:5.0');
  /** The `aws/codebuild/standard:6.0` build image. */
  public static readonly STANDARD_6_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:6.0');
  /** The `aws/codebuild/standard:7.0` build image. */
  public static readonly STANDARD_7_0 = LinuxBuildImage.codeBuildImage('aws/codebuild/standard:7.0');

  /** @deprecated Use {@link LinuxBuildImage.AMAZON_LINUX_2_5} instead. */
  public static readonly AMAZON_LINUX_2 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:1.0');
  /** @deprecated Use {@link LinuxBuildImage.AMAZON_LINUX_2_5} instead. */
  public static readonly AMAZON_LINUX_2_2 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:2.0');
  /**
   * The Amazon Linux 2 x86_64 standard image, version `3.0`.
   * @deprecated Use {@link LinuxBuildImage.AMAZON_LINUX_2_5} instead.
   * */
  public static readonly AMAZON_LINUX_2_3 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:3.0');
  /** The Amazon Linux 2 x86_64 standard image, version `4.0`. */
  public static readonly AMAZON_LINUX_2_4 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:4.0');
  /** The Amazon Linux 2023 x86_64 standard image, version `5.0`. */
  public static readonly AMAZON_LINUX_2_5 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:5.0');

  /** The Amazon Linux 2023 x86_64 standard image, version `4.0`. */
  public static readonly AMAZON_LINUX_2023_4 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux-x86_64-standard:4.0');
  /** The Amazon Linux 2023 x86_64 standard image, version `5.0`. */
  public static readonly AMAZON_LINUX_2023_5 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux-x86_64-standard:5.0');

  /** The Amazon Coretto 8 image x86_64, based on Amazon Linux 2. */
  public static readonly AMAZON_LINUX_2_CORETTO_8 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:corretto8');
  /** The Amazon Coretto 11 image x86_64, based on Amazon Linux 2. */
  public static readonly AMAZON_LINUX_2_CORETTO_11 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux2-x86_64-standard:corretto11');
  /** The Amazon Coretto 8 image x86_64, based on Amazon Linux 2023. */
  public static readonly AMAZON_LINUX_2023_CORETTO_8 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux-x86_64-standard:corretto8');
  /** The Amazon Coretto 11 image x86_64, based on Amazon Linux 2023. */
  public static readonly AMAZON_LINUX_2023_CORETTO_11 = LinuxBuildImage.codeBuildImage('aws/codebuild/amazonlinux-x86_64-standard:corretto11');

  /**
   * Image "aws/codebuild/amazonlinux2-aarch64-standard:1.0".
   * @see {LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_1_0}
   *
   * @deprecated Use {@link LinuxArmBuildImage.AMAZON_LINUX_2_ARM_3} instead.
   **/
  public static readonly AMAZON_LINUX_2_ARM = LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_1_0;
  /**
   * Image "aws/codebuild/amazonlinux2-aarch64-standard:2.0".
   * @see {LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_2_0}
   * */
  public static readonly AMAZON_LINUX_2_ARM_2 = LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_2_0;
  /**
   * Image "aws/codebuild/amazonlinux2-aarch64-standard:3.0".
   * @see {LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0}
   * */
  public static readonly AMAZON_LINUX_2_ARM_3 = LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0;

  /**
   * @returns a x86-64 Linux build image from a Docker Hub image.
   */
  public static fromDockerRegistry(name: string, options: DockerImageOptions = {}): IBuildImage {
    return new LinuxBuildImage({
      ...options,
      imageId: name,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
    });
  }

  /**
   * @returns A x86-64 Linux build image from an ECR repository.
   *
   * NOTE: if the repository is external (i.e. imported), then we won't be able to add
   * a resource policy statement for it so CodeBuild can pull the image.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-ecr.html
   *
   * @param repository The ECR repository
   * @param tagOrDigest Image tag or digest (default "latest", digests must start with `sha256:`)
   */
  public static fromEcrRepository(repository: ecr.IRepository, tagOrDigest: string = 'latest'): IBuildImage {
    return new LinuxBuildImage({
      imageId: repository.repositoryUriForTagOrDigest(tagOrDigest),
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      repository,
    });
  }

  /**
   * Uses an Docker image asset as a x86-64 Linux build image.
   */
  public static fromAsset(scope: Construct, id: string, props: DockerImageAssetProps): IBuildImage {
    const asset = new DockerImageAsset(scope, id, props);
    return new LinuxBuildImage({
      imageId: asset.imageUri,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      repository: asset.repository,
    });
  }

  /**
   * Uses a Docker image provided by CodeBuild.
   *
   * @returns A Docker image provided by CodeBuild.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
   *
   * @param id The image identifier
   * @example 'aws/codebuild/standard:4.0'
   */
  public static fromCodeBuildImageId(id: string): IBuildImage {
    return LinuxBuildImage.codeBuildImage(id);
  }

  private static codeBuildImage(name: string): IBuildImage {
    return new LinuxBuildImage({
      imageId: name,
      imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
    });
  }

  public readonly type = EnvironmentType.LINUX_CONTAINER as string;
  public readonly defaultComputeType = ComputeType.SMALL;
  public readonly imageId: string;
  public readonly imagePullPrincipalType?: ImagePullPrincipalType;
  public readonly secretsManagerCredentials?: secretsmanager.ISecret;
  public readonly repository?: ecr.IRepository;

  private constructor(props: LinuxBuildImageProps) {
    this.imageId = props.imageId;
    this.imagePullPrincipalType = props.imagePullPrincipalType;
    this.secretsManagerCredentials = props.secretsManagerCredentials;
    this.repository = props.repository;
  }

  public validate(env: BuildEnvironment): string[] {
    const errors = [];

    if (env.computeType && isLambdaComputeType(env.computeType)) {
      errors.push('x86-64 images do not support Lambda compute types');
    }

    return errors;
  }

  public runScriptBuildspec(entrypoint: string): BuildSpec {
    return runScriptLinuxBuildSpec(entrypoint);
  }
}

/**
 * Environment type for Windows Docker images
 */
export enum WindowsImageType {
  /**
   * The standard environment type, WINDOWS_CONTAINER
   */
  STANDARD = 'WINDOWS_CONTAINER',

  /**
   * The WINDOWS_SERVER_2019_CONTAINER environment type
   */
  SERVER_2019 = EnvironmentType.WINDOWS_SERVER_2019_CONTAINER,

  /**
   * The WINDOWS_SERVER_2022_CONTAINER environment type
   *
   * Notice: Cannot be used with on-demand compute, only with a {@link BuildEnvironment.fleet}.
   *
   * @see https://github.com/aws/aws-cdk/issues/29617
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/fleets.html
   */
  SERVER_2022 = EnvironmentType.WINDOWS_SERVER_2022_CONTAINER,
}

/**
 * Construction properties of `WindowsBuildImage`.
 * Module-private, as the constructor of `WindowsBuildImage` is private.
 */
interface WindowsBuildImageProps {
  readonly imageId: string;
  readonly imagePullPrincipalType?: ImagePullPrincipalType;
  readonly secretsManagerCredentials?: secretsmanager.ISecret;
  readonly repository?: ecr.IRepository;
  readonly imageType?: WindowsImageType;
}

/**
 * A CodeBuild image running Windows.
 *
 * This class has a bunch of public constants that represent the most popular images.
 *
 * You can also specify a custom image using one of the static methods:
 *
 * - WindowsBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }, imageType])
 * - WindowsBuildImage.fromEcrRepository(repo[, tag, imageType])
 * - WindowsBuildImage.fromAsset(parent, id, props, [, imageType])
 *
 * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
 */
export class WindowsBuildImage implements IBuildImage {
  /**
   * The standard CodeBuild image `aws/codebuild/windows-base:2.0`, which is
   * based off Windows Server Core 2016.
   *
   * @deprecated {@link WindowsBuildImage.WIN_SERVER_CORE_2019_BASE_3_0} should be used instead.
   */
  public static readonly WINDOWS_BASE_2_0: IBuildImage = new WindowsBuildImage({
    imageId: 'aws/codebuild/windows-base:2.0',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
  });

  /**
   * The standard CodeBuild image `aws/codebuild/windows-base:2019-1.0`, which is
   * based off Windows Server Core 2019.
   */
  public static readonly WIN_SERVER_CORE_2019_BASE: IBuildImage = new WindowsBuildImage({
    imageId: 'aws/codebuild/windows-base:2019-1.0',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
    imageType: WindowsImageType.SERVER_2019,
  });

  /**
   * The standard CodeBuild image `aws/codebuild/windows-base:2019-2.0`, which is
   * based off Windows Server Core 2019.
   */
  public static readonly WIN_SERVER_CORE_2019_BASE_2_0: IBuildImage = new WindowsBuildImage({
    imageId: 'aws/codebuild/windows-base:2019-2.0',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
    imageType: WindowsImageType.SERVER_2019,
  });

  /**
   * The standard CodeBuild image `aws/codebuild/windows-base:2019-3.0`, which is
   * based off Windows Server Core 2019.
   */
  public static readonly WIN_SERVER_CORE_2019_BASE_3_0: IBuildImage = new WindowsBuildImage({
    imageId: 'aws/codebuild/windows-base:2019-3.0',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
    imageType: WindowsImageType.SERVER_2019,
  });

  /**
   * The standard CodeBuild image `aws/codebuild/windows-base:2022-1.0`, which is
   * based off Windows Server Core 2022.
   *
   * Notice: Cannot be used with on-demand compute, only with a {@link BuildEnvironment.fleet}.
   *
   * @see https://github.com/aws/aws-cdk/issues/29617
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/fleets.html
   */
  public static readonly WIN_SERVER_CORE_2022_BASE_3_0: IBuildImage = new WindowsBuildImage({
    imageId: 'aws/codebuild/windows-base:2022-1.0',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
    imageType: WindowsImageType.SERVER_2022,
  });

  /**
   * @returns a Windows build image from a Docker Hub image.
   */
  public static fromDockerRegistry(
    name: string,
    options: DockerImageOptions = {},
    imageType: WindowsImageType = WindowsImageType.STANDARD): IBuildImage {
    return new WindowsBuildImage({
      ...options,
      imageId: name,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      imageType,
    });
  }

  /**
   * @returns A Windows build image from an ECR repository.
   *
   * NOTE: if the repository is external (i.e. imported), then we won't be able to add
   * a resource policy statement for it so CodeBuild can pull the image.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/sample-ecr.html
   *
   * @param repository The ECR repository
   * @param tagOrDigest Image tag or digest (default "latest", digests must start with `sha256:`)
   */
  public static fromEcrRepository(
    repository: ecr.IRepository,
    tagOrDigest: string = 'latest',
    imageType: WindowsImageType = WindowsImageType.STANDARD): IBuildImage {
    return new WindowsBuildImage({
      imageId: repository.repositoryUriForTagOrDigest(tagOrDigest),
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      imageType,
      repository,
    });
  }

  /**
   * Uses an Docker image asset as a Windows build image.
   */
  public static fromAsset(
    scope: Construct,
    id: string,
    props: DockerImageAssetProps,
    imageType: WindowsImageType = WindowsImageType.STANDARD): IBuildImage {
    const asset = new DockerImageAsset(scope, id, props);
    return new WindowsBuildImage({
      imageId: asset.imageUri,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      imageType,
      repository: asset.repository,
    });
  }

  public readonly type: string;
  public readonly defaultComputeType = ComputeType.MEDIUM;
  public readonly imageId: string;
  public readonly imagePullPrincipalType?: ImagePullPrincipalType;
  public readonly secretsManagerCredentials?: secretsmanager.ISecret;
  public readonly repository?: ecr.IRepository;

  private constructor(props: WindowsBuildImageProps) {
    this.type = (props.imageType ?? WindowsImageType.STANDARD).toString();
    this.imageId = props.imageId;
    this.imagePullPrincipalType = props.imagePullPrincipalType;
    this.secretsManagerCredentials = props.secretsManagerCredentials;
    this.repository = props.repository;
  }

  public validate(buildEnvironment: BuildEnvironment): string[] {
    const errors: string[] = [];

    if (buildEnvironment.privileged) {
      errors.push('Windows images do not support privileged mode');
    }

    if (buildEnvironment.computeType && isLambdaComputeType(buildEnvironment.computeType)) {
      errors.push('Windows images do not support Lambda compute types');
    }

    const unsupportedComputeTypes = [ComputeType.SMALL, ComputeType.X_LARGE, ComputeType.X2_LARGE];
    if (buildEnvironment.computeType !== undefined && unsupportedComputeTypes.includes(buildEnvironment.computeType)) {
      errors.push(`Windows images do not support the '${buildEnvironment.computeType}' compute type`);
    }

    return errors;
  }

  public runScriptBuildspec(entrypoint: string): BuildSpec {
    return BuildSpec.fromObject({
      version: '0.2',
      phases: {
        pre_build: {
          // Would love to do downloading here and executing in the next step,
          // but I don't know how to propagate the value of $TEMPDIR.
          //
          // Punting for someone who knows PowerShell well enough.
          commands: [],
        },
        build: {
          commands: [
            'Set-Variable -Name TEMPDIR -Value (New-TemporaryFile).DirectoryName',
            `aws s3 cp s3://$env:${S3_BUCKET_ENV}/$env:${S3_KEY_ENV} $TEMPDIR\\scripts.zip`,
            'New-Item -ItemType Directory -Path $TEMPDIR\\scriptdir',
            'Expand-Archive -Path $TEMPDIR/scripts.zip -DestinationPath $TEMPDIR\\scriptdir',
            '$env:SCRIPT_DIR = "$TEMPDIR\\scriptdir"',
            `& $TEMPDIR\\scriptdir\\${entrypoint}`,
          ],
        },
      },
    });
  }
}

/**
 * Construction properties of `MacBuildImage`.
 * Module-private, as the constructor of `MacBuildImage` is private.
 */
interface MacBuildImageProps {
  readonly imageId: string;
  readonly imagePullPrincipalType?: ImagePullPrincipalType;
  readonly secretsManagerCredentials?: secretsmanager.ISecret;
  readonly repository?: ecr.IRepository;
}

/**
 * A CodeBuild image running ARM MacOS.
 *
 * This class has a bunch of public constants that represent the most popular images.
 *
 * You can also specify a custom image using one of the static methods:
 *
 * - MacBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }])
 * - MacBuildImage.fromEcrRepository(repo[, tag])
 * - MacBuildImage.fromAsset(parent, id, props)
 *
 *
 * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
 */
export class MacBuildImage implements IBuildImage {
  /**
   * Corresponds to the standard CodeBuild image `aws/codebuild/macos-arm-base:14`.
   */
  public static readonly BASE_14: IBuildImage = new MacBuildImage({
    imageId: 'aws/codebuild/macos-arm-base:14',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
  });

  /**
   * Corresponds to the CodeBuild image `aws/codebuild/macos-arm-base:15`.
   */
  public static readonly BASE_15: IBuildImage = new MacBuildImage({
    imageId: 'aws/codebuild/macos-arm-base:15',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
  });

  /**
   * Corresponds to the CodeBuild image `aws/codebuild/macos-arm-base:26`.
   */
  public static readonly BASE_26: IBuildImage = new MacBuildImage({
    imageId: 'aws/codebuild/macos-arm-base:26',
    imagePullPrincipalType: ImagePullPrincipalType.CODEBUILD,
  });

  /**
   * Makes an ARM MacOS build image from a Docker Hub image.
   */
  public static fromDockerRegistry(name: string, options: DockerImageOptions = {}): IBuildImage {
    return new MacBuildImage({
      ...options,
      imageId: name,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
    });
  }

  /**
   * Makes an ARM MacOS build image from an ECR repository.
   */
  public static fromEcrRepository(repository: ecr.IRepository, tagOrDigest: string = 'latest'): IBuildImage {
    return new MacBuildImage({
      imageId: repository.repositoryUriForTagOrDigest(tagOrDigest),
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      repository,
    });
  }

  /**
   * Uses an Docker image asset as a ARM MacOS build image.
   */
  public static fromAsset(scope: Construct, id: string, props: DockerImageAssetProps): IBuildImage {
    const asset = new DockerImageAsset(scope, id, props);
    return new MacBuildImage({
      imageId: asset.imageUri,
      imagePullPrincipalType: ImagePullPrincipalType.SERVICE_ROLE,
      repository: asset.repository,
    });
  }

  public readonly type = EnvironmentType.MAC_ARM as string;
  public readonly defaultComputeType = ComputeType.MEDIUM;
  public readonly imageId: string;
  public readonly imagePullPrincipalType?: ImagePullPrincipalType;
  public readonly secretsManagerCredentials?: secretsmanager.ISecret;
  public readonly repository?: ecr.IRepository;

  private constructor(props: MacBuildImageProps) {
    this.imageId = props.imageId;
    this.imagePullPrincipalType = props.imagePullPrincipalType;
    this.secretsManagerCredentials = props.secretsManagerCredentials;
    this.repository = props.repository;
  }

  public validate(buildEnvironment: BuildEnvironment): string[] {
    const errors: string[] = [];

    if (buildEnvironment.computeType && isLambdaComputeType(buildEnvironment.computeType)) {
      errors.push('Mac images do not support Lambda compute types');
    }

    if (!buildEnvironment.fleet) {
      errors.push('Mac images must be used with a fleet');
    }

    return errors;
  }

  public runScriptBuildspec(entrypoint: string): BuildSpec {
    // Reuse Linux BuildSpec, since it is compatible
    return runScriptLinuxBuildSpec(entrypoint);
  }
}

export interface BuildEnvironmentVariable {
  /**
   * The type of environment variable.
   * @default PlainText
   */
  readonly type?: BuildEnvironmentVariableType;

  /**
   * The value of the environment variable.
   * For plain-text variables (the default), this is the literal value of variable.
   * For SSM parameter variables, pass the name of the parameter here (`parameterName` property of `IParameter`).
   * For SecretsManager variables secrets, pass either the secret name (`secretName` property of `ISecret`)
   * or the secret ARN (`secretArn` property of `ISecret`) here,
   * along with optional SecretsManager qualifiers separated by ':', like the JSON key, or the version or stage
   * (see https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html#build-spec.env.secrets-manager for details).
   */
  readonly value: any;
}

export enum BuildEnvironmentVariableType {
  /**
   * An environment variable in plaintext format.
   */
  PLAINTEXT = 'PLAINTEXT',

  /**
   * An environment variable stored in Systems Manager Parameter Store.
   */
  PARAMETER_STORE = 'PARAMETER_STORE',

  /**
   * An environment variable stored in AWS Secrets Manager.
   */
  SECRETS_MANAGER = 'SECRETS_MANAGER',
}

/**
 * Specifies the visibility of the project's builds.
 */
export enum ProjectVisibility {
  /**
   * The project builds are visible to the public.
   */
  PUBLIC_READ = 'PUBLIC_READ',

  /**
   * The project builds are not visible to the public.
   */
  PRIVATE = 'PRIVATE',
}

/**
 * The list of event types for AWS Codebuild
 * @see https://docs.aws.amazon.com/dtconsole/latest/userguide/concepts.html#events-ref-buildproject
 */
export enum ProjectNotificationEvents {
  /**
   * Trigger notification when project build state failed
   */
  BUILD_FAILED = 'codebuild-project-build-state-failed',

  /**
   * Trigger notification when project build state succeeded
   */
  BUILD_SUCCEEDED = 'codebuild-project-build-state-succeeded',

  /**
   * Trigger notification when project build state in progress
   */
  BUILD_IN_PROGRESS = 'codebuild-project-build-state-in-progress',

  /**
   * Trigger notification when project build state stopped
   */
  BUILD_STOPPED = 'codebuild-project-build-state-stopped',

  /**
   * Trigger notification when project build phase failure
   */
  BUILD_PHASE_FAILED = 'codebuild-project-build-phase-failure',

  /**
   * Trigger notification when project build phase success
   */
  BUILD_PHASE_SUCCEEDED = 'codebuild-project-build-phase-success',
}

function isBindableBuildImage(x: unknown): x is IBindableBuildImage {
  return typeof x === 'object' && !!x && !!(x as any).bind;
}
