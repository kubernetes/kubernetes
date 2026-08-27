import type { Construct } from 'constructs';
import type { IAuroraClusterInstance, IClusterInstance } from './aurora-cluster-instance';
import { InstanceType } from './aurora-cluster-instance';
import type { ClusterEngineConfig, IClusterEngine } from './cluster-engine';
import type { DatabaseClusterAttributes, IDatabaseCluster } from './cluster-ref';
import { DatabaseInsightsMode } from './database-insights-mode';
import { Endpoint } from './endpoint';
import type { NetworkType } from './instance';
import type { IParameterGroup } from './parameter-group';
import { ParameterGroup } from './parameter-group';
import { DATA_API_ACTIONS } from './perms';
import { applyDefaultRotationOptions, defaultDeletionProtection, renderCredentials, setupS3ImportExport, helperRemovalPolicy, renderUnless, renderSnapshotCredentials, validateManagedPasswordCredentials, validateManagedPasswordSnapshotCredentials } from './private/util';
import type { BackupProps, Credentials, InstanceProps, RotationSingleUserOptions, RotationMultiUserOptions, SnapshotCredentials, EngineLifecycleSupport } from './props';
import { PerformanceInsightRetention } from './props';
import type { DatabaseProxyOptions } from './proxy';
import { DatabaseProxy, ProxyTarget } from './proxy';
import type { CfnDBClusterProps } from './rds.generated';
import { CfnDBCluster, CfnDBInstance } from './rds.generated';
import type { ISubnetGroup } from './subnet-group';
import { SubnetGroup } from './subnet-group';
import { validateDatabaseClusterProps } from './validate-database-insights';
import type * as cloudwatch from '../../aws-cloudwatch';
import * as ec2 from '../../aws-ec2';
import type { IRole } from '../../aws-iam';
import * as iam from '../../aws-iam';
import { ManagedPolicy, Role, ServicePrincipal } from '../../aws-iam';
import type * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import type * as s3 from '../../aws-s3';
import * as secretsmanager from '../../aws-secretsmanager';
import * as cxschema from '../../cloud-assembly-schema';
import type { Duration } from '../../core';
import { Annotations, ArnFormat, ContextProvider, FeatureFlags, Lazy, RemovalPolicy, Resource, Stack, Token, TokenComparison } from '../../core';
import { UnscopedValidationError, ValidationError } from '../../core/lib/errors';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import * as cxapi from '../../cx-api';
import type { aws_rds } from '../../interfaces';
import { toISubnetGroup } from './private/ref-utils';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Common properties for a new database cluster or cluster from snapshot.
 */
interface DatabaseClusterBaseProps {
  /**
   * What kind of database to start
   */
  readonly engine: IClusterEngine;

  /**
   * How many replicas/instances to create
   *
   * Has to be at least 1.
   *
   * @default 2
   * @deprecated - use writer and readers instead
   */
  readonly instances?: number;

  /**
   * Settings for the individual instances that are launched
   *
   * @deprecated - use writer and readers instead
   */
  readonly instanceProps?: InstanceProps;

  /**
   * The instance to use for the cluster writer
   *
   * @default - required if instanceProps is not provided
   */
  readonly writer?: IClusterInstance;

  /**
   * A list of instances to create as cluster reader instances
   *
   * @default - no readers are created. The cluster will have a single writer/reader
   */
  readonly readers?: IClusterInstance[];

  /**
   * The maximum number of Aurora capacity units (ACUs) for a DB instance in an Aurora Serverless v2 cluster.
   * You can specify ACU values in half-step increments, such as 40, 40.5, 41, and so on.
   * The largest value that you can use is 256.
   *
   * The maximum capacity must be higher than 0.5 ACUs.
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless-v2.setting-capacity.html#aurora-serverless-v2.max_capacity_considerations
   *
   * @default 2
   */
  readonly serverlessV2MaxCapacity?: number;

  /**
   * The minimum number of Aurora capacity units (ACUs) for a DB instance in an Aurora Serverless v2 cluster.
   * You can specify ACU values in half-step increments, such as 8, 8.5, 9, and so on.
   * The smallest value that you can use is 0.
   *
   * For Aurora versions that support the Aurora Serverless v2 auto-pause feature, the smallest value that you can use is 0.
   * For versions that don't support Aurora Serverless v2 auto-pause, the smallest value that you can use is 0.5.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless-v2.setting-capacity.html#aurora-serverless-v2.min_capacity_considerations
   *
   * @default 0.5
   */
  readonly serverlessV2MinCapacity?: number;

  /**
   * Specifies the duration an Aurora Serverless v2 DB instance must be idle before Aurora attempts to automatically pause it.
   *
   * The duration must be between 300 seconds (5 minutes) and 86,400 seconds (24 hours).
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless-v2-auto-pause.html
   *
   * @default - The default is 300 seconds (5 minutes).
   */
  readonly serverlessV2AutoPauseDuration?: Duration;

  /**
   * What subnets to run the RDS instances in.
   *
   * Must be at least 2 subnets in two different AZs.
   */
  readonly vpc?: ec2.IVpc;

  /**
   * Where to place the instances within the VPC
   *
   * @default - the Vpc default strategy if not specified.
   */
  readonly vpcSubnets?: ec2.SubnetSelection;

  /**
   * Security group.
   *
   * @default - a new security group is created.
   */
  readonly securityGroups?: ec2.ISecurityGroup[];

  /**
   * The ordering of updates for instances
   *
   * @default InstanceUpdateBehaviour.BULK
   */
  readonly instanceUpdateBehaviour?: InstanceUpdateBehaviour;

  /**
   * The number of seconds to set a cluster's target backtrack window to.
   * This feature is only supported by the Aurora MySQL database engine and
   * cannot be enabled on existing clusters.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraMySQL.Managing.Backtrack.html
   * @default 0 seconds (no backtrack)
   */
  readonly backtrackWindow?: Duration;

  /**
   * Backup settings
   *
   * @default - Backup retention period for automated backups is 1 day.
   * Backup preferred window is set to a 30-minute window selected at random from an
   * 8-hour block of time for each AWS Region, occurring on a random day of the week.
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html#USER_WorkingWithAutomatedBackups.BackupWindow
   */
  readonly backup?: BackupProps;

  /**
   * What port to listen on
   *
   * @default - The default for the engine is used.
   */
  readonly port?: number;

  /**
   * An optional identifier for the cluster
   *
   * @default - A name is automatically generated.
   */
  readonly clusterIdentifier?: string;

  /**
   * Base identifier for instances
   *
   * Every replica is named by appending the replica number to this string, 1-based.
   *
   * @default - clusterIdentifier is used with the word "Instance" appended.
   * If clusterIdentifier is not provided, the identifier is automatically generated.
   */
  readonly instanceIdentifierBase?: string;

  /**
   * Name of a database which is automatically created inside the cluster
   *
   * @default - Database is not created in cluster.
   */
  readonly defaultDatabaseName?: string;

  /**
   * Indicates whether the DB cluster should have deletion protection enabled.
   *
   * @default - true if `removalPolicy` is RETAIN, `undefined` otherwise, which will not enable deletion protection.
   * To disable deletion protection after it has been enabled, you must explicitly set this value to `false`.
   */
  readonly deletionProtection?: boolean;

  /**
   * A preferred maintenance window day/time range. Should be specified as a range ddd:hh24:mi-ddd:hh24:mi (24H Clock UTC).
   *
   * Example: 'Sun:23:45-Mon:00:15'
   *
   * @default - 30-minute window selected at random from an 8-hour block of time for
   * each AWS Region, occurring on a random day of the week.
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/USER_UpgradeDBInstance.Maintenance.html#Concepts.DBMaintenance
   */
  readonly preferredMaintenanceWindow?: string;

  /**
   * Additional parameters to pass to the database engine
   *
   * @default - No parameter group.
   */
  readonly parameterGroup?: IParameterGroup;

  /**
   * The parameters in the DBClusterParameterGroup to create automatically
   *
   * You can only specify parameterGroup or parameters but not both.
   * You need to use a versioned engine to auto-generate a DBClusterParameterGroup.
   *
   * @default - None
   */
  readonly parameters?: { [key: string]: string };

  /**
   * The removal policy to apply when the cluster and its instances are removed
   * from the stack or replaced during an update.
   *
   * @default - RemovalPolicy.SNAPSHOT (remove the cluster and instances, but retain a snapshot of the data)
   */
  readonly removalPolicy?: RemovalPolicy;

  /**
   * The list of log types that need to be enabled for exporting to
   * CloudWatch Logs.
   *
   * @default - no log exports
   */
  readonly cloudwatchLogsExports?: string[];

  /**
   * The number of days log events are kept in CloudWatch Logs. When updating
   * this property, unsetting it doesn't remove the log retention policy. To
   * remove the retention policy, set the value to `Infinity`.
   *
   * @default - logs never expire
   */
  readonly cloudwatchLogsRetention?: logs.RetentionDays;

  /**
   * The IAM role for the Lambda function associated with the custom resource
   * that sets the retention policy.
   *
   * @default - a new role is created.
   */
  readonly cloudwatchLogsRetentionRole?: IRole;

  /**
   * The interval between points when Amazon RDS collects enhanced monitoring metrics.
   *
   * If you enable `enableClusterLevelEnhancedMonitoring`, this property is applied to the cluster,
   * otherwise it is applied to the instances.
   *
   * @default - no enhanced monitoring
   */
  readonly monitoringInterval?: Duration;

  /**
   * Role that will be used to manage DB monitoring.
   *
   * If you enable `enableClusterLevelEnhancedMonitoring`, this property is applied to the cluster,
   * otherwise it is applied to the instances.
   *
   * @default - A role is automatically created for you
   */
  readonly monitoringRole?: IRole;

  /**
   * Whether to enable enhanced monitoring at the cluster level.
   *
   * If set to true, `monitoringInterval` and `monitoringRole` are applied to not the instances, but the cluster.
   * `monitoringInterval` is required to be set if `enableClusterLevelEnhancedMonitoring` is set to true.
   *
   * @default - When the `monitoringInterval` is set, enhanced monitoring is enabled for each instance.
   */
  readonly enableClusterLevelEnhancedMonitoring?: boolean;

  /**
   * Role that will be associated with this DB cluster to enable S3 import.
   * This feature is only supported by the Aurora database engine.
   *
   * This property must not be used if `s3ImportBuckets` is used.
   * To use this property with Aurora PostgreSQL, it must be configured with the S3 import feature enabled when creating the DatabaseClusterEngine
   * For MySQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraMySQL.Integrating.LoadFromS3.html
   *
   * For PostgreSQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.Migrating.html
   *
   * @default - New role is created if `s3ImportBuckets` is set, no role is defined otherwise
   */
  readonly s3ImportRole?: IRole;

  /**
   * S3 buckets that you want to load data from. This feature is only supported by the Aurora database engine.
   *
   * This property must not be used if `s3ImportRole` is used.
   *
   * For MySQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraMySQL.Integrating.LoadFromS3.html
   *
   * For PostgreSQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.Migrating.html
   *
   * @default - None
   */
  readonly s3ImportBuckets?: s3.IBucket[];

  /**
   * Role that will be associated with this DB cluster to enable S3 export.
   * This feature is only supported by the Aurora database engine.
   *
   * This property must not be used if `s3ExportBuckets` is used.
   * To use this property with Aurora PostgreSQL, it must be configured with the S3 export feature enabled when creating the DatabaseClusterEngine
   * For MySQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraMySQL.Integrating.SaveIntoS3.html
   *
   * For PostgreSQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/postgresql-s3-export.html
   *
   * @default - New role is created if `s3ExportBuckets` is set, no role is defined otherwise
   */
  readonly s3ExportRole?: IRole;

  /**
   * S3 buckets that you want to load data into. This feature is only supported by the Aurora database engine.
   *
   * This property must not be used if `s3ExportRole` is used.
   *
   * For MySQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraMySQL.Integrating.SaveIntoS3.html
   *
   * For PostgreSQL:
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/postgresql-s3-export.html
   *
   * @default - None
   */
  readonly s3ExportBuckets?: s3.IBucket[];

  /**
   * Existing subnet group for the cluster.
   *
   * @default - a new subnet group will be created.
   */
  readonly subnetGroup?: aws_rds.IDBSubnetGroupRef;

  /**
   * Whether to enable mapping of AWS Identity and Access Management (IAM) accounts
   * to database accounts.
   *
   * @default false
   */
  readonly iamAuthentication?: boolean;

  /**
   * Whether to enable storage encryption.
   *
   * @default - true if storageEncryptionKey is provided, false otherwise
   */
  readonly storageEncrypted?: boolean;

  /**
   * The KMS key for storage encryption.
   * If specified, `storageEncrypted` will be set to `true`.
   *
   * @default - if storageEncrypted is true then the default master key, no key otherwise
   */
  readonly storageEncryptionKey?: kms.IKeyRef;

  /**
   * The storage type to be associated with the DB cluster.
   *
   * @default - DBClusterStorageType.AURORA
   */
  readonly storageType?: DBClusterStorageType;

  /**
   * Whether to copy tags to the snapshot when a snapshot is created.
   *
   * @default - true
   */
  readonly copyTagsToSnapshot?: boolean;

  /**
   * The network type of the DB instance.
   *
   * @default - IPV4
   */
  readonly networkType?: NetworkType;

  /**
   * Directory ID for associating the DB cluster with a specific Active Directory.
   *
   * Necessary for enabling Kerberos authentication. If specified, the DB cluster joins the given Active Directory, enabling Kerberos authentication.
   * If not specified, the DB cluster will not be associated with any Active Directory, and Kerberos authentication will not be enabled.
   *
   * @default - DB cluster is not associated with an Active Directory; Kerberos authentication is not enabled.
   */
  readonly domain?: string;

  /**
   * The IAM role to be used when making API calls to the Directory Service. The role needs the AWS-managed policy
   * `AmazonRDSDirectoryServiceAccess` or equivalent.
   *
   * @default - If `DatabaseClusterBaseProps.domain` is specified, a role with the `AmazonRDSDirectoryServiceAccess` policy is automatically created.
   */
  readonly domainRole?: iam.IRole;

  /**
   * Whether to enable the Data API for the cluster.
   *
   * @default - false
   */
  readonly enableDataApi?: boolean;

  /**
   * Whether read replicas can forward write operations to the writer DB instance in the DB cluster.
   *
   * This setting can only be enabled for Aurora MySQL 3.04 or higher, and for Aurora PostgreSQL 16.4
   * or higher (for version 16), 15.8 or higher (for version 15), and 14.13 or higher (for version 14).
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-mysql-write-forwarding.html
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-postgresql-write-forwarding.html
   *
   * @default false
   */
  readonly enableLocalWriteForwarding?: boolean;

  /**
   * Whether to enable Performance Insights for the DB cluster.
   *
   * @default - false, unless `performanceInsightRetention` or `performanceInsightEncryptionKey` is set,
   * or `databaseInsightsMode` is set to `DatabaseInsightsMode.ADVANCED`.
   */
  readonly enablePerformanceInsights?: boolean;

  /**
   * The amount of time, in days, to retain Performance Insights data.
   *
   * If you set `databaseInsightsMode` to `DatabaseInsightsMode.ADVANCED`, you must set this property to `PerformanceInsightRetention.MONTHS_15`.
   *
   * @default - 7
   */
  readonly performanceInsightRetention?: PerformanceInsightRetention;

  /**
   * The AWS KMS key for encryption of Performance Insights data.
   *
   * @default - default master key
   */
  readonly performanceInsightEncryptionKey?: kms.IKey;

  /**
   * The database insights mode.
   *
   * @default - DatabaseInsightsMode.STANDARD when performance insights are enabled and Amazon Aurora engine is used, otherwise not set.
   */
  readonly databaseInsightsMode?: DatabaseInsightsMode;

  /**
   * Specifies whether minor engine upgrades are applied automatically to the DB cluster during the maintenance window.
   *
   * @default true
   */
  readonly autoMinorVersionUpgrade?: boolean;

  /**
   * Specifies the scalability mode of the Aurora DB cluster.
   *
   * Set LIMITLESS if you want to use a limitless database; otherwise, set it to STANDARD.
   *
   * @default ClusterScalabilityType.STANDARD
   */
  readonly clusterScalabilityType?: ClusterScalabilityType;

  /**
   * [Misspelled] Specifies the scalability mode of the Aurora DB cluster.
   *
   * Set LIMITLESS if you want to use a limitless database; otherwise, set it to STANDARD.
   *
   * @default ClusterScailabilityType.STANDARD
   * @deprecated Use clusterScalabilityType instead. This will be removed in the next major version.
   */
  readonly clusterScailabilityType?: ClusterScailabilityType;

  /**
   * The life cycle type for this DB cluster.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/extended-support.html
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/extended-support.html
   *
   * @default undefined - AWS RDS default setting is `EngineLifecycleSupport.OPEN_SOURCE_RDS_EXTENDED_SUPPORT`
   */
  readonly engineLifecycleSupport?: EngineLifecycleSupport;

  /**
   * Specifies whether to remove automated backups immediately after the DB cluster is deleted.
   *
   * @default undefined - AWS RDS default is to remove automated backups immediately after the DB cluster is deleted, unless the AWS Backup policy specifies a point-in-time restore rule.
   */
  readonly deleteAutomatedBackups?: boolean;
}

/**
 * The storage type to be associated with the DB cluster.
 */
export enum DBClusterStorageType {
  /**
   * Storage type for Aurora DB standard clusters.
   */
  AURORA = 'aurora',

  /**
   * Storage type for Aurora DB I/O-Optimized clusters.
   */
  AURORA_IOPT1 = 'aurora-iopt1',
}

/**
 * The orchestration of updates of multiple instances
 */
export enum InstanceUpdateBehaviour {
  /**
   * In a bulk update, all instances of the cluster are updated at the same time.
   * This results in a faster update procedure.
   * During the update, however, all instances might be unavailable at the same time and thus a downtime might occur.
   */
  BULK = 'BULK',

  /**
   * In a rolling update, one instance after another is updated.
   * This results in at most one instance being unavailable during the update.
   * If your cluster consists of more than 1 instance, the downtime periods are limited to the time a primary switch needs.
   */
  ROLLING = 'ROLLING',
}

/**
 * The scalability mode of the Aurora DB cluster.
 */
export enum ClusterScalabilityType {
  /**
   * The cluster uses normal DB instance creation.
   */
  STANDARD = 'standard',

  /**
   * The cluster operates as an Aurora Limitless Database,
   * allowing you to create a DB shard group for horizontal scaling (sharding) capabilities.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/limitless.html
   */
  LIMITLESS = 'limitless',
}

/**
 * The scalability mode of the Aurora DB cluster.
 * @deprecated Use ClusterScalabilityType instead. This will be removed in the next major version.
 */
export enum ClusterScailabilityType {
  /**
   * The cluster uses normal DB instance creation.
   */
  STANDARD = 'standard',

  /**
   * The cluster operates as an Aurora Limitless Database,
   * allowing you to create a DB shard group for horizontal scaling (sharding) capabilities.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/limitless.html
   */
  LIMITLESS = 'limitless',
}

/**
 * Properties for looking up an existing DatabaseCluster.
 */
export interface DatabaseClusterLookupOptions {
  /**
   * The cluster identifier of the DatabaseCluster
   */
  readonly clusterIdentifier: string;
}

/**
 * A new or imported clustered database.
 */
export abstract class DatabaseClusterBase extends Resource implements IDatabaseCluster {
  // only required because of JSII bug: https://github.com/aws/jsii/issues/2040
  public abstract readonly engine?: IClusterEngine;

  /**
   * Identifier of the cluster
   */
  public abstract readonly clusterIdentifier: string;

  /**
   * The immutable identifier for the cluster; for example: cluster-ABCD1234EFGH5678IJKL90MNOP.
   *
   * This AWS Region-unique identifier is used in things like IAM authentication policies.
   */
  public abstract readonly clusterResourceIdentifier: string;

  /**
   * Identifiers of the replicas
   */
  public abstract readonly instanceIdentifiers: string[];

  /**
   * The endpoint to use for read/write operations
   */
  public abstract readonly clusterEndpoint: Endpoint;

  /**
   * Endpoint to use for load-balanced read-only operations.
   */
  public abstract readonly clusterReadEndpoint: Endpoint;

  /**
   * Endpoints which address each individual replica.
   */
  public abstract readonly instanceEndpoints: Endpoint[];

  /**
   * Access to the network connections
   */
  public abstract readonly connections: ec2.Connections;

  /**
   * The secret attached to this cluster
   */
  public abstract readonly secret?: secretsmanager.ISecret;

  protected abstract enableDataApi?: boolean;

  /**
   * The ARN of the cluster
   */
  public get clusterArn(): string {
    return Stack.of(this).formatArn({
      service: 'rds',
      resource: 'cluster',
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
      resourceName: this.clusterIdentifier,
    });
  }

  /**
   * A reference to this database cluster
   */
  public get dbClusterRef(): aws_rds.DBClusterReference {
    return {
      dbClusterIdentifier: this.clusterIdentifier,
      dbClusterArn: this.clusterArn,
    };
  }

  /**
   * Add a new db proxy to this cluster.
   */
  public addProxy(id: string, options: DatabaseProxyOptions): DatabaseProxy {
    return new DatabaseProxy(this, id, {
      proxyTarget: ProxyTarget.fromCluster(this),
      ...options,
    });
  }

  /**
   * Renders the secret attachment target specifications.
   */
  public asSecretAttachmentTarget(): secretsmanager.SecretAttachmentTargetProps {
    return {
      targetId: this.clusterIdentifier,
      targetType: secretsmanager.AttachmentTargetType.RDS_DB_CLUSTER,
    };
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantConnect(grantee: iam.IGrantable, dbUser: string): iam.Grant {
    return iam.Grant.addToPrincipal({
      actions: ['rds-db:connect'],
      grantee,
      resourceArns: [Stack.of(this).formatArn({
        service: 'rds-db',
        resource: 'dbuser',
        resourceName: `${this.clusterResourceIdentifier}/${dbUser}`,
        arnFormat: ArnFormat.COLON_RESOURCE_NAME,
      })],
    });
  }

  /**
   * Grant the given identity to access the Data API.
   *
   * [disable-awslint:no-grants]
   */
  public grantDataApiAccess(grantee: iam.IGrantable): iam.Grant {
    if (this.enableDataApi === false) {
      throw new ValidationError(lit`CannotGrantDataApiAccess`, 'Cannot grant Data API access when the Data API is disabled', this);
    }

    this.enableDataApi = true;
    const ret = iam.Grant.addToPrincipal({
      grantee,
      actions: DATA_API_ACTIONS,
      resourceArns: [this.clusterArn],
    });
    this.secret?.grantRead(grantee);
    return ret;
  }
}

/**
 * Abstract base for ``DatabaseCluster`` and ``DatabaseClusterFromSnapshot``
 */
abstract class DatabaseClusterNew extends DatabaseClusterBase {
  /**
   * The engine for this Cluster.
   * Never undefined.
   */
  public readonly engine?: IClusterEngine;

  protected readonly newCfnProps: CfnDBClusterProps;
  protected readonly securityGroups: ec2.ISecurityGroup[];
  protected readonly subnetGroupRef: aws_rds.IDBSubnetGroupRef;

  private readonly domainId?: string;
  private readonly domainRole?: iam.IRole;

  /**
   * Secret in SecretsManager to store the database cluster user credentials.
   */
  public abstract readonly secret?: secretsmanager.ISecret;

  /**
   * The VPC network to place the cluster in.
   */
  public readonly vpc: ec2.IVpc;

  /**
   * The cluster's subnets.
   */
  public readonly vpcSubnets?: ec2.SubnetSelection;

  /**
   * The log group is created when `cloudwatchLogsExports` is set.
   *
   * Each export value will create a separate log group.
   */
  public readonly cloudwatchLogGroups: {[engine: string]: logs.ILogGroup};

  /**
   * Application for single user rotation of the master password to this cluster.
   */
  public readonly singleUserRotationApplication: secretsmanager.SecretRotationApplication;

  /**
   * Application for multi user rotation to this cluster.
   */
  public readonly multiUserRotationApplication: secretsmanager.SecretRotationApplication;

  /**
   * Whether Performance Insights is enabled at cluster level.
   */
  public readonly performanceInsightsEnabled: boolean;

  /**
   * The amount of time, in days, to retain Performance Insights data.
   */
  public readonly performanceInsightRetention?: PerformanceInsightRetention;

  /**
   * The AWS KMS key for encryption of Performance Insights data.
   */
  public readonly performanceInsightEncryptionKey?: kms.IKey;

  /**
   * The database insights mode.
   */
  public readonly databaseInsightsMode?: DatabaseInsightsMode;

  /**
   * The IAM role for the enhanced monitoring.
   */
  public readonly monitoringRole?: iam.IRole;

  protected readonly serverlessV2MinCapacity: number;
  protected readonly serverlessV2MaxCapacity: number;
  protected readonly serverlessV2AutoPauseDuration?: Duration;

  protected hasServerlessInstance?: boolean;
  protected enableDataApi?: boolean;
  protected manageMasterUserPassword?: boolean;

  constructor(scope: Construct, id: string, props: DatabaseClusterBaseProps) {
    super(scope, id);

    if (props.clusterScalabilityType !== undefined && props.clusterScailabilityType !== undefined) {
      throw new ValidationError(lit`CannotSpecifyBothClusterScalabilityTypes`, 'You cannot specify both clusterScalabilityType and clusterScailabilityType (deprecated). Use clusterScalabilityType.', this);
    }

    if ((props.vpc && props.instanceProps?.vpc) || (!props.vpc && !props.instanceProps?.vpc)) {
      throw new ValidationError(lit`ProvideEitherVpcOrInstanceProps`, 'Provide either vpc or instanceProps.vpc, but not both', this);
    }
    if ((props.vpcSubnets && props.instanceProps?.vpcSubnets)) {
      throw new ValidationError(lit`ProvideEitherVpcSubnetsOrInstanceProps`, 'Provide either vpcSubnets or instanceProps.vpcSubnets, but not both', this);
    }
    this.vpc = props.instanceProps?.vpc ?? props.vpc!;
    this.vpcSubnets = props.instanceProps?.vpcSubnets ?? props.vpcSubnets;

    this.cloudwatchLogGroups = {};

    this.singleUserRotationApplication = props.engine.singleUserRotationApplication;
    this.multiUserRotationApplication = props.engine.multiUserRotationApplication;

    this.serverlessV2MaxCapacity = props.serverlessV2MaxCapacity ?? 2;
    this.serverlessV2MinCapacity = props.serverlessV2MinCapacity ?? 0.5;
    this.serverlessV2AutoPauseDuration = props.serverlessV2AutoPauseDuration;

    this.enableDataApi = props.enableDataApi;

    const { subnetIds } = this.vpc.selectSubnets(this.vpcSubnets);

    // Cannot test whether the subnets are in different AZs, but at least we can test the amount.
    if (subnetIds.length < 2) {
      Annotations.of(this)._addTrackableError(lit`InsufficientSubnets`, `Cluster requires at least 2 subnets, got ${subnetIds.length}`);
    }

    this.subnetGroupRef = props.subnetGroup ?? new SubnetGroup(this, 'Subnets', {
      description: `Subnets for ${id} database`,
      vpc: this.vpc,
      vpcSubnets: this.vpcSubnets,
      removalPolicy: renderUnless(helperRemovalPolicy(props.removalPolicy), RemovalPolicy.DESTROY),
    });

    this.securityGroups = props.instanceProps?.securityGroups ?? props.securityGroups ?? [
      new ec2.SecurityGroup(this, 'SecurityGroup', {
        description: 'RDS security group',
        vpc: this.vpc,
      }),
    ];

    const combineRoles = props.engine.combineImportAndExportRoles ?? false;
    let { s3ImportRole, s3ExportRole } = setupS3ImportExport(this, props, combineRoles);

    if (props.parameterGroup && props.parameters) {
      throw new ValidationError(lit`CannotSpecifyBothParameterGroupAndParameters`, 'You cannot specify both parameterGroup and parameters', this);
    }
    const parameterGroup = props.parameterGroup ?? (
      props.parameters
        ? new ParameterGroup(this, 'ParameterGroup', {
          engine: props.engine,
          parameters: props.parameters,
        })
        : undefined
    );
    // bind the engine to the Cluster
    const clusterEngineBindConfig = props.engine.bindToCluster(this, {
      s3ImportRole,
      s3ExportRole,
      parameterGroup,
    });

    const clusterAssociatedRoles: CfnDBCluster.DBClusterRoleProperty[] = [];
    if (s3ImportRole) {
      clusterAssociatedRoles.push({ roleArn: s3ImportRole.roleArn, featureName: clusterEngineBindConfig.features?.s3Import });
    }
    if (s3ExportRole &&
        // only add the second associated Role if it's different than the first
        // (duplicates in the associated Roles array are not allowed by the RDS service)
        (s3ExportRole !== s3ImportRole ||
        clusterEngineBindConfig.features?.s3Import !== clusterEngineBindConfig.features?.s3Export)) {
      clusterAssociatedRoles.push({ roleArn: s3ExportRole.roleArn, featureName: clusterEngineBindConfig.features?.s3Export });
    }

    const clusterParameterGroup = props.parameterGroup ?? clusterEngineBindConfig.parameterGroup;
    const clusterParameterGroupConfig = clusterParameterGroup?.bindToCluster({});
    this.engine = props.engine;

    const clusterIdentifier = FeatureFlags.of(this).isEnabled(cxapi.RDS_LOWERCASE_DB_IDENTIFIER) && !Token.isUnresolved(props.clusterIdentifier)
      ? props.clusterIdentifier?.toLowerCase()
      : props.clusterIdentifier;

    if (props.domain) {
      this.domainId = props.domain;
      this.domainRole = props.domainRole ?? new iam.Role(this, 'RDSClusterDirectoryServiceRole', {
        assumedBy: new iam.CompositePrincipal(
          new iam.ServicePrincipal('rds.amazonaws.com'),
          new iam.ServicePrincipal('directoryservice.rds.amazonaws.com'),
        ),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonRDSDirectoryServiceAccess'),
        ],
      });
    }

    validateDatabaseClusterProps(this, props);
    this.validateServerlessScalingConfig(clusterEngineBindConfig);

    const enablePerformanceInsights = props.enablePerformanceInsights
      || props.performanceInsightRetention !== undefined
      || props.performanceInsightEncryptionKey !== undefined
      || props.databaseInsightsMode === DatabaseInsightsMode.ADVANCED;
    this.performanceInsightsEnabled = enablePerformanceInsights;
    this.performanceInsightRetention = enablePerformanceInsights
      ? (props.performanceInsightRetention || PerformanceInsightRetention.DEFAULT)
      : undefined;
    this.performanceInsightEncryptionKey = props.performanceInsightEncryptionKey;
    this.databaseInsightsMode = props.databaseInsightsMode;

    // configure enhanced monitoring role for the cluster or instance
    this.monitoringRole = props.monitoringRole;
    if (!props.monitoringRole && props.monitoringInterval && props.monitoringInterval.toSeconds()) {
      this.monitoringRole = new Role(this, 'MonitoringRole', {
        assumedBy: new ServicePrincipal('monitoring.rds.amazonaws.com'),
        managedPolicies: [
          ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonRDSEnhancedMonitoringRole'),
        ],
      });
    }

    if (props.enableClusterLevelEnhancedMonitoring && !props.monitoringInterval) {
      throw new ValidationError(lit`MonitoringIntervalRequired`, '`monitoringInterval` must be set when `enableClusterLevelEnhancedMonitoring` is true.', this);
    }
    if (
      props.monitoringInterval && !props.monitoringInterval.isUnresolved() &&
      [0, 1, 5, 10, 15, 30, 60].indexOf(props.monitoringInterval.toSeconds()) === -1
    ) {
      throw new ValidationError(lit`InvalidMonitoringInterval`, `'monitoringInterval' must be one of 0, 1, 5, 10, 15, 30, or 60 seconds, got: ${props.monitoringInterval.toSeconds()} seconds.`, this);
    }

    this.newCfnProps = {
      // Basic
      engine: props.engine.engineType,
      engineVersion: props.engine.engineVersion?.fullVersion,
      dbClusterIdentifier: clusterIdentifier,
      dbSubnetGroupName: this.subnetGroupRef.dbSubnetGroupRef.dbSubnetGroupName,
      vpcSecurityGroupIds: this.securityGroups.map(sg => sg.securityGroupId),
      port: props.port ?? clusterEngineBindConfig.port,
      dbClusterParameterGroupName: clusterParameterGroupConfig?.parameterGroupName,
      associatedRoles: clusterAssociatedRoles.length > 0 ? clusterAssociatedRoles : undefined,
      deletionProtection: defaultDeletionProtection(props.deletionProtection, props.removalPolicy),
      enableIamDatabaseAuthentication: props.iamAuthentication,
      enableHttpEndpoint: Lazy.any({ produce: () => this.enableDataApi }),
      networkType: props.networkType,
      serverlessV2ScalingConfiguration: Lazy.any({
        produce: () => {
          if (this.hasServerlessInstance) {
            return {
              minCapacity: this.serverlessV2MinCapacity,
              maxCapacity: this.serverlessV2MaxCapacity,
              secondsUntilAutoPause: this.serverlessV2AutoPauseDuration?.toSeconds(),
            } satisfies CfnDBCluster.ServerlessV2ScalingConfigurationProperty;
          }
          return undefined;
        },
      }),
      storageType: props.storageType?.toString(),
      enableLocalWriteForwarding: props.enableLocalWriteForwarding,
      clusterScalabilityType: props.clusterScalabilityType ?? props.clusterScailabilityType,
      // Admin
      backtrackWindow: props.backtrackWindow?.toSeconds(),
      backupRetentionPeriod: props.backup?.retention?.toDays(),
      preferredBackupWindow: props.backup?.preferredWindow,
      preferredMaintenanceWindow: props.preferredMaintenanceWindow,
      databaseName: props.defaultDatabaseName,
      enableCloudwatchLogsExports: props.cloudwatchLogsExports,
      // Encryption
      kmsKeyId: props.storageEncryptionKey?.keyRef.keyArn,
      storageEncrypted: props.storageEncryptionKey ? true : props.storageEncrypted,
      // Tags
      copyTagsToSnapshot: props.copyTagsToSnapshot ?? true,
      domain: this.domainId,
      domainIamRoleName: this.domainRole?.roleName,
      performanceInsightsEnabled: this.performanceInsightsEnabled || props.enablePerformanceInsights, // fall back to undefined if not set
      performanceInsightsKmsKeyId: this.performanceInsightEncryptionKey?.keyArn,
      performanceInsightsRetentionPeriod: this.performanceInsightRetention,
      databaseInsightsMode: this.databaseInsightsMode,
      autoMinorVersionUpgrade: props.autoMinorVersionUpgrade,
      monitoringInterval: props.enableClusterLevelEnhancedMonitoring ? props.monitoringInterval?.toSeconds() : undefined,
      monitoringRoleArn: props.enableClusterLevelEnhancedMonitoring ? this.monitoringRole?.roleArn : undefined,
      engineLifecycleSupport: props.engineLifecycleSupport,
      deleteAutomatedBackups: props.deleteAutomatedBackups,
    };
  }

  protected get subnetGroup(): ISubnetGroup {
    return toISubnetGroup(this.subnetGroupRef);
  }

  /**
   * Create cluster instances
   *
   * @internal
   */
  protected _createInstances(props: DatabaseClusterProps): InstanceConfig {
    const instanceEndpoints: Endpoint[] = [];
    const instanceIdentifiers: string[] = [];
    const readers: IAuroraClusterInstance[] = [];

    // need to create the writer first since writer is determined by what instance is first
    const writer = props.writer!.bind(this, this, {
      // When `enableClusterLevelEnhancedMonitoring` is enabled,
      // both `monitoringInterval` and `monitoringRole` are set at cluster level so no need to re-set it in instance level.
      monitoringInterval: props.enableClusterLevelEnhancedMonitoring ? undefined : props.monitoringInterval,
      monitoringRole: props.enableClusterLevelEnhancedMonitoring ? undefined : this.monitoringRole,
      removalPolicy: props.removalPolicy ?? RemovalPolicy.SNAPSHOT,
      subnetGroup: this.subnetGroupRef,
      promotionTier: 0, // override the promotion tier so that writers are always 0
    });
    instanceIdentifiers.push(writer.instanceIdentifier);
    instanceEndpoints.push(new Endpoint(writer.dbInstanceEndpointAddress, this.clusterEndpoint.port));

    (props.readers ?? []).forEach(instance => {
      const clusterInstance = instance.bind(this, this, {
      // When `enableClusterLevelEnhancedMonitoring` is enabled,
      // both `monitoringInterval` and `monitoringRole` are set at cluster level so no need to re-set it in instance level.
        monitoringInterval: props.enableClusterLevelEnhancedMonitoring ? undefined : props.monitoringInterval,
        monitoringRole: props.enableClusterLevelEnhancedMonitoring ? undefined : this.monitoringRole,
        removalPolicy: props.removalPolicy ?? RemovalPolicy.SNAPSHOT,
        subnetGroup: this.subnetGroupRef,
      });
      readers.push(clusterInstance);
      // this makes sure the readers would always be created after the writer
      clusterInstance.node.addDependency(writer);

      if (clusterInstance.tier < 2) {
        this.validateReaderInstance(writer, clusterInstance);
      }
      instanceEndpoints.push(new Endpoint(clusterInstance.dbInstanceEndpointAddress, this.clusterEndpoint.port));
      instanceIdentifiers.push(clusterInstance.instanceIdentifier);
    });
    this.validateClusterInstances(writer, readers);

    return {
      instanceEndpoints,
      instanceIdentifiers,
    };
  }

  /**
   * Perform validations on the cluster instances
   */
  private validateClusterInstances(writer: IAuroraClusterInstance, readers: IAuroraClusterInstance[]): void {
    if (writer.type === InstanceType.SERVERLESS_V2) {
      this.hasServerlessInstance = true;
    }
    validatePerformanceInsightsSettings(this, {
      nodeId: writer.node.id,
      performanceInsightsEnabled: writer.performanceInsightsEnabled,
      performanceInsightRetention: writer.performanceInsightRetention,
      performanceInsightEncryptionKey: writer.performanceInsightEncryptionKey,
    });

    if (readers.length > 0) {
      const sortedReaders = readers.sort((a, b) => a.tier - b.tier);
      const highestTierReaders: IAuroraClusterInstance[] = [];
      const highestTier = sortedReaders[0].tier;
      let hasProvisionedReader = false;
      let noFailoverTierInstances = true;
      let serverlessInHighestTier = false;
      let hasServerlessReader = false;
      const someProvisionedReadersDontMatchWriter: IAuroraClusterInstance[] = [];
      for (const reader of sortedReaders) {
        if (reader.type === InstanceType.SERVERLESS_V2) {
          hasServerlessReader = true;
          this.hasServerlessInstance = true;
        } else {
          hasProvisionedReader = true;
          if (reader.instanceSize !== writer.instanceSize) {
            someProvisionedReadersDontMatchWriter.push(reader);
          }
        }
        if (reader.tier === highestTier) {
          if (reader.type === InstanceType.SERVERLESS_V2) {
            serverlessInHighestTier = true;
          }
          highestTierReaders.push(reader);
        }
        if (reader.tier <= 1) {
          noFailoverTierInstances = false;
        }
        validatePerformanceInsightsSettings(this, {
          nodeId: reader.node.id,
          performanceInsightsEnabled: reader.performanceInsightsEnabled,
          performanceInsightRetention: reader.performanceInsightRetention,
          performanceInsightEncryptionKey: reader.performanceInsightEncryptionKey,
        });
      }

      const hasOnlyServerlessReaders = hasServerlessReader && !hasProvisionedReader;
      if (hasOnlyServerlessReaders) {
        if (noFailoverTierInstances) {
          Annotations.of(this).addWarningV2(
            '@aws-cdk/aws-rds:noFailoverServerlessReaders',
            `Cluster ${this.node.id} only has serverless readers and no reader is in promotion tier 0-1.`+
            'Serverless readers in promotion tiers >= 2 will NOT scale with the writer, which can lead to '+
            'availability issues if a failover event occurs. It is recommended that at least one reader '+
            'has `scaleWithWriter` set to true',
          );
        }
      } else {
        if (serverlessInHighestTier && highestTier > 1) {
          Annotations.of(this).addWarningV2(
            '@aws-cdk/aws-rds:serverlessInHighestTier2-15',
            `There are serverlessV2 readers in tier ${highestTier}. Since there are no instances in a higher tier, `+
            'any instance in this tier is a failover target. Since this tier is > 1 the serverless reader will not scale '+
            'with the writer which could lead to availability issues during failover.',
          );
        }
        if (someProvisionedReadersDontMatchWriter.length > 0 && writer.type === InstanceType.PROVISIONED) {
          Annotations.of(this).addWarningV2(
            '@aws-cdk/aws-rds:provisionedReadersDontMatchWriter',
            `There are provisioned readers in the highest promotion tier ${highestTier} that do not have the same `+
            'InstanceSize as the writer. Any of these instances could be chosen as the new writer in the event '+
            'of a failover.\n'+
            `Writer InstanceSize: ${writer.instanceSize}\n`+
            `Reader InstanceSizes: ${someProvisionedReadersDontMatchWriter.map(reader => reader.instanceSize).join(', ')}`,
          );
        }
      }
    }
  }

  /**
   * Perform validations on the reader instance
   */
  private validateReaderInstance(writer: IAuroraClusterInstance, reader: IAuroraClusterInstance): void {
    if (writer.type === InstanceType.PROVISIONED) {
      if (reader.type === InstanceType.SERVERLESS_V2) {
        if (
          !Token.isUnresolved(this.serverlessV2MaxCapacity) &&
          !instanceSizeSupportedByServerlessV2(writer.instanceSize!, this.serverlessV2MaxCapacity)
        ) {
          Annotations.of(this).addWarningV2('@aws-cdk/aws-rds:serverlessInstanceCantScaleWithWriter',
            'For high availability any serverless instances in promotion tiers 0-1 '+
            'should be able to scale to match the provisioned instance capacity.\n'+
            `Serverless instance ${reader.node.id} is in promotion tier ${reader.tier},\n`+
            `But can not scale to match the provisioned writer instance (${writer.instanceSize})`,
          );
        }
      }
    }
  }

  /**
   * As a cluster-level metric, it represents the average of the ServerlessDatabaseCapacity
   * values of all the Aurora Serverless v2 DB instances in the cluster.
   */
  public metricServerlessDatabaseCapacity(props?: cloudwatch.MetricOptions) {
    return this.metric('ServerlessDatabaseCapacity', { statistic: 'Average', ...props });
  }

  /**
   * This value is represented as a percentage. It's calculated as the value of the
   * ServerlessDatabaseCapacity metric divided by the maximum ACU value of the DB cluster.
   *
   * If this metric approaches a value of 100.0, the DB instance has scaled up as high as it can.
   * Consider increasing the maximum ACU setting for the cluster.
   */
  public metricACUUtilization(props?: cloudwatch.MetricOptions) {
    return this.metric('ACUUtilization', { statistic: 'Average', ...props });
  }

  /**
   * The average number of disk read I/O operations per second.
   *
   * This metric is only available for Aurora database clusters.
   * For non-Aurora RDS clusters, this metric will not return any data
   * in CloudWatch.
   *
   * @default - average over 5 minutes
   */
  public metricVolumeReadIOPs(props?: cloudwatch.MetricOptions) {
    return this.metric('VolumeReadIOPs', { statistic: 'Average', ...props });
  }

  /**
   * The average number of disk write I/O operations per second.
   *
   * This metric is only available for Aurora database clusters.
   * For non-Aurora RDS clusters, this metric will not return any data
   * in CloudWatch.
   *
   * @default - average over 5 minutes
   */
  public metricVolumeWriteIOPs(props?: cloudwatch.MetricOptions) {
    return this.metric('VolumeWriteIOPs', { statistic: 'Average', ...props });
  }

  private validateServerlessScalingConfig(config: ClusterEngineConfig): void {
    const maxCapacityIsResolved = !Token.isUnresolved(this.serverlessV2MaxCapacity);
    const minCapacityIsResolved = !Token.isUnresolved(this.serverlessV2MinCapacity);

    if (maxCapacityIsResolved && (this.serverlessV2MaxCapacity > 256 || this.serverlessV2MaxCapacity < 1)) {
      throw new ValidationError(lit`InvalidServerlessV2MaxCapacity`, 'serverlessV2MaxCapacity must be >= 1 & <= 256', this);
    }

    if (minCapacityIsResolved && (this.serverlessV2MinCapacity > 256 || this.serverlessV2MinCapacity < 0)) {
      throw new ValidationError(lit`InvalidServerlessV2MinCapacity`, 'serverlessV2MinCapacity must be >= 0 & <= 256', this);
    }

    if (maxCapacityIsResolved && minCapacityIsResolved && this.serverlessV2MaxCapacity < this.serverlessV2MinCapacity) {
      throw new ValidationError(lit`ServerlessV2MaxCapacityTooLow`, 'serverlessV2MaxCapacity must be greater than serverlessV2MinCapacity', this);
    }

    const regexp = new RegExp(/^[0-9]+\.?5?$/);
    if (
      (maxCapacityIsResolved && !regexp.test(this.serverlessV2MaxCapacity.toString())) ||
      (minCapacityIsResolved && !regexp.test(this.serverlessV2MinCapacity.toString()))
    ) {
      throw new ValidationError(lit`InvalidServerlessV2CapacityIncrement`, 'serverlessV2MinCapacity & serverlessV2MaxCapacity must be in 0.5 step increments, received '+
      `min: ${this.serverlessV2MinCapacity}, max: ${this.serverlessV2MaxCapacity}`, this);
    }

    if (this.serverlessV2AutoPauseDuration) {
      if (!config.features?.serverlessV2AutoPauseSupported) {
        throw new ValidationError(lit`ServerlessV2AutoPauseNotSupported`, `serverlessV2 auto-pause feature is not supported by ${this.engine?.engineType} ${this.engine?.engineVersion?.fullVersion}.`, this);
      }
      if (
        !this.serverlessV2AutoPauseDuration.isUnresolved() &&
        (this.serverlessV2AutoPauseDuration.toSeconds() < 300 || this.serverlessV2AutoPauseDuration.toSeconds() > 86400)
      ) {
        throw new ValidationError(lit`InvalidServerlessV2AutoPauseDuration`, `serverlessV2AutoPause must be between 300 seconds (5 minutes) and 86,400 seconds (24 hours), received ${this.serverlessV2AutoPauseDuration.toSeconds()} seconds`, this);
      }
    }
  }

  /**
   * Adds the single user rotation of the master password to this cluster.
   * See [Single user rotation strategy](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets_strategies.html#rotating-secrets-one-user-one-password)
   */
  public addRotationSingleUser(options: RotationSingleUserOptions = {}): secretsmanager.SecretRotation {
    if (this.manageMasterUserPassword) {
      throw new ValidationError(lit`CannotAddRotationWithManageMasterUserPassword`, 'Cannot add rotation when `manageMasterUserPassword` is enabled. RDS automatically rotates the master password when it manages the secret.', this);
    }
    if (!this.secret) {
      throw new ValidationError(lit`CannotAddSingleUserRotationWithoutSecret`, 'Cannot add a single user rotation for a cluster without a secret.', this);
    }

    const id = 'RotationSingleUser';
    const existing = this.node.tryFindChild(id);
    if (existing) {
      throw new ValidationError(lit`SingleUserRotationAlreadyExists`, 'A single user rotation was already added to this cluster.', this);
    }

    return new secretsmanager.SecretRotation(this, id, {
      ...applyDefaultRotationOptions(options, this.vpcSubnets),
      secret: this.secret,
      application: this.singleUserRotationApplication,
      vpc: this.vpc,
      target: this,
    });
  }

  /**
   * Adds the multi user rotation to this cluster.
   * See [Alternating users rotation strategy](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets_strategies.html#rotating-secrets-two-users)
   */
  public addRotationMultiUser(id: string, options: RotationMultiUserOptions): secretsmanager.SecretRotation {
    if (this.manageMasterUserPassword) {
      throw new ValidationError(lit`CannotAddRotationWithManageMasterUserPassword`, 'Cannot add rotation when `manageMasterUserPassword` is enabled. RDS automatically rotates the master password when it manages the secret.', this);
    }
    if (!this.secret) {
      throw new ValidationError(lit`CannotAddMultiUserRotationWithoutSecret`, 'Cannot add a multi user rotation for a cluster without a secret.', this);
    }

    return new secretsmanager.SecretRotation(this, id, {
      ...applyDefaultRotationOptions(options, this.vpcSubnets),
      secret: options.secret,
      masterSecret: this.secret,
      application: this.multiUserRotationApplication,
      vpc: this.vpc,
      target: this,
    });
  }
}

/**
 * Represents an imported database cluster.
 */
@propertyInjectable
class ImportedDatabaseCluster extends DatabaseClusterBase implements IDatabaseCluster {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-rds.ImportedDatabaseCluster';
  public readonly clusterIdentifier: string;
  public readonly connections: ec2.Connections;
  public readonly engine?: IClusterEngine;
  public readonly secret?: secretsmanager.ISecret;

  private readonly _clusterResourceIdentifier?: string;
  private readonly _clusterEndpoint?: Endpoint;
  private readonly _clusterReadEndpoint?: Endpoint;
  private readonly _instanceIdentifiers?: string[];
  private readonly _instanceEndpoints?: Endpoint[];

  protected readonly enableDataApi: boolean;

  constructor(scope: Construct, id: string, attrs: DatabaseClusterAttributes) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, attrs);

    this.clusterIdentifier = attrs.clusterIdentifier;
    this._clusterResourceIdentifier = attrs.clusterResourceIdentifier;

    const defaultPort = attrs.port ? ec2.Port.tcp(attrs.port) : undefined;
    this.connections = new ec2.Connections({
      securityGroups: attrs.securityGroups,
      defaultPort,
    });
    this.engine = attrs.engine;
    this.secret = attrs.secret;

    this.enableDataApi = attrs.dataApiEnabled ?? false;

    this._clusterEndpoint = (attrs.clusterEndpointAddress && attrs.port) ? new Endpoint(attrs.clusterEndpointAddress, attrs.port) : undefined;
    this._clusterReadEndpoint = (attrs.readerEndpointAddress && attrs.port) ? new Endpoint(attrs.readerEndpointAddress, attrs.port) : undefined;
    this._instanceIdentifiers = attrs.instanceIdentifiers;
    this._instanceEndpoints = (attrs.instanceEndpointAddresses && attrs.port)
      ? attrs.instanceEndpointAddresses.map(addr => new Endpoint(addr, attrs.port!))
      : undefined;
  }

  public get clusterResourceIdentifier() {
    if (!this._clusterResourceIdentifier) {
      throw new ValidationError(lit`CannotAccessImportedClusterResourceIdentifier`, 'Cannot access `clusterResourceIdentifier` of an imported cluster without a clusterResourceIdentifier', this);
    }
    return this._clusterResourceIdentifier;
  }

  public get clusterEndpoint() {
    if (!this._clusterEndpoint) {
      throw new ValidationError(lit`CannotAccessImportedClusterEndpoint`, 'Cannot access `clusterEndpoint` of an imported cluster without an endpoint address and port', this);
    }
    return this._clusterEndpoint;
  }

  public get clusterReadEndpoint() {
    if (!this._clusterReadEndpoint) {
      throw new ValidationError(lit`CannotAccessImportedClusterReadEndpoint`, 'Cannot access `clusterReadEndpoint` of an imported cluster without a readerEndpointAddress and port', this);
    }
    return this._clusterReadEndpoint;
  }

  public get instanceIdentifiers() {
    if (!this._instanceIdentifiers) {
      throw new ValidationError(lit`CannotAccessImportedInstanceIdentifiers`, 'Cannot access `instanceIdentifiers` of an imported cluster without provided instanceIdentifiers', this);
    }
    return this._instanceIdentifiers;
  }

  public get instanceEndpoints() {
    if (!this._instanceEndpoints) {
      throw new ValidationError(lit`CannotAccessImportedInstanceEndpoints`, 'Cannot access `instanceEndpoints` of an imported cluster without instanceEndpointAddresses and port', this);
    }
    return this._instanceEndpoints;
  }
}

/**
 * Properties for a new database cluster
 */
export interface DatabaseClusterProps extends DatabaseClusterBaseProps {
  /**
   * Credentials for the administrative user
   *
   * @default - A username of 'admin' (or 'postgres' for PostgreSQL) and SecretsManager-generated password
   */
  readonly credentials?: Credentials;

  /**
   * The Amazon Resource Name (ARN) of the source DB instance or DB cluster if this DB cluster is created as a read replica.
   * Cannot be used with credentials.
   *
   * @default - This DB Cluster is not a read replica
   */
  readonly replicationSourceIdentifier?: string;

  /**
   * Whether to use RDS native integration with AWS Secrets Manager for master user password management.
   *
   * When enabled, RDS generates and manages the master user password in Secrets Manager.
   * Cannot be used together with credentials containing a password.
   *
   * @default false
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/rds-secrets-manager.html
   */
  readonly manageMasterUserPassword?: boolean;
}

/**
 * Create a clustered database with a given number of instances.
 *
 * @resource AWS::RDS::DBCluster
 */
@propertyInjectable
export class DatabaseCluster extends DatabaseClusterNew {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-rds.DatabaseCluster';

  /**
   * Lookup an existing DatabaseCluster using clusterIdentifier.
   */
  public static fromLookup(scope: Construct, id: string, options: DatabaseClusterLookupOptions): IDatabaseCluster {
    if (Token.isUnresolved(options.clusterIdentifier)) {
      throw new UnscopedValidationError(lit`CannotLookupClusterWithTokenizedIdentifier`, 'Cannot look up a cluster with a tokenized cluster identifier.');
    }
    const response: {[key: string]: any}[] = ContextProvider.getValue(scope, {
      provider: cxschema.ContextProvider.CC_API_PROVIDER,
      props: {
        typeName: 'AWS::RDS::DBCluster',
        exactIdentifier: options.clusterIdentifier,
        propertiesToReturn: [
          'DBClusterArn',
          'Endpoint.Address',
          'Endpoint.Port',
          'ReadEndpoint.Address',
          'DBClusterResourceId',
          'VpcSecurityGroupIds',
          'EnableHttpEndpoint',
        ],
      } as cxschema.CcApiContextQuery,
      dummyValue: [
        {
          'Identifier': 'TEST',
          'DBClusterArn': 'TESTARN',
          'Endpoint.Address': 'TESTADDRESS',
          'Endpoint.Port': '5432',
          'ReadEndpoint.Address': 'TESTREADERADDRESS',
          'DBClusterResourceId': 'TESTID',
          'VpcSecurityGroupIds': [],
          'EnableHttpEndpoint': true,
        },
      ],
    }).value;

    // getValue returns a list of result objects. We are expecting 1 result or Error.
    const cluster = response[0];

    // Get ISecurityGroup from securityGroupId
    let securityGroups: ec2.ISecurityGroup[] = [];
    const dbsg: string[] = cluster.VpcSecurityGroupIds;
    if (dbsg) {
      securityGroups = dbsg.map((securityGroupId) => {
        return ec2.SecurityGroup.fromSecurityGroupId(
          scope,
          `LSG-${securityGroupId}`,
          securityGroupId,
        );
      });
    }

    return this.fromDatabaseClusterAttributes(scope, id, {
      clusterIdentifier: options.clusterIdentifier,
      clusterEndpointAddress: cluster['Endpoint.Address'],
      readerEndpointAddress: cluster['ReadEndpoint.Address'],
      port: Number(cluster['Endpoint.Port']),
      clusterResourceIdentifier: cluster.DBClusterResourceId,
      securityGroups: securityGroups,
      dataApiEnabled: cluster.EnableHttpEndpoint,
    });
  }

  /**
   * Import an existing DatabaseCluster from properties
   */
  public static fromDatabaseClusterAttributes(scope: Construct, id: string, attrs: DatabaseClusterAttributes): IDatabaseCluster {
    return new ImportedDatabaseCluster(scope, id, attrs);
  }

  public readonly clusterIdentifier: string;
  public readonly clusterResourceIdentifier: string;
  public readonly clusterEndpoint: Endpoint;
  public readonly clusterReadEndpoint: Endpoint;
  public readonly connections: ec2.Connections;
  public readonly instanceIdentifiers: string[];
  public readonly instanceEndpoints: Endpoint[];

  /**
   * The secret attached to this cluster
   */
  public readonly secret?: secretsmanager.ISecret;

  constructor(scope: Construct, id: string, props: DatabaseClusterProps) {
    super(scope, id, props);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    // Validate manageMasterUserPassword conflicts with unsupported credential properties
    if (props.manageMasterUserPassword) {
      validateManagedPasswordCredentials(this, props.credentials);
    }

    const canHaveCredentials = props.replicationSourceIdentifier === undefined;

    // A read replica inherits the master credentials from its source cluster, so RDS does not
    // manage a master user secret for it. Reject the contradictory combination up front instead
    // of silently dropping `ManageMasterUserPassword` from the template.
    if (props.manageMasterUserPassword && !canHaveCredentials) {
      throw new ValidationError(lit`ManagedPasswordIncompatibleWithReadReplica`, 'cannot use `manageMasterUserPassword` with `replicationSourceIdentifier`; read replicas inherit credentials from the source cluster', this);
    }

    this.manageMasterUserPassword = props.manageMasterUserPassword;

    // Prepare credential-specific configuration
    let secret: secretsmanager.ISecret | undefined;
    let masterUsername: string | undefined;
    let masterUserPassword: string | undefined;
    let manageMasterUserPassword: boolean | undefined;
    let masterUserSecret: { kmsKeyId: string } | undefined;

    if (props.manageMasterUserPassword) {
      // RDS-managed approach: RDS creates and manages the Secret automatically.
      // `canHaveCredentials` is always true here because the read-replica combination is rejected above.
      masterUsername = props.credentials?.username ?? props.engine.defaultUsername ?? 'admin';
      manageMasterUserPassword = props.manageMasterUserPassword;
      masterUserSecret = props.credentials?.encryptionKey
        ? { kmsKeyId: props.credentials.encryptionKey.keyArn }
        : undefined;
    } else {
      // Standard approach: CDK creates and manages the Secret via DatabaseSecret
      const credentials = renderCredentials(this, props.engine, props.credentials);
      secret = credentials.secret;
      masterUsername = canHaveCredentials ? credentials.username : undefined;
      masterUserPassword = canHaveCredentials ? credentials.password?.unsafeUnwrap() : undefined;
    }

    // Create the cluster with the prepared configuration
    const cluster = new CfnDBCluster(this, 'Resource', {
      ...this.newCfnProps,
      masterUsername,
      masterUserPassword,
      manageMasterUserPassword,
      masterUserSecret,
      replicationSourceIdentifier: props.replicationSourceIdentifier,
    });

    this.clusterIdentifier = cluster.ref;
    this.clusterResourceIdentifier = cluster.attrDbClusterResourceId;

    // Set up the secret reference
    if (props.manageMasterUserPassword) {
      this.secret = secretsmanager.Secret.fromSecretAttributes(this, 'ManagedSecret', {
        secretCompleteArn: cluster.attrMasterUserSecretSecretArn,
        encryptionKey: props.credentials?.encryptionKey,
      });
    } else if (secret) {
      this.secret = secret.attach(this);
    }

    // create a number token that represents the port of the cluster
    const portAttribute = Token.asNumber(cluster.attrEndpointPort);
    this.clusterEndpoint = new Endpoint(cluster.attrEndpointAddress, portAttribute);
    this.clusterReadEndpoint = new Endpoint(cluster.attrReadEndpointAddress, portAttribute);
    this.connections = new ec2.Connections({
      securityGroups: this.securityGroups,
      defaultPort: ec2.Port.tcp(this.clusterEndpoint.port),
    });

    cluster.applyRemovalPolicy(props.removalPolicy ?? RemovalPolicy.SNAPSHOT);

    setLogRetention(this, props);

    // create the instances for only standard aurora clusters
    if (props.clusterScalabilityType !== ClusterScalabilityType.LIMITLESS && props.clusterScailabilityType !== ClusterScailabilityType.LIMITLESS) {
      if ((props.writer || props.readers) && (props.instances || props.instanceProps)) {
        throw new ValidationError(lit`CannotProvideWriterWithInstanceProps`, 'Cannot provide writer or readers if instances or instanceProps are provided', this);
      }

      if (!props.instanceProps && !props.writer) {
        throw new ValidationError(lit`WriterMustBeProvided`, 'writer must be provided', this);
      }

      const createdInstances = props.writer ? this._createInstances(props) : legacyCreateInstances(this, props, this.subnetGroupRef);
      this.instanceIdentifiers = createdInstances.instanceIdentifiers;
      this.instanceEndpoints = createdInstances.instanceEndpoints;
    } else {
      // Limitless database does not have instances,
      // but an empty array will be assigned to avoid destructive changes.
      this.instanceIdentifiers = [];
      this.instanceEndpoints = [];
    }
  }
}

/**
 * Mapping of instance type to memory setting on the xlarge size
 * The memory is predictable based on the xlarge size. For example
 * if m5.xlarge has 16GB memory then
 *   - m5.2xlarge will have 32 (16*2)
 *   - m5.4xlarge will have 62 (16*4)
 *   - m5.24xlarge will have 384 (16*24)
 */
const INSTANCE_TYPE_XLARGE_MEMORY_MAPPING: { [instanceType: string]: number } = {
  m5: 16,
  m5d: 16,
  m6g: 16,
  t4g: 16,
  t3: 16,
  m4: 16,
  r6g: 32,
  r5: 32,
  r5b: 32,
  r5d: 32,
  r4: 30.5,
  x2g: 64,
  x1e: 122,
  x1: 61,
  z1d: 32,
};

/**
 * This validates that the instance size falls within the maximum configured serverless capacity.
 *
 * @param instanceSize the instance size of the provisioned writer, e.g. r5.xlarge
 * @param serverlessV2MaxCapacity the maxCapacity configured on the cluster
 * @returns true if the instance size is supported by serverless v2 instances
 */
function instanceSizeSupportedByServerlessV2(instanceSize: string, serverlessV2MaxCapacity: number): boolean {
  const serverlessMaxMem = serverlessV2MaxCapacity*2;
  // i.e. r5.xlarge
  const sizeParts = instanceSize.split('.');
  if (sizeParts.length === 2) {
    const type = sizeParts[0];
    const size = sizeParts[1];
    const xlargeMem = INSTANCE_TYPE_XLARGE_MEMORY_MAPPING[type];
    if (size.endsWith('xlarge')) {
      const instanceMem = size === 'xlarge'
        ? xlargeMem
        : Number(size.slice(0, -6))*xlargeMem;
      if (instanceMem > serverlessMaxMem) {
        return false;
      }
    // smaller than xlarge
    } else {
      return true;
    }
  } else {
    // some weird non-standard instance types
    // not sure how to add automation around this so for now
    // just handling as one offs
    const unSupportedSizes = [
      'db.r5.2xlarge.tpc2.mem8x',
      'db.r5.4xlarge.tpc2.mem3x',
      'db.r5.4xlarge.tpc2.mem4x',
      'db.r5.6xlarge.tpc2.mem4x',
      'db.r5.8xlarge.tpc2.mem3x',
      'db.r5.12xlarge.tpc2.mem2x',
    ];
    if (unSupportedSizes.includes(instanceSize)) {
      return false;
    }
  }
  return true;
}

/**
 * Properties for ``DatabaseClusterFromSnapshot``
 */
export interface DatabaseClusterFromSnapshotProps extends DatabaseClusterBaseProps {
  /**
   * The identifier for the DB instance snapshot or DB cluster snapshot to restore from.
   * You can use either the name or the Amazon Resource Name (ARN) to specify a DB cluster snapshot.
   * However, you can use only the ARN to specify a DB instance snapshot.
   */
  readonly snapshotIdentifier: string;

  /**
   * Credentials for the administrative user
   *
   * Note - using this prop only works with `Credentials.fromPassword()` with the
   * username of the snapshot, `Credentials.fromUsername()` with the username and
   * password of the snapshot or `Credentials.fromSecret()` with a secret containing
   * the username and password of the snapshot.
   *
   * @default - A username of 'admin' (or 'postgres' for PostgreSQL) and SecretsManager-generated password
   * that **will not be applied** to the cluster, use `snapshotCredentials` for the correct behavior.
   *
   * @deprecated use `snapshotCredentials` which allows to generate a new password
   */
  readonly credentials?: Credentials;

  /**
   * Master user credentials.
   *
   * Note - It is not possible to change the master username for a snapshot;
   * however, it is possible to provide (or generate) a new password.
   *
   * @default - The existing username and password from the snapshot will be used.
   */
  readonly snapshotCredentials?: SnapshotCredentials;

  /**
   * Whether to use RDS native integration with AWS Secrets Manager for master user password management.
   *
   * When enabled, RDS generates and manages the master user password in Secrets Manager.
   * This is supported when restoring from snapshots, allowing migration to RDS-managed passwords.
   *
   * @default false
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/rds-secrets-manager.html
   */
  readonly manageMasterUserPassword?: boolean;
}

/**
 * A database cluster restored from a snapshot.
 *
 * @resource AWS::RDS::DBCluster
 */
@propertyInjectable
export class DatabaseClusterFromSnapshot extends DatabaseClusterNew {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-rds.DatabaseClusterFromSnapshot';

  public readonly clusterIdentifier: string;
  public readonly clusterResourceIdentifier: string;
  public readonly clusterEndpoint: Endpoint;
  public readonly clusterReadEndpoint: Endpoint;
  public readonly connections: ec2.Connections;
  public readonly instanceIdentifiers: string[];
  public readonly instanceEndpoints: Endpoint[];

  /**
   * The secret attached to this cluster
   */
  public readonly secret?: secretsmanager.ISecret;

  constructor(scope: Construct, id: string, props: DatabaseClusterFromSnapshotProps) {
    super(scope, id, props);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    if (props.credentials && !props.credentials.password && !props.credentials.secret) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-rds:useSnapshotCredentials', 'Use `snapshotCredentials` to modify password of a cluster created from a snapshot.');
    }
    if (!props.credentials && !props.snapshotCredentials) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-rds:generatedCredsNotApplied', 'Generated credentials will not be applied to cluster. Use `snapshotCredentials` instead. `addRotationSingleUser()` and `addRotationMultiUser()` cannot be used on this cluster.');
    }

    // Validate manageMasterUserPassword conflicts with unsupported snapshotCredentials properties
    if (props.manageMasterUserPassword) {
      validateManagedPasswordSnapshotCredentials(this, props.snapshotCredentials);
    }

    this.manageMasterUserPassword = props.manageMasterUserPassword;

    // When RDS manages the master password, the deprecated `credentials` rendering path must be
    // skipped — otherwise it always creates an orphan DatabaseSecret that is never bound to the cluster.
    const deprecatedCredentials = (!props.manageMasterUserPassword
      && !FeatureFlags.of(this).isEnabled(cxapi.RDS_PREVENT_RENDERING_DEPRECATED_CREDENTIALS))
      ? renderCredentials(this, props.engine, props.credentials)
      : undefined;

    const credentials = renderSnapshotCredentials(this, props.snapshotCredentials);

    const cluster = new CfnDBCluster(this, 'Resource', {
      ...this.newCfnProps,
      snapshotIdentifier: props.snapshotIdentifier,
      masterUserPassword: props.manageMasterUserPassword
        ? undefined
        : credentials?.secret?.secretValueFromJson('password')?.unsafeUnwrap() ?? credentials?.password?.unsafeUnwrap(), // Safe usage
      manageMasterUserPassword: props.manageMasterUserPassword || undefined,
      masterUserSecret: props.manageMasterUserPassword && props.snapshotCredentials?.encryptionKey
        ? { kmsKeyId: props.snapshotCredentials.encryptionKey.keyArn }
        : undefined,
    });

    this.clusterIdentifier = cluster.ref;
    this.clusterResourceIdentifier = cluster.attrDbClusterResourceId;

    if (props.manageMasterUserPassword) {
      this.secret = secretsmanager.Secret.fromSecretAttributes(this, 'ManagedSecret', {
        secretCompleteArn: cluster.attrMasterUserSecretSecretArn,
        encryptionKey: props.snapshotCredentials?.encryptionKey,
      });
    } else if (credentials?.secret) {
      this.secret = credentials.secret.attach(this);
    }

    if (deprecatedCredentials?.secret) {
      const deprecatedSecret = deprecatedCredentials.secret.attach(this);
      if (!this.secret) {
        this.secret = deprecatedSecret;
      }
    }

    // create a number token that represents the port of the cluster
    const portAttribute = Token.asNumber(cluster.attrEndpointPort);
    this.clusterEndpoint = new Endpoint(cluster.attrEndpointAddress, portAttribute);
    this.clusterReadEndpoint = new Endpoint(cluster.attrReadEndpointAddress, portAttribute);
    this.connections = new ec2.Connections({
      securityGroups: this.securityGroups,
      defaultPort: ec2.Port.tcp(this.clusterEndpoint.port),
    });

    cluster.applyRemovalPolicy(props.removalPolicy ?? RemovalPolicy.SNAPSHOT);

    setLogRetention(this, props);
    if ((props.writer || props.readers) && (props.instances || props.instanceProps)) {
      throw new ValidationError(lit`CannotProvideClusterInstancesWithInstanceProps`, 'Cannot provide clusterInstances if instances or instanceProps are provided', this);
    }
    const createdInstances = props.writer ? this._createInstances(props) : legacyCreateInstances(this, props, this.subnetGroupRef);
    this.instanceIdentifiers = createdInstances.instanceIdentifiers;
    this.instanceEndpoints = createdInstances.instanceEndpoints;
  }
}

/**
 * Sets up CloudWatch log retention if configured.
 * A function rather than protected member to prevent exposing ``DatabaseClusterBaseProps``.
 */
function setLogRetention(cluster: DatabaseClusterNew, props: DatabaseClusterBaseProps) {
  if (props.cloudwatchLogsExports) {
    const unsupportedLogTypes = props.cloudwatchLogsExports.filter(logType => !props.engine.supportedLogTypes.includes(logType));
    if (unsupportedLogTypes.length > 0) {
      throw new ValidationError(lit`UnsupportedLogTypesForEngine`, `Unsupported logs for the current engine type: ${unsupportedLogTypes.join(',')}`, cluster);
    }

    if (props.cloudwatchLogsRetention) {
      for (const log of props.cloudwatchLogsExports) {
        const logGroupName = `/aws/rds/cluster/${cluster.clusterIdentifier}/${log}`;
        new logs.LogRetention(cluster, `LogRetention${log}`, {
          logGroupName,
          retention: props.cloudwatchLogsRetention,
          role: props.cloudwatchLogsRetentionRole,
        });
        cluster.cloudwatchLogGroups[log] = logs.LogGroup.fromLogGroupName(cluster, `LogGroup${cluster.clusterIdentifier}${log}`, logGroupName);
      }
    }
  }
}

/** Output from the createInstances method; used to set instance identifiers and endpoints */
interface InstanceConfig {
  readonly instanceIdentifiers: string[];
  readonly instanceEndpoints: Endpoint[];
}

/**
 * Creates the instances for the cluster.
 * A function rather than a protected method on ``DatabaseClusterNew`` to avoid exposing
 * ``DatabaseClusterNew`` and ``DatabaseClusterBaseProps`` in the API.
 */
function legacyCreateInstances(cluster: DatabaseClusterNew, props: DatabaseClusterBaseProps, subnetGroup: aws_rds.IDBSubnetGroupRef): InstanceConfig {
  const instanceCount = props.instances != null ? props.instances : 2;
  const instanceUpdateBehaviour = props.instanceUpdateBehaviour ?? InstanceUpdateBehaviour.BULK;
  if (Token.isUnresolved(instanceCount)) {
    throw new ValidationError(lit`InstanceCountCannotBeDeployTimeValue`, 'The number of instances an RDS Cluster consists of cannot be provided as a deploy-time only value!', cluster);
  }
  if (instanceCount < 1) {
    throw new ValidationError(lit`AtLeastOneInstanceRequired`, 'At least one instance is required', cluster);
  }

  const instanceIdentifiers: string[] = [];
  const instanceEndpoints: Endpoint[] = [];
  const portAttribute = cluster.clusterEndpoint.port;
  const instanceProps = props.instanceProps!;

  // Get the actual subnet objects so we can depend on internet connectivity.
  const internetConnected = instanceProps.vpc.selectSubnets(instanceProps.vpcSubnets).internetConnectivityEstablished;

  const enablePerformanceInsights = instanceProps.enablePerformanceInsights
    || instanceProps.performanceInsightRetention !== undefined || instanceProps.performanceInsightEncryptionKey !== undefined;
  if (enablePerformanceInsights && instanceProps.enablePerformanceInsights === false) {
    throw new ValidationError(lit`PerformanceInsightsDisabledButRetentionOrKeySet`, '`enablePerformanceInsights` disabled, but `performanceInsightRetention` or `performanceInsightEncryptionKey` was set', cluster);
  }
  const performanceInsightRetention = enablePerformanceInsights
    ? (instanceProps.performanceInsightRetention || PerformanceInsightRetention.DEFAULT)
    : undefined;
  validatePerformanceInsightsSettings(
    cluster,
    {
      performanceInsightsEnabled: enablePerformanceInsights,
      performanceInsightRetention,
      performanceInsightEncryptionKey: instanceProps.performanceInsightEncryptionKey,
    },
  );

  const instanceType = instanceProps.instanceType ?? ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM);

  if (instanceProps.parameterGroup && instanceProps.parameters) {
    throw new ValidationError(lit`CannotSpecifyBothParameterGroupAndParameters`, 'You cannot specify both parameterGroup and parameters', cluster);
  }

  const instanceParameterGroup = instanceProps.parameterGroup ?? (
    instanceProps.parameters
      ? new ParameterGroup(cluster, 'InstanceParameterGroup', {
        engine: props.engine,
        parameters: instanceProps.parameters,
      })
      : undefined
  );
  const instanceParameterGroupConfig = instanceParameterGroup?.bindToInstance({});

  const instances: CfnDBInstance[] = [];

  for (let i = 0; i < instanceCount; i++) {
    const instanceIndex = i + 1;
    const instanceIdentifier = props.instanceIdentifierBase != null ? `${props.instanceIdentifierBase}${instanceIndex}` :
      props.clusterIdentifier != null ? `${props.clusterIdentifier}instance${instanceIndex}` :
        undefined;

    const instance = new CfnDBInstance(cluster, `Instance${instanceIndex}`, {
      // Link to cluster
      engine: props.engine.engineType,
      dbClusterIdentifier: cluster.clusterIdentifier,
      dbInstanceIdentifier: instanceIdentifier,
      // Instance properties
      dbInstanceClass: databaseInstanceType(instanceType),
      publiclyAccessible: instanceProps.publiclyAccessible ??
        (instanceProps.vpcSubnets && instanceProps.vpcSubnets.subnetType === ec2.SubnetType.PUBLIC),
      enablePerformanceInsights: enablePerformanceInsights || instanceProps.enablePerformanceInsights, // fall back to undefined if not set
      performanceInsightsKmsKeyId: instanceProps.performanceInsightEncryptionKey?.keyArn,
      performanceInsightsRetentionPeriod: performanceInsightRetention,
      // This is already set on the Cluster. Unclear to me whether it should be repeated or not. Better yes.
      dbSubnetGroupName: subnetGroup.dbSubnetGroupRef.dbSubnetGroupName,
      dbParameterGroupName: instanceParameterGroupConfig?.parameterGroupName,
      // When `enableClusterLevelEnhancedMonitoring` is enabled,
      // both `monitoringInterval` and `monitoringRole` are set at cluster level so no need to re-set it in instance level.
      monitoringInterval: props.enableClusterLevelEnhancedMonitoring ? undefined : props.monitoringInterval?.toSeconds(),
      monitoringRoleArn: props.enableClusterLevelEnhancedMonitoring ? undefined : cluster.monitoringRole?.roleArn,
      autoMinorVersionUpgrade: instanceProps.autoMinorVersionUpgrade,
      allowMajorVersionUpgrade: instanceProps.allowMajorVersionUpgrade,
      deleteAutomatedBackups: instanceProps.deleteAutomatedBackups,
      preferredMaintenanceWindow: instanceProps.preferredMaintenanceWindow,
    });

    // For instances that are part of a cluster:
    //
    //  Cluster DESTROY or SNAPSHOT -> DESTROY (snapshot is good enough to recreate)
    //  Cluster RETAIN              -> RETAIN (otherwise cluster state will disappear)
    instance.applyRemovalPolicy(helperRemovalPolicy(props.removalPolicy));

    // We must have a dependency on the NAT gateway provider here to create
    // things in the right order.
    instance.node.addDependency(internetConnected);

    instanceIdentifiers.push(instance.ref);
    instanceEndpoints.push(new Endpoint(instance.attrEndpointAddress, portAttribute));
    instances.push(instance);
  }

  // Adding dependencies here to ensure that the instances are updated one after the other.
  if (instanceUpdateBehaviour === InstanceUpdateBehaviour.ROLLING) {
    for (let i = 1; i < instanceCount; i++) {
      instances[i].node.addDependency(instances[i-1]);
    }
  }

  return { instanceEndpoints, instanceIdentifiers };
}

/**
 * Turn a regular instance type into a database instance type
 */
function databaseInstanceType(instanceType: ec2.InstanceType) {
  return 'db.' + instanceType.toString();
}

/**
 * Validate Performance Insights settings
 */
function validatePerformanceInsightsSettings(
  cluster: DatabaseClusterNew,
  instance: {
    nodeId?: string;
    performanceInsightsEnabled?: boolean;
    performanceInsightRetention?: PerformanceInsightRetention;
    performanceInsightEncryptionKey?: kms.IKeyRef;
  },
): void {
  const target = instance.nodeId ? `instance \'${instance.nodeId}\'` : '`instanceProps`';

  // If Performance Insights is enabled on the cluster, the one for each instance will be enabled as well.
  if (cluster.performanceInsightsEnabled && instance.performanceInsightsEnabled === false) {
    Annotations.of(cluster).addWarningV2(
      '@aws-cdk/aws-rds:instancePerformanceInsightsOverridden',
      `Performance Insights is enabled on cluster '${cluster.node.id}' at cluster level, but disabled for ${target}. `+
      'However, Performance Insights for this instance will also be automatically enabled if enabled at cluster level.',
    );
  }

  // If `performanceInsightRetention` is enabled on the cluster, the same parameter for each instance must be
  // undefined or the same as the value at cluster level.
  if (
    cluster.performanceInsightRetention &&
    instance.performanceInsightRetention &&
    instance.performanceInsightRetention !== cluster.performanceInsightRetention
  ) {
    throw new ValidationError(lit`PerformanceInsightRetentionMismatch`, `\`performanceInsightRetention\` for each instance must be the same as the one at cluster level, got ${target}: ${instance.performanceInsightRetention}, cluster: ${cluster.performanceInsightRetention}`, cluster);
  }

  // If `performanceInsightEncryptionKey` is enabled on the cluster, the same parameter for each instance must be
  // undefined or the same as the value at cluster level.
  if (cluster.performanceInsightEncryptionKey && instance.performanceInsightEncryptionKey) {
    const clusterKeyArn = cluster.performanceInsightEncryptionKey.keyArn;
    const instanceKeyArn = instance.performanceInsightEncryptionKey.keyRef.keyArn;
    const compared = Token.compareStrings(clusterKeyArn, instanceKeyArn);

    if (compared === TokenComparison.DIFFERENT) {
      throw new ValidationError(lit`PerformanceInsightEncryptionKeyMismatch`, `\`performanceInsightEncryptionKey\` for each instance must be the same as the one at cluster level, got ${target}: '${instance.performanceInsightEncryptionKey.keyRef.keyArn}', cluster: '${cluster.performanceInsightEncryptionKey.keyRef.keyArn}'`, cluster);
    }
    // Even if both of cluster and instance keys are unresolved, check if they are the same token.
    if (compared === TokenComparison.BOTH_UNRESOLVED && clusterKeyArn !== instanceKeyArn) {
      throw new ValidationError(lit`PerformanceInsightEncryptionKeyMustMatch`, '`performanceInsightEncryptionKey` for each instance must be the same as the one at cluster level', cluster);
    }
  }
}
