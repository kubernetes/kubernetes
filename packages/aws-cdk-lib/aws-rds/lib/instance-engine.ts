import type { Construct } from 'constructs';
import type { IEngine } from './engine';
import type { EngineVersion } from './engine-version';
import type { IOptionGroup } from './option-group';
import { OptionGroup } from './option-group';
import type * as iam from '../../aws-iam';
import * as secretsmanager from '../../aws-secretsmanager';
import { ValidationError } from '../../core/lib/errors';
import { lit } from '../../core/lib/private/literal-string';

/**
 * The options passed to `IInstanceEngine.bind`.
 */
export interface InstanceEngineBindOptions {
  /**
   * The Active Directory directory ID to create the DB instance in.
   *
   * @default - none (it's an optional field)
   */
  readonly domain?: string;

  /**
   * The timezone of the database, set by the customer.
   *
   * @default - none (it's an optional field)
   */
  readonly timezone?: string;

  /**
   * The role used for S3 importing.
   *
   * @default - none
   */
  readonly s3ImportRole?: iam.IRoleRef;

  /**
   * The role used for S3 exporting.
   *
   * @default - none
   */
  readonly s3ExportRole?: iam.IRoleRef;

  /**
   * The option group of the database
   *
   * @default - none
   */
  readonly optionGroup?: IOptionGroup;
}

/**
 * The type returned from the `IInstanceEngine.bind` method.
 */
export interface InstanceEngineConfig {
  /**
   * Features supported by the database engine.
   *
   * @see https://docs.aws.amazon.com/AmazonRDS/latest/APIReference/API_DBEngineVersion.html
   *
   * @default - no features
   */
  readonly features?: InstanceEngineFeatures;

  /**
   * Option group of the database.
   *
   * @default - none
   */
  readonly optionGroup?: IOptionGroup;
}

/**
 * Represents Database Engine features
 */
export interface InstanceEngineFeatures {
  /**
   * Feature name for the DB instance that the IAM role to access the S3 bucket for import
   * is to be associated with.
   *
   * @default - no s3Import feature name
   */
  readonly s3Import?: string;

  /**
   * Feature name for the DB instance that the IAM role to export to S3 bucket is to be
   * associated with.
   *
   * @default - no s3Export feature name
   */
  readonly s3Export?: string;
}

/**
 * Interface representing a database instance (as opposed to cluster) engine.
 */
export interface IInstanceEngine extends IEngine {
  /** The application used by this engine to perform rotation for a single-user scenario. */
  readonly singleUserRotationApplication: secretsmanager.SecretRotationApplication;

  /** The application used by this engine to perform rotation for a multi-user scenario. */
  readonly multiUserRotationApplication: secretsmanager.SecretRotationApplication;

  /**
   * Whether this engine supports automatic backups of a read replica instance.
   *
   * @default false
   */
  readonly supportsReadReplicaBackups?: boolean;

  /**
   * Method called when the engine is used to create a new instance.
   */
  bindToInstance(scope: Construct, options: InstanceEngineBindOptions): InstanceEngineConfig;
}

interface InstanceEngineBaseProps {
  readonly engineType: string;
  readonly singleUserRotationApplication: secretsmanager.SecretRotationApplication;
  readonly multiUserRotationApplication: secretsmanager.SecretRotationApplication;
  readonly version?: EngineVersion;
  readonly parameterGroupFamily?: string;
  readonly engineFamily?: string;
  readonly features?: InstanceEngineFeatures;
}

abstract class InstanceEngineBase implements IInstanceEngine {
  public readonly engineType: string;
  public readonly engineVersion?: EngineVersion;
  public readonly parameterGroupFamily?: string;
  public readonly singleUserRotationApplication: secretsmanager.SecretRotationApplication;
  public readonly multiUserRotationApplication: secretsmanager.SecretRotationApplication;
  public readonly engineFamily?: string;
  public readonly supportsReadReplicaBackups?: boolean;

  private readonly features?: InstanceEngineFeatures;

  constructor(props: InstanceEngineBaseProps) {
    this.engineType = props.engineType;
    this.features = props.features;
    this.singleUserRotationApplication = props.singleUserRotationApplication;
    this.multiUserRotationApplication = props.multiUserRotationApplication;
    this.engineVersion = props.version;
    this.parameterGroupFamily = props.parameterGroupFamily ??
      (this.engineVersion ? `${this.engineType}${this.engineVersion.majorVersion}` : undefined);
    this.engineFamily = props.engineFamily;
  }

  public bindToInstance(scope: Construct, options: InstanceEngineBindOptions): InstanceEngineConfig {
    if (options.timezone && !this.supportsTimezone) {
      throw new ValidationError(lit`TimezonePropertyConfigured`, `timezone property can not be configured for ${this.engineType}`, scope);
    }
    return {
      features: this.features,
      optionGroup: options.optionGroup,
    };
  }

  /** Defines whether this Instance Engine can support timezone properties. */
  protected get supportsTimezone() { return false; }
}

/**
 * The versions for the MariaDB instance engines
 * (those returned by `DatabaseInstanceEngine.mariaDb`).
 */
export class MariaDbEngineVersion {
  /**
   * Version "10.0" (only a major version, without a specific minor version).
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0 = MariaDbEngineVersion.of('10.0', '10.0');
  /**
   * Version "10.0.17".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_17 = MariaDbEngineVersion.of('10.0.17', '10.0');
  /**
   * Version "10.0.24".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_24 = MariaDbEngineVersion.of('10.0.24', '10.0');
  /**
   * Version "10.0.28".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_28 = MariaDbEngineVersion.of('10.0.28', '10.0');
  /**
   * Version "10.0.31".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_31 = MariaDbEngineVersion.of('10.0.31', '10.0');
  /**
   * Version "10.0.32".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_32 = MariaDbEngineVersion.of('10.0.32', '10.0');
  /**
   * Version "10.0.34".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_34 = MariaDbEngineVersion.of('10.0.34', '10.0');
  /**
   * Version "10.0.35".
   * @deprecated MariaDB 10.0 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_0_35 = MariaDbEngineVersion.of('10.0.35', '10.0');

  /**
   * Version "10.1" (only a major version, without a specific minor version).
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1 = MariaDbEngineVersion.of('10.1', '10.1');
  /**
   * Version "10.1.14".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_14 = MariaDbEngineVersion.of('10.1.14', '10.1');
  /**
   * Version "10.1.19".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_19 = MariaDbEngineVersion.of('10.1.19', '10.1');
  /**
   * Version "10.1.23".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_23 = MariaDbEngineVersion.of('10.1.23', '10.1');
  /**
   * Version "10.1.26".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_26 = MariaDbEngineVersion.of('10.1.26', '10.1');
  /**
   * Version "10.1.31".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_31 = MariaDbEngineVersion.of('10.1.31', '10.1');
  /**
   * Version "10.1.34".
   * @deprecated MariaDB 10.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1_34 = MariaDbEngineVersion.of('10.1.34', '10.1');

  /**
   * Version "10.2" (only a major version, without a specific minor version)
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2 = MariaDbEngineVersion.of('10.2', '10.2');
  /**
   * Version "10.2.11".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_11 = MariaDbEngineVersion.of('10.2.11', '10.2');
  /**
   * Version "10.2.12".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_12 = MariaDbEngineVersion.of('10.2.12', '10.2');
  /**
   * Version "10.2.15".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_15 = MariaDbEngineVersion.of('10.2.15', '10.2');
  /**
   * Version "10.2.21".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_21 = MariaDbEngineVersion.of('10.2.21', '10.2');
  /**
   * Version "10.2.32".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_32 = MariaDbEngineVersion.of('10.2.32', '10.2');
  /**
   * Version "10.2.37".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_37 = MariaDbEngineVersion.of('10.2.37', '10.2');
  /**
   * Version "10.2.39".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_39 = MariaDbEngineVersion.of('10.2.39', '10.2');
  /**
   * Version "10.2.40".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_40 = MariaDbEngineVersion.of('10.2.40', '10.2');
  /**
   * Version "10.2.41".
   * @deprecated MariaDB 10.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_41 = MariaDbEngineVersion.of('10.2.41', '10.2');
  /**
   * Version "10.2.43"
   * @deprecated MariaDB 10.2.43 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_43 = MariaDbEngineVersion.of('10.2.43', '10.2');
  /**
   * Version "10.2.44"
   * @deprecated MariaDB 10.2.44 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_2_44 = MariaDbEngineVersion.of('10.2.44', '10.2');

  /**
   * Version "10.3" (only a major version, without a specific minor version).
   * @deprecated MariaDB 10.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3 = MariaDbEngineVersion.of('10.3', '10.3');
  /**
   * Version "10.3.8"
   * @deprecated MariaDB 10.3.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_8 = MariaDbEngineVersion.of('10.3.8', '10.3');
  /**
   * Version "10.3.13"
   * @deprecated MariaDB 10.3.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_13 = MariaDbEngineVersion.of('10.3.13', '10.3');
  /**
   * Version "10.3.20"
   * @deprecated MariaDB 10.3.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_20 = MariaDbEngineVersion.of('10.3.20', '10.3');
  /**
   * Version "10.3.23"
   * @deprecated MariaDB 10.3.23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_23 = MariaDbEngineVersion.of('10.3.23', '10.3');
  /**
   * Version "10.3.28"
   * @deprecated MariaDB 10.3.28 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_28 = MariaDbEngineVersion.of('10.3.28', '10.3');
  /**
   * Version "10.3.31"
   * @deprecated MariaDB 10.3.31 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_31 = MariaDbEngineVersion.of('10.3.31', '10.3');
  /**
   * Version "10.3.32"
   * @deprecated MariaDB 10.3.32 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_32 = MariaDbEngineVersion.of('10.3.32', '10.3');
  /**
   * Version "10.3.34"
   * @deprecated MariaDB 10.3.34 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_34 = MariaDbEngineVersion.of('10.3.34', '10.3');
  /**
   * Version "10.3.35"
   * @deprecated MariaDB 10.3.35 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_35 = MariaDbEngineVersion.of('10.3.35', '10.3');
  /**
   * Version "10.3.36"
   * @deprecated MariaDB 10.3.36 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_36 = MariaDbEngineVersion.of('10.3.36', '10.3');
  /**
   * Version "10.3.37"
   * @deprecated MariaDB 10.3.37 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_37 = MariaDbEngineVersion.of('10.3.37', '10.3');
  /**
   * Version "10.3.38"
   * @deprecated MariaDB 10.3.38 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_38 = MariaDbEngineVersion.of('10.3.38', '10.3');
  /**
   * Version "10.3.39"
   * @deprecated MariaDB 10.3.39 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3_39 = MariaDbEngineVersion.of('10.3.39', '10.3');

  /**
   * Version "10.4" (only a major version, without a specific minor version)
   * @deprecated MariaDB 10.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4 = MariaDbEngineVersion.of('10.4', '10.4');
  /**
   * Version "10.4.8"
   * @deprecated MariaDB 10.4.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_8 = MariaDbEngineVersion.of('10.4.8', '10.4');
  /**
   * Version "10.4.13"
   * @deprecated MariaDB 10.4.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_13 = MariaDbEngineVersion.of('10.4.13', '10.4');
  /**
   * Version "10.4.18"
   * @deprecated MariaDB 10.4.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_18 = MariaDbEngineVersion.of('10.4.18', '10.4');
  /**
   * Version "10.4.21"
   * @deprecated MariaDB 10.4.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_21 = MariaDbEngineVersion.of('10.4.21', '10.4');
  /**
   * Version "10.4.22"
   * @deprecated MariaDB 10.4.22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_22 = MariaDbEngineVersion.of('10.4.22', '10.4');
  /**
   * Version "10.4.24"
   * @deprecated MariaDB 10.4.24 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_24 = MariaDbEngineVersion.of('10.4.24', '10.4');
  /**
   * Version "10.4.25"
   * @deprecated MariaDB 10.4.25 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_25 = MariaDbEngineVersion.of('10.4.25', '10.4');
  /**
   * Version "10.4.26"
   * @deprecated MariaDB 10.4.26 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_26 = MariaDbEngineVersion.of('10.4.26', '10.4');
  /**
   * Version "10.4.27"
   * @deprecated MariaDB 10.4.27 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_27 = MariaDbEngineVersion.of('10.4.27', '10.4');
  /**
   * Version "10.4.28"
   * @deprecated MariaDB 10.4.28 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_28 = MariaDbEngineVersion.of('10.4.28', '10.4');
  /**
   * Version "10.4.29"
   * @deprecated MariaDB 10.4.29 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_29 = MariaDbEngineVersion.of('10.4.29', '10.4');
  /**
   * Version "10.4.30"
   * @deprecated MariaDB 10.4.30 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_30 = MariaDbEngineVersion.of('10.4.30', '10.4');
  /**
   * Version "10.4.31"
   * @deprecated MariaDB 10.4.31 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_31 = MariaDbEngineVersion.of('10.4.31', '10.4');
  /**
   * Version "10.4.32"
   * @deprecated MariaDB 10.4.32 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_32 = MariaDbEngineVersion.of('10.4.32', '10.4');
  /**
   * Version "10.4.33"
   * @deprecated MariaDB 10.4.33 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_33 = MariaDbEngineVersion.of('10.4.33', '10.4');
  /**
   * Version "10.4.34"
   * @deprecated MariaDB 10.4.34 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4_34 = MariaDbEngineVersion.of('10.4.34', '10.4');

  /** Version "10.5" (only a major version, without a specific minor version). */
  public static readonly VER_10_5 = MariaDbEngineVersion.of('10.5', '10.5');
  /**
   * Version "10.5.8"
   * @deprecated MariaDB 10.5.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_8 = MariaDbEngineVersion.of('10.5.8', '10.5');
  /**
   * Version "10.5.9"
   * @deprecated MariaDB 10.5.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_9 = MariaDbEngineVersion.of('10.5.9', '10.5');
  /**
   * Version "10.5.12"
   * @deprecated MariaDB 10.5.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_12 = MariaDbEngineVersion.of('10.5.12', '10.5');
  /**
   * Version "10.5.13"
   * @deprecated MariaDB 10.5.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_13 = MariaDbEngineVersion.of('10.5.13', '10.5');
  /**
   * Version "10.5.15"
   * @deprecated MariaDB 10.5.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_15 = MariaDbEngineVersion.of('10.5.15', '10.5');
  /**
   * Version "10.5.16"
   * @deprecated MariaDB 10.5.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_16 = MariaDbEngineVersion.of('10.5.16', '10.5');
  /**
   * Version "10.5.17"
   * @deprecated MariaDB 10.5.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_17 = MariaDbEngineVersion.of('10.5.17', '10.5');
  /**
   * Version "10.5.18"
   * @deprecated MariaDB 10.5.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_18 = MariaDbEngineVersion.of('10.5.18', '10.5');
  /**
   * Version "10.5.19"
   * @deprecated MariaDB 10.5.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_19 = MariaDbEngineVersion.of('10.5.19', '10.5');
  /**
   * Version "10.5.20"
   * @deprecated MariaDB 10.5.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_20 = MariaDbEngineVersion.of('10.5.20', '10.5');
  /**
   * Version "10.5.21"
   * @deprecated MariaDB 10.5.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_21 = MariaDbEngineVersion.of('10.5.21', '10.5');
  /**
   * Version "10.5.22"
   * @deprecated MariaDB 10.5.22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_22 = MariaDbEngineVersion.of('10.5.22', '10.5');
  /**
   * Version "10.5.23"
   * @deprecated MariaDB 10.5.23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_23 = MariaDbEngineVersion.of('10.5.23', '10.5');
  /**
   * Version "10.5.24"
   * @deprecated MariaDB 10.5.24 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_24 = MariaDbEngineVersion.of('10.5.24', '10.5');
  /**
   * Version "10.5.25".
   * @deprecated MariaDB 10.5.25 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_25 = MariaDbEngineVersion.of('10.5.25', '10.5');
  /**
   * Version "10.5.26".
   * @deprecated MariaDB 10.5.26 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5_26 = MariaDbEngineVersion.of('10.5.26', '10.5');
  /** Version "10.5.27". */
  public static readonly VER_10_5_27 = MariaDbEngineVersion.of('10.5.27', '10.5');
  /** Version "10.5.28". */
  public static readonly VER_10_5_28 = MariaDbEngineVersion.of('10.5.28', '10.5');
  /** Version "10.5.29". */
  public static readonly VER_10_5_29 = MariaDbEngineVersion.of('10.5.29', '10.5');

  /** Version "10.6" (only a major version, without a specific minor version). */
  public static readonly VER_10_6 = MariaDbEngineVersion.of('10.6', '10.6');
  /**
   * Version "10.6.5"
   * @deprecated MariaDB 10.6.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_5 = MariaDbEngineVersion.of('10.6.5', '10.6');
  /**
   * Version "10.6.7"
   * @deprecated MariaDB 10.6.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_7 = MariaDbEngineVersion.of('10.6.7', '10.6');
  /**
   * Version "10.6.8"
   * @deprecated MariaDB 10.6.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_8 = MariaDbEngineVersion.of('10.6.8', '10.6');
  /**
   * Version "10.6.10"
   * @deprecated MariaDB 10.6.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_10 = MariaDbEngineVersion.of('10.6.10', '10.6');
  /**
   * Version "10.6.11"
   * @deprecated MariaDB 10.6.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_11 = MariaDbEngineVersion.of('10.6.11', '10.6');
  /**
   * Version "10.6.12"
   * @deprecated MariaDB 10.6.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_12 = MariaDbEngineVersion.of('10.6.12', '10.6');
  /**
   * Version "10.6.13"
   * @deprecated MariaDB 10.6.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_13 = MariaDbEngineVersion.of('10.6.13', '10.6');
  /**
   * Version "10.6.14"
   * @deprecated MariaDB 10.6.14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_14 = MariaDbEngineVersion.of('10.6.14', '10.6');
  /**
   * Version "10.6.15"
   * @deprecated MariaDB 10.6.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_15 = MariaDbEngineVersion.of('10.6.15', '10.6');
  /**
   * Version "10.6.16"
   * @deprecated MariaDB 10.6.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_16 = MariaDbEngineVersion.of('10.6.16', '10.6');
  /**
   * Version "10.6.17"
   * @deprecated MariaDB 10.6.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_17 = MariaDbEngineVersion.of('10.6.17', '10.6');
  /**
   * Version "10.6.18".
   * @deprecated MariaDB 10.6.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6_18 = MariaDbEngineVersion.of('10.6.18', '10.6');
  /** Version "10.6.19". */
  public static readonly VER_10_6_19 = MariaDbEngineVersion.of('10.6.19', '10.6');
  /** Version "10.6.20". */
  public static readonly VER_10_6_20 = MariaDbEngineVersion.of('10.6.20', '10.6');
  /** Version "10.6.21". */
  public static readonly VER_10_6_21 = MariaDbEngineVersion.of('10.6.21', '10.6');
  /** Version "10.6.22". */
  public static readonly VER_10_6_22 = MariaDbEngineVersion.of('10.6.22', '10.6');
  /** Version "10.6.23". */
  public static readonly VER_10_6_23 = MariaDbEngineVersion.of('10.6.23', '10.6');
  /** Version "10.6.24". */
  public static readonly VER_10_6_24 = MariaDbEngineVersion.of('10.6.24', '10.6');
  /** Version "10.6.25". */
  public static readonly VER_10_6_25 = MariaDbEngineVersion.of('10.6.25', '10.6');

  /** Version "10.11" (only a major version, without a specific minor version). */
  public static readonly VER_10_11 = MariaDbEngineVersion.of('10.11', '10.11');
  /**
   * Version "10.11.4"
   * @deprecated MariaDB 10.11.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11_4 = MariaDbEngineVersion.of('10.11.4', '10.11');
  /**
   * Version "10.11.5"
   * @deprecated MariaDB 10.11.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11_5 = MariaDbEngineVersion.of('10.11.5', '10.11');
  /**
   * Version "10.11.6"
   * @deprecated MariaDB 10.11.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11_6 = MariaDbEngineVersion.of('10.11.6', '10.11');
  /**
   * Version "10.11.7"
   * @deprecated MariaDB 10.11.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11_7 = MariaDbEngineVersion.of('10.11.7', '10.11');
  /**
   * Version "10.11.8".
   * @deprecated MariaDB 10.11.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11_8 = MariaDbEngineVersion.of('10.11.8', '10.11');
  /** Version "10.11.9". */
  public static readonly VER_10_11_9 = MariaDbEngineVersion.of('10.11.9', '10.11');
  /** Version "10.11.10". */
  public static readonly VER_10_11_10 = MariaDbEngineVersion.of('10.11.10', '10.11');
  /** Version "10.11.11". */
  public static readonly VER_10_11_11 = MariaDbEngineVersion.of('10.11.11', '10.11');
  /** Version "10.11.13". */
  public static readonly VER_10_11_13 = MariaDbEngineVersion.of('10.11.13', '10.11');
  /** Version "10.11.14". */
  public static readonly VER_10_11_14 = MariaDbEngineVersion.of('10.11.14', '10.11');
  /** Version "10.11.15". */
  public static readonly VER_10_11_15 = MariaDbEngineVersion.of('10.11.15', '10.11');
  /** Version "10.11.16". */
  public static readonly VER_10_11_16 = MariaDbEngineVersion.of('10.11.16', '10.11');

  /** Version "11.4.3". */
  public static readonly VER_11_4_3 = MariaDbEngineVersion.of('11.4.3', '11.4');
  /** Version "11.4.4". */
  public static readonly VER_11_4_4 = MariaDbEngineVersion.of('11.4.4', '11.4');
  /** Version "11.4.5". */
  public static readonly VER_11_4_5 = MariaDbEngineVersion.of('11.4.5', '11.4');
  /** Version "11.4.7". */
  public static readonly VER_11_4_7 = MariaDbEngineVersion.of('11.4.7', '11.4');
  /** Version "11.4.8". */
  public static readonly VER_11_4_8 = MariaDbEngineVersion.of('11.4.8', '11.4');
  /** Version "11.4.9". */
  public static readonly VER_11_4_9 = MariaDbEngineVersion.of('11.4.9', '11.4');
  /** Version "11.4.10". */
  public static readonly VER_11_4_10 = MariaDbEngineVersion.of('11.4.10', '11.4');

  /** Version "11.8.3". */
  public static readonly VER_11_8_3 = MariaDbEngineVersion.of('11.8.3', '11.8');
  /** Version "11.8.5". */
  public static readonly VER_11_8_5 = MariaDbEngineVersion.of('11.8.5', '11.8');
  /** Version "11.8.6". */
  public static readonly VER_11_8_6 = MariaDbEngineVersion.of('11.8.6', '11.8');

  /**
   * Create a new MariaDbEngineVersion with an arbitrary version.
   *
   * @param mariaDbFullVersion the full version string,
   *   for example "10.5.28"
   * @param mariaDbMajorVersion the major version of the engine,
   *   for example "10.5"
   */
  public static of(mariaDbFullVersion: string, mariaDbMajorVersion: string): MariaDbEngineVersion {
    return new MariaDbEngineVersion(mariaDbFullVersion, mariaDbMajorVersion);
  }

  /** The full version string, for example, "10.5.28". */
  public readonly mariaDbFullVersion: string;
  /** The major version of the engine, for example, "10.5". */
  public readonly mariaDbMajorVersion: string;

  private constructor(mariaDbFullVersion: string, mariaDbMajorVersion: string) {
    this.mariaDbFullVersion = mariaDbFullVersion;
    this.mariaDbMajorVersion = mariaDbMajorVersion;
  }
}

/**
 * Properties for MariaDB instance engines.
 * Used in `DatabaseInstanceEngine.mariaDb`.
 */
export interface MariaDbInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: MariaDbEngineVersion;
}

class MariaDbInstanceEngine extends InstanceEngineBase {
  public readonly supportsReadReplicaBackups = true;

  constructor(version?: MariaDbEngineVersion) {
    super({
      engineType: 'mariadb',
      singleUserRotationApplication: secretsmanager.SecretRotationApplication.MARIADB_ROTATION_SINGLE_USER,
      multiUserRotationApplication: secretsmanager.SecretRotationApplication.MARIADB_ROTATION_MULTI_USER,
      version: version
        ? {
          fullVersion: version.mariaDbFullVersion,
          majorVersion: version.mariaDbMajorVersion,
        }
        : undefined,
      engineFamily: 'MYSQL',
    });
  }

  public bindToInstance(scope: Construct, options: InstanceEngineBindOptions): InstanceEngineConfig {
    if (options.domain) {
      throw new ValidationError(lit`DomainPropertyCannotConfigured`, `domain property cannot be configured for ${this.engineType}`, scope);
    }
    return super.bindToInstance(scope, options);
  }
}

/**
 * The versions for the MySQL instance engines
 * (those returned by `DatabaseInstanceEngine.mysql`).
 */
export class MysqlEngineVersion {
  /**
   * Version "5.5" (only a major version, without a specific minor version).
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5 = MysqlEngineVersion.of('5.5', '5.5');
  /**
   * Version "5.5.46".
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_46 = MysqlEngineVersion.of('5.5.46', '5.5');
  /**
   * Version "5.5.53".
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_53 = MysqlEngineVersion.of('5.5.53', '5.5');
  /**
   * Version "5.5.54"
   * @deprecated MySQL 5.5.54 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_54 = MysqlEngineVersion.of('5.5.54', '5.5');
  /**
   * Version "5.5.57".
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_57 = MysqlEngineVersion.of('5.5.57', '5.5');
  /**
   * Version "5.5.59".
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_59 = MysqlEngineVersion.of('5.5.59', '5.5');
  /**
   * Version "5.5.61".
   * @deprecated MySQL 5.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_5_61 = MysqlEngineVersion.of('5.5.61', '5.5');

  /**
   * Version "5.6" (only a major version, without a specific minor version).
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6 = MysqlEngineVersion.of('5.6', '5.6');
  /**
   * Version "5.6.34".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_34 = MysqlEngineVersion.of('5.6.34', '5.6');
  /**
   * Version "5.6.35".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_35 = MysqlEngineVersion.of('5.6.35', '5.6');
  /**
   * Version "5.6.37".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_37 = MysqlEngineVersion.of('5.6.37', '5.6');
  /**
   * Version "5.6.39".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_39 = MysqlEngineVersion.of('5.6.39', '5.6');
  /**
   * Version "5.6.40".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_40 = MysqlEngineVersion.of('5.6.40', '5.6');
  /**
   * Version "5.6.41".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_41 = MysqlEngineVersion.of('5.6.41', '5.6');
  /**
   * Version "5.6.43".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_43 = MysqlEngineVersion.of('5.6.43', '5.6');
  /**
   * Version "5.6.44".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_44 = MysqlEngineVersion.of('5.6.44', '5.6');
  /**
   * Version "5.6.46".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_46 = MysqlEngineVersion.of('5.6.46', '5.6');
  /**
   * Version "5.6.48".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_48 = MysqlEngineVersion.of('5.6.48', '5.6');
  /**
   * Version "5.6.49".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_49 = MysqlEngineVersion.of('5.6.49', '5.6');
  /**
   * Version "5.6.51".
   * @deprecated MySQL 5.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_6_51 = MysqlEngineVersion.of('5.6.51', '5.6');

  /** Version "5.7" (only a major version, without a specific minor version). */
  public static readonly VER_5_7 = MysqlEngineVersion.of('5.7', '5.7');
  /**
   * Version "5.7.16"
   * @deprecated MySQL 5.7.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_16 = MysqlEngineVersion.of('5.7.16', '5.7');
  /**
   * Version "5.7.17"
   * @deprecated MySQL 5.7.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_17 = MysqlEngineVersion.of('5.7.17', '5.7');
  /**
   * Version "5.7.19"
   * @deprecated MySQL 5.7.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_19 = MysqlEngineVersion.of('5.7.19', '5.7');
  /**
   * Version "5.7.21"
   * @deprecated MySQL 5.7.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_21 = MysqlEngineVersion.of('5.7.21', '5.7');
  /**
   * Version "5.7.22"
   * @deprecated MySQL 5.7.22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_22 = MysqlEngineVersion.of('5.7.22', '5.7');
  /**
   * Version "5.7.23"
   * @deprecated MySQL 5.7.23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_23 = MysqlEngineVersion.of('5.7.23', '5.7');
  /**
   * Version "5.7.24"
   * @deprecated MySQL 5.7.24 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_24 = MysqlEngineVersion.of('5.7.24', '5.7');
  /**
   * Version "5.7.25"
   * @deprecated MySQL 5.7.25 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_25 = MysqlEngineVersion.of('5.7.25', '5.7');
  /**
   * Version "5.7.26"
   * @deprecated MySQL 5.7.26 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_26 = MysqlEngineVersion.of('5.7.26', '5.7');
  /**
   * Version "5.7.28"
   * @deprecated MySQL 5.7.28 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_28 = MysqlEngineVersion.of('5.7.28', '5.7');
  /**
   * Version "5.7.30"
   * @deprecated MySQL 5.7.30 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_30 = MysqlEngineVersion.of('5.7.30', '5.7');
  /**
   * Version "5.7.31"
   * @deprecated MySQL 5.7.31 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_31 = MysqlEngineVersion.of('5.7.31', '5.7');
  /**
   * Version "5.7.33"
   * @deprecated MySQL 5.7.33 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_33 = MysqlEngineVersion.of('5.7.33', '5.7');
  /**
   * Version "5.7.34"
   * @deprecated MySQL 5.7.34 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_34 = MysqlEngineVersion.of('5.7.34', '5.7');
  /**
   * Version "5.7.35"
   * @deprecated MySQL 5.7.35 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_35 = MysqlEngineVersion.of('5.7.35', '5.7');
  /**
   * Version "5.7.36"
   * @deprecated MySQL 5.7.36 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_36 = MysqlEngineVersion.of('5.7.36', '5.7');
  /**
   * Version "5.7.37"
   * @deprecated MySQL 5.7.37 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_37 = MysqlEngineVersion.of('5.7.37', '5.7');
  /**
   * Version "5.7.38"
   * @deprecated MySQL 5.7.38 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_38 = MysqlEngineVersion.of('5.7.38', '5.7');
  /**
   * Version "5.7.39"
   * @deprecated MySQL 5.7.39 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_39 = MysqlEngineVersion.of('5.7.39', '5.7');
  /**
   * Version "5.7.40"
   * @deprecated MySQL 5.7.40 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_40 = MysqlEngineVersion.of('5.7.40', '5.7');
  /**
   * Version "5.7.41"
   * @deprecated MySQL 5.7.41 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_41 = MysqlEngineVersion.of('5.7.41', '5.7');
  /**
   * Version "5.7.42"
   * @deprecated MySQL 5.7.42 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_42 = MysqlEngineVersion.of('5.7.42', '5.7');
  /**
   * Version "5.7.43"
   * @deprecated MySQL 5.7.43 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_43 = MysqlEngineVersion.of('5.7.43', '5.7');
  /**
   * Version "5.7.44"
   * @deprecated MySQL 5.7.44 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_44 = MysqlEngineVersion.of('5.7.44', '5.7');
  /**
   * Version "5.7.44-rds.20240408"
   * @deprecated MySQL 5.7.44-rds.20240408 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_44_RDS_20240408 = MysqlEngineVersion.of('5.7.44-rds.20240408', '5.7');
  /**
   * Version "5.7.44-rds.20240529"
   * @deprecated MySQL 5.7.44-rds.20240529 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_44_RDS_20240529 = MysqlEngineVersion.of('5.7.44-rds.20240529', '5.7');
  /**
   * Version "5.7.44-rds.20240808"
   * @deprecated MySQL 5.7.44-rds.20240808 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_44_RDS_20240808 = MysqlEngineVersion.of('5.7.44-rds.20240808', '5.7');
  /**
   * Version "5.7.44-rds.20250103"
   * @deprecated MySQL 5.7.44-rds.20250103 is no longer supported by Amazon RDS.
   */
  public static readonly VER_5_7_44_RDS_20250103 = MysqlEngineVersion.of('5.7.44-rds.20250103', '5.7');
  /** Version "5.7.44-rds.20250213". */
  public static readonly VER_5_7_44_RDS_20250213 = MysqlEngineVersion.of('5.7.44-rds.20250213', '5.7');
  /** Version "5.7.44-rds.20250508". */
  public static readonly VER_5_7_44_RDS_20250508 = MysqlEngineVersion.of('5.7.44-rds.20250508', '5.7');
  /** Version "5.7.44-rds.20250818". */
  public static readonly VER_5_7_44_RDS_20250818 = MysqlEngineVersion.of('5.7.44-rds.20250818', '5.7');
  /** Version "5.7.44-rds.20251212". */
  public static readonly VER_5_7_44_RDS_20251212 = MysqlEngineVersion.of('5.7.44-rds.20251212', '5.7');
  /** Version "5.7.44-rds.20260212". */
  public static readonly VER_5_7_44_RDS_20260212 = MysqlEngineVersion.of('5.7.44-rds.20260212', '5.7');
  /** Version "5.7.44-rds.20260521". */
  public static readonly VER_5_7_44_RDS_20260521 = MysqlEngineVersion.of('5.7.44-rds.20260521', '5.7');

  /** Version "8.0" (only a major version, without a specific minor version). */
  public static readonly VER_8_0 = MysqlEngineVersion.of('8.0', '8.0');
  /**
   * Version "8.0.11"
   * @deprecated MySQL 8.0.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_11 = MysqlEngineVersion.of('8.0.11', '8.0');
  /**
   * Version "8.0.13"
   * @deprecated MySQL 8.0.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_13 = MysqlEngineVersion.of('8.0.13', '8.0');
  /**
   * Version "8.0.15"
   * @deprecated MySQL 8.0.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_15 = MysqlEngineVersion.of('8.0.15', '8.0');
  /**
   * Version "8.0.16"
   * @deprecated MySQL 8.0.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_16 = MysqlEngineVersion.of('8.0.16', '8.0');
  /**
   * Version "8.0.17"
   * @deprecated MySQL 8.0.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_17 = MysqlEngineVersion.of('8.0.17', '8.0');
  /**
   * Version "8.0.19"
   * @deprecated MySQL 8.0.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_19 = MysqlEngineVersion.of('8.0.19', '8.0');
  /**
   * Version "8.0.20"
   * @deprecated MySQL 8.0.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_20 = MysqlEngineVersion.of('8.0.20', '8.0');
  /**
   * Version "8.0.21"
   * @deprecated MySQL 8.0.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_21 = MysqlEngineVersion.of('8.0.21', '8.0');
  /**
   * Version "8.0.23"
   * @deprecated MySQL 8.0.23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_23 = MysqlEngineVersion.of('8.0.23', '8.0');
  /**
   * Version "8.0.25"
   * @deprecated MySQL 8.0.25 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_25 = MysqlEngineVersion.of('8.0.25', '8.0');
  /**
   * Version "8.0.26"
   * @deprecated MySQL 8.0.26 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_26 = MysqlEngineVersion.of('8.0.26', '8.0');
  /**
   * Version "8.0.27"
   * @deprecated MySQL 8.0.27 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_27 = MysqlEngineVersion.of('8.0.27', '8.0');
  /**
   * Version "8.0.28"
   * @deprecated MySQL 8.0.28 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_28 = MysqlEngineVersion.of('8.0.28', '8.0');
  /**
   * Version "8.0.29"
   * @deprecated MySQL 8.0.29 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_29 = MysqlEngineVersion.of('8.0.29', '8.0');
  /**
   * Version "8.0.30"
   * @deprecated MySQL 8.0.30 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_30 = MysqlEngineVersion.of('8.0.30', '8.0');
  /**
   * Version "8.0.31"
   * @deprecated MySQL 8.0.31 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_31 = MysqlEngineVersion.of('8.0.31', '8.0');
  /**
   * Version "8.0.32"
   * @deprecated MySQL 8.0.32 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_32 = MysqlEngineVersion.of('8.0.32', '8.0');
  /**
   * Version "8.0.33"
   * @deprecated MySQL 8.0.33 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_33 = MysqlEngineVersion.of('8.0.33', '8.0');
  /**
   * Version "8.0.34"
   * @deprecated MySQL 8.0.34 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_34 = MysqlEngineVersion.of('8.0.34', '8.0');
  /**
   * Version "8.0.35"
   * @deprecated MySQL 8.0.35 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_35 = MysqlEngineVersion.of('8.0.35', '8.0');
  /**
   * Version "8.0.36"
   * @deprecated MySQL 8.0.36 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_36 = MysqlEngineVersion.of('8.0.36', '8.0');
  /**
   * Version "8.0.37".
   * @deprecated MySQL 8.0.37 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_37 = MysqlEngineVersion.of('8.0.37', '8.0');
  /**
   * Version "8.0.39".
   * @deprecated MySQL 8.0.39 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_39 = MysqlEngineVersion.of('8.0.39', '8.0');
  /**
   * Version "8.0.40"
   * @deprecated MySQL 8.0.40 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_40 = MysqlEngineVersion.of('8.0.40', '8.0');
  /**
   * Version "8.0.41"
   * @deprecated MySQL 8.0.41 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_0_41 = MysqlEngineVersion.of('8.0.41', '8.0');
  /** Version "8.0.42". */
  public static readonly VER_8_0_42 = MysqlEngineVersion.of('8.0.42', '8.0');
  /** Version "8.0.43". */
  public static readonly VER_8_0_43 = MysqlEngineVersion.of('8.0.43', '8.0');
  /** Version "8.0.44". */
  public static readonly VER_8_0_44 = MysqlEngineVersion.of('8.0.44', '8.0');
  /** Version "8.0.45". */
  public static readonly VER_8_0_45 = MysqlEngineVersion.of('8.0.45', '8.0');
  /** Version "8.0.46". */
  public static readonly VER_8_0_46 = MysqlEngineVersion.of('8.0.46', '8.0');
  /**
   * Version "8.4.3"
   * @deprecated MySQL 8.4.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_4_3 = MysqlEngineVersion.of('8.4.3', '8.4');
  /**
   * Version "8.4.4"
   * @deprecated MySQL 8.4.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_8_4_4 = MysqlEngineVersion.of('8.4.4', '8.4');
  /** Version "8.4.5". */
  public static readonly VER_8_4_5 = MysqlEngineVersion.of('8.4.5', '8.4');
  /** Version "8.4.6". */
  public static readonly VER_8_4_6 = MysqlEngineVersion.of('8.4.6', '8.4');
  /** Version "8.4.7". */
  public static readonly VER_8_4_7 = MysqlEngineVersion.of('8.4.7', '8.4');
  /** Version "8.4.8". */
  public static readonly VER_8_4_8 = MysqlEngineVersion.of('8.4.8', '8.4');
  /** Version "8.4.9". */
  public static readonly VER_8_4_9 = MysqlEngineVersion.of('8.4.9', '8.4');
  /** Version "8.4.10". */
  public static readonly VER_8_4_10 = MysqlEngineVersion.of('8.4.10', '8.4');

  /**
   * Create a new MysqlEngineVersion with an arbitrary version.
   *
   * @param mysqlFullVersion the full version string,
   *   for example "8.1.43"
   * @param mysqlMajorVersion the major version of the engine,
   *   for example "8.1"
   */
  public static of(mysqlFullVersion: string, mysqlMajorVersion: string): MysqlEngineVersion {
    return new MysqlEngineVersion(mysqlFullVersion, mysqlMajorVersion);
  }

  /** The full version string, for example, "10.5.28". */
  public readonly mysqlFullVersion: string;
  /** The major version of the engine, for example, "10.5". */
  public readonly mysqlMajorVersion: string;

  private constructor(mysqlFullVersion: string, mysqlMajorVersion: string) {
    this.mysqlFullVersion = mysqlFullVersion;
    this.mysqlMajorVersion = mysqlMajorVersion;
  }
}

/**
 * Properties for MySQL instance engines.
 * Used in `DatabaseInstanceEngine.mysql`.
 */
export interface MySqlInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: MysqlEngineVersion;
}

class MySqlInstanceEngine extends InstanceEngineBase {
  public readonly supportsReadReplicaBackups = true;

  constructor(version?: MysqlEngineVersion) {
    super({
      engineType: 'mysql',
      singleUserRotationApplication: secretsmanager.SecretRotationApplication.MYSQL_ROTATION_SINGLE_USER,
      multiUserRotationApplication: secretsmanager.SecretRotationApplication.MYSQL_ROTATION_MULTI_USER,
      version: version
        ? {
          fullVersion: version.mysqlFullVersion,
          majorVersion: version.mysqlMajorVersion,
        }
        : undefined,
      engineFamily: 'MYSQL',
    });
  }
}

/**
 * Features supported by the Postgres database engine
 */
export interface PostgresEngineFeatures {
  /**
   * Whether this version of the Postgres engine supports the S3 data import feature.
   *
   * @default false
   */
  readonly s3Import?: boolean;

  /**
   * Whether this version of the Postgres engine supports the S3 data export feature.
   *
   * @default false
   */
  readonly s3Export?: boolean;
}

/**
 * The versions for the PostgreSQL instance engines
 * (those returned by `DatabaseInstanceEngine.postgres`).
 */
export class PostgresEngineVersion {
  /**
   * Version "9.5" (only a major version, without a specific minor version).
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5 = PostgresEngineVersion.of('9.5', '9.5');
  /**
   * Version "9.5.2".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_2 = PostgresEngineVersion.of('9.5.2', '9.5');
  /**
   * Version "9.5.4".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_4 = PostgresEngineVersion.of('9.5.4', '9.5');
  /**
   * Version "9.5.6".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_6 = PostgresEngineVersion.of('9.5.6', '9.5');
  /**
   * Version "9.5.7".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_7 = PostgresEngineVersion.of('9.5.7', '9.5');
  /**
   * Version "9.5.9".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_9 = PostgresEngineVersion.of('9.5.9', '9.5');
  /**
   * Version "9.5.10".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_10 = PostgresEngineVersion.of('9.5.10', '9.5');
  /**
   * Version "9.5.12".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_12 = PostgresEngineVersion.of('9.5.12', '9.5');
  /**
   * Version "9.5.13".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_13 = PostgresEngineVersion.of('9.5.13', '9.5');
  /**
   * Version "9.5.14".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_14 = PostgresEngineVersion.of('9.5.14', '9.5');
  /**
   * Version "9.5.15".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_15 = PostgresEngineVersion.of('9.5.15', '9.5');
  /**
   * Version "9.5.16".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_16 = PostgresEngineVersion.of('9.5.16', '9.5');
  /**
   * Version "9.5.18".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_18 = PostgresEngineVersion.of('9.5.18', '9.5');
  /**
   * Version "9.5.19".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_19 = PostgresEngineVersion.of('9.5.19', '9.5');
  /**
   * Version "9.5.20".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_20 = PostgresEngineVersion.of('9.5.20', '9.5');
  /**
   * Version "9.5.21".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_21 = PostgresEngineVersion.of('9.5.21', '9.5');
  /**
   * Version "9.5.22".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_22 = PostgresEngineVersion.of('9.5.22', '9.5');
  /**
   * Version "9.5.23".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_23 = PostgresEngineVersion.of('9.5.23', '9.5');
  /**
   * Version "9.5.24".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_24 = PostgresEngineVersion.of('9.5.24', '9.5');
  /**
   * Version "9.5.25".
   * @deprecated PostgreSQL 9.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_5_25 = PostgresEngineVersion.of('9.5.25', '9.5');

  /**
   * Version "9.6" (only a major version, without a specific minor version).
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6 = PostgresEngineVersion.of('9.6', '9.6');
  /**
   * Version "9.6.1".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_1 = PostgresEngineVersion.of('9.6.1', '9.6');
  /**
   * Version "9.6.2".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_2 = PostgresEngineVersion.of('9.6.2', '9.6');
  /**
   * Version "9.6.3".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_3 = PostgresEngineVersion.of('9.6.3', '9.6');
  /**
   * Version "9.6.5".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_5 = PostgresEngineVersion.of('9.6.5', '9.6');
  /**
   * Version "9.6.6".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_6 = PostgresEngineVersion.of('9.6.6', '9.6');
  /**
   * Version "9.6.8".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_8 = PostgresEngineVersion.of('9.6.8', '9.6');
  /**
   * Version "9.6.9".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_9 = PostgresEngineVersion.of('9.6.9', '9.6');
  /**
   * Version "9.6.10".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_10 = PostgresEngineVersion.of('9.6.10', '9.6');
  /**
   * Version "9.6.11".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_11 = PostgresEngineVersion.of('9.6.11', '9.6');
  /**
   * Version "9.6.12".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_12 = PostgresEngineVersion.of('9.6.12', '9.6');
  /**
   * Version "9.6.14".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_14 = PostgresEngineVersion.of('9.6.14', '9.6');
  /**
   * Version "9.6.15".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_15 = PostgresEngineVersion.of('9.6.15', '9.6');
  /**
   * Version "9.6.16".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_16 = PostgresEngineVersion.of('9.6.16', '9.6');
  /**
   * Version "9.6.17".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_17 = PostgresEngineVersion.of('9.6.17', '9.6');
  /**
   * Version "9.6.18".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_18 = PostgresEngineVersion.of('9.6.18', '9.6');
  /**
   * Version "9.6.19".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_19 = PostgresEngineVersion.of('9.6.19', '9.6');
  /**
   * Version "9.6.20".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_20 = PostgresEngineVersion.of('9.6.20', '9.6');
  /**
   * Version "9.6.21".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_21 = PostgresEngineVersion.of('9.6.21', '9.6');
  /**
   * Version "9.6.22".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_22 = PostgresEngineVersion.of('9.6.22', '9.6');
  /**
   * Version "9.6.23".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_23 = PostgresEngineVersion.of('9.6.23', '9.6');
  /**
   * Version "9.6.24".
   * @deprecated PostgreSQL 9.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_9_6_24 = PostgresEngineVersion.of('9.6.24', '9.6');

  /**
   * Version "10" (only a major version, without a specific minor version).
   * @deprecated PostgreSQL 10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10 = PostgresEngineVersion.of('10', '10');
  /**
   * Version "10.1".
   * @deprecated PostgreSQL 10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_1 = PostgresEngineVersion.of('10.1', '10');
  /**
   * Version "10.3".
   * @deprecated PostgreSQL 10.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_3 = PostgresEngineVersion.of('10.3', '10');
  /**
   * Version "10.4".
   * @deprecated PostgreSQL 10.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_4 = PostgresEngineVersion.of('10.4', '10');
  /**
   * Version "10.5".
   * @deprecated PostgreSQL 10.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_5 = PostgresEngineVersion.of('10.5', '10');
  /**
   * Version "10.6".
   * @deprecated PostgreSQL 10.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_6 = PostgresEngineVersion.of('10.6', '10');
  /**
   * Version "10.7".
   * @deprecated PostgreSQL 10.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_7 = PostgresEngineVersion.of('10.7', '10', { s3Import: true });
  /**
   * Version "10.9".
   * @deprecated PostgreSQL 10.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_9 = PostgresEngineVersion.of('10.9', '10', { s3Import: true });
  /**
   * Version "10.10".
   * @deprecated PostgreSQL 10.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_10 = PostgresEngineVersion.of('10.10', '10', { s3Import: true });
  /**
   * Version "10.11".
   * @deprecated PostgreSQL 10.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_11 = PostgresEngineVersion.of('10.11', '10', { s3Import: true });
  /**
   * Version "10.12".
   * @deprecated PostgreSQL 10.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_12 = PostgresEngineVersion.of('10.12', '10', { s3Import: true });
  /**
   * Version "10.13".
   * @deprecated PostgreSQL 10.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_13 = PostgresEngineVersion.of('10.13', '10', { s3Import: true });
  /**
   * Version "10.14".
   * @deprecated PostgreSQL 10.14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_14 = PostgresEngineVersion.of('10.14', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.15".
   * @deprecated PostgreSQL 10.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_15 = PostgresEngineVersion.of('10.15', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.16".
   * @deprecated PostgreSQL 10.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_16 = PostgresEngineVersion.of('10.16', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.17".
   * @deprecated PostgreSQL 10.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_17 = PostgresEngineVersion.of('10.17', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.18".
   * @deprecated PostgreSQL 10.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_18 = PostgresEngineVersion.of('10.18', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.19".
   * @deprecated PostgreSQL 10.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_19 = PostgresEngineVersion.of('10.19', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.20".
   * @deprecated PostgreSQL 10.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_20 = PostgresEngineVersion.of('10.20', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.21".
   * @deprecated PostgreSQL 10.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_21 = PostgresEngineVersion.of('10.21', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.22".
   * @deprecated PostgreSQL 10.22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_22 = PostgresEngineVersion.of('10.22', '10', { s3Import: true, s3Export: true });
  /**
   * Version "10.23".
   * @deprecated PostgreSQL 10.23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_10_23 = PostgresEngineVersion.of('10.23', '10', { s3Import: true, s3Export: true });

  /** Version "11" (only a major version, without a specific minor version). */
  public static readonly VER_11 = PostgresEngineVersion.of('11', '11', { s3Import: true });
  /**
   * Version "11.1".
   * @deprecated PostgreSQL 11.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_1 = PostgresEngineVersion.of('11.1', '11', { s3Import: true });
  /**
   * Version "11.2".
   * @deprecated PostgreSQL 11.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_2 = PostgresEngineVersion.of('11.2', '11', { s3Import: true });
  /**
   * Version "11.4".
   * @deprecated PostgreSQL 11.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_4 = PostgresEngineVersion.of('11.4', '11', { s3Import: true });
  /**
   * Version "11.5".
   * @deprecated PostgreSQL 11.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_5 = PostgresEngineVersion.of('11.5', '11', { s3Import: true });
  /**
   * Version "11.6".
   * @deprecated PostgreSQL 11.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_6 = PostgresEngineVersion.of('11.6', '11', { s3Import: true });
  /**
   * Version "11.7".
   * @deprecated PostgreSQL 11.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_7 = PostgresEngineVersion.of('11.7', '11', { s3Import: true });
  /**
   * Version "11.8".
   * @deprecated PostgreSQL 11.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_8 = PostgresEngineVersion.of('11.8', '11', { s3Import: true });
  /**
   * Version "11.9".
   * @deprecated PostgreSQL 11.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_9 = PostgresEngineVersion.of('11.9', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.10"
   * @deprecated PostgreSQL 11.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_10 = PostgresEngineVersion.of('11.10', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.11"
   * @deprecated PostgreSQL 11.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_11 = PostgresEngineVersion.of('11.11', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.12"
   * @deprecated PostgreSQL 11.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_12 = PostgresEngineVersion.of('11.12', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.13"
   * @deprecated PostgreSQL 11.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_13 = PostgresEngineVersion.of('11.13', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.14"
   * @deprecated PostgreSQL 11.14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_14 = PostgresEngineVersion.of('11.14', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.15"
   * @deprecated PostgreSQL 11.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_15 = PostgresEngineVersion.of('11.15', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.16"
   * @deprecated PostgreSQL 11.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_16 = PostgresEngineVersion.of('11.16', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.17"
   * @deprecated PostgreSQL 11.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_17 = PostgresEngineVersion.of('11.17', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.18"
   * @deprecated PostgreSQL 11.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_18 = PostgresEngineVersion.of('11.18', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.19"
   * @deprecated PostgreSQL 11.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_19 = PostgresEngineVersion.of('11.19', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.20"
   * @deprecated PostgreSQL 11.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_20 = PostgresEngineVersion.of('11.20', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.21"
   * @deprecated PostgreSQL 11.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_21 = PostgresEngineVersion.of('11.21', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22"
   * @deprecated PostgreSQL 11.22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22 = PostgresEngineVersion.of('11.22', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20240418"
   */
  public static readonly VER_11_22_RDS_20240418 = PostgresEngineVersion.of('11.22-rds.20240418', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20240509"
   */
  public static readonly VER_11_22_RDS_20240509 = PostgresEngineVersion.of('11.22-rds.20240509', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20240808"
   * @deprecated PostgreSQL 11.22-rds.20240808 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22_RDS_20240808 = PostgresEngineVersion.of('11.22-RDS.20240808', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-RDS.20241121"
   * @deprecated PostgreSQL 11.22-RDS.20241121 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22_RDS_20241121 = PostgresEngineVersion.of('11.22-RDS.20241121', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20250220"
   * @deprecated PostgreSQL 11.22-rds.20250220 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22_RDS_20250220 = PostgresEngineVersion.of('11.22-rds.20250220', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20250508"
   * @deprecated PostgreSQL 11.22-rds.20250508 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22_RDS_20250508 = PostgresEngineVersion.of('11.22-rds.20250508', '11', { s3Import: true, s3Export: true });
  /**
   * Version "11.22-rds.20250814"
   * @deprecated PostgreSQL 11.22-rds.20250814 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_22_RDS_20250814 = PostgresEngineVersion.of('11.22-rds.20250814', '11', { s3Import: true, s3Export: true });
  /** Version "11.22-rds.20251114". */
  public static readonly VER_11_22_RDS_20251114 = PostgresEngineVersion.of('11.22-rds.20251114', '11', { s3Import: true, s3Export: true });

  /** Version "12" (only a major version, without a specific minor version). */
  public static readonly VER_12 = PostgresEngineVersion.of('12', '12', { s3Import: true });
  /**
   * Version "12.2".
   * @deprecated PostgreSQL 12.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2 = PostgresEngineVersion.of('12.2', '12', { s3Import: true });
  /**
   * Version "12.3".
   * @deprecated PostgreSQL 12.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_3 = PostgresEngineVersion.of('12.3', '12', { s3Import: true });
  /**
   * Version "12.4".
   * @deprecated PostgreSQL 12.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_4 = PostgresEngineVersion.of('12.4', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.5".
   * @deprecated PostgreSQL 12.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_5 = PostgresEngineVersion.of('12.5', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.6".
   * @deprecated PostgreSQL 12.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_6 = PostgresEngineVersion.of('12.6', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.7".
   * @deprecated PostgreSQL 12.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_7 = PostgresEngineVersion.of('12.7', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.8".
   * @deprecated PostgreSQL 12.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_8 = PostgresEngineVersion.of('12.8', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.9".
   * @deprecated PostgreSQL 12.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_9 = PostgresEngineVersion.of('12.9', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.10"
   * @deprecated PostgreSQL 12.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_10 = PostgresEngineVersion.of('12.10', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.11"
   * @deprecated PostgreSQL 12.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_11 = PostgresEngineVersion.of('12.11', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.12"
   * @deprecated PostgreSQL 12.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_12 = PostgresEngineVersion.of('12.12', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.13"
   * @deprecated PostgreSQL 12.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_13 = PostgresEngineVersion.of('12.13', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.14"
   * @deprecated PostgreSQL 12.14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_14 = PostgresEngineVersion.of('12.14', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.15"
   * @deprecated PostgreSQL 12.15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_15 = PostgresEngineVersion.of('12.15', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.16"
   * @deprecated PostgreSQL 12.16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_16 = PostgresEngineVersion.of('12.16', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.17"
   * @deprecated PostgreSQL 12.17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_17 = PostgresEngineVersion.of('12.17', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.18"
   * @deprecated PostgreSQL 12.18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_18 = PostgresEngineVersion.of('12.18', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.19"
   * @deprecated PostgreSQL 12.19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_19 = PostgresEngineVersion.of('12.19', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.20"
   * @deprecated PostgreSQL 12.20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_20 = PostgresEngineVersion.of('12.20', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.21"
   * @deprecated PostgreSQL 12.21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_21 = PostgresEngineVersion.of('12.21', '12', { s3Import: true, s3Export: true });
  /** Version "12.22". */
  public static readonly VER_12_22 = PostgresEngineVersion.of('12.22', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.22-rds.20250220".
   * @deprecated PostgreSQL 12.22-rds.20250220 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_22_RDS_20250220 = PostgresEngineVersion.of('12.22-rds.20250220', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.22-rds.20250508".
   * @deprecated PostgreSQL 12.22-rds.20250508 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_22_RDS_20250508 = PostgresEngineVersion.of('12.22-rds.20250508', '12', { s3Import: true, s3Export: true });
  /**
   * Version "12.22-rds.20250814".
   * @deprecated PostgreSQL 12.22-rds.20250814 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_22_RDS_20250814 = PostgresEngineVersion.of('12.22-rds.20250814', '12', { s3Import: true, s3Export: true });
  /** Version "12.22-rds.20251114". */
  public static readonly VER_12_22_RDS_20251114 = PostgresEngineVersion.of('12.22-rds.20251114', '12', { s3Import: true, s3Export: true });

  /** Version "13" (only a major version, without a specific minor version). */
  public static readonly VER_13 = PostgresEngineVersion.of('13', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.1".
   * @deprecated PostgreSQL 13.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_1 = PostgresEngineVersion.of('13.1', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.2".
   * @deprecated PostgreSQL 13.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_2 = PostgresEngineVersion.of('13.2', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.3".
   * @deprecated PostgreSQL 13.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_3 = PostgresEngineVersion.of('13.3', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.4".
   * @deprecated PostgreSQL 13.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_4 = PostgresEngineVersion.of('13.4', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.5".
   * @deprecated PostgreSQL 13.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_5 = PostgresEngineVersion.of('13.5', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.6".
   * @deprecated PostgreSQL 13.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_6 = PostgresEngineVersion.of('13.6', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.7"
   * @deprecated PostgreSQL 13.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_7 = PostgresEngineVersion.of('13.7', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.8"
   * @deprecated PostgreSQL 13.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_8 = PostgresEngineVersion.of('13.8', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.9"
   * @deprecated PostgreSQL 13.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_9 = PostgresEngineVersion.of('13.9', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.10"
   * @deprecated PostgreSQL 13.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_10 = PostgresEngineVersion.of('13.10', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.11".
   * @deprecated PostgreSQL 13.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_11 = PostgresEngineVersion.of('13.11', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.12".
   * @deprecated PostgreSQL 13.12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_12 = PostgresEngineVersion.of('13.12', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.13".
   * @deprecated PostgreSQL 13.13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_13 = PostgresEngineVersion.of('13.13', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.14".
   * @deprecated PostgreSQL 13.14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_14 = PostgresEngineVersion.of('13.14', '13', { s3Import: true, s3Export: true });
  /** Version "13.15". */
  public static readonly VER_13_15 = PostgresEngineVersion.of('13.15', '13', { s3Import: true, s3Export: true });
  /** Version "13.16". */
  public static readonly VER_13_16 = PostgresEngineVersion.of('13.16', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.17".
   * @deprecated PostgreSQL 13.17 is no longer supported by Amazon RDS
   */
  public static readonly VER_13_17 = PostgresEngineVersion.of('13.17', '13', { s3Import: true, s3Export: true });
  /** Version "13.18". */
  public static readonly VER_13_18 = PostgresEngineVersion.of('13.18', '13', { s3Import: true, s3Export: true });
  /**
   * Version "13.19".
   * @deprecated PostgreSQL 13.19 is no longer supported by Amazon RDS
   */
  public static readonly VER_13_19 = PostgresEngineVersion.of('13.19', '13', { s3Import: true, s3Export: true });
  /** Version "13.20". */
  public static readonly VER_13_20 = PostgresEngineVersion.of('13.20', '13', { s3Import: true, s3Export: true });
  /** Version "13.21". */
  public static readonly VER_13_21 = PostgresEngineVersion.of('13.21', '13', { s3Import: true, s3Export: true });
  /** Version "13.22". */
  public static readonly VER_13_22 = PostgresEngineVersion.of('13.22', '13', { s3Import: true, s3Export: true });
  /** Version "13.23". */
  public static readonly VER_13_23 = PostgresEngineVersion.of('13.23', '13', { s3Import: true, s3Export: true });

  /** Version "14" (only a major version, without a specific minor version). */
  public static readonly VER_14 = PostgresEngineVersion.of('14', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.1".
   * @deprecated PostgreSQL 14.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_1 = PostgresEngineVersion.of('14.1', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.2".
   * @deprecated PostgreSQL 14.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_2 = PostgresEngineVersion.of('14.2', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.3"
   * @deprecated PostgreSQL 14.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_3 = PostgresEngineVersion.of('14.3', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.4"
   * @deprecated PostgreSQL 14.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_4 = PostgresEngineVersion.of('14.4', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.5"
   * @deprecated PostgreSQL 14.5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_5 = PostgresEngineVersion.of('14.5', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.6"
   * @deprecated PostgreSQL 14.6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_6 = PostgresEngineVersion.of('14.6', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.7"
   * @deprecated PostgreSQL 14.7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_7 = PostgresEngineVersion.of('14.7', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.8"
   * @deprecated PostgreSQL 14.8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_8 = PostgresEngineVersion.of('14.8', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.9".
   * @deprecated PostgreSQL 14.9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_9 = PostgresEngineVersion.of('14.9', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.10".
   * @deprecated PostgreSQL 14.10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_10 = PostgresEngineVersion.of('14.10', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.11".
   * @deprecated PostgreSQL 14.11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_11 = PostgresEngineVersion.of('14.11', '14', { s3Import: true, s3Export: true });
  /** Version "14.12". */
  public static readonly VER_14_12 = PostgresEngineVersion.of('14.12', '14', { s3Import: true, s3Export: true });
  /** Version "14.13". */
  public static readonly VER_14_13 = PostgresEngineVersion.of('14.13', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.14".
   * @deprecated PostgreSQL 14.14 is no longer supported by Amazon RDS
   */
  public static readonly VER_14_14 = PostgresEngineVersion.of('14.14', '14', { s3Import: true, s3Export: true });
  /** Version "14.15". */
  public static readonly VER_14_15 = PostgresEngineVersion.of('14.15', '14', { s3Import: true, s3Export: true });
  /**
   * Version "14.16".
   * @deprecated PostgreSQL 14.16 is no longer supported by Amazon RDS
   */
  public static readonly VER_14_16 = PostgresEngineVersion.of('14.16', '14', { s3Import: true, s3Export: true });
  /** Version "14.17". */
  public static readonly VER_14_17 = PostgresEngineVersion.of('14.17', '14', { s3Import: true, s3Export: true });
  /** Version "14.18". */
  public static readonly VER_14_18 = PostgresEngineVersion.of('14.18', '14', { s3Import: true, s3Export: true });
  /** Version "14.19". */
  public static readonly VER_14_19 = PostgresEngineVersion.of('14.19', '14', { s3Import: true, s3Export: true });
  /** Version "14.20". */
  public static readonly VER_14_20 = PostgresEngineVersion.of('14.20', '14', { s3Import: true, s3Export: true });
  /** Version "14.21". */
  public static readonly VER_14_21 = PostgresEngineVersion.of('14.21', '14', { s3Import: true, s3Export: true });
  /** Version "14.22". */
  public static readonly VER_14_22 = PostgresEngineVersion.of('14.22', '14', { s3Import: true, s3Export: true });

  /** Version "15" (only a major version, without a specific minor version). */
  public static readonly VER_15 = PostgresEngineVersion.of('15', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.2"
   * @deprecated PostgreSQL 15.2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_15_2 = PostgresEngineVersion.of('15.2', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.3"
   * @deprecated PostgreSQL 15.3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_15_3 = PostgresEngineVersion.of('15.3', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.4".
   * @deprecated PostgreSQL 15.4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_15_4 = PostgresEngineVersion.of('15.4', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.5".
   * @deprecated PostgreSQL 15.5 is no longer supported by Amazon RDS
   */
  public static readonly VER_15_5 = PostgresEngineVersion.of('15.5', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.6".
   * @deprecated PostgreSQL 15.6 is no longer supported by Amazon RDS
   */
  public static readonly VER_15_6 = PostgresEngineVersion.of('15.6', '15', { s3Import: true, s3Export: true });
  /** Version "15.7". */
  public static readonly VER_15_7 = PostgresEngineVersion.of('15.7', '15', { s3Import: true, s3Export: true });
  /** Version "15.8". */
  public static readonly VER_15_8 = PostgresEngineVersion.of('15.8', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.9".
   * @deprecated PostgreSQL 15.9 is no longer supported by Amazon RDS
   */
  public static readonly VER_15_9 = PostgresEngineVersion.of('15.9', '15', { s3Import: true, s3Export: true });
  /** Version "15.10". */
  public static readonly VER_15_10 = PostgresEngineVersion.of('15.10', '15', { s3Import: true, s3Export: true });
  /**
   * Version "15.11".
   * @deprecated PostgreSQL 15.11 is no longer supported by Amazon RDS
   */
  public static readonly VER_15_11 = PostgresEngineVersion.of('15.11', '15', { s3Import: true, s3Export: true });
  /** Version "15.12". */
  public static readonly VER_15_12 = PostgresEngineVersion.of('15.12', '15', { s3Import: true, s3Export: true });
  /** Version "15.13". */
  public static readonly VER_15_13 = PostgresEngineVersion.of('15.13', '15', { s3Import: true, s3Export: true });
  /** Version "15.14". */
  public static readonly VER_15_14 = PostgresEngineVersion.of('15.14', '15', { s3Import: true, s3Export: true });
  /** Version "15.15". */
  public static readonly VER_15_15 = PostgresEngineVersion.of('15.15', '15', { s3Import: true, s3Export: true });
  /** Version "15.16". */
  public static readonly VER_15_16 = PostgresEngineVersion.of('15.16', '15', { s3Import: true, s3Export: true });
  /** Version "15.17". */
  public static readonly VER_15_17 = PostgresEngineVersion.of('15.17', '15', { s3Import: true, s3Export: true });

  /** Version "16" (only a major version, without a specific minor version). */
  public static readonly VER_16 = PostgresEngineVersion.of('16', '16', { s3Import: true, s3Export: true });
  /**
   * Version "16.1".
   * @deprecated PostgreSQL 16.1 is no longer supported by Amazon RDS
   */
  public static readonly VER_16_1 = PostgresEngineVersion.of('16.1', '16', { s3Import: true, s3Export: true });
  /**
   * Version "16.2".
   * @deprecated PostgreSQL 16.2 is no longer supported by Amazon RDS
   */
  public static readonly VER_16_2 = PostgresEngineVersion.of('16.2', '16', { s3Import: true, s3Export: true });
  /** Version "16.3". */
  public static readonly VER_16_3 = PostgresEngineVersion.of('16.3', '16', { s3Import: true, s3Export: true });
  /** Version "16.4". */
  public static readonly VER_16_4 = PostgresEngineVersion.of('16.4', '16', { s3Import: true, s3Export: true });
  /**
   * Version "16.5".
   * @deprecated PostgreSQL 16.5 is no longer supported by Amazon RDS
   */
  public static readonly VER_16_5 = PostgresEngineVersion.of('16.5', '16', { s3Import: true, s3Export: true });
  /** Version "16.6". */
  public static readonly VER_16_6 = PostgresEngineVersion.of('16.6', '16', { s3Import: true, s3Export: true });
  /**
   * Version "16.7".
   * @deprecated PostgreSQL 16.7 is no longer supported by Amazon RDS
   */
  public static readonly VER_16_7 = PostgresEngineVersion.of('16.7', '16', { s3Import: true, s3Export: true });
  /** Version "16.8". */
  public static readonly VER_16_8 = PostgresEngineVersion.of('16.8', '16', { s3Import: true, s3Export: true });
  /** Version "16.9". */
  public static readonly VER_16_9 = PostgresEngineVersion.of('16.9', '16', { s3Import: true, s3Export: true });
  /** Version "16.10". */
  public static readonly VER_16_10 = PostgresEngineVersion.of('16.10', '16', { s3Import: true, s3Export: true });
  /** Version "16.11". */
  public static readonly VER_16_11 = PostgresEngineVersion.of('16.11', '16', { s3Import: true, s3Export: true });
  /** Version "16.12". */
  public static readonly VER_16_12 = PostgresEngineVersion.of('16.12', '16', { s3Import: true, s3Export: true });
  /** Version "16.13". */
  public static readonly VER_16_13 = PostgresEngineVersion.of('16.13', '16', { s3Import: true, s3Export: true });

  /** Version "17" (only a major version, without a specific minor version). */
  public static readonly VER_17 = PostgresEngineVersion.of('17', '17', { s3Import: true, s3Export: true });
  /**
   * Version "17.1".
   * @deprecated PostgreSQL 17.1 is no longer supported by Amazon RDS
   */
  public static readonly VER_17_1 = PostgresEngineVersion.of('17.1', '17', { s3Import: true, s3Export: true });
  /** Version "17.2". */
  public static readonly VER_17_2 = PostgresEngineVersion.of('17.2', '17', { s3Import: true, s3Export: true });
  /**
   * Version "17.3".
   * @deprecated PostgreSQL 17.3 is no longer supported by Amazon RDS
   */
  public static readonly VER_17_3 = PostgresEngineVersion.of('17.3', '17', { s3Import: true, s3Export: true });
  /** Version "17.4". */
  public static readonly VER_17_4 = PostgresEngineVersion.of('17.4', '17', { s3Import: true, s3Export: true });
  /** Version "17.5". */
  public static readonly VER_17_5 = PostgresEngineVersion.of('17.5', '17', { s3Import: true, s3Export: true });
  /** Version "17.6". */
  public static readonly VER_17_6 = PostgresEngineVersion.of('17.6', '17', { s3Import: true, s3Export: true });
  /** Version "17.7". */
  public static readonly VER_17_7 = PostgresEngineVersion.of('17.7', '17', { s3Import: true, s3Export: true });
  /** Version "17.8". */
  public static readonly VER_17_8 = PostgresEngineVersion.of('17.8', '17', { s3Import: true, s3Export: true });
  /** Version "17.9". */
  public static readonly VER_17_9 = PostgresEngineVersion.of('17.9', '17', { s3Import: true, s3Export: true });

  /** Version "18" (only a major version, without a specific minor version). */
  public static readonly VER_18 = PostgresEngineVersion.of('18', '18', { s3Import: true, s3Export: true });
  /** Version "18.1". */
  public static readonly VER_18_1 = PostgresEngineVersion.of('18.1', '18', { s3Import: true, s3Export: true });
  /** Version "18.2". */
  public static readonly VER_18_2 = PostgresEngineVersion.of('18.2', '18', { s3Import: true, s3Export: true });
  /** Version "18.3". */
  public static readonly VER_18_3 = PostgresEngineVersion.of('18.3', '18', { s3Import: true, s3Export: true });

  /**
   * Create a new PostgresEngineVersion with an arbitrary version.
   *
   * @param postgresFullVersion the full version string,
   *   for example "13.11"
   * @param postgresMajorVersion the major version of the engine,
   *   for example "13"
   */
  public static of(postgresFullVersion: string, postgresMajorVersion: string, postgresFeatures?: PostgresEngineFeatures): PostgresEngineVersion {
    return new PostgresEngineVersion(postgresFullVersion, postgresMajorVersion, postgresFeatures);
  }

  /** The full version string, for example, "13.11". */
  public readonly postgresFullVersion: string;
  /** The major version of the engine, for example, "13". */
  public readonly postgresMajorVersion: string;

  /**
   * The supported features for the DB engine
   * @internal
   */
  public readonly _features: InstanceEngineFeatures;

  private constructor(postgresFullVersion: string, postgresMajorVersion: string, postgresFeatures?: PostgresEngineFeatures) {
    this.postgresFullVersion = postgresFullVersion;
    this.postgresMajorVersion = postgresMajorVersion;
    this._features = {
      s3Import: postgresFeatures?.s3Import ? 's3Import' : undefined,
      s3Export: postgresFeatures?.s3Export ? 's3Export' : undefined,
    };
  }
}

/**
 * Properties for PostgreSQL instance engines.
 * Used in `DatabaseInstanceEngine.postgres`.
 */
export interface PostgresInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: PostgresEngineVersion;
}

/**
 * The instance engine for PostgreSQL.
 */
class PostgresInstanceEngine extends InstanceEngineBase {
  public readonly defaultUsername = 'postgres';

  constructor(version?: PostgresEngineVersion) {
    super({
      engineType: 'postgres',
      singleUserRotationApplication: secretsmanager.SecretRotationApplication.POSTGRES_ROTATION_SINGLE_USER,
      multiUserRotationApplication: secretsmanager.SecretRotationApplication.POSTGRES_ROTATION_MULTI_USER,
      version: version
        ? {
          fullVersion: version.postgresFullVersion,
          majorVersion: version.postgresMajorVersion,
        }
        : undefined,
      features: version ? version?._features : { s3Import: 's3Import' },
      engineFamily: 'POSTGRESQL',
    });
  }
}

/**
 * The versions for the legacy Oracle instance engines
 * (those returned by `DatabaseInstanceEngine.oracleSe`
 * and `DatabaseInstanceEngine.oracleSe1`).
 * Note: RDS will stop allowing creating new databases with this version in August 2020.
 *
 * @deprecated instances can no longer be created with these engine versions. See https://forums.aws.amazon.com/ann.jspa?annID=7341
 */
export class OracleLegacyEngineVersion {
  /** Version "11.2" (only a major version, without a specific minor version). */
  public static readonly VER_11_2 = OracleLegacyEngineVersion.of('11.2', '11.2');
  /** Version "11.2.0.2.v2". */
  public static readonly VER_11_2_0_2_V2 = OracleLegacyEngineVersion.of('11.2.0.2.v2', '11.2');
  /** Version "11.2.0.4.v1". */
  public static readonly VER_11_2_0_4_V1 = OracleLegacyEngineVersion.of('11.2.0.4.v1', '11.2');
  /** Version "11.2.0.4.v3". */
  public static readonly VER_11_2_0_4_V3 = OracleLegacyEngineVersion.of('11.2.0.4.v3', '11.2');
  /** Version "11.2.0.4.v4". */
  public static readonly VER_11_2_0_4_V4 = OracleLegacyEngineVersion.of('11.2.0.4.v4', '11.2');
  /** Version "11.2.0.4.v5". */
  public static readonly VER_11_2_0_4_V5 = OracleLegacyEngineVersion.of('11.2.0.4.v5', '11.2');
  /** Version "11.2.0.4.v6". */
  public static readonly VER_11_2_0_4_V6 = OracleLegacyEngineVersion.of('11.2.0.4.v6', '11.2');
  /** Version "11.2.0.4.v7". */
  public static readonly VER_11_2_0_4_V7 = OracleLegacyEngineVersion.of('11.2.0.4.v7', '11.2');
  /** Version "11.2.0.4.v8". */
  public static readonly VER_11_2_0_4_V8 = OracleLegacyEngineVersion.of('11.2.0.4.v8', '11.2');
  /** Version "11.2.0.4.v9". */
  public static readonly VER_11_2_0_4_V9 = OracleLegacyEngineVersion.of('11.2.0.4.v9', '11.2');
  /** Version "11.2.0.4.v10". */
  public static readonly VER_11_2_0_4_V10 = OracleLegacyEngineVersion.of('11.2.0.4.v10', '11.2');
  /** Version "11.2.0.4.v11". */
  public static readonly VER_11_2_0_4_V11 = OracleLegacyEngineVersion.of('11.2.0.4.v11', '11.2');
  /** Version "11.2.0.4.v12". */
  public static readonly VER_11_2_0_4_V12 = OracleLegacyEngineVersion.of('11.2.0.4.v12', '11.2');
  /** Version "11.2.0.4.v13". */
  public static readonly VER_11_2_0_4_V13 = OracleLegacyEngineVersion.of('11.2.0.4.v13', '11.2');
  /** Version "11.2.0.4.v14". */
  public static readonly VER_11_2_0_4_V14 = OracleLegacyEngineVersion.of('11.2.0.4.v14', '11.2');
  /** Version "11.2.0.4.v15". */
  public static readonly VER_11_2_0_4_V15 = OracleLegacyEngineVersion.of('11.2.0.4.v15', '11.2');
  /** Version "11.2.0.4.v16". */
  public static readonly VER_11_2_0_4_V16 = OracleLegacyEngineVersion.of('11.2.0.4.v16', '11.2');
  /** Version "11.2.0.4.v17". */
  public static readonly VER_11_2_0_4_V17 = OracleLegacyEngineVersion.of('11.2.0.4.v17', '11.2');
  /** Version "11.2.0.4.v18". */
  public static readonly VER_11_2_0_4_V18 = OracleLegacyEngineVersion.of('11.2.0.4.v18', '11.2');
  /** Version "11.2.0.4.v19". */
  public static readonly VER_11_2_0_4_V19 = OracleLegacyEngineVersion.of('11.2.0.4.v19', '11.2');
  /** Version "11.2.0.4.v20". */
  public static readonly VER_11_2_0_4_V20 = OracleLegacyEngineVersion.of('11.2.0.4.v20', '11.2');
  /** Version "11.2.0.4.v21". */
  public static readonly VER_11_2_0_4_V21 = OracleLegacyEngineVersion.of('11.2.0.4.v21', '11.2');
  /** Version "11.2.0.4.v22". */
  public static readonly VER_11_2_0_4_V22 = OracleLegacyEngineVersion.of('11.2.0.4.v22', '11.2');
  /** Version "11.2.0.4.v23". */
  public static readonly VER_11_2_0_4_V23 = OracleLegacyEngineVersion.of('11.2.0.4.v23', '11.2');
  /** Version "11.2.0.4.v24". */
  public static readonly VER_11_2_0_4_V24 = OracleLegacyEngineVersion.of('11.2.0.4.v24', '11.2');
  /** Version "11.2.0.4.v25". */
  public static readonly VER_11_2_0_4_V25 = OracleLegacyEngineVersion.of('11.2.0.4.v25', '11.2');

  private static of(oracleLegacyFullVersion: string, oracleLegacyMajorVersion: string): OracleLegacyEngineVersion {
    return new OracleLegacyEngineVersion(oracleLegacyFullVersion, oracleLegacyMajorVersion);
  }

  /** The full version string, for example, "11.2.0.4.v24". */
  public readonly oracleLegacyFullVersion: string;
  /** The major version of the engine, for example, "11.2". */
  public readonly oracleLegacyMajorVersion: string;

  private constructor(oracleLegacyFullVersion: string, oracleLegacyMajorVersion: string) {
    this.oracleLegacyFullVersion = oracleLegacyFullVersion;
    this.oracleLegacyMajorVersion = oracleLegacyMajorVersion;
  }
}

/**
 * The versions for the Oracle instance engines.
 * Those returned by the following list.
 * - `DatabaseInstanceEngine.oracleSe2`
 * - `DatabaseInstanceEngine.oracleSe2Cdb`
 * - `DatabaseInstanceEngine.oracleEe`
 * - `DatabaseInstanceEngine.oracleEeCdb`.
 */
export class OracleEngineVersion {
  /**
   * Version "12.1" (only a major version, without a specific minor version).
   * @deprecated Oracle 12.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1 = OracleEngineVersion.of('12.1', '12.1');
  /**
   * Version "12.1.0.2.v1"
   * @deprecated Oracle 12.1.0.2.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V1 = OracleEngineVersion.of('12.1.0.2.v1', '12.1');
  /**
   * Version "12.1.0.2.v2"
   * @deprecated Oracle 12.1.0.2.v2 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V2 = OracleEngineVersion.of('12.1.0.2.v2', '12.1');
  /**
   * Version "12.1.0.2.v3"
   * @deprecated Oracle 12.1.0.2.v3 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V3 = OracleEngineVersion.of('12.1.0.2.v3', '12.1');
  /**
   * Version "12.1.0.2.v4"
   * @deprecated Oracle 12.1.0.2.v4 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V4 = OracleEngineVersion.of('12.1.0.2.v4', '12.1');
  /**
   * Version "12.1.0.2.v5"
   * @deprecated Oracle 12.1.0.2.v5 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V5 = OracleEngineVersion.of('12.1.0.2.v5', '12.1');
  /**
   * Version "12.1.0.2.v6"
   * @deprecated Oracle 12.1.0.2.v6 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V6 = OracleEngineVersion.of('12.1.0.2.v6', '12.1');
  /**
   * Version "12.1.0.2.v7"
   * @deprecated Oracle 12.1.0.2.v7 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V7 = OracleEngineVersion.of('12.1.0.2.v7', '12.1');
  /**
   * Version "12.1.0.2.v8"
   * @deprecated Oracle 12.1.0.2.v8 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V8 = OracleEngineVersion.of('12.1.0.2.v8', '12.1');
  /**
   * Version "12.1.0.2.v9"
   * @deprecated Oracle 12.1.0.2.v9 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V9 = OracleEngineVersion.of('12.1.0.2.v9', '12.1');
  /**
   * Version "12.1.0.2.v10"
   * @deprecated Oracle 12.1.0.2.v10 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V10 = OracleEngineVersion.of('12.1.0.2.v10', '12.1');
  /**
   * Version "12.1.0.2.v11"
   * @deprecated Oracle 12.1.0.2.v11 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V11 = OracleEngineVersion.of('12.1.0.2.v11', '12.1');
  /**
   * Version "12.1.0.2.v12"
   * @deprecated Oracle 12.1.0.2.v12 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V12 = OracleEngineVersion.of('12.1.0.2.v12', '12.1');
  /**
   * Version "12.1.0.2.v13"
   * @deprecated Oracle 12.1.0.2.v13 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V13 = OracleEngineVersion.of('12.1.0.2.v13', '12.1');
  /**
   * Version "12.1.0.2.v14"
   * @deprecated Oracle 12.1.0.2.v14 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V14 = OracleEngineVersion.of('12.1.0.2.v14', '12.1');
  /**
   * Version "12.1.0.2.v15"
   * @deprecated Oracle 12.1.0.2.v15 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V15 = OracleEngineVersion.of('12.1.0.2.v15', '12.1');
  /**
   * Version "12.1.0.2.v16"
   * @deprecated Oracle 12.1.0.2.v16 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V16 = OracleEngineVersion.of('12.1.0.2.v16', '12.1');
  /**
   * Version "12.1.0.2.v17"
   * @deprecated Oracle 12.1.0.2.v17 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V17 = OracleEngineVersion.of('12.1.0.2.v17', '12.1');
  /**
   * Version "12.1.0.2.v18"
   * @deprecated Oracle 12.1.0.2.v18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V18 = OracleEngineVersion.of('12.1.0.2.v18', '12.1');
  /**
   * Version "12.1.0.2.v19"
   * @deprecated Oracle 12.1.0.2.v19 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V19 = OracleEngineVersion.of('12.1.0.2.v19', '12.1');
  /**
   * Version "12.1.0.2.v20"
   * @deprecated Oracle 12.1.0.2.v20 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V20 = OracleEngineVersion.of('12.1.0.2.v20', '12.1');
  /**
   * Version "12.1.0.2.v21"
   * @deprecated Oracle 12.1.0.2.v21 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V21 = OracleEngineVersion.of('12.1.0.2.v21', '12.1');
  /**
   * Version "12.1.0.2.v22"
   * @deprecated Oracle 12.1.0.2.v22 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V22 = OracleEngineVersion.of('12.1.0.2.v22', '12.1');
  /**
   * Version "12.1.0.2.v23"
   * @deprecated Oracle 12.1.0.2.v23 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V23 = OracleEngineVersion.of('12.1.0.2.v23', '12.1');
  /**
   * Version "12.1.0.2.v24"
   * @deprecated Oracle 12.1.0.2.v24 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V24 = OracleEngineVersion.of('12.1.0.2.v24', '12.1');
  /**
   * Version "12.1.0.2.v25"
   * @deprecated Oracle 12.1.0.2.v25 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V25 = OracleEngineVersion.of('12.1.0.2.v25', '12.1');
  /**
   * Version "12.1.0.2.v26"
   * @deprecated Oracle 12.1.0.2.v26 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V26 = OracleEngineVersion.of('12.1.0.2.v26', '12.1');
  /**
   * Version "12.1.0.2.v27"
   * @deprecated Oracle 12.1.0.2.v27 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V27 = OracleEngineVersion.of('12.1.0.2.v27', '12.1');
  /**
   * Version "12.1.0.2.v28"
   * @deprecated Oracle 12.1.0.2.v28 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V28 = OracleEngineVersion.of('12.1.0.2.v28', '12.1');
  /**
   * Version "12.1.0.2.v29"
   * @deprecated Oracle 12.1.0.2.v29 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_1_0_2_V29 = OracleEngineVersion.of('12.1.0.2.v29', '12.1');

  /**
   * Version "12.1" (only a major version, without a specific minor version).
   * @deprecated Oracle 12.1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2 = OracleEngineVersion.of('12.2', '12.2');
  /**
   * Version "12.2.0.1.ru-2018-10.rur-2018-10.r1"
   * @deprecated Oracle 12.2.0.1.ru-2018-10.rur-2018-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2018_10_R1 = OracleEngineVersion.of('12.2.0.1.ru-2018-10.rur-2018-10.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2019-01.rur-2019-01.r1"
   * @deprecated Oracle 12.2.0.1.ru-2019-01.rur-2019-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2019_01_R1 = OracleEngineVersion.of('12.2.0.1.ru-2019-01.rur-2019-01.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2019-04.rur-2019-04.r1"
   * @deprecated Oracle 12.2.0.1.ru-2019-04.rur-2019-04.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2019_04_R1 = OracleEngineVersion.of('12.2.0.1.ru-2019-04.rur-2019-04.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2019-07.rur-2019-07.r1"
   * @deprecated Oracle 12.2.0.1.ru-2019-07.rur-2019-07.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2019_07_R1 = OracleEngineVersion.of('12.2.0.1.ru-2019-07.rur-2019-07.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2019-10.rur-2019-10.r1"
   * @deprecated Oracle 12.2.0.1.ru-2019-10.rur-2019-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2019_10_R1 = OracleEngineVersion.of('12.2.0.1.ru-2019-10.rur-2019-10.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2020-01.rur-2020-01.r1"
   * @deprecated Oracle 12.2.0.1.ru-2020-01.rur-2020-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2020_01_R1 = OracleEngineVersion.of('12.2.0.1.ru-2020-01.rur-2020-01.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2020-04.rur-2020-04.r1"
   * @deprecated Oracle 12.2.0.1.ru-2020-04.rur-2020-04.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2020_04_R1 = OracleEngineVersion.of('12.2.0.1.ru-2020-04.rur-2020-04.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2020-07.rur-2020-07.r1"
   * @deprecated Oracle 12.2.0.1.ru-2020-07.rur-2020-07.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2020_07_R1 = OracleEngineVersion.of('12.2.0.1.ru-2020-07.rur-2020-07.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2021-10.rur-2020-10.r1"
   * @deprecated Oracle 12.2.0.1.ru-2021-10.rur-2020-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2020_10_R1 = OracleEngineVersion.of('12.2.0.1.ru-2020-10.rur-2020-10.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2021-01.rur-2021-01.r1"
   * @deprecated Oracle 12.2.0.1.ru-2021-01.rur-2021-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2021_01_R1 = OracleEngineVersion.of('12.2.0.1.ru-2021-01.rur-2021-01.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2021-04.rur-2021-04.r1"
   * @deprecated Oracle 12.2.0.1.ru-2021-04.rur-2021-04.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2021_04_R1 = OracleEngineVersion.of('12.2.0.1.ru-2021-04.rur-2021-04.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2021-07.rur-2021-07.r1"
   * @deprecated Oracle 12.2.0.1.ru-2021-07.rur-2021-07.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2021_07_R1 = OracleEngineVersion.of('12.2.0.1.ru-2021-07.rur-2021-07.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2021-10.rur-2021-10.r1"
   * @deprecated Oracle 12.2.0.1.ru-2021-10.rur-2021-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2021_10_R1 = OracleEngineVersion.of('12.2.0.1.ru-2021-10.rur-2021-10.r1', '12.2');
  /**
   * Version "12.2.0.1.ru-2022-01.rur-2022-01.r1"
   * @deprecated Oracle 12.2.0.1.ru-2022-01.rur-2022-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_2_0_1_2022_01_R1 = OracleEngineVersion.of('12.2.0.1.ru-2022-01.rur-2022-01.r1', '12.2');

  /**
   * Version "18" (only a major version, without a specific minor version).
   * @deprecated Oracle 18 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18 = OracleEngineVersion.of('18', '18');
  /**
   * Version "18.0.0.0.ru-2019-07.rur-2019-07.r1"
   * @deprecated Oracle 18.0.0.0.ru-2019-07.rur-2019-07.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2019_07_R1 = OracleEngineVersion.of('18.0.0.0.ru-2019-07.rur-2019-07.r1', '18');
  /**
   * Version "18.0.0.0.ru-2019-10.rur-2019-10.r1"
   * @deprecated Oracle 18.0.0.0.ru-2019-10.rur-2019-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2019_10_R1 = OracleEngineVersion.of('18.0.0.0.ru-2019-10.rur-2019-10.r1', '18');
  /**
   * Version "18.0.0.0.ru-2020-01.rur-2020-01.r1"
   * @deprecated Oracle 18.0.0.0.ru-2020-01.rur-2020-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2020_01_R1 = OracleEngineVersion.of('18.0.0.0.ru-2020-01.rur-2020-01.r1', '18');
  /**
   * Version "18.0.0.0.ru-2020-04.rur-2020-04.r1"
   * @deprecated Oracle 18.0.0.0.ru-2020-04.rur-2020-04.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2020_04_R1 = OracleEngineVersion.of('18.0.0.0.ru-2020-04.rur-2020-04.r1', '18');
  /**
   * Version "18.0.0.0.ru-2020-07.rur-2020-07.r1"
   * @deprecated Oracle 18.0.0.0.ru-2020-07.rur-2020-07.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2020_07_R1 = OracleEngineVersion.of('18.0.0.0.ru-2020-07.rur-2020-07.r1', '18');
  /**
   * Version "18.0.0.0.ru-2020-10.rur-2020-10.r1"
   * @deprecated Oracle 18.0.0.0.ru-2020-10.rur-2020-10.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2020_10_R1 = OracleEngineVersion.of('18.0.0.0.ru-2020-10.rur-2020-10.r1', '18');
  /**
   * Version "18.0.0.0.ru-2021-01.rur-2021-01.r1"
   * @deprecated Oracle 18.0.0.0.ru-2021-01.rur-2021-01.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2021_01_R1 = OracleEngineVersion.of('18.0.0.0.ru-2021-01.rur-2021-01.r1', '18');
  /**
   * Version "18.0.0.0.ru-2021-04.rur-2021-04.r1"
   * @deprecated Oracle 18.0.0.0.ru-2021-04.rur-2021-04.r1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_18_0_0_0_2021_04_R1 = OracleEngineVersion.of('18.0.0.0.ru-2021-04.rur-2021-04.r1', '18');

  /** Version "19" (only a major version, without a specific minor version). */
  public static readonly VER_19 = OracleEngineVersion.of('19', '19');
  /** Version "19.0.0.0.ru-2019-07.rur-2019-07.r1". */
  public static readonly VER_19_0_0_0_2019_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2019-07.rur-2019-07.r1', '19');
  /** Version "19.0.0.0.ru-2019-10.rur-2019-10.r1". */
  public static readonly VER_19_0_0_0_2019_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2019-10.rur-2019-10.r1', '19');
  /** Version "19.0.0.0.ru-2020-01.rur-2020-01.r1". */
  public static readonly VER_19_0_0_0_2020_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2020-01.rur-2020-01.r1', '19');
  /** Version "19.0.0.0.ru-2020-04.rur-2020-04.r1". */
  public static readonly VER_19_0_0_0_2020_04_R1 = OracleEngineVersion.of('19.0.0.0.ru-2020-04.rur-2020-04.r1', '19');
  /** Version "19.0.0.0.ru-2020-07.rur-2020-07.r1". */
  public static readonly VER_19_0_0_0_2020_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2020-07.rur-2020-07.r1', '19');
  /** Version "19.0.0.0.ru-2020-07.rur-2020-10.r1". */
  public static readonly VER_19_0_0_0_2020_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2020-10.rur-2020-10.r1', '19');
  /** Version "19.0.0.0.ru-2021-01.rur-2021-01.r1". */
  public static readonly VER_19_0_0_0_2021_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2021-01.rur-2021-01.r1', '19');
  /** Version "19.0.0.0.ru-2021-01.rur-2021-01.r2". */
  public static readonly VER_19_0_0_0_2021_01_R2 = OracleEngineVersion.of('19.0.0.0.ru-2021-01.rur-2021-01.r2', '19');
  /** Version "19.0.0.0.ru-2021-01.rur-2021-04.r1". */
  public static readonly VER_19_0_0_0_2021_04_R1 = OracleEngineVersion.of('19.0.0.0.ru-2021-04.rur-2021-04.r1', '19');
  /** Version "19.0.0.0.ru-2021-07.rur-2021-07.r1". */
  public static readonly VER_19_0_0_0_2021_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2021-07.rur-2021-07.r1', '19');
  /** Version "19.0.0.0.ru-2021-10.rur-2021-10.r1". */
  public static readonly VER_19_0_0_0_2021_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2021-10.rur-2021-10.r1', '19');
  /** Version "19.0.0.0.ru-2022-01.rur-2022-01.r1". */
  public static readonly VER_19_0_0_0_2022_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2022-01.rur-2022-01.r1', '19');
  /** Version "19.0.0.0.ru-2022-04.rur-2022-04.r1". */
  public static readonly VER_19_0_0_0_2022_04_R1 = OracleEngineVersion.of('19.0.0.0.ru-2022-04.rur-2022-04.r1', '19');
  /** Version "19.0.0.0.ru-2022-07.rur-2022-07.r1". */
  public static readonly VER_19_0_0_0_2022_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2022-07.rur-2022-07.r1', '19');
  /** Version "19.0.0.0.ru-2022-10.rur-2022-10.r1". */
  public static readonly VER_19_0_0_0_2022_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2022-10.rur-2022-10.r1', '19');
  /** Version "19.0.0.0.ru-2023-01.rur-2023-01.r1". */
  public static readonly VER_19_0_0_0_2023_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2023-01.rur-2023-01.r1', '19');
  /** Version "19.0.0.0.ru-2023-01.rur-2023-01.r2". */
  public static readonly VER_19_0_0_0_2023_01_R2 = OracleEngineVersion.of('19.0.0.0.ru-2023-01.rur-2023-01.r2', '19');
  /** Version "19.0.0.0.ru-2023-04.rur-2023-04.r1". */
  public static readonly VER_19_0_0_0_2023_04_R1 = OracleEngineVersion.of('19.0.0.0.ru-2023-04.rur-2023-04.r1', '19');
  /** Version "19.0.0.0.ru-2023-07.rur-2023-07.r1"  */
  public static readonly VER_19_0_0_0_2023_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2023-07.rur-2023-07.r1', '19');
  /** Version "19.0.0.0.ru-2023-10.rur-2023-10.r1"  */
  public static readonly VER_19_0_0_0_2023_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2023-10.rur-2023-10.r1', '19');
  /** Version "19.0.0.0.ru-2024-01.rur-2024-01.r1". */
  public static readonly VER_19_0_0_0_2024_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2024-01.rur-2024-01.r1', '19');
  /** Version "19.0.0.0.ru-2024-04.rur-2024-04.r1". */
  public static readonly VER_19_0_0_0_2024_04_R1 = OracleEngineVersion.of('19.0.0.0.ru-2024-04.rur-2024-04.r1', '19');
  /** Version "19.0.0.0.ru-2024-07.rur-2024-07.r1". */
  public static readonly VER_19_0_0_0_2024_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2024-07.rur-2024-07.r1', '19');
  /** Version "19.0.0.0.ru-2024-10.rur-2024-10.r1". */
  public static readonly VER_19_0_0_0_2024_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2024-10.rur-2024-10.r1', '19');
  /** Version "19.0.0.0.ru-2025-01.rur-2025-01.r1". */
  public static readonly VER_19_0_0_0_2025_01_R1 = OracleEngineVersion.of('19.0.0.0.ru-2025-01.rur-2025-01.r1', '19');
  /** Version "19.0.0.0.ru-2025-07.rur-2025-07.r1". */
  public static readonly VER_19_0_0_0_2025_07_R1 = OracleEngineVersion.of('19.0.0.0.ru-2025-07.rur-2025-07.r1', '19');
  /** Version "19.0.0.0.ru-2025-10.rur-2025-10.r1". */
  public static readonly VER_19_0_0_0_2025_10_R1 = OracleEngineVersion.of('19.0.0.0.ru-2025-10.rur-2025-10.r1', '19');

  /** Version "21" (only a major version, without a specific minor version). */
  public static readonly VER_21 = OracleEngineVersion.of('21', '21');
  /** Version "21.0.0.0.ru-2022-01.rur-2022-01.r1". */
  public static readonly VER_21_0_0_0_2022_01_R1 = OracleEngineVersion.of('21.0.0.0.ru-2022-01.rur-2022-01.r1', '21');
  /** Version "21.0.0.0.ru-2022-04.rur-2022-04.r1". */
  public static readonly VER_21_0_0_0_2022_04_R1 = OracleEngineVersion.of('21.0.0.0.ru-2022-04.rur-2022-04.r1', '21');
  /** Version "21.0.0.0.ru-2022-07.rur-2022-07.r1". */
  public static readonly VER_21_0_0_0_2022_07_R1 = OracleEngineVersion.of('21.0.0.0.ru-2022-07.rur-2022-07.r1', '21');
  /** Version "21.0.0.0.ru-2022-10.rur-2022-10.r1". */
  public static readonly VER_21_0_0_0_2022_10_R1 = OracleEngineVersion.of('21.0.0.0.ru-2022-10.rur-2022-10.r1', '21');
  /** Version "21.0.0.0.ru-2023-01.rur-2023-01.r1". */
  public static readonly VER_21_0_0_0_2023_01_R1 = OracleEngineVersion.of('21.0.0.0.ru-2023-01.rur-2023-01.r1', '21');
  /** Version "21.0.0.0.ru-2023-01.rur-2023-01.r2". */
  public static readonly VER_21_0_0_0_2023_01_R2 = OracleEngineVersion.of('21.0.0.0.ru-2023-01.rur-2023-01.r2', '21');
  /** Version "21.0.0.0.ru-2023-04.rur-2023-04.r1". */
  public static readonly VER_21_0_0_0_2023_04_R1 = OracleEngineVersion.of('21.0.0.0.ru-2023-04.rur-2023-04.r1', '21');
  /** Version "21.0.0.0.ru-2023-07.rur-2023-07.r1". */
  public static readonly VER_21_0_0_0_2023_07_R1 = OracleEngineVersion.of('21.0.0.0.ru-2023-07.rur-2023-07.r1', '21');
  /** Version "21.0.0.0.ru-2023-10.rur-2023-10.r1". */
  public static readonly VER_21_0_0_0_2023_10_R1 = OracleEngineVersion.of('21.0.0.0.ru-2023-10.rur-2023-10.r1', '21');
  /** Version "21.0.0.0.ru-2024-01.rur-2024-01.r1". */
  public static readonly VER_21_0_0_0_2024_01_R1 = OracleEngineVersion.of('21.0.0.0.ru-2024-01.rur-2024-01.r1', '21');
  /** Version "21.0.0.0.ru-2024-04.rur-2024-04.r1". */
  public static readonly VER_21_0_0_0_2024_04_R1 = OracleEngineVersion.of('21.0.0.0.ru-2024-04.rur-2024-04.r1', '21');
  /** Version "21.0.0.0.ru-2024-07.rur-2024-07.r1". */
  public static readonly VER_21_0_0_0_2024_07_R1 = OracleEngineVersion.of('21.0.0.0.ru-2024-07.rur-2024-07.r1', '21');
  /** Version "21.0.0.0.ru-2024-10.rur-2024-10.r1". */
  public static readonly VER_21_0_0_0_2024_10_R1 = OracleEngineVersion.of('21.0.0.0.ru-2024-10.rur-2024-10.r1', '21');
  /** Version "21.0.0.0.ru-2025-01.rur-2025-01.r1". */
  public static readonly VER_21_0_0_0_2025_01_R1 = OracleEngineVersion.of('21.0.0.0.ru-2025-01.rur-2025-01.r1', '21');
  /** Version "21.0.0.0.ru-2025-07.rur-2025-07.r1". */
  public static readonly VER_21_0_0_0_2025_07_R1 = OracleEngineVersion.of('21.0.0.0.ru-2025-07.rur-2025-07.r1', '21');

  /**
   * Creates a new OracleEngineVersion with an arbitrary version.
   *
   * @param oracleFullVersion the full version string,
   *   for example "19.0.0.0.ru-2019-10.rur-2019-10.r1"
   * @param oracleMajorVersion the major version of the engine,
   *   for example "19"
   */
  public static of(oracleFullVersion: string, oracleMajorVersion: string): OracleEngineVersion {
    return new OracleEngineVersion(oracleFullVersion, oracleMajorVersion);
  }

  /** The full version string, for example, "19.0.0.0.ru-2019-10.rur-2019-10.r1". */
  public readonly oracleFullVersion: string;
  /** The major version of the engine, for example, "19". */
  public readonly oracleMajorVersion: string;

  private constructor(oracleFullVersion: string, oracleMajorVersion: string) {
    this.oracleFullVersion = oracleFullVersion;
    this.oracleMajorVersion = oracleMajorVersion;
  }
}

interface OracleInstanceEngineBaseProps {
  readonly engineType: string;
  readonly version?: EngineVersion;
}

abstract class OracleInstanceEngineBase extends InstanceEngineBase {
  constructor(props: OracleInstanceEngineBaseProps) {
    super({
      ...props,
      singleUserRotationApplication: secretsmanager.SecretRotationApplication.ORACLE_ROTATION_SINGLE_USER,
      multiUserRotationApplication: secretsmanager.SecretRotationApplication.ORACLE_ROTATION_MULTI_USER,
      parameterGroupFamily: props.version ? `${props.engineType}-${props.version.majorVersion}` : undefined,
      features: {
        s3Import: 'S3_INTEGRATION',
        s3Export: 'S3_INTEGRATION',
      },
    });
  }

  public bindToInstance(scope: Construct, options: InstanceEngineBindOptions): InstanceEngineConfig {
    const config = super.bindToInstance(scope, options);

    let optionGroup = options.optionGroup;
    if (options.s3ImportRole || options.s3ExportRole) {
      if (!optionGroup) {
        optionGroup = new OptionGroup(scope, 'InstanceOptionGroup', {
          engine: this,
          configurations: [],
        });
      }
      // https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/oracle-s3-integration.html
      optionGroup.addConfiguration({
        name: 'S3_INTEGRATION',
        version: '1.0',
      });
    }

    return {
      ...config,
      optionGroup,
    };
  }
}

interface OracleInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: OracleEngineVersion;
}

/**
 * Properties for Oracle Standard Edition instance engines.
 * Used in `DatabaseInstanceEngine.oracleSe`.
 *
 * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
 */
export interface OracleSeInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: OracleLegacyEngineVersion;
}

/** @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341 */
class OracleSeInstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleLegacyEngineVersion) {
    super({
      engineType: 'oracle-se',
      version: version
        ? {
          fullVersion: version.oracleLegacyFullVersion,
          majorVersion: version.oracleLegacyMajorVersion,
        }
        : {
          majorVersion: '11.2',
        },
    });
  }
}

/**
 * Properties for Oracle Standard Edition 1 instance engines.
 * Used in `DatabaseInstanceEngine.oracleSe1`.
 *
 * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
 */
export interface OracleSe1InstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: OracleLegacyEngineVersion;
}

/** @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341 */
class OracleSe1InstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleLegacyEngineVersion) {
    super({
      engineType: 'oracle-se1',
      version: version
        ? {
          fullVersion: version.oracleLegacyFullVersion,
          majorVersion: version.oracleLegacyMajorVersion,
        }
        : {
          majorVersion: '11.2',
        },
    });
  }
}

/**
 * Properties for Oracle Standard Edition 2 instance engines.
 * Used in `DatabaseInstanceEngine.oracleSe2`.
 */
export interface OracleSe2InstanceEngineProps extends OracleInstanceEngineProps {
}

class OracleSe2InstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleEngineVersion) {
    super({
      engineType: 'oracle-se2',
      version: version
        ? {
          fullVersion: version.oracleFullVersion,
          majorVersion: version.oracleMajorVersion,
        }
        : undefined,
    });
  }
}

/**
 * Properties for Oracle Standard Edition 2 (CDB) instance engines.
 * Used in `DatabaseInstanceEngine.oracleSe2Cdb`.
 */
export interface OracleSe2CdbInstanceEngineProps extends OracleInstanceEngineProps {
}

class OracleSe2CdbInstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleEngineVersion) {
    super({
      engineType: 'oracle-se2-cdb',
      version: version
        ? {
          fullVersion: version.oracleFullVersion,
          majorVersion: version.oracleMajorVersion,
        }
        : undefined,
    });
  }
}

/**
 * Properties for Oracle Enterprise Edition instance engines.
 * Used in `DatabaseInstanceEngine.oracleEe`.
 */
export interface OracleEeInstanceEngineProps extends OracleInstanceEngineProps {
}

class OracleEeInstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleEngineVersion) {
    super({
      engineType: 'oracle-ee',
      version: version
        ? {
          fullVersion: version.oracleFullVersion,
          majorVersion: version.oracleMajorVersion,
        }
        : undefined,
    });
  }
}

/**
 * Properties for Oracle Enterprise Edition (CDB) instance engines.
 * Used in `DatabaseInstanceEngine.oracleEeCdb`.
 */
export interface OracleEeCdbInstanceEngineProps extends OracleInstanceEngineProps {
}

class OracleEeCdbInstanceEngine extends OracleInstanceEngineBase {
  constructor(version?: OracleEngineVersion) {
    super({
      engineType: 'oracle-ee-cdb',
      version: version
        ? {
          fullVersion: version.oracleFullVersion,
          majorVersion: version.oracleMajorVersion,
        }
        : undefined,
    });
  }
}

/**
 * The versions for the SQL Server instance engines
 * (those returned by `DatabaseInstanceEngine.sqlServerSe`,
 * `DatabaseInstanceEngine.sqlServerEx`, `DatabaseInstanceEngine.sqlServerWeb`
 * and `DatabaseInstanceEngine.sqlServerEe`).
 */
export class SqlServerEngineVersion {
  /**
   * Version "11.00" (only a major version, without a specific minor version).
   * @deprecated SQL Server 11.00 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11 = SqlServerEngineVersion.of('11.00', '11.00');
  /**
   * Version "11.00.5058.0.v1".
   * @deprecated SQL Server 11.00.5058.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_00_5058_0_V1 = SqlServerEngineVersion.of('11.00.5058.0.v1', '11.00');
  /**
   * Version "11.00.6020.0.v1".
   * @deprecated SQL Server 11.00.6020.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_00_6020_0_V1 = SqlServerEngineVersion.of('11.00.6020.0.v1', '11.00');
  /**
   * Version "11.00.6594.0.v1".
   * @deprecated SQL Server 11.00.6594.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_00_6594_0_V1 = SqlServerEngineVersion.of('11.00.6594.0.v1', '11.00');
  /**
   * Version "11.00.7462.6.v1".
   * @deprecated SQL Server 11.00.7462.6.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_00_7462_6_V1 = SqlServerEngineVersion.of('11.00.7462.6.v1', '11.00');
  /**
   * Version "11.00.7493.4.v1".
   * @deprecated SQL Server 11.00.7493.4.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_11_00_7493_4_V1 = SqlServerEngineVersion.of('11.00.7493.4.v1', '11.00');

  /**
   * Version "12.00" (only a major version, without a specific minor version).
   * @deprecated SQL Server 12.00 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12 = SqlServerEngineVersion.of('12.00', '12.00');
  /**
   * Version "12.00.4422.0.v1"
   * @deprecated SQL Server 12.00.4422.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_4422_0_V1 = SqlServerEngineVersion.of('12.00.4422.0.v1', '12.00');
  /**
   * Version "12.00.5000.0.v1".
   * @deprecated SQL Server 12.00.5000.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_5000_0_V1 = SqlServerEngineVersion.of('12.00.5000.0.v1', '12.00');
  /**
   * Version "12.00.5546.0.v1".
   * @deprecated SQL Server 12.00.5546.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_5546_0_V1 = SqlServerEngineVersion.of('12.00.5546.0.v1', '12.00');
  /**
   * Version "12.00.5571.0.v1".
   * @deprecated SQL Server 12.00.5571.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_5571_0_V1 = SqlServerEngineVersion.of('12.00.5571.0.v1', '12.00');
  /**
   * Version "12.00.6293.0.v1".
   * @deprecated SQL Server 12.00.6293.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6293_0_V1 = SqlServerEngineVersion.of('12.00.6293.0.v1', '12.00');
  /**
   * Version "12.00.6329.1.v1".
   * @deprecated SQL Server 12.00.6329.1.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6329_1_V1 = SqlServerEngineVersion.of('12.00.6329.1.v1', '12.00');
  /**
   * Version "12.00.6433.1.v1".
   * @deprecated SQL Server 12.00.6433.1.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6433_1_V1 = SqlServerEngineVersion.of('12.00.6433.1.v1', '12.00');
  /**
   * Version "12.00.6439.10.v1".
   * @deprecated SQL Server 12.00.6439.10.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6439_10_V1 = SqlServerEngineVersion.of('12.00.6439.10.v1', '12.00');
  /**
   * Version "12.00.6444.4.v1".
   * @deprecated SQL Server 12.00.6444.4.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6444_4_V1 = SqlServerEngineVersion.of('12.00.6444.4.v1', '12.00');
  /**
   * Version "12.00.6449.1.v1".
   * @deprecated SQL Server 12.00.6449.1.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_12_00_6449_1_V1 = SqlServerEngineVersion.of('12.00.6449.1.v1', '12.00');

  /** Version "13.00" (only a major version, without a specific minor version). */
  public static readonly VER_13 = SqlServerEngineVersion.of('13.00', '13.00');
  /**
   * Version "13.00.2164.0.v1".
   * @deprecated SQL Server 13.00.2164.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_2164_0_V1 = SqlServerEngineVersion.of('13.00.2164.0.v1', '13.00');
  /**
   * Version "13.00.4422.0.v1".
   * @deprecated SQL Server 13.00.4422.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_4422_0_V1 = SqlServerEngineVersion.of('13.00.4422.0.v1', '13.00');
  /**
   * Version "13.00.4451.0.v1".
   * @deprecated SQL Server 13.00.4451.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_4451_0_V1 = SqlServerEngineVersion.of('13.00.4451.0.v1', '13.00');
  /**
   * Version "13.00.4466.4.v1".
   * @deprecated SQL Server 13.00.4466.4.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_4466_4_V1 = SqlServerEngineVersion.of('13.00.4466.4.v1', '13.00');
  /**
   * Version "13.00.4522.0.v1".
   * @deprecated SQL Server 13.00.4522.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_4522_0_V1 = SqlServerEngineVersion.of('13.00.4522.0.v1', '13.00');
  /**
   * Version "13.00.5216.0.v1".
   * @deprecated SQL Server 13.00.5216.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5216_0_V1 = SqlServerEngineVersion.of('13.00.5216.0.v1', '13.00');
  /**
   * Version "13.00.5292.0.v1".
   * @deprecated SQL Server 13.00.5292.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5292_0_V1 = SqlServerEngineVersion.of('13.00.5292.0.v1', '13.00');
  /**
   * Version "13.00.5366.0.v1".
   * @deprecated SQL Server 13.00.5366.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5366_0_V1 = SqlServerEngineVersion.of('13.00.5366.0.v1', '13.00');
  /**
   * Version "13.00.5426.0.v1".
   * @deprecated SQL Server 13.00.5426.0.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5426_0_V1 = SqlServerEngineVersion.of('13.00.5426.0.v1', '13.00');
  /**
   * Version "13.00.5598.27.v1".
   * @deprecated SQL Server 13.00.5598.27.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5598_27_V1 = SqlServerEngineVersion.of('13.00.5598.27.v1', '13.00');
  /**
   * Version "13.00.5820.21.v1".
   * @deprecated SQL Server 13.00.5820.21.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5820_21_V1 = SqlServerEngineVersion.of('13.00.5820.21.v1', '13.00');
  /**
   * Version "13.00.5850.14.v1".
   * @deprecated SQL Server 13.00.5850.14.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5850_14_V1 = SqlServerEngineVersion.of('13.00.5850.14.v1', '13.00');
  /**
   * Version "13.00.5882.1.v1".
   * @deprecated SQL Server 13.00.5882.1.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_13_00_5882_1_V1 = SqlServerEngineVersion.of('13.00.5882.1.v1', '13.00');
  /** Version "13.00.6300.2.v1". */
  public static readonly VER_13_00_6300_2_V1 = SqlServerEngineVersion.of('13.00.6300.2.v1', '13.00');
  /** Version "13.00.6419.1.v1". */
  public static readonly VER_13_00_6419_1_V1 = SqlServerEngineVersion.of('13.00.6419.1.v1', '13.00');
  /** Version "13.00.6430.49.v1". */
  public static readonly VER_13_00_6430_49_V1 = SqlServerEngineVersion.of('13.00.6430.49.v1', '13.00');
  /** Version "13.00.6435.1.v1". */
  public static readonly VER_13_00_6435_1_V1 = SqlServerEngineVersion.of('13.00.6435.1.v1', '13.00');
  /** Version "13.00.6441.1.v1". */
  public static readonly VER_13_00_6441_1_V1 = SqlServerEngineVersion.of('13.00.6441.1.v1', '13.00');
  /** Version "13.00.6445.1.v1". */
  public static readonly VER_13_00_6445_1_V1 = SqlServerEngineVersion.of('13.00.6445.1.v1', '13.00');
  /** Version "13.00.6450.1.v1". */
  public static readonly VER_13_00_6450_1_V1 = SqlServerEngineVersion.of('13.00.6450.1.v1', '13.00');
  /** Version "13.00.6455.2.v1". */
  public static readonly VER_13_00_6455_2_V1 = SqlServerEngineVersion.of('13.00.6455.2.v1', '13.00');
  /** Version "13.00.6460.7.v1". */
  public static readonly VER_13_00_6460_7_V1 = SqlServerEngineVersion.of('13.00.6460.7.v1', '13.00');
  /** Version "13.00.6465.1.v1". */
  public static readonly VER_13_00_6465_1_V1 = SqlServerEngineVersion.of('13.00.6465.1.v1', '13.00');
  /** Version "13.00.6470.1.v1". */
  public static readonly VER_13_00_6470_1_V1 = SqlServerEngineVersion.of('13.00.6470.1.v1', '13.00');

  /** Version "14.00" (only a major version, without a specific minor version). */
  public static readonly VER_14 = SqlServerEngineVersion.of('14.00', '14.00');
  /**
   * Version "14.00.1000.169.v1".
   * @deprecated SQL Server 14.00.1000.169.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_1000_169_V1 = SqlServerEngineVersion.of('14.00.1000.169.v1', '14.00');
  /**
   * Version "14.00.3015.40.v1".
   * @deprecated SQL Server 14.00.3015.40.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_3015_40_V1 = SqlServerEngineVersion.of('14.00.3015.40.v1', '14.00');
  /**
   * Version "14.00.3035.2.v1".
   * @deprecated SQL Server 14.00.3035.2.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_3035_2_V1 = SqlServerEngineVersion.of('14.00.3035.2.v1', '14.00');
  /**
   * Version "14.00.3049.1.v1".
   * @deprecated SQL Server 14.00.3049.1.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_3049_1_V1 = SqlServerEngineVersion.of('14.00.3049.1.v1', '14.00');
  /**
   * Version "14.00.3192.2.v1".
   * @deprecated SQL Server 14.00.3192.2.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_3192_2_V1 = SqlServerEngineVersion.of('14.00.3192.2.v1', '14.00');
  /**
   * Version "14.00.3223.3.v1".
   * @deprecated SQL Server 14.00.3223.3.v1 is no longer supported by Amazon RDS.
   */
  public static readonly VER_14_00_3223_3_V1 = SqlServerEngineVersion.of('14.00.3223.3.v1', '14.00');
  /** Version "14.00.3281.6.v1". */
  public static readonly VER_14_00_3281_6_V1 = SqlServerEngineVersion.of('14.00.3281.6.v1', '14.00');
  /** Version "14.00.3294.2.v1". */
  public static readonly VER_14_00_3294_2_V1 = SqlServerEngineVersion.of('14.00.3294.2.v1', '14.00');
  /** Version "14.00.3356.20.v1". */
  public static readonly VER_14_00_3356_20_V1 = SqlServerEngineVersion.of('14.00.3356.20.v1', '14.00');
  /** Version "14.00.3381.3.v1". */
  public static readonly VER_14_00_3381_3_V1 = SqlServerEngineVersion.of('14.00.3381.3.v1', '14.00');
  /** Version "14.00.3401.7.v1". */
  public static readonly VER_14_00_3401_7_V1 = SqlServerEngineVersion.of('14.00.3401.7.v1', '14.00');
  /** Version "14.00.3421.10.v1". */
  public static readonly VER_14_00_3421_10_V1 = SqlServerEngineVersion.of('14.00.3421.10.v1', '14.00');
  /** Version "14.00.3451.2.v1". */
  public static readonly VER_14_00_3451_2_V1 = SqlServerEngineVersion.of('14.00.3451.2.v1', '14.00');
  /** Version "14.00.3460.9.v1". */
  public static readonly VER_14_00_3460_9_V1 = SqlServerEngineVersion.of('14.00.3460.9.v1', '14.00');
  /** Version "14.00.3465.1.v1". */
  public static readonly VER_14_00_3465_1_V1 = SqlServerEngineVersion.of('14.00.3465.1.v1', '14.00');
  /** Version "14.00.3471.2.v1 ". */
  public static readonly VER_14_00_3471_2_V1 = SqlServerEngineVersion.of('14.00.3471.2.v1', '14.00');
  /** Version "14.00.3475.1.v1 ". */
  public static readonly VER_14_00_3475_1_V1 = SqlServerEngineVersion.of('14.00.3475.1.v1', '14.00');
  /** Version "14.00.3480.1.v1 ". */
  public static readonly VER_14_00_3480_1_V1 = SqlServerEngineVersion.of('14.00.3480.1.v1', '14.00');
  /** Version "14.00.3485.1.v1 ". */
  public static readonly VER_14_00_3485_1_V1 = SqlServerEngineVersion.of('14.00.3485.1.v1', '14.00');
  /** Version "14.00.3495.9.v1 ". */
  public static readonly VER_14_00_3495_9_V1 = SqlServerEngineVersion.of('14.00.3495.9.v1', '14.00');
  /** Version "14.00.3500.1.v1". */
  public static readonly VER_14_00_3500_1_V1 = SqlServerEngineVersion.of('14.00.3500.1.v1', '14.00');
  /** Version "14.00.3505.1.v1". */
  public static readonly VER_14_00_3505_1_V1 = SqlServerEngineVersion.of('14.00.3505.1.v1', '14.00');

  /** Version "15.00" (only a major version, without a specific minor version). */
  public static readonly VER_15 = SqlServerEngineVersion.of('15.00', '15.00');
  /** Version "15.00.4043.16.v1". */
  public static readonly VER_15_00_4043_16_V1 = SqlServerEngineVersion.of('15.00.4043.16.v1', '15.00');
  /** Version "15.00.4355.3.v1". */
  public static readonly VER_15_00_4355_3_V1 = SqlServerEngineVersion.of('15.00.4355.3.v1', '15.00');
  /**
   * Version "15.00.4043.23.v1".
   * @deprecated This version is erroneous. You might be looking for `SqlServerEngineVersion.VER_15_00_4073_23_V1`, instead.
   */
  public static readonly VER_15_00_4043_23_V1 = SqlServerEngineVersion.of('15.00.4043.23.v1', '15.00');
  /** Version "15.00.4073.23.v1". */
  public static readonly VER_15_00_4073_23_V1 = SqlServerEngineVersion.of('15.00.4073.23.v1', '15.00');
  /** Version "15.00.4153.1.v1". */
  public static readonly VER_15_00_4153_1_V1 = SqlServerEngineVersion.of('15.00.4153.1.v1', '15.00');
  /** Version "15.00.4198.2.v1". */
  public static readonly VER_15_00_4198_2_V1 = SqlServerEngineVersion.of('15.00.4198.2.v1', '15.00');
  /** Version "15.00.4236.7.v1". */
  public static readonly VER_15_00_4236_7_V1 = SqlServerEngineVersion.of('15.00.4236.7.v1', '15.00');
  /** Version "15.00.4312.2.v1". */
  public static readonly VER_15_00_4312_2_V1 = SqlServerEngineVersion.of('15.00.4312.2.v1', '15.00');
  /** Version "15.00.4316.3.v1". */
  public static readonly VER_15_00_4316_3_V1 = SqlServerEngineVersion.of('15.00.4316.3.v1', '15.00');
  /** Version "15.00.4322.2.v1". */
  public static readonly VER_15_00_4322_2_V1 = SqlServerEngineVersion.of('15.00.4322.2.v1', '15.00');
  /** Version "15.00.4335.1.v1". */
  public static readonly VER_15_00_4335_1_V1 = SqlServerEngineVersion.of('15.00.4335.1.v1', '15.00');
  /** Version "15.00.4345.5.v1". */
  public static readonly VER_15_00_4345_5_V1 = SqlServerEngineVersion.of('15.00.4345.5.v1', '15.00');
  /** Version "15.00.4365.2.v1". */
  public static readonly VER_15_00_4365_2_V1 = SqlServerEngineVersion.of('15.00.4365.2.v1', '15.00');
  /** Version "15.00.4375.4.v1". */
  public static readonly VER_15_00_4375_4_V1 = SqlServerEngineVersion.of('15.00.4375.4.v1', '15.00');
  /** Version "15.00.4382.1.v1". */
  public static readonly VER_15_00_4382_1_V1 = SqlServerEngineVersion.of('15.00.4382.1.v1', '15.00');
  /** Version "15.00.4385.2.v1". */
  public static readonly VER_15_00_4385_2_V1 = SqlServerEngineVersion.of('15.00.4385.2.v1', '15.00');
  /** Version "15.00.4390.2.v1". */
  public static readonly VER_15_00_4390_2_V1 = SqlServerEngineVersion.of('15.00.4390.2.v1', '15.00');
  /** Version "15.00.4395.2.v1". */
  public static readonly VER_15_00_4395_2_V1 = SqlServerEngineVersion.of('15.00.4395.2.v1', '15.00');
  /** Version "15.00.4410.1.v1". */
  public static readonly VER_15_00_4410_1_V1 = SqlServerEngineVersion.of('15.00.4410.1.v1', '15.00');
  /** Version "15.00.4415.2.v1". */
  public static readonly VER_15_00_4415_2_V1 = SqlServerEngineVersion.of('15.00.4415.2.v1', '15.00');
  /** Version "15.00.4420.2.v1". */
  public static readonly VER_15_00_4420_2_V1 = SqlServerEngineVersion.of('15.00.4420.2.v1', '15.00');
  /** Version "15.00.4430.1.v1". */
  public static readonly VER_15_00_4430_1_V1 = SqlServerEngineVersion.of('15.00.4430.1.v1', '15.00');
  /** Version "15.00.4435.7.v1". */
  public static readonly VER_15_00_4435_7_V1 = SqlServerEngineVersion.of('15.00.4435.7.v1', '15.00');
  /** Version "15.00.4440.1.v1". */
  public static readonly VER_15_00_4440_1_V1 = SqlServerEngineVersion.of('15.00.4440.1.v1', '15.00');
  /** Version "15.00.4445.1.v1". */
  public static readonly VER_15_00_4445_1_V1 = SqlServerEngineVersion.of('15.00.4445.1.v1', '15.00');
  /** Version "15.00.4455.2.v1". */
  public static readonly VER_15_00_4455_2_V1 = SqlServerEngineVersion.of('15.00.4455.2.v1', '15.00');

  /** Version "16.00" (only a major version, without a specific minor version). */
  public static readonly VER_16 = SqlServerEngineVersion.of('16.00', '16.00');
  /** Version "16.00.4085.2.v1". */
  public static readonly VER_16_00_4085_2_V1 = SqlServerEngineVersion.of('16.00.4085.2.v1', '16.00');
  /** Version "16.00.4095.4.v1". */
  public static readonly VER_16_00_4095_4_V1 = SqlServerEngineVersion.of('16.00.4095.4.v1', '16.00');
  /** Version "16.00.4105.2.v1". */
  public static readonly VER_16_00_4105_2_V1 = SqlServerEngineVersion.of('16.00.4105.2.v1', '16.00');
  /** Version "16.00.4115.5.v1". */
  public static readonly VER_16_00_4115_5_V1 = SqlServerEngineVersion.of('16.00.4115.5.v1', '16.00');
  /** Version "16.00.4120.1.v1". */
  public static readonly VER_16_00_4120_1_V1 = SqlServerEngineVersion.of('16.00.4120.1.v1', '16.00');
  /** Version "16.00.4125.3.v1". */
  public static readonly VER_16_00_4125_3_V1 = SqlServerEngineVersion.of('16.00.4125.3.v1', '16.00');
  /** Version "16.00.4131.2.v1". */
  public static readonly VER_16_00_4131_2_V1 = SqlServerEngineVersion.of('16.00.4131.2.v1', '16.00');
  /** Version "16.00.4135.4.v1". */
  public static readonly VER_16_00_4135_4_V1 = SqlServerEngineVersion.of('16.00.4135.4.v1', '16.00');
  /** Version "16.00.4140.3.v1". */
  public static readonly VER_16_00_4140_3_V1 = SqlServerEngineVersion.of('16.00.4140.3.v1', '16.00');
  /** Version "16.00.4150.1.v1". */
  public static readonly VER_16_00_4150_1_V1 = SqlServerEngineVersion.of('16.00.4150.1.v1', '16.00');
  /** Version "16.00.4165.4.v1". */
  public static readonly VER_16_00_4165_4_V1 = SqlServerEngineVersion.of('16.00.4165.4.v1', '16.00');
  /** Version "16.00.4175.1.v1". */
  public static readonly VER_16_00_4175_1_V1 = SqlServerEngineVersion.of('16.00.4175.1.v1', '16.00');
  /** Version "16.00.4185.3.v1". */
  public static readonly VER_16_00_4185_3_V1 = SqlServerEngineVersion.of('16.00.4185.3.v1', '16.00');
  /** Version "16.00.4195.2.v1". */
  public static readonly VER_16_00_4195_2_V1 = SqlServerEngineVersion.of('16.00.4195.2.v1', '16.00');
  /** Version "16.00.4205.1.v1". */
  public static readonly VER_16_00_4205_1_V1 = SqlServerEngineVersion.of('16.00.4205.1.v1', '16.00');
  /** Version "16.00.4210.1.v1". */
  public static readonly VER_16_00_4210_1_V1 = SqlServerEngineVersion.of('16.00.4210.1.v1', '16.00');
  /** Version "16.00.4215.2.v1". */
  public static readonly VER_16_00_4215_2_V1 = SqlServerEngineVersion.of('16.00.4215.2.v1', '16.00');
  /** Version "16.00.4225.2.v1". */
  public static readonly VER_16_00_4225_2_V1 = SqlServerEngineVersion.of('16.00.4225.2.v1', '16.00');
  /** Version "16.00.4230.2.v1". */
  public static readonly VER_16_00_4230_2_V1 = SqlServerEngineVersion.of('16.00.4230.2.v1', '16.00');
  /** Version "16.00.4236.2.v1". */
  public static readonly VER_16_00_4236_2_V1 = SqlServerEngineVersion.of('16.00.4236.2.v1', '16.00');

  /**
   * Create a new SqlServerEngineVersion with an arbitrary version.
   *
   * @param sqlServerFullVersion the full version string,
   *   for example "15.00.3049.1.v1"
   * @param sqlServerMajorVersion the major version of the engine,
   *   for example "15.00"
   */
  public static of(sqlServerFullVersion: string, sqlServerMajorVersion: string): SqlServerEngineVersion {
    return new SqlServerEngineVersion(sqlServerFullVersion, sqlServerMajorVersion);
  }

  /** The full version string, for example, "15.00.3049.1.v1". */
  public readonly sqlServerFullVersion: string;
  /** The major version of the engine, for example, "15.00". */
  public readonly sqlServerMajorVersion: string;

  private constructor(sqlServerFullVersion: string, sqlServerMajorVersion: string) {
    this.sqlServerFullVersion = sqlServerFullVersion;
    this.sqlServerMajorVersion = sqlServerMajorVersion;
  }
}

interface SqlServerInstanceEngineProps {
  /** The exact version of the engine to use. */
  readonly version: SqlServerEngineVersion;
}

interface SqlServerInstanceEngineBaseProps {
  readonly engineType: string;
  readonly version?: SqlServerEngineVersion;
}

abstract class SqlServerInstanceEngineBase extends InstanceEngineBase {
  constructor(props: SqlServerInstanceEngineBaseProps) {
    super({
      ...props,
      singleUserRotationApplication: secretsmanager.SecretRotationApplication.SQLSERVER_ROTATION_SINGLE_USER,
      multiUserRotationApplication: secretsmanager.SecretRotationApplication.SQLSERVER_ROTATION_MULTI_USER,
      version: props.version
        ? {
          fullVersion: props.version.sqlServerFullVersion,
          majorVersion: props.version.sqlServerMajorVersion,
        }
        : undefined,
      parameterGroupFamily: props.version
        // for some reason, even though SQL Server major versions usually end in '.00',
        // the ParameterGroup family has to end in '.0'
        ? `${props.engineType}-${props.version.sqlServerMajorVersion.endsWith('.00')
          ? props.version.sqlServerMajorVersion.slice(0, -1)
          : props.version.sqlServerMajorVersion}`
        : undefined,
      features: {
        s3Import: 'S3_INTEGRATION',
        s3Export: 'S3_INTEGRATION',
      },
      engineFamily: 'SQLSERVER',
    });
  }

  public bindToInstance(scope: Construct, options: InstanceEngineBindOptions): InstanceEngineConfig {
    const config = super.bindToInstance(scope, options);

    let optionGroup = options.optionGroup;
    const s3Role = options.s3ImportRole ?? options.s3ExportRole;
    if (s3Role) {
      if (options.s3ImportRole && options.s3ExportRole && options.s3ImportRole !== options.s3ExportRole) {
        throw new ValidationError(lit`ImportExportRolesServerEngines`, 'S3 import and export roles must be the same for SQL Server engines', scope);
      }

      if (!optionGroup) {
        optionGroup = new OptionGroup(scope, 'InstanceOptionGroup', {
          engine: this,
          configurations: [],
        });
      }
      // https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Appendix.SQLServer.Options.BackupRestore.html
      optionGroup.addConfiguration({
        name: 'SQLSERVER_BACKUP_RESTORE',
        settings: { IAM_ROLE_ARN: s3Role.roleRef.roleArn },
      });
    }

    return {
      ...config,
      optionGroup,
    };
  }

  protected get supportsTimezone() { return true; }
}

/**
 * Properties for SQL Server Standard Edition instance engines.
 * Used in `DatabaseInstanceEngine.sqlServerSe`.
 */
export interface SqlServerSeInstanceEngineProps extends SqlServerInstanceEngineProps {
}

class SqlServerSeInstanceEngine extends SqlServerInstanceEngineBase {
  constructor(version?: SqlServerEngineVersion) {
    super({
      engineType: 'sqlserver-se',
      version,
    });
  }
}

/**
 * Properties for SQL Server Express Edition instance engines.
 * Used in `DatabaseInstanceEngine.sqlServerEx`.
 */
export interface SqlServerExInstanceEngineProps extends SqlServerInstanceEngineProps {
}

class SqlServerExInstanceEngine extends SqlServerInstanceEngineBase {
  constructor(version?: SqlServerEngineVersion) {
    super({
      engineType: 'sqlserver-ex',
      version,
    });
  }
}

/**
 * Properties for SQL Server Web Edition instance engines.
 * Used in `DatabaseInstanceEngine.sqlServerWeb`.
 */
export interface SqlServerWebInstanceEngineProps extends SqlServerInstanceEngineProps {
}

class SqlServerWebInstanceEngine extends SqlServerInstanceEngineBase {
  constructor(version?: SqlServerEngineVersion) {
    super({
      engineType: 'sqlserver-web',
      version,
    });
  }
}

/**
 * Properties for SQL Server Enterprise Edition instance engines.
 * Used in `DatabaseInstanceEngine.sqlServerEe`.
 */
export interface SqlServerEeInstanceEngineProps extends SqlServerInstanceEngineProps {
}

class SqlServerEeInstanceEngine extends SqlServerInstanceEngineBase {
  constructor(version?: SqlServerEngineVersion) {
    super({
      engineType: 'sqlserver-ee',
      version,
    });
  }
}

/**
 * A database instance engine. Provides mapping to DatabaseEngine used for
 * secret rotation.
 */
export class DatabaseInstanceEngine {
  /**
   * The unversioned 'mariadb' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `mariaDb()` method
   */
  public static readonly MARIADB: IInstanceEngine = new MariaDbInstanceEngine();

  /**
   * The unversioned 'mysql' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `mysql()` method
   */
  public static readonly MYSQL: IInstanceEngine = new MySqlInstanceEngine();

  /**
   * The unversioned 'oracle-ee' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `oracleEe()` method
   */
  public static readonly ORACLE_EE: IInstanceEngine = new OracleEeInstanceEngine();

  /**
   * The unversioned 'oracle-ee-cdb' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `oracleEeCdb()` method
   */
  public static readonly ORACLE_EE_CDB: IInstanceEngine = new OracleEeCdbInstanceEngine();

  /**
   * The unversioned 'oracle-se2' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `oracleSe2()` method
   */
  public static readonly ORACLE_SE2: IInstanceEngine = new OracleSe2InstanceEngine();

  /**
   * The unversioned 'oracle-se2-cdb' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `oracleSe2Cdb()` method
   */
  public static readonly ORACLE_SE2_CDB: IInstanceEngine = new OracleSe2CdbInstanceEngine();

  /**
   * The unversioned 'oracle-se1' instance engine.
   *
   * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
   */
  public static readonly ORACLE_SE1: IInstanceEngine = new OracleSe1InstanceEngine();

  /**
   * The unversioned 'oracle-se' instance engine.
   *
   * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
   */
  public static readonly ORACLE_SE: IInstanceEngine = new OracleSeInstanceEngine();

  /**
   * The unversioned 'postgres' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `postgres()` method
   */
  public static readonly POSTGRES: IInstanceEngine = new PostgresInstanceEngine();

  /**
   * The unversioned 'sqlserver-ee' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `sqlServerEe()` method
   */
  public static readonly SQL_SERVER_EE: IInstanceEngine = new SqlServerEeInstanceEngine();

  /**
   * The unversioned 'sqlserver-se' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `sqlServerSe()` method
   */
  public static readonly SQL_SERVER_SE: IInstanceEngine = new SqlServerSeInstanceEngine();

  /**
   * The unversioned 'sqlserver-ex' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `sqlServerEx()` method
   */
  public static readonly SQL_SERVER_EX: IInstanceEngine = new SqlServerExInstanceEngine();

  /**
   * The unversioned 'sqlserver-web' instance engine.
   *
   * NOTE: using unversioned engines is an availability risk.
   *   We recommend using versioned engines created using the `sqlServerWeb()` method
   */
  public static readonly SQL_SERVER_WEB: IInstanceEngine = new SqlServerWebInstanceEngine();

  /** Creates a new MariaDB instance engine. */
  public static mariaDb(props: MariaDbInstanceEngineProps): IInstanceEngine {
    return new MariaDbInstanceEngine(props.version);
  }

  /** Creates a new MySQL instance engine. */
  public static mysql(props: MySqlInstanceEngineProps): IInstanceEngine {
    return new MySqlInstanceEngine(props.version);
  }

  /** Creates a new PostgreSQL instance engine. */
  public static postgres(props: PostgresInstanceEngineProps): IInstanceEngine {
    return new PostgresInstanceEngine(props.version);
  }

  /**
   * Creates a new Oracle Standard Edition instance engine.
   * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
   */
  public static oracleSe(props: OracleSeInstanceEngineProps): IInstanceEngine {
    return new OracleSeInstanceEngine(props.version);
  }

  /**
   * Creates a new Oracle Standard Edition 1 instance engine.
   * @deprecated instances can no longer be created with this engine. See https://forums.aws.amazon.com/ann.jspa?annID=7341
   */
  public static oracleSe1(props: OracleSe1InstanceEngineProps): IInstanceEngine {
    return new OracleSe1InstanceEngine(props.version);
  }

  /** Creates a new Oracle Standard Edition 2 instance engine. */
  public static oracleSe2(props: OracleSe2InstanceEngineProps): IInstanceEngine {
    return new OracleSe2InstanceEngine(props.version);
  }

  /** Creates a new Oracle Standard Edition 2 (CDB) instance engine. */
  public static oracleSe2Cdb(props: OracleSe2CdbInstanceEngineProps): IInstanceEngine {
    return new OracleSe2CdbInstanceEngine(props.version);
  }

  /** Creates a new Oracle Enterprise Edition instance engine. */
  public static oracleEe(props: OracleEeInstanceEngineProps): IInstanceEngine {
    return new OracleEeInstanceEngine(props.version);
  }

  /** Creates a new Oracle Enterprise Edition (CDB) instance engine. */
  public static oracleEeCdb(props: OracleEeCdbInstanceEngineProps): IInstanceEngine {
    return new OracleEeCdbInstanceEngine(props.version);
  }

  /** Creates a new SQL Server Standard Edition instance engine. */
  public static sqlServerSe(props: SqlServerSeInstanceEngineProps): IInstanceEngine {
    return new SqlServerSeInstanceEngine(props.version);
  }

  /** Creates a new SQL Server Express Edition instance engine. */
  public static sqlServerEx(props: SqlServerExInstanceEngineProps): IInstanceEngine {
    return new SqlServerExInstanceEngine(props.version);
  }

  /** Creates a new SQL Server Web Edition instance engine. */
  public static sqlServerWeb(props: SqlServerWebInstanceEngineProps): IInstanceEngine {
    return new SqlServerWebInstanceEngine(props.version);
  }

  /** Creates a new SQL Server Enterprise Edition instance engine. */
  public static sqlServerEe(props: SqlServerEeInstanceEngineProps): IInstanceEngine {
    return new SqlServerEeInstanceEngine(props.version);
  }
}
