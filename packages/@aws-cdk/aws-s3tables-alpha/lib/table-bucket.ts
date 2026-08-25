import { EOL } from 'os';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as s3tables from 'aws-cdk-lib/aws-s3tables';
import type { IResource, ITaggableV2, RemovalPolicy, TagManager } from 'aws-cdk-lib/core';
import { Resource, UnscopedValidationError, Token } from 'aws-cdk-lib/core';
import { memoizedGetter, lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import * as perms from './permissions';
import { TableBucketPolicy } from './table-bucket-policy';
import { validateTableBucketAttributes } from './util';

/**
 * Interface definition for S3 Table Buckets
 */
export interface ITableBucket extends IResource {
  /**
   * The ARN of the table bucket.
   * @attribute
   */
  readonly tableBucketArn: string;

  /**
   * The name of the table bucket.
   * @attribute
   */
  readonly tableBucketName: string;

  /**
   * The accountId containing the table bucket.
   * @attribute
   */
  readonly account?: string;

  /**
   * The region containing the table bucket.
   * @attribute
   */
  readonly region?: string;

  /**
   * Optional KMS encryption key associated with this table bucket.
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * Adds a statement to the resource policy for a principal (i.e.
   * account/role/service) to perform actions on this table bucket and/or its
   * tables.
   *
   * Note that the policy statement may or may not be added to the policy.
   * For example, when an `ITableBucket` is created from an existing table bucket,
   * it's not possible to tell whether the bucket already has a policy
   * attached, let alone to re-use that policy to add more statements to it.
   * So it's safest to do nothing in these cases.
   *
   * @param statement the policy statement to be added to the bucket's
   * policy.
   * @returns metadata about the execution of this method. If the policy
   * was not added, the value of `statementAdded` will be `false`. You
   * should always check this value to make sure that the operation was
   * actually carried out. Otherwise, synthesis and deploy will terminate
   * silently, which may be confusing.
   */
  addToResourcePolicy(statement: iam.PolicyStatement): iam.AddToResourcePolicyResult;

  /**
   * Grant read permissions for this table bucket and its tables
   * to an IAM principal (Role/Group/User).
   *
   * If encryption is used, permission to use the key to decrypt the contents
   * of the bucket will also be granted to the same principal.
   *
   * @param identity The principal to allow read permissions to
   * @param tableId Allow the permissions to all tables using '*' or to single table by its unique ID.
   */
  grantRead(identity: iam.IGrantable, tableId: string): iam.Grant;

  /**
   * Grant write permissions for this table bucket and its tables
   * to an IAM principal (Role/Group/User).
   *
   * If encryption is used, permission to use the key to encrypt the contents
   * of the bucket will also be granted to the same principal.
   *
   * @param identity The principal to allow write permissions to
   * @param tableId Allow the permissions to all tables using '*' or to single table by its unique ID.
   */
  grantWrite(identity: iam.IGrantable, tableId: string): iam.Grant;

  /**
   * Grant read and write permissions for this table bucket and its tables
   * to an IAM principal (Role/Group/User).
   *
   * If encryption is used, permission to use the key to encrypt/decrypt the contents
   * of the bucket will also be granted to the same principal.
   *
   * @param identity The principal to allow read and write permissions to
   * @param tableId Allow the permissions to all tables using '*' or to single table by its unique ID.
   */
  grantReadWrite(identity: iam.IGrantable, tableId: string): iam.Grant;
}

/**
 * Unreferenced file removal settings for the this table bucket.
 */
export interface UnreferencedFileRemoval {
  /**
   * Duration after which noncurrent files should be removed. Should be at least one day.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-table-buckets-maintenance.html
   *
   * @default - See S3 Tables User Guide
   */
  readonly noncurrentDays?: number;

  /**
   * Status of unreferenced file removal. Can be Enabled or Disabled.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-table-buckets-maintenance.html
   *
   * @default - See S3 Tables User Guide
   */
  readonly status?: UnreferencedFileRemovalStatus;

  /**
   * Duration after which unreferenced files should be removed. Should be at least one day.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-table-buckets-maintenance.html
   *
   * @default - See S3 Tables User Guide
   */
  readonly unreferencedDays?: number;
}

/**
 * Controls whether unreferenced file removal is enabled or disabled.
 */
export enum UnreferencedFileRemovalStatus {
  /**
   * Enable unreferenced file removal.
   */
  ENABLED = 'Enabled',

  /**
   * Disable unreferenced file removal.
   */
  DISABLED = 'Disabled',
}

/**
 * Controls whether CloudWatch request metrics are enabled or disabled for the table bucket.
 */
export enum RequestMetricsStatus {
  /**
   * Enable CloudWatch request metrics for the table bucket.
   */
  ENABLED = 'Enabled',

  /**
   * Disable CloudWatch request metrics for the table bucket.
   */
  DISABLED = 'Disabled',
}

/**
 * Storage class for S3 Tables.
 *
 * Determines the storage tier used for table data, allowing optimization
 * for different access patterns and cost profiles.
 */
export enum StorageClass {
  /**
   * Standard storage class for frequently accessed data.
   *
   * Use this for tables with consistent, high-frequency access patterns.
   */
  STANDARD = 'STANDARD',

  /**
   * Intelligent-Tiering storage class that automatically moves data between access tiers.
   *
   * Use this for tables with unknown or changing access patterns to optimize costs
   * while maintaining performance for frequently accessed data.
   */
  INTELLIGENT_TIERING = 'INTELLIGENT_TIERING',
}

/**
 * Controls Server Side Encryption (SSE) for this TableBucket.
 */
export enum TableBucketEncryption {
  /**
   * Use a customer defined KMS key for encryption
   * If `encryptionKey` is specified, this key will be used, otherwise, one will be defined.
   */
  KMS = 'aws:kms',

  /**
   * Use S3 managed encryption keys with AES256 encryption
   */
  S3_MANAGED = 'AES256',
}

abstract class TableBucketBase extends Resource implements ITableBucket {
  public abstract readonly tableBucketArn: string;
  public abstract readonly tableBucketName: string;

  /**
   * The resource policy associated with this table bucket.
   *
   * If `autoCreatePolicy` is true, a `TableBucketPolicy` will be created upon the
   * first call to addToResourcePolicy(s).
   */
  public abstract tableBucketPolicy?: TableBucketPolicy;

  /**
   * Indicates if a table bucket resource policy should automatically created upon
   * the first call to `addToResourcePolicy`.
   */
  protected abstract autoCreatePolicy: boolean;

  public abstract encryptionKey?: kms.IKey | undefined;

  /**
   * Adds a statement to the resource policy for a principal (i.e.
   * account/role/service) to perform actions on this table bucket and/or its
   * contents. Use `tableBucketArn` and `arnForObjects(keys)` to obtain ARNs for
   * this bucket or objects.
   *
   * Note that the policy statement may or may not be added to the policy.
   * For example, when an `ITableBucket` is created from an existing table bucket,
   * it's not possible to tell whether the bucket already has a policy
   * attached, let alone to re-use that policy to add more statements to it.
   * So it's safest to do nothing in these cases.
   *
   * @param statement the policy statement to be added to the bucket's
   * policy.
   * @returns metadata about the execution of this method. If the policy
   * was not added, the value of `statementAdded` will be `false`. You
   * should always check this value to make sure that the operation was
   * actually carried out. Otherwise, synthesis and deploy will terminate
   * silently, which may be confusing.
   */
  public addToResourcePolicy(
    statement: iam.PolicyStatement,
  ): iam.AddToResourcePolicyResult {
    if (!this.tableBucketPolicy && this.autoCreatePolicy) {
      this.tableBucketPolicy = new TableBucketPolicy(this, 'DefaultPolicy', {
        tableBucket: this,
      });
    }

    if (this.tableBucketPolicy) {
      this.tableBucketPolicy.document.addStatements(statement);
      return { statementAdded: true, policyDependable: this.tableBucketPolicy };
    }

    return { statementAdded: false };
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantRead(identity: iam.IGrantable, tableId: string) {
    return this.grant(
      identity,
      perms.TABLE_BUCKET_READ_ACCESS,
      perms.KEY_READ_ACCESS,
      this.tableBucketArn,
      this.getTableArn(tableId),
    );
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantWrite(identity: iam.IGrantable, tableId: string) {
    return this.grant(
      identity,
      perms.TABLE_BUCKET_WRITE_ACCESS,
      perms.KEY_READ_WRITE_ACCESS,
      this.tableBucketArn,
      this.getTableArn(tableId),
    );
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantReadWrite(identity: iam.IGrantable, tableId: string) {
    return this.grant(
      identity,
      perms.TABLE_BUCKET_READ_WRITE_ACCESS,
      perms.KEY_WRITE_ACCESS,
      this.tableBucketArn,
      this.getTableArn(tableId),
    );
  }

  /**
   * Grants the given s3tables permissions to the provided principal
   * @returns Grant object
   */
  private grant(
    grantee: iam.IGrantable,
    tableBucketActions: string[],
    keyActions: string[],
    resourceArn: string,
    ...otherResourceArns: (string | undefined)[]) {
    const resources = [resourceArn, ...otherResourceArns].filter(arn => arn != undefined);

    const grant = iam.Grant.addToPrincipalOrResource({
      grantee,
      actions: tableBucketActions,
      resourceArns: resources,
      resource: this,
    });

    if (this.encryptionKey && keyActions && keyActions.length !== 0) {
      this.encryptionKey.grant(grantee, ...keyActions);
    }

    return grant;
  }

  private getTableArn(tableId: string | undefined) {
    return tableId ? `${this.tableBucketArn}/table/${tableId}` : undefined;
  }
}

/**
 * Parameters for constructing a TableBucket
 */
export interface TableBucketProps {
  /**
   * Name of the S3 TableBucket.
   * @link https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-buckets-naming.html#table-buckets-naming-rules
   */
  readonly tableBucketName: string;

  /**
   * Unreferenced file removal settings for the S3 TableBucket.
   * @link https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3tables-tablebucket-unreferencedfileremoval.html
   * @default Enabled with default values
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-table-buckets-maintenance.html
   */
  readonly unreferencedFileRemoval?: UnreferencedFileRemoval;

  /**
   * AWS region that the table bucket exists in.
   *
   * @default - it's assumed the bucket is in the same region as the scope it's being imported into
   */
  readonly region?: string;

  /**
   * AWS Account ID of the table bucket owner.
   *
   * @default - it's assumed the bucket belongs to the same account as the scope it's being imported into
   */
  readonly account?: string;

  /**
   * The kind of server-side encryption to apply to this bucket.
   *
   * If you choose KMS, you can specify a KMS key via `encryptionKey`. If
   * encryption key is not specified, a key will automatically be created.
   *
   * @default - `KMS` if `encryptionKey` is specified, or `S3_MANAGED` otherwise.
   */
  readonly encryption?: TableBucketEncryption;

  /**
   * External KMS key to use for bucket encryption.
   *
   * The `encryption` property must be either not specified or set to `KMS`.
   * An error will be emitted if `encryption` is set to `S3_MANAGED`.
   *
   * @default - If `encryption` is set to `KMS` and this property is undefined,
   * a new KMS key will be created and associated with this bucket.
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * Controls what happens to this table bucket it it stoped being managed by cloudformation.
   *
   * @default RETAIN
   */
  readonly removalPolicy?: RemovalPolicy;

  /**
   * CloudWatch request metrics configuration for the table bucket.
   *
   * When enabled, S3 Tables publishes CloudWatch request metrics for the table bucket.
   * Request metrics provide insight into Amazon S3 Tables requests.
   *
   * @default - Request metrics are disabled
   */
  readonly requestMetricsStatus?: RequestMetricsStatus;

  /**
   * The storage class for the table bucket.
   *
   * Determines the storage tier used for table data, allowing optimization
   * for different access patterns and cost profiles.
   *
   * @default - STANDARD storage class
   */
  readonly storageClass?: StorageClass;
}

/**
 * A reference to a table bucket outside this stack
 *
 * The tableBucketName, region, and account can be provided explicitly
 * or will be inferred from the tableBucketArn
 */
export interface TableBucketAttributes {
  /**
   * AWS region this table bucket exists in
   * @default region inferred from scope
   */
  readonly region?: string;

  /**
   * The accountId containing this table bucket
   * @default account inferred from scope
   */
  readonly account?: string;

  /**
   * The table bucket name, unique per region
   * @default tableBucketName inferred from arn
   */
  readonly tableBucketName?: string;

  /**
   * The table bucket's ARN.
   * @default tableBucketArn constructed from region, account and tableBucketName are provided
   */
  readonly tableBucketArn?: string;

  /**
   * Optional KMS encryption key associated with this bucket.
   * @default - undefined
   */
  readonly encryptionKey?: kms.IKey;
}

/**
 * An S3 table bucket with helpers for associated resource policies
 *
 * This bucket may not yet have all features that exposed by the underlying CfnTableBucket.
 *
 * @stateful
 * @example
 * const sampleTableBucket = new TableBucket(scope, 'ExampleTableBucket', {
 *   tableBucketName: 'example-bucket',
 *   // Optional fields:
 *   unreferencedFileRemoval: {
 *     noncurrentDays: 123,
 *     status: UnreferencedFileRemovalStatus.ENABLED,
 *     unreferencedDays: 123,
 *   },
 * });
 */
@propertyInjectable
export class TableBucket extends TableBucketBase implements ITaggableV2 {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-s3tables-alpha.TableBucket';

  /**
   * Defines a TableBucket construct from an external table bucket ARN.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param tableBucketArn Amazon Resource Name (arn) of the table bucket
   */
  public static fromTableBucketArn(scope: Construct, id: string, tableBucketArn: string): ITableBucket {
    return TableBucket.fromTableBucketAttributes(scope, id, { tableBucketArn });
  }

  /**
   * Defines a TableBucket construct that represents an external table bucket.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param attrs A `TableBucketAttributes` object. Can be manually created.
   */
  public static fromTableBucketAttributes(
    scope: Construct,
    id: string,
    attrs: TableBucketAttributes,
  ): ITableBucket {
    const { tableBucketName, region, account, tableBucketArn } = validateTableBucketAttributes(scope, attrs);
    TableBucket.validateTableBucketName(tableBucketName);
    class Import extends TableBucketBase {
      public readonly tableBucketName = tableBucketName!;
      public readonly tableBucketArn = tableBucketArn;
      public readonly tableBucketPolicy?: TableBucketPolicy;
      public readonly region = region;
      public readonly account = account;
      public readonly encryptionKey?: kms.IKey = attrs.encryptionKey;
      protected autoCreatePolicy: boolean = false;

      /**
       * Exports this bucket from the stack.
       */
      public export() {
        return attrs;
      }
    }

    return new Import(scope, id, {
      account,
      region,
      physicalName: tableBucketName,
    });
  }

  /**
   * Throws an exception if the given table bucket name is not valid.
   *
   * @param bucketName name of the bucket.
   */
  public static validateTableBucketName(
    bucketName: string | undefined,
  ) {
    if (bucketName == undefined || Token.isUnresolved(bucketName)) {
      // the name is a late-bound value, not a defined string, so skip validation
      return;
    }

    const errors: string[] = [];

    // Length validation
    if (bucketName.length < 3 || bucketName.length > 63) {
      errors.push(
        'Bucket name must be at least 3 and no more than 63 characters',
      );
    }

    // Character set validation
    const illegalCharsetRegEx = /[^a-z0-9-]/;
    const allowedEdgeCharsetRegEx = /[a-z0-9]/;

    const illegalCharMatch = bucketName.match(illegalCharsetRegEx);
    if (illegalCharMatch) {
      errors.push(
        'Bucket name must only contain lowercase characters, numbers, and hyphens (-)' +
          ` (offset: ${illegalCharMatch.index})`,
      );
    }

    // Edge character validation
    if (!allowedEdgeCharsetRegEx.test(bucketName.charAt(0))) {
      errors.push(
        'Bucket name must start with a lowercase letter or number (offset: 0)',
      );
    }
    if (
      !allowedEdgeCharsetRegEx.test(bucketName.charAt(bucketName.length - 1))
    ) {
      errors.push(
        `Bucket name must end with a lowercase letter or number (offset: ${
          bucketName.length - 1
        })`,
      );
    }

    if (errors.length > 0) {
      throw new UnscopedValidationError(
        lit`InvalidTableBucketName`,
        `Invalid S3 table bucket name (value: ${bucketName})${EOL}${errors.join(EOL)}`,
      );
    }
  }

  /**
   * Throws an exception if the given unreferencedFileRemovalProperty is not valid.
   * @param unreferencedFileRemoval configuration for the table bucket
   */
  public static validateUnreferencedFileRemoval(
    unreferencedFileRemoval?: UnreferencedFileRemoval,
  ): void {
    // Skip validation if property is not defined
    if (!unreferencedFileRemoval) {
      return;
    }

    const { noncurrentDays, status, unreferencedDays } = unreferencedFileRemoval;

    const errors: string[] = [];

    if (noncurrentDays != undefined) {
      if (noncurrentDays < 1) {
        errors.push('noncurrentDays must be at least 1 day');
      }
      if (!Number.isInteger(noncurrentDays)) {
        errors.push('noncurrentDays must be a whole number');
      }
    }

    if (unreferencedDays != undefined) {
      if (unreferencedDays < 1) {
        errors.push('unreferencedDays must be at least 1 day');
      }
      if (!Number.isInteger(noncurrentDays)) {
        errors.push('unreferencedDays must be a whole number');
      }
    }

    const allowedStatus = ['Enabled', 'Disabled'];
    if (status != undefined && !allowedStatus.includes(status)) {
      errors.push('status must be one of \'Enabled\' or \'Disabled\'');
    }

    if (errors.length > 0) {
      throw new UnscopedValidationError(
        lit`InvalidUnreferencedFileRemovalProperty`,
        `Invalid UnreferencedFileRemovalProperty})${EOL}${errors.join(EOL)}`,
      );
    }
  }

  /**
   * The underlying CfnTableBucket L1 resource
   * @internal
   */
  private readonly resource: s3tables.CfnTableBucket;

  /**
   * The tag manager for this resource.
   */
  public readonly cdkTagManager: TagManager;

  /**
   * The resource policy for this tableBucket.
   */
  public readonly tableBucketPolicy?: TableBucketPolicy;

  public readonly encryptionKey?: kms.IKey | undefined;

  protected autoCreatePolicy: boolean = true;

  constructor(scope: Construct, id: string, props: TableBucketProps) {
    super(scope, id, {
      physicalName: props.tableBucketName,
    });

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    TableBucket.validateTableBucketName(props.tableBucketName);
    TableBucket.validateUnreferencedFileRemoval(props.unreferencedFileRemoval);
    const { bucketEncryption, encryptionKey } = this.parseEncryption(props);
    this.encryptionKey = encryptionKey;

    this.resource = new s3tables.CfnTableBucket(this, id, {
      tableBucketName: props.tableBucketName,
      unreferencedFileRemoval: {
        ...props.unreferencedFileRemoval,
        noncurrentDays: props.unreferencedFileRemoval?.noncurrentDays,
        unreferencedDays: props.unreferencedFileRemoval?.unreferencedDays,
      },
      encryptionConfiguration: bucketEncryption,
      metricsConfiguration: props.requestMetricsStatus ? { status: props.requestMetricsStatus } : undefined,
      storageClassConfiguration: props.storageClass ? { storageClass: props.storageClass } : undefined,
    });

    this.cdkTagManager = this.resource.cdkTagManager;
    this.resource.applyRemovalPolicy(props.removalPolicy);
  }

  /**
   * The name of this table bucket
   */
  @memoizedGetter
  public get tableBucketName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  /**
   * The unique Amazon Resource Name (arn) of this table bucket
   */
  @memoizedGetter
  public get tableBucketArn(): string {
    return this.resource.attrTableBucketArn;
  }

  /**
   * Set up key properties and return the Bucket encryption property from the
   * user's configuration, according to the following table:
   *
   * | props.encryption | props.encryptionKey | bucketEncryption (return value) | encryptionKey (return value)  |
   * |------------------|---------------------|---------------------------------|-------------------------------|
   * | undefined        | undefined           | undefined                       | undefined                     |
   * | undefined        | k                   | aws:kms                         | k                             |
   * | KMS              | undefined           | aws:kms                         | new key (allow maintenance SP)|
   * | KMS              | k                   | aws:kms                         | k                             |
   * | S3_MANAGED       | undefined           | AES256                          | undefined                     |
   * | S3_MANAGED       | k                   | ERROR!                          | ERROR!                        |
   */
  private parseEncryption(props: TableBucketProps): {
    bucketEncryption?: s3tables.CfnTableBucket.EncryptionConfigurationProperty;
    encryptionKey?: kms.IKey;
  } {
    const encryptionType = props.encryption;
    let key = props.encryptionKey;

    if (encryptionType === undefined) {
      if ( key === undefined ) {
        return { bucketEncryption: undefined, encryptionKey: undefined };
      } else {
        return {
          bucketEncryption: {
            kmsKeyArn: key.keyArn,
            sseAlgorithm: TableBucketEncryption.KMS,
          },
          encryptionKey: key,
        };
      }
    }

    if (encryptionType === TableBucketEncryption.KMS) {
      if ( key === undefined ) {
        key = new kms.Key(this, 'Key', {
          description: `Created by ${this.node.path}`,
          enableKeyRotation: true,
        });
        this.allowTablesMaintenanceAccessToKey(key, props.tableBucketName);
      }
      return {
        bucketEncryption: {
          kmsKeyArn: key.keyArn,
          sseAlgorithm: TableBucketEncryption.KMS,
        },
        encryptionKey: key,
      };
    }

    if (encryptionType === TableBucketEncryption.S3_MANAGED) {
      if ( key === undefined ) {
        return {
          bucketEncryption: {
            sseAlgorithm: TableBucketEncryption.S3_MANAGED,
          },
        };
      } else {
        throw new UnscopedValidationError(lit`InvalidEncryptionConfiguration`, 'Expected encryption = `KMS` with user provided encryption key');
      }
    }
    throw new UnscopedValidationError(lit`UnknownEncryptionConfiguration`, `Unknown encryption configuration detected: ${props.encryption} with key ${props.encryptionKey}`);
  }

  /**
   * Allowlist S3 Tables Maintenance to access this table bucket's encryption key
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-kms-permissions.html
   * @param encryptionKey The key to provide access to
   */
  private allowTablesMaintenanceAccessToKey(encryptionKey: kms.IKey, tableBucketName: string) {
    const region = this.stack.region;
    const account = this.stack.account;
    const partition = this.stack.partition;

    encryptionKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowS3TablesMaintenanceAccess',
      effect: iam.Effect.ALLOW,
      principals: [
        new iam.ServicePrincipal('maintenance.s3tables.amazonaws.com'),
      ],
      actions: [
        'kms:GenerateDataKey',
        'kms:Decrypt',
      ],
      resources: ['*'],
      conditions: {
        StringLike: {
          'kms:EncryptionContext:aws:s3:arn': `arn:${partition}:s3tables:${region}:${account}:bucket/${tableBucketName}/*`,
        },
      },
    }));
  }
}
