import { Annotations, ValidationError } from 'aws-cdk-lib';
import { CfnTable } from 'aws-cdk-lib/aws-glue';
import type * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { memoizedGetter, lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { Column } from './schema';
import type { PartitionIndex, TableBaseProps } from './table-base';
import { TableBase } from './table-base';

/**
 * Server-side encryption for the S3 bucket that a managed `S3Table` creates.
 *
 * Applies only when the table manages its own bucket (via
 * `S3TableStorage.managedBucket`). An existing bucket keeps whatever encryption
 * it was created with.
 */
export class S3TableEncryption {
  /**
   * Server-side encryption (SSE-S3) with an Amazon S3-managed key.
   */
  public static s3Managed(): S3TableEncryption {
    return new S3TableEncryption(s3.BucketEncryption.S3_MANAGED);
  }

  /**
   * Server-side encryption (SSE-KMS) with an AWS KMS key managed by the account owner.
   *
   * @param key the KMS key used to encrypt the data. A key is created if one is not provided.
   */
  public static kms(key?: kms.IKey): S3TableEncryption {
    return new S3TableEncryption(s3.BucketEncryption.KMS, key);
  }

  /**
   * Server-side encryption (SSE-KMS) with an AWS KMS key managed by the KMS service.
   */
  public static kmsManaged(): S3TableEncryption {
    return new S3TableEncryption(s3.BucketEncryption.KMS_MANAGED);
  }

  /** @internal */
  public readonly _bucketEncryption: s3.BucketEncryption;

  /**
   * @internal
   * Typed `kms.IKey` (not `IKeyRef`) because it is forwarded to `s3.Bucket`, whose `encryptionKey` prop requires `IKey`.
   */
  public readonly _kmsKey?: kms.IKey;

  private constructor(bucketEncryption: s3.BucketEncryption, kmsKey?: kms.IKey) {
    this._bucketEncryption = bucketEncryption;
    this._kmsKey = kmsKey;
  }
}

/**
 * Where an `S3Table` stores its data.
 *
 * The two paths are mutually exclusive: a managed bucket may specify its
 * server-side encryption, while an existing bucket keeps its own encryption — so
 * an encryption choice can never be paired with a bring-your-own bucket.
 */
export class S3TableStorage {
  /**
   * Store the table's data in a bucket created and managed by the table.
   *
   * @param encryption the server-side encryption for the created bucket.
   * @default - S3-managed (SSE-S3) encryption
   */
  public static managedBucket(encryption?: S3TableEncryption): S3TableStorage {
    return new S3TableStorage(undefined, encryption);
  }

  /**
   * Store the table's data in an existing bucket. CDK does not manage the
   * bucket's encryption.
   *
   * The bucket can be one you don't own, imported with
   * `Bucket.fromBucketArn()` or `Bucket.fromBucketAttributes()`. If that bucket
   * is KMS-encrypted, import it with `Bucket.fromBucketAttributes()` and supply
   * the `encryptionKey` attribute. Otherwise, CDK has no reference to the key,
   * which means that `S3Table.grantRead()`/`grantWrite()` will correctly grant
   * S3 access but silently skip the KMS permissions on the key. As a consequence,
   * at runtime, reads and writes will fail with access denied on the key.
   *
   * @param bucket the bucket that holds the table's data.
   */
  public static fromBucket(bucket: s3.IBucket): S3TableStorage {
    return new S3TableStorage(bucket, undefined);
  }

  /** @internal */
  public readonly _bucket?: s3.IBucket;

  /** @internal */
  public readonly _encryption?: S3TableEncryption;

  private constructor(bucket?: s3.IBucket, encryption?: S3TableEncryption) {
    this._bucket = bucket;
    this._encryption = encryption;
  }
}

/**
 * Client-side encryption for an `S3Table`'s data.
 *
 * Independent of the bucket's server-side encryption and of who owns the bucket:
 * the data is encrypted by the client before it is written to S3. When set, the
 * `grant*` methods also grant the relevant KMS permissions on the key.
 */
export class TableClientSideEncryption {
  /**
   * Client-side encryption (CSE-KMS) with an AWS KMS key managed by the account owner.
   *
   * @param key the KMS key used to encrypt the data. A key is created if one is not provided.
   */
  public static kms(key?: kms.IKeyRef): TableClientSideEncryption {
    return new TableClientSideEncryption(key);
  }

  /** @internal */
  public readonly _kmsKey?: kms.IKeyRef;

  private constructor(kmsKey?: kms.IKeyRef) {
    this._kmsKey = kmsKey;
  }
}

export interface S3TableProps extends TableBaseProps {
  /**
   * Where the table's data is stored: a bucket created and managed by the table,
   * or an existing bucket you provide.
   *
   * @default - a managed bucket with S3-managed (SSE-S3) encryption
   */
  readonly storage?: S3TableStorage;

  /**
   * S3 prefix under which table objects are stored.
   *
   * When the table shares a bucket with other tables or consumers, set this so
   * that the `grant*` methods scope S3 access to this table's data. Without a
   * prefix, those grants cover the entire bucket.
   *
   * @default - No prefix. The data will be stored under the root of the bucket.
   */
  readonly s3Prefix?: string;

  /**
   * Client-side encryption (CSE-KMS) for the table's data.
   *
   * Independent of the bucket's server-side encryption, and valid whether the
   * bucket is managed or provided.
   *
   * @default - no client-side encryption
   */
  readonly clientSideEncryption?: TableClientSideEncryption;
}

/**
 * A Glue table that targets a S3 dataset.
 * @resource AWS::Glue::Table
 */
@propertyInjectable
export class S3Table extends TableBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.S3Table';

  private resource: CfnTable;

  /**
   * S3 bucket in which the table's data resides.
   */
  public readonly bucket: s3.IBucket;

  /**
   * S3 Key Prefix under which this table's files are stored in S3.
   */
  public readonly s3Prefix: string;

  /**
   * The KMS key used for client-side encryption of the table's data, if
   * `clientSideEncryption` was configured. Otherwise, `undefined`.
   *
   * For server-side (bucket) encryption, read `bucket.encryptionKey` instead.
   */
  public readonly clientSideEncryptionKey?: kms.IKeyRef;

  /**
   * This table's partition indexes.
   */
  public readonly partitionIndexes?: PartitionIndex[];

  protected readonly tableResource: CfnTable;

  /**
   * Whether the data bucket was supplied by the user (as opposed to created by
   * this construct). A user-supplied bucket may hold data for other tables, so
   * granting access to the whole bucket can over-grant.
   */
  private readonly userProvidedBucket: boolean;

  constructor(scope: Construct, id: string, props: S3TableProps) {
    super(scope, id, props);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);
    this.s3Prefix = props.s3Prefix ?? '';
    const storage = props.storage ?? S3TableStorage.managedBucket();
    this.userProvidedBucket = storage._bucket !== undefined;
    this.bucket = storage._bucket ?? this.createManagedBucket(storage._encryption);
    if (props.clientSideEncryption) {
      // CSE-KMS: use the provided key or create one automatically. The bucket's
      // own server-side encryption is independent and comes from `storage`.
      this.clientSideEncryptionKey = props.clientSideEncryption._kmsKey ?? new kms.Key(this, 'Key');
    }

    this.resource = new CfnTable(this, 'Table', {
      catalogId: props.database.catalog.catalogId,

      databaseName: props.database.databaseName,

      tableInput: {
        name: this.physicalName,
        description: props.description || `${this.physicalName} generated by CDK`,

        partitionKeys: renderColumns(props.partitionKeys),

        parameters: {
          'classification': props.dataFormat.classificationString?.value,
          'partition_filtering.enabled': props.enablePartitionFiltering,
          ...this.parameters,
          // Managed keys are emitted last so free-form `parameters` cannot
          // silently override them. Conflicts are rejected in `TableBase`.
          'has_encrypted_data': this.hasEncryptedData,
        },
        storageDescriptor: {
          location: `s3://${this.bucket.bucketName}/${this.s3Prefix}`,
          compressed: this.compressed,
          storedAsSubDirectories: props.storedAsSubDirectories ?? false,
          columns: renderColumns(props.columns),
          inputFormat: props.dataFormat.inputFormat.className,
          outputFormat: props.dataFormat.outputFormat.className,
          serdeInfo: {
            serializationLibrary: props.dataFormat.serializationLibrary.className,
          },
          parameters: props.storageParameters ? props.storageParameters.reduce((acc, param) => {
            if (param.key in acc) {
              throw new ValidationError(lit`DuplicateStorageParameterKey`, `Duplicate storage parameter key: ${param.key}`, this);
            }
            const key = param.key;
            acc[key] = param.value;
            return acc;
          }, {} as { [key: string]: string }) : undefined,
        },

        tableType: 'EXTERNAL_TABLE',
      },
    });

    this.tableResource = this.resource;
    this.node.defaultChild = this.resource;

    // Partition index creation relies on created table.
    if (props.partitionIndexes) {
      this.partitionIndexes = props.partitionIndexes;
      this.partitionIndexes.forEach((index) => this.addPartitionIndex(index));
    }
  }

  /**
   * Name of this table.
   */
  @memoizedGetter
  public get tableName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  /**
   * ARN of this table.
   */
  @memoizedGetter
  public get tableArn(): string {
    return this.stack.formatArn({
      service: 'glue',
      resource: 'table',
      resourceName: `${this.database.databaseName}/${this.tableName}`,
    });
  }

  /**
   * Grant read permissions to the table and the underlying data stored in S3 to an IAM principal.
   * [disable-awslint:no-grants]
   *
   * @param grantee the principal
   */
  @MethodMetadata()
  public grantRead(grantee: iam.IGrantable): iam.Grant {
    const ret = this.grant(grantee, readPermissions);
    if (this.clientSideEncryptionKey) { kms.KeyGrants.fromKey(this.clientSideEncryptionKey).decrypt(grantee); }
    this.bucket.grantRead(grantee, this.generateS3PrefixForGrant());
    return ret;
  }

  /**
   * Grant write permissions to the table and the underlying data stored in S3 to an IAM principal.
   * [disable-awslint:no-grants]
   *
   * @param grantee the principal
   */
  @MethodMetadata()
  public grantWrite(grantee: iam.IGrantable): iam.Grant {
    const ret = this.grant(grantee, writePermissions);
    if (this.clientSideEncryptionKey) { kms.KeyGrants.fromKey(this.clientSideEncryptionKey).encrypt(grantee); }
    this.bucket.grantWrite(grantee, this.generateS3PrefixForGrant());
    return ret;
  }

  /**
   * Grant read and write permissions to the table and the underlying data stored in S3 to an IAM principal.
   * [disable-awslint:no-grants]
   *
   * @param grantee the principal
   */
  @MethodMetadata()
  public grantReadWrite(grantee: iam.IGrantable): iam.Grant {
    const ret = this.grant(grantee, [...readPermissions, ...writePermissions]);
    if (this.clientSideEncryptionKey) { kms.KeyGrants.fromKey(this.clientSideEncryptionKey).encryptDecrypt(grantee); }
    this.bucket.grantReadWrite(grantee, this.generateS3PrefixForGrant());
    return ret;
  }

  protected generateS3PrefixForGrant() {
    // When the user supplied their own bucket and did not scope the table to a
    // prefix, the grant covers every object in the bucket - which may include
    // data owned by other tables or consumers sharing that bucket. Warn so the
    // over-grant is a deliberate choice rather than a silent surprise.
    if (this.userProvidedBucket && this.s3Prefix === '') {
      Annotations.of(this).addWarningV2(
        '@aws-cdk/aws-glue-alpha:grantScopedToWholeBucket',
        'granting access to the entire data bucket because `s3Prefix` is empty and a shared bucket was provided; ' +
          'set `s3Prefix` to scope grants to this table\'s data and avoid granting access to other tables sharing the bucket.',
      );
    }
    return this.s3Prefix + '*';
  }

  private createManagedBucket(encryption?: S3TableEncryption): s3.Bucket {
    const enc = encryption ?? S3TableEncryption.s3Managed();
    return new s3.Bucket(this, 'Bucket', {
      encryption: enc._bucketEncryption,
      encryptionKey: enc._kmsKey,
      enforceSSL: true,
    });
  }
}

const readPermissions = [
  'glue:BatchGetPartition',
  'glue:GetPartition',
  'glue:GetPartitions',
  'glue:GetTable',
  'glue:GetTables',
  'glue:GetTableVersion',
  'glue:GetTableVersions',
];

const writePermissions = [
  'glue:BatchCreatePartition',
  'glue:BatchDeletePartition',
  'glue:CreatePartition',
  'glue:DeletePartition',
  'glue:UpdatePartition',
];

function renderColumns(columns?: Column[]) {
  if (columns === undefined) {
    return undefined;
  }
  return columns.map(column => {
    return {
      name: column.name,
      type: column.type.inputString,
      comment: column.comment,
    };
  });
}
