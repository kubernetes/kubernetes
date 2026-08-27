import { EOL } from 'os';
import * as iam from 'aws-cdk-lib/aws-iam';
import { CfnTable, CfnTablePolicy } from 'aws-cdk-lib/aws-s3tables';
import type {
  IResource,
  ITaggableV2,
  RemovalPolicy,
  TagManager,
} from 'aws-cdk-lib/core';
import {
  Resource,
  UnscopedValidationError,
  Token,
} from 'aws-cdk-lib/core';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { INamespace } from './namespace';
import * as perms from './permissions';
import type { StorageClass } from './table-bucket';

/**
 * Represents an S3 Table.
 */
export interface ITable extends IResource {
  /**
   * The ARN of this table.
   * @attribute
   */
  readonly tableArn: string;

  /**
   * The name of this table.
   * @attribute
   */
  readonly tableName: string;

  /**
   * The accountId containing this table.
   * @attribute
   */
  readonly account?: string;

  /**
   * The region containing this table.
   * @attribute
   */
  readonly region?: string;

  /**
   * Adds a statement to the resource policy for a principal (i.e.
   * account/role/service) to perform actions on this table.
   *
   * Note that the policy statement may or may not be added to the policy.
   * For example, when an `ITable` is created from an existing table,
   * it's not possible to tell whether the table already has a policy
   * attached, let alone to re-use that policy to add more statements to it.
   * So it's safest to do nothing in these cases.
   *
   * @param statement the policy statement to be added to the table's
   * policy.
   * @returns metadata about the execution of this method. If the policy
   * was not added, the value of `statementAdded` will be `false`. You
   * should always check this value to make sure that the operation was
   * actually carried out. Otherwise, synthesis and deploy will terminate
   * silently, which may be confusing.
   */
  addToResourcePolicy(statement: iam.PolicyStatement): iam.AddToResourcePolicyResult;

  /**
   * Grant read permissions for this table to an IAM principal (Role/Group/User).
   *
   * If the parent TableBucket of this table has encryption,
   * you should grant kms:Decrypt permission to use this key to the same principal.
   *
   * @param identity The principal to allow read permissions to
   */
  grantRead(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant write permissions for this table to an IAM principal (Role/Group/User).
   *
   * If the parent TableBucket of this table has encryption,
   * you should grant kms:GenerateDataKey and kms:Decrypt permission
   * to use this key to the same principal.
   *
   * @param identity The principal to allow write permissions to
   */
  grantWrite(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read and write permissions for this table to an IAM principal (Role/Group/User).
   *
   * If the parent TableBucket of this table has encryption,
   * you should grant kms:GenerateDataKey and kms:Decrypt permission
   * to use this key to the same principal.
   *
   * @param identity The principal to allow read and write permissions to
   */
  grantReadWrite(identity: iam.IGrantable): iam.Grant;
}

/**
 * Base class for Table implementations.
 */
abstract class TableBase extends Resource implements ITable {
  public abstract readonly tableName: string;
  public abstract readonly tableArn: string;

  /**
   * The resource policy associated with this table.
   *
   * If `autoCreatePolicy` is true, a `TablePolicy` will be created upon the
   * first call to addToResourcePolicy(s).
   */
  public abstract tablePolicy?: CfnTablePolicy;

  /**
   * Indicates if a table resource policy should automatically created upon
   * the first call to `addToResourcePolicy`.
   */
  protected abstract autoCreatePolicy: boolean;

  public addToResourcePolicy(
    statement: iam.PolicyStatement,
  ): iam.AddToResourcePolicyResult {
    if (!this.tablePolicy && this.autoCreatePolicy) {
      this.tablePolicy = new CfnTablePolicy(this, 'DefaultPolicy', {
        tableArn: this.tableArn,
        resourcePolicy: new iam.PolicyDocument({}),
      });
    }

    if (this.tablePolicy) {
      this.tablePolicy.resourcePolicy.addStatements(statement);
      return { statementAdded: true, policyDependable: this.tablePolicy };
    }

    return { statementAdded: false };
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantRead(identity: iam.IGrantable) {
    return this.grant(
      identity,
      perms.TABLE_READ_ACCESS,
      this.tableArn,
    );
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantWrite(identity: iam.IGrantable) {
    return this.grant(
      identity,
      perms.TABLE_WRITE_ACCESS,
      this.tableArn,
    );
  }

  /**
   * [disable-awslint:no-grants]
   */
  public grantReadWrite(identity: iam.IGrantable) {
    return this.grant(
      identity,
      perms.TABLE_READ_WRITE_ACCESS,
      this.tableArn,
    );
  }

  /**
   * Grants the given s3tables permissions to the provided principal
   * @returns Grant object
   */
  private grant(
    grantee: iam.IGrantable,
    tableActions: string[],
    resourceArn: string,
    ...otherResourceArns: (string | undefined)[]) {
    const resources = [resourceArn, ...otherResourceArns].filter(arn => arn != undefined);

    const grant = iam.Grant.addToPrincipalOrResource({
      grantee,
      actions: tableActions,
      resourceArns: resources,
      resource: this,
    });

    return grant;
  }
}

/**
 * Properties for creating a new S3 Table.
 */
export interface TableProps {
  /**
   * Name of this table, unique within the namespace
   */
  readonly tableName: string;
  /**
   * The namespace under which this table is created
   */
  readonly namespace: INamespace;
  /**
   * Format of this table. Currently, the only supported value is OpenTableFormat.ICEBERG.
   */
  readonly openTableFormat: OpenTableFormat;
  /**
   * Settings governing the Compaction maintenance action.
   * @default Amazon S3 selects the best compaction strategy based on your table sort order.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-maintenance.html
   */
  readonly compaction?: CompactionProperty;
  /**
   * Contains details about the metadata for an Iceberg table.
   * @default table is created without any metadata
   */
  readonly icebergMetadata?: IcebergMetadataProperty;
  /**
   * Contains details about the snapshot management settings for an Iceberg table.
   * @default enabled: MinimumSnapshots is 1 by default and MaximumSnapshotAge is 120 hours by default.
   */
  readonly snapshotManagement?: SnapshotManagementProperty;
  /**
   * Controls what happens to this table it it stoped being managed by cloudformation.
   *
   * @default RETAIN
   */
  readonly removalPolicy?: RemovalPolicy;
  /**
   * If true, indicates that you don't want to specify a schema for the table.
   *
   * This property is mutually exclusive to 'IcebergMetadata'.
   *
   * @default false
   */
  readonly withoutMetadata?: boolean;

  /**
   * The storage class for the table.
   *
   * Determines the storage tier used for table data, allowing optimization
   * for different access patterns and cost profiles.
   *
   * Note: This is a create-only property and cannot be changed after the table is created.
   *
   * @default - Inherits from the table bucket's storage class configuration
   */
  readonly storageClass?: StorageClass;
}

/**
 * Supported open table formats.
 */
export enum OpenTableFormat {
  /**
   * Apache Iceberg table format.
   */
  ICEBERG = 'ICEBERG',
}

/**
 * Settings governing the Compaction maintenance action.
 *
 * @default - No compaction settings
 */
export interface CompactionProperty {
  /**
   * Status of the compaction maintenance action.
   */
  readonly status: Status;
  /**
   * Target file size in megabytes for compaction.
   */
  readonly targetFileSizeMb: number;
}

/**
 * Status values for maintenance actions.
 */
export enum Status {
  /**
   * Enable the maintenance action.
   */
  ENABLED = 'enabled',
  /**
   * Disable the maintenance action.
   */
  DISABLED = 'disabled',
}

/**
 * Iceberg transform values for partition and sort fields.
 */
export class IcebergTransform {
  /**
   * Use the column value as the transform value without transformation.
   */
  public static readonly IDENTITY = new IcebergTransform('identity');

  /**
   * Transform a timestamp/date field to year.
   */
  public static readonly YEAR = new IcebergTransform('year');

  /**
   * Transform a timestamp/date field to month.
   */
  public static readonly MONTH = new IcebergTransform('month');

  /**
   * Transform a timestamp/date field to day.
   */
  public static readonly DAY = new IcebergTransform('day');

  /**
   * Transform a timestamp field to hour.
   */
  public static readonly HOUR = new IcebergTransform('hour');

  /**
   * Transform values into a fixed number of buckets.
   *
   * @param n The number of buckets (must be a positive integer)
   */
  public static bucket(n: number): IcebergTransform {
    if (n <= 0 || !Number.isInteger(n)) {
      throw new UnscopedValidationError(
        lit`InvalidBucketCount`,
        'Bucket count must be a positive integer.',
      );
    }
    return new IcebergTransform(`bucket[${n}]`);
  }

  /**
   * Truncate values to a fixed width.
   *
   * @param width The truncation width (must be a positive integer)
   */
  public static truncate(width: number): IcebergTransform {
    if (width <= 0 || !Number.isInteger(width)) {
      throw new UnscopedValidationError(
        lit`InvalidTruncateWidth`,
        'Truncate width must be a positive integer.',
      );
    }
    return new IcebergTransform(`truncate[${width}]`);
  }

  /**
   * Create a custom transform from a string value.
   *
   * @param value The transform string value
   */
  public static of(value: string): IcebergTransform {
    return new IcebergTransform(value);
  }

  /**
   * The string value of the transform.
   */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }

  /**
   * Returns the string representation of the transform.
   */
  public toString(): string {
    return this.value;
  }
}

/**
 * Sort direction values for Iceberg sort fields.
 */
export enum SortDirection {
  /**
   * Sort values in ascending order.
   */
  ASC = 'asc',
  /**
   * Sort values in descending order.
   */
  DESC = 'desc',
}

/**
 * Null ordering values for Iceberg sort fields.
 */
export enum NullOrder {
  /**
   * Place null values before non-null values.
   */
  NULLS_FIRST = 'nulls-first',
  /**
   * Place null values after non-null values.
   */
  NULLS_LAST = 'nulls-last',
}

/**
 * Partition field definition for Iceberg table.
 *
 * Defines a single partition column. Multiple partition fields can be combined
 * in an IcebergPartitionSpec to create multi-level partitioning.
 */
export interface IcebergPartitionField {
  /**
   * The source field ID from the schema.
   */
  readonly sourceId: number;

  /**
   * The partition transform function.
   *
   * Use `IcebergTransform` static properties for common transforms (e.g., `IcebergTransform.IDENTITY`)
   * or methods for parameterized transforms (e.g., `IcebergTransform.bucket(16)`).
   */
  readonly transform: IcebergTransform;
  /**
   * The name of the partition field.
   */
  readonly name: string;

  /**
   * The unique identifier for the partition field.
   *
   * @default - Auto-assigned starting from 1000
   */
  readonly fieldId?: number;
}

/**
 * Partition specification for Iceberg table.
 *
 * Contains the complete partitioning configuration for a table, including all partition fields.
 * Use this to define multi-level partitioning (e.g., partition by date, then by region).
 */
export interface IcebergPartitionSpec {
  /**
   * The list of partition fields.
   */
  readonly fields: IcebergPartitionField[];

  /**
   * The unique identifier for the partition specification.
   *
   * @default 0
   */
  readonly specId?: number;
}

/**
 * Sort field definition for Iceberg table.
 */
export interface IcebergSortField {
  /**
   * The source field ID from the schema.
   */
  readonly sourceId: number;

  /**
   * The sort transform function.
   *
   * Use `IcebergTransform` static properties for common transforms (e.g., `IcebergTransform.IDENTITY`)
   * or methods for parameterized transforms (e.g., `IcebergTransform.bucket(16)`).
   */
  readonly transform: IcebergTransform;

  /**
   * The sort direction.
   */
  readonly direction: SortDirection;

  /**
   * The null ordering.
   */
  readonly nullOrder: NullOrder;
}

/**
 * Sort order specification for Iceberg table.
 */
export interface IcebergSortOrder {
  /**
   * The list of sort fields.
   */
  readonly fields: IcebergSortField[];

  /**
   * The unique identifier for the sort order.
   *
   * @default 1
   */
  readonly orderId?: number;
}

/**
 * A single table property key-value pair for Iceberg table configuration.
 */
export interface TablePropertyEntry {
  /**
   * The property key.
   */
  readonly key: string;

  /**
   * The property value.
   */
  readonly value: string;
}

/**
 * Contains details about the metadata for an Iceberg table.
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3tables-table-icebergmetadata.html
 */
export interface IcebergMetadataProperty {
  /**
   * Contains details about the schema for an Iceberg table.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3tables-table-icebergmetadata.html#cfn-s3tables-table-icebergmetadata-icebergschema
   */
  readonly icebergSchema: IcebergSchemaProperty;

  /**
   * The partition specification for the Iceberg table.
   *
   * @default - No partition specification
   */
  readonly icebergPartitionSpec?: IcebergPartitionSpec;

  /**
   * The sort order for the Iceberg table.
   *
   * @default - No sort order
   */
  readonly icebergSortOrder?: IcebergSortOrder;

  /**
   * Custom properties for the Iceberg table.
   * Each entry represents a key-value pair for Iceberg table configuration.
   *
   * @default - No custom properties
   */
  readonly tableProperties?: TablePropertyEntry[];
}

/**
 * Contains details about the schema for an Iceberg table.
 */
export interface IcebergSchemaProperty {
  /**
   * Contains details about the schema for an Iceberg table.
   */
  readonly schemaFieldList: SchemaFieldProperty[];
}

/**
 * Contains details about a schema field.
 */
export interface SchemaFieldProperty {
  /**
   * The unique identifier for the field.
   *
   * @default - Auto-assigned by S3 Tables
   */
  readonly id?: number;

  /**
   * The name of the field.
   */
  readonly name: string;

  /**
   * A Boolean value that specifies whether values are required for each row in this field.
   *
   * By default, this is `false` and null values are allowed in the field. If this is `true`, the field does not allow null values.
   *
   * @default false
   */
  readonly required?: boolean;

  /**
   * The field type.
   *
   * S3 Tables supports all Apache Iceberg primitive types. For more information, see the [Apache Iceberg documentation](https://iceberg.apache.org/spec/#primitive-types).
   */
  readonly type: string;
}

/**
 * Contains details about the snapshot management settings for an Iceberg table.
 *
 * A snapshot is expired when it exceeds MinSnapshotsToKeep and MaxSnapshotAgeHours.
 *
 * @default - No snapshot management settings
 */
export interface SnapshotManagementProperty {
  /**
   * The maximum age of a snapshot before it can be expired.
   *
   * @default - No maximum age
   */
  readonly maxSnapshotAgeHours?: number;

  /**
   * The minimum number of snapshots to keep.
   *
   * @default - No minimum number
   */
  readonly minSnapshotsToKeep?: number;

  /**
   * Indicates whether the SnapshotManagement maintenance action is enabled.
   *
   * @default - Not specified
   */
  readonly status?: Status;
}

/**
 * A reference to a table outside this stack
 *
 * The tableName, region, and account can be provided explicitly
 * or will be inferred from the tableArn
 */
export interface TableAttributes {
  /**
   * Name of this table
   */
  readonly tableName: string;

  /**
   * The table's ARN.
   */
  readonly tableArn: string;
}

/**
 * An S3 Table with helpers.
 */
@propertyInjectable
export class Table extends TableBase implements ITaggableV2 {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-s3tables-alpha.Table';

  /**
   * Defines a Table construct that represents an external table.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param attrs A `TableAttributes` object containing the table name and ARN.
   */
  public static fromTableAttributes(
    scope: Construct,
    id: string,
    attrs: TableAttributes,
  ): ITable {
    const tableArn = attrs.tableArn;
    Table.validateTableName(attrs.tableName);

    class Import extends TableBase {
      public readonly tableName = attrs.tableName;
      public readonly tableArn = tableArn;
      public readonly tablePolicy?: CfnTablePolicy;
      protected autoCreatePolicy: boolean = false;

      /**
       * Exports this  from the stack.
       */
      public export() {
        return attrs;
      }
    }

    return new Import(scope, id, {
      environmentFromArn: tableArn,
      physicalName: tableArn,
    });
  }

  /**
   * See https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-buckets-naming.html
   * @param tableName Name of the table
   * @throws UnscopedValidationError if any naming errors are detected
   */
  public static validateTableName(tableName: string) {
    if (tableName == undefined || Token.isUnresolved(tableName)) {
      // the name is a late-bound value, not a defined string, so skip validation
      return;
    }

    const errors: string[] = [];

    // Length validation
    if (tableName.length < 1 || tableName.length > 255) {
      errors.push(
        'Table name must be at least 1 and no more than 255 characters',
      );
    }

    // Character set validation
    const illegalCharsetRegEx = /[^a-z0-9_]/;
    const allowedEdgeCharsetRegEx = /[a-z0-9]/;

    const illegalCharMatch = tableName.match(illegalCharsetRegEx);
    if (illegalCharMatch) {
      errors.push(
        'Table name must only contain lowercase characters, numbers, and underscores (_)' +
          ` (offset: ${illegalCharMatch.index})`,
      );
    }

    // Edge character validation
    if (!allowedEdgeCharsetRegEx.test(tableName.charAt(0))) {
      errors.push(
        'Table name must start with a lowercase letter or number (offset: 0)',
      );
    }
    if (!allowedEdgeCharsetRegEx.test(tableName.charAt(tableName.length - 1))) {
      errors.push(
        `Table name must end with a lowercase letter or number (offset: ${
          tableName.length - 1
        })`,
      );
    }

    if (errors.length > 0) {
      throw new UnscopedValidationError(
        lit`InvalidTableName`,
        `Invalid S3 table name (value: ${tableName})${EOL}${errors.join(EOL)}`,
      );
    }
  }

  /**
   * The unique Amazon Resource Name (arn) of this table
   */
  public readonly tableArn: string;

  /**
   * The underlying CfnTable L1 resource
   * @internal
   */
  private readonly _resource: CfnTable;

  /**
   * The tag manager for this resource.
   */
  public readonly cdkTagManager: TagManager;

  /**
   * The name of this table
   */
  public readonly tableName: string;

  /**
   * The namespace containing this table
   */
  public readonly namespace: INamespace;

  /**
   * The resource policy for this table.
   */
  public readonly tablePolicy?: CfnTablePolicy;

  protected autoCreatePolicy: boolean = true;

  constructor(scope: Construct, id: string, props: TableProps) {
    super(scope, id, {});

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    if (props.withoutMetadata && props.icebergMetadata) {
      throw new UnscopedValidationError(
        lit`MutuallyExclusiveProps`,
        "TableProps: 'withoutMetadata' and 'icebergMetadata' are mutually exclusive. Specify only one.",
      );
    }

    Table.validateTableName(props.tableName);

    this._resource = new CfnTable(this, id, {
      tableName: props.tableName,
      openTableFormat: props.openTableFormat,
      tableBucketArn: props.namespace.tableBucket.tableBucketArn,
      namespace: props.namespace.namespaceName,
      compaction: props.compaction,
      icebergMetadata: this.buildIcebergMetadata(props.icebergMetadata),
      snapshotManagement: props.snapshotManagement,
      withoutMetadata: props.withoutMetadata ? 'Yes' : undefined,
      storageClassConfiguration: props.storageClass ? { storageClass: props.storageClass } : undefined,
    });

    this.namespace = props.namespace;
    this.tableName = props.tableName;
    this.tableArn = this._resource.attrTableArn;
    this.cdkTagManager = this._resource.cdkTagManager;
    this._resource.applyRemovalPolicy(props.removalPolicy);
    this.node.addDependency(this.namespace);
  }

  /**
   * Builds the CloudFormation Iceberg metadata property from the CDK metadata model.
   */
  private buildIcebergMetadata(metadata?: IcebergMetadataProperty): CfnTable.IcebergMetadataProperty | undefined {
    if (!metadata) {
      return undefined;
    }

    return {
      icebergSchema: this.buildIcebergSchema(metadata.icebergSchema),
      icebergPartitionSpec: metadata.icebergPartitionSpec
        ? this.buildIcebergPartitionSpec(metadata.icebergPartitionSpec)
        : undefined,
      icebergSortOrder: metadata.icebergSortOrder
        ? this.buildIcebergSortOrder(metadata.icebergSortOrder)
        : undefined,
      tableProperties: metadata.tableProperties
        ? this.buildTableProperties(metadata.tableProperties)
        : undefined,
    };
  }

  /**
   * Builds the CloudFormation Iceberg schema property.
   */
  private buildIcebergSchema(schema: IcebergSchemaProperty): CfnTable.IcebergSchemaProperty {
    return {
      schemaFieldList: schema.schemaFieldList.map(field => ({
        name: field.name,
        type: field.type,
        ...(field.required !== undefined && { required: field.required }),
        ...(field.id !== undefined && { id: field.id }),
      })),
    };
  }

  /**
   * Builds the CloudFormation Iceberg partition specification property.
   */
  private buildIcebergPartitionSpec(spec: IcebergPartitionSpec): CfnTable.IcebergPartitionSpecProperty {
    return {
      ...(spec.specId !== undefined && { specId: spec.specId }),
      fields: spec.fields.map(field => ({
        sourceId: field.sourceId,
        transform: field.transform.value,
        name: field.name,
        ...(field.fieldId !== undefined && { fieldId: field.fieldId }),
      })),
    };
  }

  /**
   * Builds the CloudFormation Iceberg sort order property.
   */
  private buildIcebergSortOrder(sortOrder: IcebergSortOrder): CfnTable.IcebergSortOrderProperty {
    return {
      ...(sortOrder.orderId !== undefined && { orderId: sortOrder.orderId }),
      fields: sortOrder.fields.map(field => ({
        sourceId: field.sourceId,
        transform: field.transform.value,
        direction: field.direction,
        nullOrder: field.nullOrder,
      })),
    };
  }

  /**
   * Builds the CloudFormation table properties map from key-value entries.
   */
  private buildTableProperties(tableProperties: TablePropertyEntry[]): Record<string, string> {
    const keys = tableProperties.map(entry => entry.key);
    const duplicateKeys = keys.filter((key, index) => keys.indexOf(key) !== index);
    if (duplicateKeys.length > 0) {
      throw new UnscopedValidationError(
        lit`DuplicateTablePropertyKey`,
        `Duplicate table property keys are not allowed: ${[...new Set(duplicateKeys)].join(', ')}`,
      );
    }

    return tableProperties.reduce(
      (acc, entry) => ({
        ...acc,
        [entry.key]: entry.value,
      }),
      {} as Record<string, string>,
    );
  }
}
