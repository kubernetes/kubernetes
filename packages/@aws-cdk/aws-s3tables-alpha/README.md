# Amazon S3 Tables Construct Library
<!--BEGIN STABILITY BANNER-->

---

![cdk-constructs: Experimental](https://img.shields.io/badge/cdk--constructs-experimental-important.svg?style=for-the-badge)

> The APIs of higher level constructs in this module are experimental and under active development.
> They are subject to non-backward compatible changes or removal in any future version. These are
> not subject to the [Semantic Versioning](https://semver.org/) model and breaking changes will be
> announced in the release notes. This means that while you may use them, you may need to update
> your source code when upgrading to a newer version of this package.

---

<!--END STABILITY BANNER-->

## Amazon S3 Tables

Amazon S3 Tables deliver the first cloud object store with built-in Apache Iceberg support and streamline storing tabular data at scale.

[Product Page](https://aws.amazon.com/s3/features/tables/) | [User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables.html)


## Usage

### Define an S3 Table Bucket

```ts
// Build a Table bucket
const sampleTableBucket = new TableBucket(scope, 'ExampleTableBucket', {
    tableBucketName: 'example-bucket-1',
    // optional fields:
    unreferencedFileRemoval: {
        status: UnreferencedFileRemovalStatus.ENABLED,
        noncurrentDays: 20,
        unreferencedDays: 20,
    }
});
```

### Define an S3 Tables Namespace

```ts
// Build a namespace
const sampleNamespace = new Namespace(scope, 'ExampleNamespace', {
    namespaceName: 'example-namespace-1',
    tableBucket: tableBucket,
});
```

### Define an S3 Table

```ts
// Build a table
const sampleTable = new Table(scope, 'ExampleTable', {
    tableName: 'example_table',
    namespace: namespace,
    openTableFormat: OpenTableFormat.ICEBERG,
    withoutMetadata: true,
});

// Build a table with an Iceberg Schema
const sampleTableWithSchema = new Table(scope, 'ExampleSchemaTable', {
    tableName: 'example_table_with_schema',
    namespace: namespace,
    openTableFormat: OpenTableFormat.ICEBERG,
    icebergMetadata: {
        icebergSchema: {
            schemaFieldList: [
            {
                name: 'id',
                type: 'int',
                required: true,
            },
            {
                name: 'name',
                type: 'string',
            },
            ],
        },
    },
    compaction: {
        status: Status.ENABLED,
        targetFileSizeMb: 128,
    },
    snapshotManagement: {
        status: Status.ENABLED,
        maxSnapshotAgeHours: 48,
        minSnapshotsToKeep: 5,
    },
});
```

Learn more about table buckets maintenance operations and default behavior from the [S3 Tables User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-table-buckets-maintenance.html)

### Advanced Iceberg Table Configuration

You can configure partition specifications, sort orders, and table properties for optimized query performance.

The simplest way to add partitioning to your table:

```ts
// Build a table with partition spec (minimal configuration)
const partitionedTable = new Table(scope, 'PartitionedTable', {
    tableName: 'partitioned_table',
    namespace: namespace,
    openTableFormat: OpenTableFormat.ICEBERG,
    icebergMetadata: {
        icebergSchema: {
            schemaFieldList: [
                { name: 'event_date', type: 'date', required: true },
                { name: 'event_name', type: 'string' },
            ],
        },
        icebergPartitionSpec: {
            fields: [
                {
                    sourceId: 1,
                    transform: IcebergTransform.IDENTITY,
                    name: 'date_partition',
                },
            ],
        },
    },
});
```

For full control, you can also configure sort orders and table properties:

```ts
// Build a table with partition spec, sort order, and table properties
const advancedTable = new Table(scope, 'AdvancedTable', {
    tableName: 'advanced_table',
    namespace: namespace,
    openTableFormat: OpenTableFormat.ICEBERG,
    icebergMetadata: {
        icebergSchema: {
            schemaFieldList: [
                { id: 1, name: 'event_date', type: 'date', required: true },
                { id: 2, name: 'user_id', type: 'string', required: true },
            ],
        },
        icebergPartitionSpec: {
            specId: 0,
            fields: [
                {
                    sourceId: 1,
                    transform: IcebergTransform.IDENTITY,
                    name: 'date_partition',
                    fieldId: 1000,
                },
            ],
        },
        icebergSortOrder: {
            orderId: 1,
            fields: [
                {
                    sourceId: 1,
                    transform: IcebergTransform.IDENTITY,
                    direction: SortDirection.ASC,
                    nullOrder: NullOrder.NULLS_LAST,
                },
            ],
        },
        tableProperties: [
            { key: 'write.format.default', value: 'parquet' },
        ],
    },
});
```

### Controlling Table Bucket Permissions

```ts
// Grant the principal read permissions to the bucket and all tables within
const accountId = '123456789012'
tableBucket.grantRead(new iam.AccountPrincipal(accountId), '*');

// Grant the role write permissions to the bucket and all tables within
const role = new iam.Role(stack, 'MyRole', { assumedBy: new iam.ServicePrincipal('sample') });
tableBucket.grantWrite(role, '*');

// Grant the user read and write permissions to the bucket and all tables within 
tableBucket.grantReadWrite(new iam.User(stack, 'MyUser'), '*');

// Grant permissions to the bucket and a particular table within it
const tableId = '6ba046b2-26de-44cf-9144-0c7862593a7b'
tableBucket.grantReadWrite(new iam.AccountPrincipal(accountId), tableId);

// Add custom resource policy statements
const permissions = new iam.PolicyStatement({
    effect: iam.Effect.ALLOW,
    actions: ['s3tables:*'],
    principals: [ new iam.ServicePrincipal('example.aws.internal') ],
    resources: ['*']
});

tableBucket.addToResourcePolicy(permissions);
```

### Controlling Table Bucket Encryption Settings

S3 TableBuckets have SSE (server-side encryption with AES-256) enabled by default with S3 managed keys.
You can also bring your own KMS key for KMS-SSE or have S3 create a KMS key for you.

If a bucket is encrypted with KMS, grant functions on the bucket will also grant access
to the TableBucket's associated KMS key.

```ts
// Provide a user defined KMS Key:
const key = new kms.Key(scope, 'UserKey', {});
const encryptedBucket = new TableBucket(scope, 'EncryptedTableBucket', {
    tableBucketName: 'table-bucket-1',
    encryption: TableBucketEncryption.KMS,
    encryptionKey: key,
});
// This account principal will also receive kms:Decrypt access to the KMS key
encryptedBucket.grantRead(new iam.AccountPrincipal('123456789012'), '*');

// Use S3 managed server side encryption (default)
const encryptedBucketDefault = new TableBucket(scope, 'EncryptedTableBucketDefault', {
    tableBucketName: 'table-bucket-3',
    encryption: TableBucketEncryption.S3_MANAGED, // Uses AES-256 encryption by default
});
```

When using KMS encryption (`TableBucketEncryption.KMS`), if no encryption key is provided, CDK will automatically create a new KMS key for the table bucket with necessary permissions.

```ts
// If no key is provided, one will be created automatically
const encryptedBucketAuto = new TableBucket(scope, 'EncryptedTableBucketAuto', {
    tableBucketName: 'table-bucket-2',
    encryption: TableBucketEncryption.KMS,
});
```

### Enabling CloudWatch Request Metrics

You can enable CloudWatch request metrics for your table bucket. Request metrics provide insight into Amazon S3 Tables requests, helping you monitor and optimize your table bucket usage.

For more information about S3 Tables CloudWatch metrics, see the [S3 Tables CloudWatch Metrics documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-cloudwatch-metrics.html).

```ts
// Enable CloudWatch request metrics for the table bucket
const tableBucketWithMetrics = new TableBucket(scope, 'TableBucketWithMetrics', {
    tableBucketName: 'metrics-enabled-bucket',
    requestMetricsStatus: RequestMetricsStatus.ENABLED,
});
```

### Configuring Storage Class

You can configure the storage class for your table bucket and tables. Storage class determines how data is stored and billed, allowing you to optimize for different access patterns.

```ts
// Create a table bucket with INTELLIGENT_TIERING storage class
const tableBucketWithStorageClass = new TableBucket(scope, 'TableBucketWithStorageClass', {
    tableBucketName: 'storage-class-bucket',
    storageClass: StorageClass.INTELLIGENT_TIERING,
});
```

Tables inherit the storage class from their parent bucket by default. You can also override the storage class at the table level:

```ts
// Create a table with explicit storage class (overrides bucket default)
const tableWithStorageClass = new Table(scope, 'TableWithStorageClass', {
    tableName: 'storage_class_table',
    namespace: namespace,
    openTableFormat: OpenTableFormat.ICEBERG,
    withoutMetadata: true,
    storageClass: StorageClass.STANDARD,
});
```

Available storage classes:

- `StorageClass.STANDARD` - For frequently accessed data
- `StorageClass.INTELLIGENT_TIERING` - Automatically moves data between access tiers based on usage patterns

Note: Table storage class is a create-only property and cannot be changed after the table is created.

### Controlling Table Permissions

```ts
// Grant the principal read permissions to the table
const accountId = '123456789012'
table.grantRead(new iam.AccountPrincipal(accountId));

// Grant the role write permissions to the table
const role = new iam.Role(stack, 'MyRole', { assumedBy: new iam.ServicePrincipal('sample') });
table.grantWrite(role);

// Grant the user read and write permissions to the table 
table.grantReadWrite(new iam.User(stack, 'MyUser'));

// Grant an account permissions to the table
table.grantReadWrite(new iam.AccountPrincipal(accountId));

// Add custom resource policy statements
const permissions = new iam.PolicyStatement({
    effect: iam.Effect.ALLOW,
    actions: ['s3tables:*'],
    principals: [ new iam.ServicePrincipal('example.aws.internal') ],
    resources: ['*']
});

table.addToResourcePolicy(permissions);
```

### Tagging

Both `TableBucket` and `Table` support tagging through CDK's standard tagging mechanism:

```ts
Tags.of(tableBucket).add('Environment', 'Production');
Tags.of(table).add('Team', 'DataEngineering');

// Stack-level tags propagate to all resources
Tags.of(stack).add('Project', 'DataLake');
```

## Coming Soon

L2 Construct support for:

- KMS encryption support for Tables
