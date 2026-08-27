import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as core from 'aws-cdk-lib/core';
import type { Construct } from 'constructs';
import * as s3tables from '../../lib';

/**
 * Test stack for table bucket with STANDARD storage class
 */
class StandardStorageClassTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    this.tableBucket = new s3tables.TableBucket(this, 'StandardStorageBucket', {
      tableBucketName: 'sc-std-integ-test',
      storageClass: s3tables.StorageClass.STANDARD,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

/**
 * Test stack for table bucket with INTELLIGENT_TIERING storage class
 */
class IntelligentTieringStorageClassTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    this.tableBucket = new s3tables.TableBucket(this, 'IntelligentTieringBucket', {
      tableBucketName: 'sc-it-integ-test',
      storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

/**
 * Test stack for table with storage class configuration (override)
 */
class TableStorageClassOverrideTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;
  public readonly namespace: s3tables.Namespace;
  public readonly table: s3tables.Table;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    this.tableBucket = new s3tables.TableBucket(this, 'TableStorageClassBucket', {
      tableBucketName: 'sc-override-integ-test',
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace = new s3tables.Namespace(this, 'StorageClassNamespace', {
      namespaceName: 'storage_class_namespace',
      tableBucket: this.tableBucket,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    // Ensure namespace waits for bucket to be fully created
    this.namespace.node.addDependency(this.tableBucket);

    this.table = new s3tables.Table(this, 'IntelligentTieringTable', {
      tableName: 'intelligent_tiering_table',
      namespace: this.namespace,
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
      withoutMetadata: true,
      storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

/**
 * Test stack for table inheriting storage class from bucket
 */
class TableStorageClassInheritanceTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;
  public readonly namespace: s3tables.Namespace;
  public readonly table: s3tables.Table;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    // Bucket with INTELLIGENT_TIERING storage class
    this.tableBucket = new s3tables.TableBucket(this, 'InheritanceBucket', {
      tableBucketName: 'sc-inherit-integ-test',
      storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace = new s3tables.Namespace(this, 'InheritanceNamespace', {
      namespaceName: 'inheritance_namespace',
      tableBucket: this.tableBucket,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    // Ensure namespace waits for bucket to be fully created
    this.namespace.node.addDependency(this.tableBucket);

    // Table WITHOUT storage class - should inherit INTELLIGENT_TIERING from bucket
    this.table = new s3tables.Table(this, 'InheritingTable', {
      tableName: 'inheriting_table',
      namespace: this.namespace,
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
      withoutMetadata: true,
      // No storageClass specified - should inherit from bucket
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

/**
 * Test stack for reverse override: bucket=INTELLIGENT_TIERING, table=STANDARD
 * Verifies that table can override bucket's storage class in either direction
 */
class TableStorageClassReverseOverrideTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;
  public readonly namespace: s3tables.Namespace;
  public readonly table: s3tables.Table;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    // Bucket with INTELLIGENT_TIERING
    this.tableBucket = new s3tables.TableBucket(this, 'ReverseOverrideBucket', {
      tableBucketName: 'sc-reverse-integ-test',
      storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace = new s3tables.Namespace(this, 'ReverseOverrideNamespace', {
      namespaceName: 'reverse_override_ns',
      tableBucket: this.tableBucket,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace.node.addDependency(this.tableBucket);

    // Table explicitly sets STANDARD, overriding bucket's INTELLIGENT_TIERING
    this.table = new s3tables.Table(this, 'StandardOverrideTable', {
      tableName: 'standard_override_table',
      namespace: this.namespace,
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
      withoutMetadata: true,
      storageClass: s3tables.StorageClass.STANDARD,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

/**
 * Test stack for default behavior: both bucket and table have no storage class
 * Both should default to STANDARD
 */
class DefaultStorageClassTestStack extends core.Stack {
  public readonly tableBucket: s3tables.TableBucket;
  public readonly namespace: s3tables.Namespace;
  public readonly table: s3tables.Table;

  constructor(scope: Construct, id: string, props?: core.StackProps) {
    super(scope, id, props);

    // Bucket without storage class - defaults to STANDARD
    this.tableBucket = new s3tables.TableBucket(this, 'DefaultBucket', {
      tableBucketName: 'sc-default-integ-test',
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace = new s3tables.Namespace(this, 'DefaultNamespace', {
      namespaceName: 'default_ns',
      tableBucket: this.tableBucket,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });

    this.namespace.node.addDependency(this.tableBucket);

    // Table without storage class - inherits bucket's default (STANDARD)
    this.table = new s3tables.Table(this, 'DefaultTable', {
      tableName: 'default_table',
      namespace: this.namespace,
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
      withoutMetadata: true,
      removalPolicy: core.RemovalPolicy.DESTROY,
    });
  }
}

const app = new core.App();

const standardStorageClassTestStack = new StandardStorageClassTestStack(app, 'StandardStorageClassTestStack');
const intelligentTieringStorageClassTestStack = new IntelligentTieringStorageClassTestStack(app, 'IntelligentTieringStorageClassTestStack');
const tableStorageClassOverrideTestStack = new TableStorageClassOverrideTestStack(app, 'TableStorageClassOverrideTestStack');
const tableStorageClassInheritanceTestStack = new TableStorageClassInheritanceTestStack(app, 'TableStorageClassInheritanceTestStack');
const tableStorageClassReverseOverrideTestStack = new TableStorageClassReverseOverrideTestStack(app, 'TableStorageClassReverseOverrideTestStack');
const defaultStorageClassTestStack = new DefaultStorageClassTestStack(app, 'DefaultStorageClassTestStack');

new IntegTest(app, 'TableBucketStorageClassIntegTest', {
  testCases: [
    standardStorageClassTestStack,
    intelligentTieringStorageClassTestStack,
    tableStorageClassOverrideTestStack,
    tableStorageClassInheritanceTestStack,
    tableStorageClassReverseOverrideTestStack,
    defaultStorageClassTestStack,
  ],
});

// Note: GetTableBucketStorageClassCommand and GetTableStorageClassCommand are not yet
// available in the SDK used by integ-tests-alpha. The deployment itself verifies that
// the StorageClassConfiguration property works correctly with CloudFormation.

app.synth();
