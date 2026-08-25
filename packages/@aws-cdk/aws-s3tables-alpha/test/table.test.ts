import { Match, Template } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as core from 'aws-cdk-lib/core';
import * as s3tables from '../lib';

/* Allow quotes in the object keys used for CloudFormation template assertions */
/* eslint-disable @stylistic/quote-props */

describe('Table', () => {
  const TABLE_CFN_RESOURCE = 'AWS::S3Tables::Table';
  const TABLE_POLICY_CFN_RESOURCE = 'AWS::S3Tables::TablePolicy';

  let stack: core.Stack;
  let namespace: s3tables.Namespace;

  beforeEach(() => {
    stack = new core.Stack();
    const tableBucket = new s3tables.TableBucket(stack, 'TestTableBucket', {
      tableBucketName: 'test-table-bucket',
    });
    namespace = new s3tables.Namespace(stack, 'TestNamespace', {
      namespaceName: 'test_namespace',
      tableBucket,
    });
  });

  describe('created with default properties', () => {
    const DEFAULT_PROPS: s3tables.TableProps = {
      tableName: 'example_table',
      namespace: {} as s3tables.Namespace, // Will be replaced in beforeEach
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
    };
    let table: s3tables.Table;

    beforeEach(() => {
      table;
      const props = { ...DEFAULT_PROPS, namespace };
      table = new s3tables.Table(stack, 'ExampleTable', props);
    });

    test(`creates a ${TABLE_CFN_RESOURCE} resource`, () => {
      Template.fromStack(stack).resourceCountIs(TABLE_CFN_RESOURCE, 1);
    });

    test('with tableName property', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'TableName': DEFAULT_PROPS.tableName,
      });
    });

    test('with openTableFormat property', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'OpenTableFormat': DEFAULT_PROPS.openTableFormat,
      });
    });

    test('has removalPolicy set to "Retain"', () => {
      Template.fromStack(stack).hasResource(TABLE_CFN_RESOURCE, {
        'DeletionPolicy': 'Retain',
      });
    });

    test('returns true from addToResourcePolicy', () => {
      const result = table.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toBe(true);
      expect(result.policyDependable).toBe(table.tablePolicy);
    });
  });

  describe('created with all properties', () => {
    const TABLE_PROPS: s3tables.TableProps = {
      tableName: 'example_table',
      namespace: {} as s3tables.Namespace, // Will be replaced in beforeEach
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
      compaction: {
        status: 'enabled' as any,
        targetFileSizeMb: 128,
      },
      icebergMetadata: {
        icebergSchema: {
          schemaFieldList: [
            {
              id: 1,
              name: 'id',
              type: 'int',
              required: true,
            },
            {
              id: 2,
              name: 'name',
              type: 'string',
            },
          ],
        },
        icebergPartitionSpec: {
          specId: 0,
          fields: [
            {
              sourceId: 1,
              transform: s3tables.IcebergTransform.IDENTITY,
              name: 'id_partition',
            },
          ],
        },
        icebergSortOrder: {
          orderId: 1,
          fields: [
            {
              sourceId: 2,
              transform: s3tables.IcebergTransform.IDENTITY,
              direction: s3tables.SortDirection.ASC,
              nullOrder: s3tables.NullOrder.NULLS_FIRST,
            },
          ],
        },
        tableProperties: [
          { key: 'write.parquet.compression-codec', value: 'snappy' },
          { key: 'write.metadata.compression-codec', value: 'gzip' },
        ],
      },
      snapshotManagement: {
        maxSnapshotAgeHours: 24,
        minSnapshotsToKeep: 5,
        status: 'enabled' as any,
      },
      removalPolicy: core.RemovalPolicy.DESTROY,
    };
    let table: s3tables.Table;

    beforeEach(() => {
      const props = { ...TABLE_PROPS, namespace };
      table = new s3tables.Table(stack, 'ExampleTable', props);
    });

    test(`creates a ${TABLE_CFN_RESOURCE} resource`, () => {
      table;
      Template.fromStack(stack).resourceCountIs(TABLE_CFN_RESOURCE, 1);
    });

    test('has all properties', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'TableName': TABLE_PROPS.tableName,
        'OpenTableFormat': TABLE_PROPS.openTableFormat,
        'Compaction': {
          'Status': TABLE_PROPS.compaction?.status,
          'TargetFileSizeMB': TABLE_PROPS.compaction?.targetFileSizeMb,
        },
        'IcebergMetadata': {
          'IcebergSchema': {
            'SchemaFieldList': [
              {
                'Id': 1,
                'Name': 'id',
                'Type': 'int',
                'Required': true,
              },
              {
                'Id': 2,
                'Name': 'name',
                'Type': 'string',
              },
            ],
          },
          'IcebergPartitionSpec': {
            'SpecId': 0,
            'Fields': [
              {
                'SourceId': 1,
                'Transform': 'identity',
                'Name': 'id_partition',
              },
            ],
          },
          'IcebergSortOrder': {
            'OrderId': 1,
            'Fields': [
              {
                'SourceId': 2,
                'Transform': 'identity',
                'Direction': 'asc',
                'NullOrder': 'nulls-first',
              },
            ],
          },
          'TableProperties': {
            'write.parquet.compression-codec': 'snappy',
            'write.metadata.compression-codec': 'gzip',
          },
        },
        'SnapshotManagement': {
          'MaxSnapshotAgeHours': TABLE_PROPS.snapshotManagement?.maxSnapshotAgeHours,
          'MinSnapshotsToKeep': TABLE_PROPS.snapshotManagement?.minSnapshotsToKeep,
          'Status': TABLE_PROPS.snapshotManagement?.status,
        },
      });
    });

    test('has removalPolicy set to "Delete"', () => {
      Template.fromStack(stack).hasResource(TABLE_CFN_RESOURCE, {
        'DeletionPolicy': 'Delete',
      });
    });
  });

  describe('created with withoutMetadata property', () => {
    let table: s3tables.Table;

    beforeEach(() => {
      table;
      table = new s3tables.Table(stack, 'ExampleTable', {
        tableName: 'example_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
      });
    });

    test('has withoutMetadata set to "Yes"', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'WithoutMetadata': 'Yes',
      });
    });
  });

  describe('created with partition spec and sort order', () => {
    beforeEach(() => {
      new s3tables.Table(stack, 'PartitionedTable', {
        tableName: 'partitioned_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        icebergMetadata: {
          icebergSchema: {
            schemaFieldList: [
              { id: 1, name: 'event_id', type: 'string' },
              { id: 2, name: 'event_time', type: 'timestamp' },
              { id: 3, name: 'category', type: 'string' },
            ],
          },
          icebergPartitionSpec: {
            specId: 0,
            fields: [
              { sourceId: 2, transform: s3tables.IcebergTransform.DAY, name: 'event_day' },
              { sourceId: 3, transform: s3tables.IcebergTransform.IDENTITY, name: 'category_partition' },
            ],
          },
          icebergSortOrder: {
            orderId: 1,
            fields: [
              {
                sourceId: 1,
                transform: s3tables.IcebergTransform.IDENTITY,
                direction: s3tables.SortDirection.ASC,
                nullOrder: s3tables.NullOrder.NULLS_FIRST,
              },
            ],
          },
          tableProperties: [
            { key: 'write.parquet.compression-codec', value: 'snappy' },
          ],
        },
      });
    });

    test('has partition spec', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'IcebergMetadata': {
          'IcebergPartitionSpec': {
            'SpecId': 0,
            'Fields': [
              { 'SourceId': 2, 'Transform': 'day', 'Name': 'event_day' },
              { 'SourceId': 3, 'Transform': 'identity', 'Name': 'category_partition' },
            ],
          },
        },
      });
    });

    test('has sort order', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'IcebergMetadata': {
          'IcebergSortOrder': {
            'OrderId': 1,
            'Fields': [
              { 'SourceId': 1, 'Transform': 'identity', 'Direction': 'asc', 'NullOrder': 'nulls-first' },
            ],
          },
        },
      });
    });

    test('has table properties', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'IcebergMetadata': {
          'TableProperties': {
            'write.parquet.compression-codec': 'snappy',
          },
        },
      });
    });
  });

  describe('defined with resource policy', () => {
    const DEFAULT_PROPS: s3tables.TableProps = {
      tableName: 'example_table',
      namespace: {} as s3tables.Namespace,
      openTableFormat: s3tables.OpenTableFormat.ICEBERG,
    };
    let table: s3tables.Table;

    beforeEach(() => {
      const props = { ...DEFAULT_PROPS, namespace };
      table = new s3tables.Table(stack, 'ExampleTable', props);
      table.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));
    });

    test('resourcePolicy contains statement', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_POLICY_CFN_RESOURCE, {
        'ResourcePolicy': {
          'Statement': [
            {
              'Action': 's3tables:*',
              'Effect': 'Allow',
              'Resource': '*',
            },
          ],
        },
      });
    });

    test('calling multiple times appends statements', () => {
      table.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:*'],
        effect: iam.Effect.DENY,
        resources: ['*'],
      }));
      Template.fromStack(stack).hasResourceProperties(TABLE_POLICY_CFN_RESOURCE, {
        'ResourcePolicy': {
          'Statement': [
            {
              'Action': 's3tables:*',
              'Effect': 'Allow',
              'Resource': '*',
            },
            {
              'Action': 's3:*',
              'Effect': 'Deny',
              'Resource': '*',
            },
          ],
        },
      });
    });
  });

  describe('import existing table with attributes', () => {
    const TABLE_ATTRS = {
      tableName: 'example_table',
      tableArn: 'arn:aws:s3tables:us-west-2:123456789012:table/example_table',
    };
    let table: s3tables.ITable;

    beforeEach(() => {
      table = s3tables.Table.fromTableAttributes(stack, 'ImportedTable', TABLE_ATTRS);
    });

    test('has the same name as it was imported with', () => {
      expect(table.tableName).toEqual(TABLE_ATTRS.tableName);
    });

    test('has the same ARN as it was imported with', () => {
      expect(table.tableArn).toEqual(TABLE_ATTRS.tableArn);
    });

    test('validates table name during import', () => {
      expect(() => {
        s3tables.Table.fromTableAttributes(stack, 'InvalidImport', {
          tableName: 'Invalid-Table',
          tableArn: 'arn:aws:s3tables:us-west-2:123456789012:table/Invalid-Table',
        });
      }).toThrow('Table name must only contain lowercase characters, numbers, and underscores (_)');
    });

    test('creates resource with correct physical name', () => {
      expect(table.node.id).toBe('ImportedTable');
    });

    test('addToResourcePolicy does not add a policy', () => {
      const result = table.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toEqual(false);
      expect(result.policyDependable).toBeUndefined();
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
    });
  });

  describe('validateTableName', () => {
    it('should accept valid table names', () => {
      const validNames = [
        'my_table_123',
        'test_table',
        'abc',
        'a'.repeat(255),
        '123_table',
      ];

      validNames.forEach(name => {
        expect(() => s3tables.Table.validateTableName(name)).not.toThrow();
      });
    });

    it('should skip validation for unresolved tokens', () => {
      const isUnresolved = core.Token.isUnresolved;
      core.Token.isUnresolved = jest.fn().mockReturnValue(true);
      expect(() => s3tables.Table.validateTableName('unresolved')).not.toThrow();
      // Cleanup
      core.Token.isUnresolved = isUnresolved;
    });

    it('should reject table names that are too short', () => {
      expect(() => s3tables.Table.validateTableName('')).toThrow(
        /Table name must be at least 1/,
      );
    });

    it('should reject table names that are too long', () => {
      const longName = 'a'.repeat(256);
      expect(() => s3tables.Table.validateTableName(longName)).toThrow(
        /no more than 255 characters/,
      );
    });

    it('should reject table names with illegal characters', () => {
      const invalidNames = [
        'My-Table', // uppercase
        'table!123', // special character
        'table-123', // hyphen
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.Table.validateTableName(name)).toThrow(
          /must only contain lowercase characters, numbers, and underscores/,
        );
      });
    });

    it('should reject table names that start with invalid characters', () => {
      const invalidNames = [
        '_table',
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.Table.validateTableName(name)).toThrow(
          /must start with a lowercase letter or number/,
        );
      });
    });

    it('should reject table names that end with invalid characters', () => {
      const invalidNames = [
        'table_',
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.Table.validateTableName(name)).toThrow(
          /must end with a lowercase letter or number/,
        );
      });
    });

    it('should include the invalid table name in the error message', () => {
      const invalidName = 'Invalid-Table!';
      expect(() => s3tables.Table.validateTableName(invalidName)).toThrow(
        /Invalid-Table!/,
      );
    });
  });

  describe('table name validation through Table creation', () => {
    test('rejects table creation with invalid table name', () => {
      expect(() => {
        new s3tables.Table(stack, 'TestTable', {
          tableName: 'Invalid-Table',
          namespace,
          openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        });
      }).toThrow('Table name must only contain lowercase characters, numbers, and underscores (_)');
    });

    test('rejects table creation with empty table name', () => {
      expect(() => {
        new s3tables.Table(stack, 'TestTable', {
          tableName: '',
          namespace,
          openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        });
      }).toThrow('Table name must be at least 1 and no more than 255 characters');
    });

    test('rejects table creation with table name starting with underscore', () => {
      expect(() => {
        new s3tables.Table(stack, 'TestTable', {
          tableName: '_invalid',
          namespace,
          openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        });
      }).toThrow('Table name must start with a lowercase letter or number');
    });
  });

  describe('tagging', () => {
    test('implements ITaggableV2', () => {
      const table = new s3tables.Table(stack, 'TaggedTable', {
        tableName: 'tagged_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
      });
      expect(core.TagManager.of(table)).toBeDefined();
    });

    test('tags are applied to the table', () => {
      const table = new s3tables.Table(stack, 'TaggedTable', {
        tableName: 'tagged_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
      });

      core.Tags.of(table).add('Environment', 'Production');

      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        Tags: Match.arrayWith([
          Match.objectLike({ Key: 'Environment', Value: 'Production' }),
        ]),
      });
    });

    test('stack-level tags propagate to table', () => {
      new s3tables.Table(stack, 'TaggedTable', {
        tableName: 'tagged_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
      });

      core.Tags.of(stack).add('StackTag', 'Propagated');

      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        Tags: Match.arrayWith([
          Match.objectLike({ Key: 'StackTag', Value: 'Propagated' }),
        ]),
      });
    });
  });

  describe('IcebergTransform validation', () => {
    test('bucket rejects zero', () => {
      expect(() => s3tables.IcebergTransform.bucket(0)).toThrow('Bucket count must be a positive integer.');
    });

    test('bucket rejects negative numbers', () => {
      expect(() => s3tables.IcebergTransform.bucket(-5)).toThrow('Bucket count must be a positive integer.');
    });

    test('bucket rejects non-integers', () => {
      expect(() => s3tables.IcebergTransform.bucket(3.5)).toThrow('Bucket count must be a positive integer.');
    });

    test('bucket accepts positive integers', () => {
      const transform = s3tables.IcebergTransform.bucket(16);
      expect(transform.value).toBe('bucket[16]');
    });

    test('truncate rejects zero', () => {
      expect(() => s3tables.IcebergTransform.truncate(0)).toThrow('Truncate width must be a positive integer.');
    });

    test('truncate rejects negative numbers', () => {
      expect(() => s3tables.IcebergTransform.truncate(-10)).toThrow('Truncate width must be a positive integer.');
    });

    test('truncate rejects non-integers', () => {
      expect(() => s3tables.IcebergTransform.truncate(2.5)).toThrow('Truncate width must be a positive integer.');
    });

    test('truncate accepts positive integers', () => {
      const transform = s3tables.IcebergTransform.truncate(8);
      expect(transform.value).toBe('truncate[8]');
    });
  });

  describe('created with STANDARD storage class', () => {
    beforeEach(() => {
      new s3tables.Table(stack, 'StandardStorageTable', {
        tableName: 'standard_storage_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
        storageClass: s3tables.StorageClass.STANDARD,
      });
    });

    test('has StorageClassConfiguration with STANDARD storage class', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'TableName': 'standard_storage_table',
        'StorageClassConfiguration': {
          'StorageClass': 'STANDARD',
        },
      });
    });
  });

  describe('created with INTELLIGENT_TIERING storage class', () => {
    beforeEach(() => {
      new s3tables.Table(stack, 'IntelligentTieringTable', {
        tableName: 'intelligent_tiering_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
        storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
      });
    });

    test('has StorageClassConfiguration with INTELLIGENT_TIERING storage class', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_CFN_RESOURCE, {
        'TableName': 'intelligent_tiering_table',
        'StorageClassConfiguration': {
          'StorageClass': 'INTELLIGENT_TIERING',
        },
      });
    });
  });

  describe('created without storage class configuration', () => {
    beforeEach(() => {
      new s3tables.Table(stack, 'NoStorageClassTable', {
        tableName: 'no_storage_class_table',
        namespace,
        openTableFormat: s3tables.OpenTableFormat.ICEBERG,
        withoutMetadata: true,
      });
    });

    test('does not have StorageClassConfiguration property', () => {
      const template = Template.fromStack(stack);
      const resources = template.findResources(TABLE_CFN_RESOURCE);
      // Find the table resource (not the table bucket)
      const tableResourceKey = Object.keys(resources).find(key =>
        resources[key].Properties.TableName === 'no_storage_class_table',
      );
      expect(tableResourceKey).toBeDefined();
      expect(resources[tableResourceKey!].Properties.StorageClassConfiguration).toBeUndefined();
    });
  });

  describe('tableProperties validation', () => {
    test('rejects duplicate keys in tableProperties', () => {
      expect(() => {
        new s3tables.Table(stack, 'TestTable', {
          tableName: 'test_table',
          namespace,
          openTableFormat: s3tables.OpenTableFormat.ICEBERG,
          icebergMetadata: {
            icebergSchema: {
              schemaFieldList: [{ name: 'id', type: 'int' }],
            },
            tableProperties: [
              { key: 'write.format.default', value: 'parquet' },
              { key: 'write.format.default', value: 'avro' },
            ],
          },
        });
      }).toThrow('Duplicate table property keys are not allowed: write.format.default');
    });

    test('accepts unique keys in tableProperties', () => {
      expect(() => {
        new s3tables.Table(stack, 'TestTable', {
          tableName: 'test_table',
          namespace,
          openTableFormat: s3tables.OpenTableFormat.ICEBERG,
          icebergMetadata: {
            icebergSchema: {
              schemaFieldList: [{ name: 'id', type: 'int' }],
            },
            tableProperties: [
              { key: 'write.format.default', value: 'parquet' },
              { key: 'write.parquet.compression-codec', value: 'zstd' },
            ],
          },
        });
      }).not.toThrow();
    });
  });
});
