import { Match, Template } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as core from 'aws-cdk-lib/core';
import * as s3tables from '../lib';

/* Allow quotes in the object keys used for CloudFormation template assertions */
/* eslint-disable @stylistic/quote-props */

describe('TableBucket', () => {
  const TABLE_BUCKET_CFN_RESOURCE = 'AWS::S3Tables::TableBucket';
  const TABLE_BUCKET_POLICY_CFN_RESOURCE = 'AWS::S3Tables::TableBucketPolicy';

  let stack: core.Stack;

  beforeEach(() => {
    stack = new core.Stack();
  });

  describe('created with default properties', () => {
    const DEFAULT_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'example-table-bucket',
    };
    let tableBucket: s3tables.TableBucket;

    beforeEach(() => {
      tableBucket = new s3tables.TableBucket(stack, 'ExampleTableBucket', DEFAULT_PROPS);
    });

    test(`creates a ${TABLE_BUCKET_CFN_RESOURCE} resource`, () => {
      Template.fromStack(stack).resourceCountIs(TABLE_BUCKET_CFN_RESOURCE, 1);
    });

    test('with tableBucketName property', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': DEFAULT_PROPS.tableBucketName,
      });
    });

    test('returns true from addToResourcePolicy', () => {
      const result = tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toBe(true);
    });

    test('has removalPolicy set to "Retain"', () => {
      Template.fromStack(stack).hasResource(TABLE_BUCKET_CFN_RESOURCE, {
        'DeletionPolicy': 'Retain',
      });
    });
  });

  describe('created with unreferenced file removal properties', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      account: '0123456789012',
      region: 'us-west-2',
      tableBucketName: 'example-table-bucket',
      unreferencedFileRemoval: {
        noncurrentDays: 10,
        unreferencedDays: 10,
        status: s3tables.UnreferencedFileRemovalStatus.ENABLED,
      },
      removalPolicy: core.RemovalPolicy.DESTROY,
    };
    let tableBucket: s3tables.TableBucket;

    beforeEach(() => {
      tableBucket = new s3tables.TableBucket(stack, 'ExampleTableBucket', TABLE_BUCKET_PROPS);
    });

    test(`creates a ${TABLE_BUCKET_CFN_RESOURCE} resource`, () => {
      tableBucket;
      Template.fromStack(stack).resourceCountIs(TABLE_BUCKET_CFN_RESOURCE, 1);
    });

    test('has UnreferencedFileRemoval properties', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': TABLE_BUCKET_PROPS.tableBucketName,
        'UnreferencedFileRemoval': {
          'NoncurrentDays': TABLE_BUCKET_PROPS.unreferencedFileRemoval?.noncurrentDays,
          'Status': TABLE_BUCKET_PROPS.unreferencedFileRemoval?.status,
          'UnreferencedDays': TABLE_BUCKET_PROPS.unreferencedFileRemoval?.unreferencedDays,
        },
      });
    });

    test('has removalPolicy set to "Delete"', () => {
      Template.fromStack(stack).hasResource(TABLE_BUCKET_CFN_RESOURCE, {
        'DeletionPolicy': 'Delete',
      });
    });
  });

  describe('defined with resource policy', () => {
    const DEFAULT_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'example-table-bucket',
    };
    let tableBucket: s3tables.TableBucket;

    beforeEach(() => {
      tableBucket = new s3tables.TableBucket(stack, 'ExampleTableBucket', DEFAULT_PROPS);
      tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));
    });

    test('resourcePolicy contains statement', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_POLICY_CFN_RESOURCE, {
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
      tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:*'],
        effect: iam.Effect.DENY,
        resources: ['*'],
      }));
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_POLICY_CFN_RESOURCE, {
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

  describe('import existing table bucket with name', () => {
    const BUCKET_PROPS = {
      tableBucketName: 'example-table-bucket',
    };
    let tableBucket: s3tables.ITableBucket;

    beforeEach(() => {
      tableBucket = s3tables.TableBucket.fromTableBucketAttributes(stack, 'ExampleTableBucket', BUCKET_PROPS);
    });

    test('has the same name as it was imported with', () => {
      expect(tableBucket.tableBucketName).toEqual(BUCKET_PROPS.tableBucketName);
      tableBucket.grantRead(new iam.ServicePrincipal(''), '*');
    });

    test('renders the correct ARN for Example Resource', () => {
      const arn = stack.resolve(tableBucket.tableBucketArn);
      expect(arn).toEqual({
        'Fn::Join': ['', [
          'arn:',
          { 'Ref': 'AWS::Partition' },
          ':s3tables:',
          { 'Ref': 'AWS::Region' },
          ':',
          { 'Ref': 'AWS::AccountId' },
          `:bucket/${BUCKET_PROPS.tableBucketName}`,
        ]],
      });
    });

    test('returns false from addToResourcePolicy', () => {
      const result = tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toEqual(false);
    });
  });

  describe('import existing table bucket with arn', () => {
    const BUCKET_NAME = 'test-bucket';
    const ACCOUNT_ID = '123456789012';
    const REGION = 'us-west-2';
    const BUCKET_ARN = `arn:aws:s3tables:${REGION}:${ACCOUNT_ID}:bucket/${BUCKET_NAME}`;
    let tableBucket: s3tables.ITableBucket;

    beforeEach(() => {
      tableBucket = s3tables.TableBucket.fromTableBucketArn(stack, 'ExampleTableBucket', BUCKET_ARN);
    });

    test('has the same name as it was imported with', () => {
      expect(tableBucket.tableBucketName).toEqual(BUCKET_NAME);
    });

    test('has the same region as it was imported with', () => {
      expect(tableBucket.region).toEqual(REGION);
    });

    test('has the same account as it was imported with', () => {
      expect(tableBucket.account).toEqual(ACCOUNT_ID);
    });

    test('returns false from addToResourcePolicy', () => {
      const result = tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toEqual(false);
    });
  });

  describe('import existing table bucket with name, region and account', () => {
    const BUCKET_PROPS = {
      tableBucketName: 'example-table-bucket',
      region: 'us-east-2',
      account: '123456789012',
    };
    let tableBucket: s3tables.ITableBucket;

    beforeEach(() => {
      tableBucket = s3tables.TableBucket.fromTableBucketAttributes(stack, 'ExampleTableBucket', BUCKET_PROPS);
    });

    test('has the same name as it was imported with', () => {
      expect(tableBucket.tableBucketName).toEqual(BUCKET_PROPS.tableBucketName);
    });

    test('has the same account as it was imported with', () => {
      expect(tableBucket.account).toEqual(BUCKET_PROPS.account);
    });

    test('has the same region as it was imported with', () => {
      expect(tableBucket.region).toEqual(BUCKET_PROPS.region);
    });

    test('renders the correct ARN for Example Resource', () => {
      const arn = stack.resolve(tableBucket.tableBucketArn);
      expect(arn).toEqual({
        'Fn::Join': ['', [
          'arn:',
          { 'Ref': 'AWS::Partition' },
          `:s3tables:${BUCKET_PROPS.region}:${BUCKET_PROPS.account}:bucket/${BUCKET_PROPS.tableBucketName}`,
        ]],
      });
    });

    test('returns false from addToResourcePolicy', () => {
      const result = tableBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3tables:*'],
        resources: ['*'],
      }));

      expect(result.statementAdded).toEqual(false);
      Template.fromStack(stack).resourceCountIs('AWS::S3Tables::TableBucketPolicy', 0);
    });
  });

  describe('created with request metrics enabled', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'metrics-enabled-bucket',
      requestMetricsStatus: s3tables.RequestMetricsStatus.ENABLED,
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'MetricsEnabledBucket', TABLE_BUCKET_PROPS);
    });

    test(`creates a ${TABLE_BUCKET_CFN_RESOURCE} resource`, () => {
      Template.fromStack(stack).resourceCountIs(TABLE_BUCKET_CFN_RESOURCE, 1);
    });

    test('has MetricsConfiguration with Enabled status', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': TABLE_BUCKET_PROPS.tableBucketName,
        'MetricsConfiguration': {
          'Status': 'Enabled',
        },
      });
    });
  });

  describe('created with request metrics disabled', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'metrics-disabled-bucket',
      requestMetricsStatus: s3tables.RequestMetricsStatus.DISABLED,
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'MetricsDisabledBucket', TABLE_BUCKET_PROPS);
    });

    test('has MetricsConfiguration with Disabled status', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': TABLE_BUCKET_PROPS.tableBucketName,
        'MetricsConfiguration': {
          'Status': 'Disabled',
        },
      });
    });
  });

  describe('created without request metrics configuration', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'no-metrics-bucket',
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'NoMetricsBucket', TABLE_BUCKET_PROPS);
    });

    test('does not have MetricsConfiguration property', () => {
      const template = Template.fromStack(stack);
      const resources = template.findResources(TABLE_BUCKET_CFN_RESOURCE);
      const resourceKey = Object.keys(resources)[0];
      expect(resources[resourceKey].Properties.MetricsConfiguration).toBeUndefined();
    });
  });

  describe('created with STANDARD storage class', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'standard-storage-bucket',
      storageClass: s3tables.StorageClass.STANDARD,
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'StandardStorageBucket', TABLE_BUCKET_PROPS);
    });

    test(`creates a ${TABLE_BUCKET_CFN_RESOURCE} resource`, () => {
      Template.fromStack(stack).resourceCountIs(TABLE_BUCKET_CFN_RESOURCE, 1);
    });

    test('has StorageClassConfiguration with STANDARD storage class', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': TABLE_BUCKET_PROPS.tableBucketName,
        'StorageClassConfiguration': {
          'StorageClass': 'STANDARD',
        },
      });
    });
  });

  describe('created with INTELLIGENT_TIERING storage class', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'intelligent-tiering-bucket',
      storageClass: s3tables.StorageClass.INTELLIGENT_TIERING,
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'IntelligentTieringBucket', TABLE_BUCKET_PROPS);
    });

    test('has StorageClassConfiguration with INTELLIGENT_TIERING storage class', () => {
      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        'TableBucketName': TABLE_BUCKET_PROPS.tableBucketName,
        'StorageClassConfiguration': {
          'StorageClass': 'INTELLIGENT_TIERING',
        },
      });
    });
  });

  describe('created without storage class configuration', () => {
    const TABLE_BUCKET_PROPS: s3tables.TableBucketProps = {
      tableBucketName: 'no-storage-class-bucket',
    };

    beforeEach(() => {
      new s3tables.TableBucket(stack, 'NoStorageClassBucket', TABLE_BUCKET_PROPS);
    });

    test('does not have StorageClassConfiguration property', () => {
      const template = Template.fromStack(stack);
      const resources = template.findResources(TABLE_BUCKET_CFN_RESOURCE);
      const resourceKey = Object.keys(resources)[0];
      expect(resources[resourceKey].Properties.StorageClassConfiguration).toBeUndefined();
    });
  });

  describe('validateUnreferencedFileRemoval', () => {
    it('should not throw error when unreferencedFileRemovalProperty is undefined', () => {
      expect(() => s3tables.TableBucket.validateUnreferencedFileRemoval(undefined)).not.toThrow();
    });

    it('should not throw error for valid property values', () => {
      const validProperty = {
        noncurrentDays: 1,
        unreferencedDays: 1,
        status: s3tables.UnreferencedFileRemovalStatus.ENABLED,
      };
      expect(() => s3tables.TableBucket.validateUnreferencedFileRemoval(validProperty)).not.toThrow();
    });

    it('should throw error when noncurrentDays is less than 1', () => {
      const invalidProperty = {
        noncurrentDays: 0,
        unreferencedDays: 1,
        status: s3tables.UnreferencedFileRemovalStatus.ENABLED,
      };
      expect(() => s3tables.TableBucket.validateUnreferencedFileRemoval(invalidProperty))
        .toThrow(
          /noncurrentDays must be at least 1/,
        );
    });

    it('should throw error when unreferencedDays is less than 1', () => {
      const invalidProperty = {
        noncurrentDays: 1,
        unreferencedDays: 0,
        status: s3tables.UnreferencedFileRemovalStatus.ENABLED,
      };
      expect(() => s3tables.TableBucket.validateUnreferencedFileRemoval(invalidProperty))
        .toThrow(
          /unreferencedDays must be at least 1/,
        );
    });

    it('should not throw error when optional fields are undefined', () => {
      const partialProperty = {};
      expect(() => s3tables.TableBucket.validateUnreferencedFileRemoval(partialProperty)).not.toThrow();
    });
  });

  describe('validateBucketName', () => {
    it('should accept valid bucket names', () => {
      const validNames = [
        'my-bucket-123',
        'test-bucket',
        'abc',
        'a'.repeat(63),
        '123-bucket',
      ];

      validNames.forEach(name => {
        expect(() => s3tables.TableBucket.validateTableBucketName(name)).not.toThrow();
      });
    });

    it('should skip validation for unresolved tokens', () => {
      const isUnresolved = core.Token.isUnresolved;
      core.Token.isUnresolved = jest.fn().mockReturnValue(true);
      expect(() => s3tables.TableBucket.validateTableBucketName('unresolved')).not.toThrow();
      // Cleanup
      core.Token.isUnresolved = isUnresolved;
    });

    it('should skip validation for undefined name', () => {
      expect(() => s3tables.TableBucket.validateTableBucketName(undefined)).not.toThrow();
    });

    it('should reject bucket names that are too short', () => {
      expect(() => s3tables.TableBucket.validateTableBucketName('XX')).toThrow(
        /Bucket name must be at least 3/,
      );
    });

    it('should reject bucket names that are too long', () => {
      const longName = 'a'.repeat(64);
      expect(() => s3tables.TableBucket.validateTableBucketName(longName)).toThrow(
        /no more than 63 characters/,
      );
    });

    it('should reject bucket names with illegal characters', () => {
      const invalidNames = [
        'My-Bucket', // uppercase
        'bucket!123', // special character
        'bucket.123', // period
        'bucket_123', // underscore
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.TableBucket.validateTableBucketName(name)).toThrow(
          /must only contain lowercase characters, numbers, and hyphens/,
        );
      });
    });

    it('should reject bucket names that start with invalid characters', () => {
      const invalidNames = [
        '-bucket',
        '.bucket',
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.TableBucket.validateTableBucketName(name)).toThrow(
          /must start with a lowercase letter or number/,
        );
      });
    });

    it('should reject bucket names that end with invalid characters', () => {
      const invalidNames = [
        'bucket-',
        'bucket.',
      ];

      invalidNames.forEach(name => {
        expect(() => s3tables.TableBucket.validateTableBucketName(name)).toThrow(
          /must end with a lowercase letter or number/,
        );
      });
    });

    it('should include the invalid bucket name in the error message', () => {
      const invalidName = 'Invalid-Bucket!';
      expect(() => s3tables.TableBucket.validateTableBucketName(invalidName)).toThrow(
        /Invalid-Bucket!/,
      );
    });

    it('should handle empty bucket names', () => {
      expect(() => s3tables.TableBucket.validateTableBucketName('')).toThrow(
        /Bucket name must be at least 3/,
      );
    });
  });

  describe('tagging', () => {
    test('implements ITaggableV2', () => {
      const tableBucket = new s3tables.TableBucket(stack, 'TaggedBucket', {
        tableBucketName: 'tagged-bucket',
      });
      expect(core.TagManager.of(tableBucket)).toBeDefined();
    });

    test('tags are applied to the table bucket', () => {
      const tableBucket = new s3tables.TableBucket(stack, 'TaggedBucket', {
        tableBucketName: 'tagged-bucket',
      });

      core.Tags.of(tableBucket).add('Environment', 'Production');
      core.Tags.of(tableBucket).add('Team', 'DataEng');

      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        Tags: Match.arrayWith([
          Match.objectLike({ Key: 'Environment', Value: 'Production' }),
          Match.objectLike({ Key: 'Team', Value: 'DataEng' }),
        ]),
      });
    });

    test('stack-level tags propagate to table bucket', () => {
      new s3tables.TableBucket(stack, 'TaggedBucket', {
        tableBucketName: 'tagged-bucket',
      });

      core.Tags.of(stack).add('StackTag', 'Propagated');

      Template.fromStack(stack).hasResourceProperties(TABLE_BUCKET_CFN_RESOURCE, {
        Tags: Match.arrayWith([
          Match.objectLike({ Key: 'StackTag', Value: 'Propagated' }),
        ]),
      });
    });
  });
});
