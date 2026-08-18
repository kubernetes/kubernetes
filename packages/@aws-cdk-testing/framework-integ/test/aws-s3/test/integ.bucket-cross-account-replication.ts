import * as cdk from 'aws-cdk-lib';
import type { AwsApiCall } from '@aws-cdk/integ-tests-alpha';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as s3 from 'aws-cdk-lib/aws-s3';
import type { Construct } from 'constructs';
import { SET_UNIQUE_REPLICATION_ROLE_NAME } from 'aws-cdk-lib/cx-api';

/**
 * Notes on how to run this integ test
 * (All regions are flexible, my testing used account A with af-south-1 not enabled)
 * Replace 123456789012 and 234567890123 with your own account numbers
 *
 *  * 1. Configure Accounts
 *   a. Account A (123456789012) should be bootstrapped for us-east-1
 *      and needs to set trust permissions for account B (234567890123)
 *      - `cdk bootstrap --trust 234567890123 --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess'`
 *      - assuming this is the default profile for aws credentials
 *   b. Account B (234567890123) should be bootstrapped for us-east-1 and af-south-1
 *     - note Account B needs to have af-south-1 enabled as it is an opt-in region
 *     - assuming this account is configured with the profile 'cross-account' for aws credentials
 *
 * 2. Set environment variables
 *   a. `export CDK_INTEG_ACCOUNT=123456789012`
 *   b. `export CDK_INTEG_CROSS_ACCOUNT=234567890123`
 *
 * 3. Run the integ test (from the @aws-cdk-testing/framework-integ/test directory)
 *   a. Get temporary console access credentials for account B
 *     - `yarn integ aws-s3/test/integ.bucket-cross-account-replication.js`
 *   b. Fall back if temp credentials do not work (account info may be in snapshot)
 *     - `yarn integ aws-s3/test/integ.bucket-cross-account-replication.js --profiles cross-account`
 * */

const app = new cdk.App({
  postCliContext: {
    [SET_UNIQUE_REPLICATION_ROLE_NAME]: false,
  },
});

const destinationAccount = process.env.CDK_INTEG_ACCOUNT || '123456789012';
const sourceAccount = process.env.CDK_INTEG_CROSS_ACCOUNT || '234567890123';

class DestinationBucketStack extends cdk.Stack {
  public readonly bucket: s3.IBucket;
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.bucket = new s3.Bucket(this, 'DestinationBucketnew', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      objectOwnership: s3.ObjectOwnership.OBJECT_WRITER,
    });
  }
}

interface SourceBucketStackProps extends cdk.StackProps {
  bucket: s3.IBucket;
}

class SourceBucketStack extends cdk.Stack {
  public readonly bucket: s3.IBucket;
  constructor(scope: Construct, id: string, props: SourceBucketStackProps) {
    super(scope, id, props);
    this.bucket = new s3.Bucket(this, 'SourceBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      encryption: s3.BucketEncryption.S3_MANAGED,
      replicationRules: [
        {
          destination: props.bucket,
          priority: 1,
          accessControlTransition: true,
          replicationTimeControl: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
          metrics: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
          storageClass: s3.StorageClass.INFREQUENT_ACCESS,
          replicaModifications: true,
          deleteMarkerReplication: true,
          id: 'full-settings-rule',
        },
      ],
    });
  }
}

const destinationBucketStack = new DestinationBucketStack(app, 'parent-stack', {
  env: {
    account: destinationAccount,
    region: 'us-east-1',
  },
});

const sourceBucketStack = new SourceBucketStack(app, 'child-stack', {
  env: {
    account: sourceAccount,
    region: 'us-east-1',
  },
  bucket: destinationBucketStack.bucket,
});

sourceBucketStack.addStackDependency(destinationBucketStack);
// Add permissions to the destination bucket policy after the replication role is created
// Comment these lines while deploying this integ test for the first time to avoid failure
if (sourceBucketStack.bucket.replicationRoleArn) {
  destinationBucketStack.bucket.addReplicationPolicy(
    sourceBucketStack.bucket.replicationRoleArn,
    true,
    sourceAccount,
  );
}

const integ = new IntegTest(app, 'ReplicationInteg', {
  testCases: [destinationBucketStack, sourceBucketStack],
});

// Assertion to put object into the bucket and test for replication
// Manually Verify that object test-object is replicated in account A (destination account)
const firstAssertion = integ.assertions
  .awsApiCall('S3', 'putObject', {
    Bucket: sourceBucketStack.bucket.bucketName,
    Key: 'test-object',
    Body: 'test-object-body',
    ContentType: 'text/plain',
  })
  .waitForAssertions({
    totalTimeout: cdk.Duration.minutes(5),
  }) as AwsApiCall;
firstAssertion.waiterProvider?.addToRolePolicy({
  Effect: 'Allow',
  Action: ['kms:*'],
  Resource: ['*'],
});
