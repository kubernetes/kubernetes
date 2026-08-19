import * as cdk from 'aws-cdk-lib';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import type { Construct } from 'constructs';
import { SET_UNIQUE_REPLICATION_ROLE_NAME } from 'aws-cdk-lib/cx-api';

/**
 * This test covers cross-account replication to a destination bucket that is *imported* into the
 * source stack with `Bucket.fromBucketAttributes()` and the `account` attribute, as documented in
 * the aws-s3 README. Because CDK cannot modify the bucket policy of a referenced bucket, the
 * required bucket policy is created by a separate stack in the destination account. That stack
 * deploys after the source stack, so the replication role already exists when the policy
 * references it as a principal.
 *
 * Notes on how to run this integ test
 * (Same two-account setup as integ.bucket-cross-account-replication.ts; all stacks use us-east-1.
 * Replace 123456789012 and 234567890123 with your own account numbers.)
 *
 * 1. Configure Accounts
 *   a. Account A (123456789012) hosts the destination bucket and its bucket policy. It should be
 *      bootstrapped for us-east-1 and needs to set trust permissions for account B (234567890123)
 *      - `cdk bootstrap --trust 234567890123 --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess'`
 *      - assuming this is the default profile for aws credentials
 *   b. Account B (234567890123) hosts the source bucket and should be bootstrapped for us-east-1
 *     - assuming this account is configured with the profile 'cross-account' for aws credentials
 *
 * 2. Set environment variables
 *   a. `export CDK_INTEG_ACCOUNT=123456789012`
 *   b. `export CDK_INTEG_CROSS_ACCOUNT=234567890123`
 *
 * 3. Run the integ test (from the @aws-cdk-testing/framework-integ/test directory)
 *   a. Get temporary console access credentials for account B
 *     - `yarn integ aws-s3/test/integ.bucket-cross-account-replication-imported.js`
 *   b. Fall back if temp credentials do not work (account info may be in snapshot)
 *     - `yarn integ aws-s3/test/integ.bucket-cross-account-replication-imported.js --profiles cross-account`
 *
 * 4. Manually verify that the object 'test-object' is replicated to the destination bucket in
 *    account A and that the replica is owned by account A (accessControlTransition).
 * */

const app = new cdk.App({
  postCliContext: {
    [SET_UNIQUE_REPLICATION_ROLE_NAME]: true,
  },
});

const destinationAccount = process.env.CDK_INTEG_ACCOUNT || '123456789012';
const sourceAccount = process.env.CDK_INTEG_CROSS_ACCOUNT || '234567890123';

class DestinationBucketStack extends cdk.Stack {
  public readonly bucket: s3.Bucket;
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.bucket = new s3.Bucket(this, 'DestinationBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      objectOwnership: s3.ObjectOwnership.OBJECT_WRITER,
    });
  }
}

interface SourceBucketStackProps extends cdk.StackProps {
  destinationBucketName: string;
}

class SourceBucketStack extends cdk.Stack {
  public readonly bucket: s3.Bucket;
  constructor(scope: Construct, id: string, props: SourceBucketStackProps) {
    super(scope, id, props);

    // The destination bucket exists in another account, so it is imported by name and the
    // `account` attribute identifies the destination bucket owner's account. Without the
    // `account` attribute the bucket would be treated as belonging to this stack's account.
    const destinationBucket = s3.Bucket.fromBucketAttributes(this, 'ImportedDestinationBucket', {
      bucketName: props.destinationBucketName,
      account: destinationAccount,
    });

    this.bucket = new s3.Bucket(this, 'SourceBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      bucketName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      encryption: s3.BucketEncryption.S3_MANAGED,
      replicationRules: [
        {
          destination: destinationBucket,
          priority: 1,
          accessControlTransition: true,
          deleteMarkerReplication: true,
          id: 'imported-destination-rule',
        },
      ],
    });
  }
}

interface DestinationBucketPolicyStackProps extends cdk.StackProps {
  destinationBucket: s3.Bucket;
  replicationRoleArn: string;
}

class DestinationBucketPolicyStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: DestinationBucketPolicyStackProps) {
    super(scope, id, props);

    // CDK cannot modify the bucket policy of a bucket that was imported into the source stack,
    // so the destination account configures the policy itself. The statements mirror what
    // Bucket.addReplicationPolicy() would add.
    // https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-walkthrough-2.html
    const bucketPolicy = new s3.BucketPolicy(this, 'DestinationBucketPolicy', {
      bucket: props.destinationBucket,
    });
    bucketPolicy.document.addStatements(
      new iam.PolicyStatement({
        actions: ['s3:GetBucketVersioning', 's3:PutBucketVersioning'],
        resources: [props.destinationBucket.bucketArn],
        principals: [new iam.ArnPrincipal(props.replicationRoleArn)],
      }),
      new iam.PolicyStatement({
        actions: ['s3:ReplicateObject', 's3:ReplicateDelete'],
        resources: [props.destinationBucket.arnForObjects('*')],
        principals: [new iam.ArnPrincipal(props.replicationRoleArn)],
      }),
      new iam.PolicyStatement({
        actions: ['s3:ObjectOwnerOverrideToBucketOwner'],
        resources: [props.destinationBucket.arnForObjects('*')],
        principals: [new iam.AccountPrincipal(sourceAccount)],
      }),
    );
  }
}

const destinationBucketStack = new DestinationBucketStack(app, 'destination-bucket-stack', {
  env: {
    account: destinationAccount,
    region: 'us-east-1',
  },
});

const sourceBucketStack = new SourceBucketStack(app, 'source-bucket-stack', {
  env: {
    account: sourceAccount,
    region: 'us-east-1',
  },
  destinationBucketName: destinationBucketStack.bucket.bucketName,
});
sourceBucketStack.addStackDependency(destinationBucketStack);

const replicationRoleArn = sourceBucketStack.bucket.replicationRoleArn;
if (!replicationRoleArn) {
  throw new Error('expected the source bucket to create a replication role');
}

const destinationPolicyStack = new DestinationBucketPolicyStack(app, 'destination-policy-stack', {
  env: {
    account: destinationAccount,
    region: 'us-east-1',
  },
  destinationBucket: destinationBucketStack.bucket,
  replicationRoleArn,
});
// Deploy the bucket policy after the source stack so that the replication role it references as
// a principal already exists.
destinationPolicyStack.addStackDependency(sourceBucketStack);

const integ = new IntegTest(app, 'ReplicationImportedDestinationInteg', {
  testCases: [destinationBucketStack, sourceBucketStack, destinationPolicyStack],
});

// Assertion to put an object into the source bucket and test for replication.
// Manually verify that 'test-object' is replicated to the destination bucket in account A and
// owned by the destination account.
integ.assertions
  .awsApiCall('S3', 'putObject', {
    Bucket: sourceBucketStack.bucket.bucketName,
    Key: 'test-object',
    Body: 'test-object-body',
    ContentType: 'text/plain',
  })
  .waitForAssertions({
    totalTimeout: cdk.Duration.minutes(5),
  });
