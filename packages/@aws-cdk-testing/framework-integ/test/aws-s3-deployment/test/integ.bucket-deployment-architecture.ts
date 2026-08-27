import * as path from 'path';
import type * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cdk from 'aws-cdk-lib';
import { ExpectedResult, IntegTest, Match } from '@aws-cdk/integ-tests-alpha';
import type { Construct } from 'constructs';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';

/**
 * Integration test for the architecture of the deployment resource handler.
 *
 * The handler is CDK-managed internal code and always runs on ARM_64 (Graviton).
 * This test deploys against a real account to verify that:
 * - the handler function is actually created on `arm64`
 * - the handler succeeds on `arm64`, i.e. the bundled AWS CLI layer runs there and
 *   the source files really do land in the destination bucket
 * - deploy-time marker substitution, which round-trips the sources through the
 *   handler, also works on `arm64`
 */
class TestBucketDeploymentArchitectureStack extends cdk.Stack {
  public readonly destinationBucket: s3.IBucket;
  public readonly substitutedBucket: s3.IBucket;
  public readonly handlerFunctionName: string;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const commonBucketProps = {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true, // needed for integration test cleanup
    };

    // A plain asset deployment: exercises unzip + `aws s3 sync` inside the arm64 handler.
    this.destinationBucket = new s3.Bucket(this, 'Destination', commonBucketProps);

    const deployment = new s3deploy.BucketDeployment(this, 'Deploy', {
      sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
      destinationBucket: this.destinationBucket,
      retainOnDelete: false,
    });

    // A deployment with deploy-time markers: exercises the handler's substitution code path,
    // which rewrites file contents in the Lambda before uploading.
    this.substitutedBucket = new s3.Bucket(this, 'Substituted', commonBucketProps);

    new s3deploy.BucketDeployment(this, 'DeployWithMarkers', {
      sources: [
        s3deploy.Source.jsonData('config.json', {
          bucketName: this.destinationBucket.bucketName,
        }),
      ],
      destinationBucket: this.substitutedBucket,
      retainOnDelete: false,
    });

    // Both deployments share the same singleton handler, so either one gives us its name.
    const handler = deployment.node.findChild('CustomResourceHandler') as lambda.SingletonFunction;
    this.handlerFunctionName = handler.functionName;
  }
}

const app = new cdk.App();
const testStack = new TestBucketDeploymentArchitectureStack(app, 'test-bucket-deployment-architecture');

const integTest = new IntegTest(app, 'integ-test-bucket-deployment-architecture', {
  testCases: [testStack],
});

// The deployed handler really is arm64, not just in the template.
integTest.assertions.awsApiCall('Lambda', 'getFunction', {
  FunctionName: testStack.handlerFunctionName,
}).expect(ExpectedResult.objectLike({
  Configuration: Match.objectLike({
    Architectures: ['arm64'],
  }),
}));

// The arm64 handler succeeded: the asset was unzipped and synced to the bucket.
const listObjects = integTest.assertions.awsApiCall('S3', 'listObjects', {
  Bucket: testStack.destinationBucket.bucketName,
});
listObjects.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['s3:GetObject', 's3:ListBucket'],
  Resource: ['*'],
});
listObjects.expect(ExpectedResult.objectLike({
  Contents: Match.arrayWith([
    Match.objectLike({ Key: 'index.html' }),
  ]),
}));

// The arm64 handler also performed deploy-time marker substitution correctly.
const getSubstituted = integTest.assertions.awsApiCall('S3', 'getObject', {
  Bucket: testStack.substitutedBucket.bucketName,
  Key: 'config.json',
});
getSubstituted.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['s3:GetObject', 's3:ListBucket'],
  Resource: ['*'],
});
getSubstituted.expect(ExpectedResult.objectLike({
  Body: Match.stringLikeRegexp(testStack.destinationBucket.bucketName),
}));
