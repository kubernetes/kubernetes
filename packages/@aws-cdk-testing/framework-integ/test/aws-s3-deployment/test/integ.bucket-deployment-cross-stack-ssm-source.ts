import * as cdk from 'aws-cdk-lib';
import type { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as integ from '@aws-cdk/integ-tests-alpha';

/**
 * Integration test for bucket deployment with cross-stack SSM parameter references:
 * - Tests that SSM StringListParameter tokens are resolved in Source.jsonData()
 * - Validates cross-nested-stack parameter references work correctly
 * - Tests that parameter values are properly serialized in JSON output
 */
class SsmStack extends cdk.NestedStack {
  public readonly ssmParam: ssm.StringListParameter;

  constructor(scope: Construct, id: string, props?: cdk.NestedStackProps) {
    super(scope, id, props);

    const testSubnets = ['subnet-12345', 'subnet-67890'];

    this.ssmParam = new ssm.StringListParameter(this, 'TestParam', {
      parameterName: '/repro/subnets',
      stringListValue: testSubnets,
      description: 'Test parameter for reproduction',
    });
  }
}

class S3Stack extends cdk.NestedStack {
  public readonly bucket: s3.Bucket;

  constructor(scope: Construct, id: string, props?: cdk.NestedStackProps) {
    super(scope, id, props);

    const readParam = ssm.StringListParameter.fromStringListParameterName(
      this,
      'ReadParam',
      '/repro/subnets',
    );

    this.bucket = new s3.Bucket(this, 'ReproBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    new s3deploy.BucketDeployment(this, 'DeployWithSsmParameter', {
      sources: [
        s3deploy.Source.jsonData('config.json', {
          subnets: readParam.stringListValue,
          expectedValues: ['subnet-12345', 'subnet-67890'],
          version: '2.207.0',
          issue: 'StringListParameter tokens not resolved in Source.jsonData',
          timestamp: new Date().toISOString(),
        }),
      ],
      destinationBucket: this.bucket,
    });
  }
}

export class MainStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const ssmStack = new SsmStack(this, 'SsmStack');
    const s3Stack = new S3Stack(this, 'S3Stack');
    s3Stack.addStackDependency(ssmStack);

    new cdk.CfnOutput(this, 'BucketName', {
      value: s3Stack.bucket.bucketName,
      description: 'Check config.json in this bucket',
    });

    new cdk.CfnOutput(this, 'ParameterName', {
      value: ssmStack.ssmParam.parameterName,
      description: 'SSM parameter name',
    });

    new cdk.CfnOutput(this, 'ExpectedValues', {
      value: JSON.stringify(['subnet-12345', 'subnet-67890']),
      description: 'Expected subnet values in config.json',
    });

    new cdk.CfnOutput(this, 'VerificationCommand', {
      value: `aws s3 cp s3://${s3Stack.bucket.bucketName}/config.json - | jq .`,
      description: 'Command to check the deployed JSON',
    });
  }
}

const app = new cdk.App();
const stack = new MainStack(app, 'integ-bucket-deployment-cross-stack-ssm');

new integ.IntegTest(app, 'integ-bucket-deployment-cross-stack-ssm-source', {
  testCases: [stack],
});
