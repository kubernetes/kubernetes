import type { StackProps, StageProps } from 'aws-cdk-lib';
import { App, Duration, RemovalPolicy, Stack, Stage } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import type { Construct } from 'constructs';
import * as pipelines from 'aws-cdk-lib/pipelines';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';

/**
 * Notes on how to run this integ test
 * Replace 123456789012 and 234567890123 with your own account numbers
 *
 * 1. Configure Accounts
 *   a. Account A (123456789012) should be bootstrapped for us-east-1
 *      and needs to set trust permissions for account B (234567890123)
 *      - `cdk bootstrap --trust 234567890123 --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess'`
 *      - assuming this is the default profile for aws credentials
 *   b. Account B (234567890123) should be bootstrapped for us-east-1
 *     - assuming this account is configured with the profile 'cross-account' for aws credentials
 *
 * 2. Set environment variables
 *   a. `export CDK_INTEG_ACCOUNT=123456789012`
 *   b. `export CDK_INTEG_CROSS_ACCOUNT=234567890123`
 *
 * 3. Run the integ test (from the @aws-cdk-testing/framework-integ/test directory)
 *   a. Get temporary console access credentials for account B
 *     - `yarn integ pipelines/test/integ.cross-account-pipeline-action.js`
 *   b. Fall back if temp credentials do not work (account info may be in snapshot)
 *     - `yarn integ pipelines/test/integ.cross-account-pipeline-action.js --profiles cross-account`
 *
 * 4. Before you commit, set both accounts to dummy values, run integ test in dry run mode, and then push the snapshot.
 */

const account = process.env.CDK_INTEG_ACCOUNT || '123456789012';
const crossAccount = process.env.CDK_INTEG_CROSS_ACCOUNT || '234567890123';
const region = process.env.CDK_INTEG_REGION || process.env.CDK_DEFAULT_REGION;

class SourceStack extends Stack {
  public readonly sourceBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const bucketName = `cross-account-source-bucket-${account}-${this.region}`;

    this.sourceBucket = new s3.Bucket(this, 'SourceBucket', {
      bucketName,
      versioned: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    new s3deploy.BucketDeployment(this, 'DeploySource', {
      sources: [s3deploy.Source.data('source.zip', 'dummy content')],
      destinationBucket: this.sourceBucket,
    });
  }
}

class ProdStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    new sqs.Queue(this, 'MyQueue', {
      queueName: 'prod-queue',
      visibilityTimeout: Duration.seconds(300),
    });
  }
}

class ProdStage extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    new ProdStack(this, 'ProdStack', props);
  }
}

class PipelineStack extends Stack {
  constructor(scope: Construct, id: string, sourceBucket: s3.IBucket, props?: StackProps) {
    super(scope, id, props);

    const pipeline = new pipelines.CodePipeline(this, 'Pipeline', {
      pipelineName: 'cross-account-pipeline',
      crossAccountKeys: true,
      useChangeSets: false,
      synth: new pipelines.ShellStep('Synth', {
        input: pipelines.CodePipelineSource.s3(sourceBucket, 'source.zip', {
          actionName: 'S3Source',
        }),
        commands: ['npm ci && npm run build'],
      }),
      dockerEnabledForSynth: true,
      codeBuildDefaults: {
        buildEnvironment: {
          buildImage: codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
          computeType: codebuild.ComputeType.SMALL,
        },
        cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER),
      },
    });

    pipeline.addStage(new ProdStage(this, 'Prod'), {
      pre: [new pipelines.ManualApprovalStep('PromoteToProd')],
    });
  }
}

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
    '@aws-cdk/aws-codepipeline:defaultPipelineTypeToV2': false,
    '@aws-cdk/pipelines:reduceCrossAccountActionRoleTrustScope': true,
  },
});

const sourceStack = new SourceStack(app, 'CrossAccountSourceStack', {
  env: {
    account: crossAccount,
    region,
  },
});

const pipelineStack = new PipelineStack(app, 'CdkPipelineInvestigationStack',
  sourceStack.sourceBucket,
  {
    env: {
      account: account,
      region,
    },
  },
);

pipelineStack.addStackDependency(sourceStack);

new integ.IntegTest(app, 'CdkPipelineInvestigationTest', {
  testCases: [sourceStack, pipelineStack],
  diffAssets: true,
});

app.synth();
