import type { StackProps } from 'aws-cdk-lib';
import { App, Fn, PhysicalName, RemovalPolicy, Stack } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import type { Construct } from 'constructs';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as codepipeline from 'aws-cdk-lib/aws-codepipeline';
import * as codepipeline_actions from 'aws-cdk-lib/aws-codepipeline-actions';
import { AccountPrincipal, Role } from 'aws-cdk-lib/aws-iam';
import { Key } from 'aws-cdk-lib/aws-kms';
import * as path from 'path';

const account = process.env.CDK_INTEG_ACCOUNT || '123456789012';
const crossAccount = process.env.CDK_INTEG_CROSS_ACCOUNT || '234567890123';
const region = process.env.CDK_INTEG_REGION || process.env.CDK_DEFAULT_REGION;

class StepFunctionStack extends Stack {
  public readonly stateMachine: sfn.StateMachine;
  public readonly crossAccountRole: Role;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // Create a simple Step Function
    this.stateMachine = new sfn.StateMachine(this, 'CrossAccountStateMachine', {
      definitionBody: sfn.DefinitionBody.fromChainable(new sfn.Pass(this, 'PassState', {
        result: sfn.Result.fromObject({ message: 'Hello from cross-account Step Function!' }),
      })),
      stateMachineName: 'CrossAccountStateMachine',
    });

    this.crossAccountRole = new Role(this, 'CrossAccountRole', {
      roleName: PhysicalName.GENERATE_IF_NEEDED,
      assumedBy: new AccountPrincipal(account),
    });

    this.stateMachine.grantStartExecution(this.crossAccountRole);
  }
}

class PipelineStack extends Stack {
  constructor(scope: Construct, id: string, stateMachine: sfn.IStateMachine, props?: StackProps) {
    super(scope, id, props);

    const sourceBucketKey = new Key(this, 'SourceBucketKey', {
      description: 'SourceBucketKey',
    });
    const bucket = new s3.Bucket(this, 'SourceBucket', {
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      versioned: true,
      encryptionKey: sourceBucketKey,
    });

    const bucketDeployment = new s3deploy.BucketDeployment(this, 'BucketDeployment', {
      sources: [s3deploy.Source.asset(path.join(__dirname, 'assets', 'nodejs.zip'))],
      destinationBucket: bucket,
      extract: false,
    });
    const zipKey = Fn.select(0, bucketDeployment.objectKeys);

    const sourceOutput = new codepipeline.Artifact('SourceArtifact');
    const sourceAction = new codepipeline_actions.S3SourceAction({
      actionName: 'Source',
      output: sourceOutput,
      bucket,
      bucketKey: zipKey,
    });
    const buildOutput = new codepipeline.Artifact('BuildOutput');

    // Create the pipeline
    const pipeline = new codepipeline.Pipeline(this, 'CrossAccountStepFunctionsPipeline', {
      pipelineName: 'cross-account-sfn-pipeline',
      crossAccountKeys: true,
      artifactBucket: new s3.Bucket(this, 'ArtifactBucket', {
        encryption: s3.BucketEncryption.KMS,
        removalPolicy: RemovalPolicy.DESTROY,
        autoDeleteObjects: true,
      }),
    });

    // Add the source stage
    pipeline.addStage({
      stageName: 'Source',
      actions: [
        sourceAction,
      ],
    });

    // Add a build stage
    pipeline.addStage({
      stageName: 'Build',
      actions: [
        new codepipeline_actions.CodeBuildAction({
          actionName: 'BuildAction',
          project: new codebuild.PipelineProject(this, 'BuildProject', {
            environment: {
              buildImage: codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
              computeType: codebuild.ComputeType.SMALL,
            },
            buildSpec: codebuild.BuildSpec.fromObject({
              version: '0.2',
              phases: {
                install: {
                  commands: [
                    'npm install -y jq',
                  ],
                },
                build: {
                  commands: [
                    'echo "Starting build..."',
                  ],
                },
              },
            }),
          }),
          input: sourceOutput,
          outputs: [buildOutput],
        }),
      ],
    });

    // Add the Step Function invoke stage
    pipeline.addStage({
      stageName: 'InvokeStepFunction',
      actions: [
        new codepipeline_actions.StepFunctionInvokeAction({
          actionName: 'InvokeStateMachine',
          stateMachine,
          stateMachineInput: codepipeline_actions.StateMachineInput.literal({
            source: 'codepipeline',
          }),
        }),
      ],
    });
  }
}

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
    '@aws-cdk/pipelines:reduceStageRoleTrustScope': true,
  },
});

// Create the Step Function stack in the cross account
const stepFunctionStack = new StepFunctionStack(app, 'CrossAccountStepFunctionStack', {
  env: {
    account: crossAccount,
    region,
  },
});

const pipelineStack = new PipelineStack(app, 'CdkPipelineStepFunctionsActionStack',
  stepFunctionStack.stateMachine,
  {
    env: {
      account: account,
      region,
    },
  },
);

pipelineStack.addStackDependency(stepFunctionStack);

new integ.IntegTest(app, 'integ-cross-account-pipeline-sfn-action', {
  testCases: [stepFunctionStack, pipelineStack],
  diffAssets: true,
});

app.synth();
