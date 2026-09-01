import type { Construct } from 'constructs';
import { Match, Template } from '../../../assertions';
import * as cb from '../../../aws-codebuild';
import * as secretsmanager from '../../../aws-secretsmanager';
import { Stack } from '../../../core';
import * as cdkp from '../../lib';
import { CodeBuildStep } from '../../lib';
import { CDKP_DEFAULT_CODEBUILD_IMAGE } from '../../lib/private/default-codebuild-image';
import { PIPELINE_ENV, TestApp, ModernTestGitHubNpmPipeline, DockerAssetApp, stringLike } from '../testhelpers';

const secretSynthArn = 'arn:aws:secretsmanager:eu-west-1:012345678912:secret:synth-012345';
const secretUpdateArn = 'arn:aws:secretsmanager:eu-west-1:012345678912:secret:update-012345';
const secretPublishArn = 'arn:aws:secretsmanager:eu-west-1:012345678912:secret:publish-012345';

let app: TestApp;
let pipelineStack: Stack;
let secretSynth: secretsmanager.ISecret;
let secretUpdate: secretsmanager.ISecret;
let secretPublish: secretsmanager.ISecret;

beforeEach(() => {
  app = new TestApp();
  pipelineStack = new Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  secretSynth = secretsmanager.Secret.fromSecretCompleteArn(pipelineStack, 'Secret1', secretSynthArn);
  secretUpdate = secretsmanager.Secret.fromSecretCompleteArn(pipelineStack, 'Secret2', secretUpdateArn);
  secretPublish = secretsmanager.Secret.fromSecretCompleteArn(pipelineStack, 'Secret3', secretPublishArn);
});

afterEach(() => {
  app.cleanup();
});

test('synth action receives install commands and access to relevant credentials', () => {
  const pipeline = new ModernPipelineWithCreds(pipelineStack, 'Cdk');
  pipeline.addStage(new DockerAssetApp(app, 'App1'));

  const expectedCredsConfig = JSON.stringify({
    version: '1.0',
    domainCredentials: { 'synth.example.com': { secretsManagerSecretId: secretSynthArn } },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: { Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          pre_build: {
            commands: Match.arrayWith([
              'mkdir $HOME/.cdk',
              `echo '${expectedCredsConfig}' > $HOME/.cdk/cdk-docker-creds.json`,
            ]),
          },
          // Prove we're looking at the Synth project
          build: {
            commands: Match.arrayWith([stringLike('*cdk*synth*')]),
          },
        },
      })),
    },
  });
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret'],
        Effect: 'Allow',
        Resource: secretSynthArn,
      }]),
      Version: '2012-10-17',
    },
    Roles: [{ Ref: stringLike('Cdk*BuildProjectRole*') }],
  });
});

test('synth action receives Windows install commands if a Windows image is detected', () => {
  const pipeline = new ModernPipelineWithCreds(pipelineStack, 'Cdk2', {
    synth: new CodeBuildStep('Synth', {
      commands: ['cdk synth'],
      primaryOutputDirectory: 'cdk.out',
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      buildEnvironment: {
        buildImage: cb.WindowsBuildImage.WINDOWS_BASE_2_0,
        computeType: cb.ComputeType.MEDIUM,
      },
    }),
  });
  pipeline.addStage(new DockerAssetApp(app, 'App1'));

  const expectedCredsConfig = JSON.stringify({
    version: '1.0',
    domainCredentials: { 'synth.example.com': { secretsManagerSecretId: secretSynthArn } },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: { Image: 'aws/codebuild/windows-base:2.0' },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          pre_build: {
            commands: Match.arrayWith([
              'mkdir %USERPROFILE%\\.cdk',
              `echo '${expectedCredsConfig}' > %USERPROFILE%\\.cdk\\cdk-docker-creds.json`,
            ]),
          },
          // Prove we're looking at the Synth project
          build: {
            commands: Match.arrayWith([stringLike('*cdk*synth*')]),
          },
        },
      })),
    },
  });
});

test('self-update receives install commands and access to relevant credentials', () => {
  const pipeline = new ModernPipelineWithCreds(pipelineStack, 'Cdk');
  pipeline.addStage(new DockerAssetApp(app, 'App1'));

  const expectedCredsConfig = JSON.stringify({
    version: '1.0',
    domainCredentials: { 'selfupdate.example.com': { secretsManagerSecretId: secretUpdateArn } },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: { Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          pre_build: {
            commands: Match.arrayWith([
              'mkdir $HOME/.cdk',
              `echo '${expectedCredsConfig}' > $HOME/.cdk/cdk-docker-creds.json`,
            ]),
          },
          // Prove we're looking at the SelfMutate project
          build: {
            commands: Match.arrayWith([
              stringLike('cdk * deploy PipelineStack*'),
            ]),
          },
        },
      })),
    },
  });
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret'],
        Effect: 'Allow',
        Resource: secretUpdateArn,
      }]),
      Version: '2012-10-17',
    },
    Roles: [{ Ref: stringLike('*SelfMutat*Role*') }],
  });
});

test('asset publishing receives install commands and access to relevant credentials', () => {
  const pipeline = new ModernPipelineWithCreds(pipelineStack, 'Cdk');
  pipeline.addStage(new DockerAssetApp(app, 'App1'));

  const expectedCredsConfig = JSON.stringify({
    version: '1.0',
    domainCredentials: { 'publish.example.com': { secretsManagerSecretId: secretPublishArn } },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: { Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          pre_build: {
            commands: Match.arrayWith([
              'mkdir $HOME/.cdk',
              `echo '${expectedCredsConfig}' > $HOME/.cdk/cdk-docker-creds.json`,
            ]),
          },
          // Prove we're looking at the Publishing project
          build: {
            commands: Match.arrayWith([stringLike('cdk-assets*')]),
          },
        },
      })),
    },
  });
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret'],
        Effect: 'Allow',
        Resource: secretPublishArn,
      }]),
      Version: '2012-10-17',
    },
    Roles: [{ Ref: 'CdkAssetsDockerRole484B6DD3' }],
  });
});

class ModernPipelineWithCreds extends ModernTestGitHubNpmPipeline {
  constructor(scope: Construct, id: string, props?: ConstructorParameters<typeof ModernTestGitHubNpmPipeline>[2]) {
    super(scope, id, {
      dockerCredentials: [
        cdkp.DockerCredential.customRegistry('synth.example.com', secretSynth, {
          usages: [cdkp.DockerCredentialUsage.SYNTH],
        }),
        cdkp.DockerCredential.customRegistry('selfupdate.example.com', secretUpdate, {
          usages: [cdkp.DockerCredentialUsage.SELF_UPDATE],
        }),
        cdkp.DockerCredential.customRegistry('publish.example.com', secretPublish, {
          usages: [cdkp.DockerCredentialUsage.ASSET_PUBLISHING],
        }),
      ],
      ...props,
    });
  }
}
