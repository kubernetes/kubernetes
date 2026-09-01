import { Capture, Match, Template } from '../../../assertions';
import * as cbuild from '../../../aws-codebuild';
import * as codepipeline from '../../../aws-codepipeline';
import * as ec2 from '../../../aws-ec2';
import * as ecr from '../../../aws-ecr';
import * as iam from '../../../aws-iam';
import * as s3 from '../../../aws-s3';
import { Stack } from '../../../core';
import * as cdkp from '../../lib';
import { CodeBuildStep } from '../../lib';
import { CDKP_DEFAULT_CODEBUILD_IMAGE } from '../../lib/private/default-codebuild-image';
import type { ModernTestGitHubNpmPipelineProps } from '../testhelpers';
import { PIPELINE_ENV, TestApp, ModernTestGitHubNpmPipeline, OneStackApp } from '../testhelpers';

let app: TestApp;
let pipelineStack: Stack;

// Must be unique across all test files, but preferably also consistent
const OUTDIR = 'testcdk0.out';

beforeEach(() => {
  app = new TestApp({ outdir: OUTDIR });
  pipelineStack = new Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
});

afterEach(() => {
  app.cleanup();
});

test('synth takes arrays of commands', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    installCommands: ['install1', 'install2'],
    commands: ['build1', 'build2', 'test1', 'test2', 'cdk synth'],
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: [
              'install1',
              'install2',
            ],
          },
          build: {
            commands: [
              'build1',
              'build2',
              'test1',
              'test2',
              'cdk synth',
            ],
          },
        },
      })),
    },
  });
});

test('synth sets artifact base-directory to cdk.out', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        artifacts: {
          'base-directory': 'cdk.out',
        },
      })),
    },
  });
});

test('synth supports setting subdirectory', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    installCommands: ['cd subdir'],
    commands: ['true'],
    primaryOutputDirectory: 'subdir/cdk.out',
  });
  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: Match.arrayWith(['cd subdir']),
          },
        },
        artifacts: {
          'base-directory': 'subdir/cdk.out',
        },
      })),
    },
  });
});

test('npm synth sets, or allows setting, UNSAFE_PERM=true', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    env: {
      NPM_CONFIG_UNSAFE_PERM: 'true',
    },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      EnvironmentVariables: [
        {
          Name: 'NPM_CONFIG_UNSAFE_PERM',
          Type: 'PLAINTEXT',
          Value: 'true',
        },
      ],
    },
  });
});

test('Magic CodePipeline variables passed to synth envvars must be rendered in the action', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    env: {
      VERSION: codepipeline.GlobalVariables.executionId,
    },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Build',
      Actions: [
        Match.objectLike({
          Name: 'Synth',
          Configuration: Match.objectLike({
            EnvironmentVariables: Match.serializedJson(Match.arrayWith([
              {
                name: 'VERSION',
                type: 'PLAINTEXT',
                value: '#{codepipeline.PipelineExecutionId}',
              },
            ])),
          }),
        }),
      ],
    }]),
  });
});

test('CodeBuild: environment variables specified in multiple places are correctly merged', () => {
  const vpc = new ec2.Vpc(pipelineStack, 'Vpc');
  const securityGroup = new ec2.SecurityGroup(pipelineStack, 'SecurityGroup', {
    vpc,
  });
  const bucket = s3.Bucket.fromBucketArn(pipelineStack, 'Bucket', 'arn:aws:s3:::this-particular-bucket');
  const fleet = new cbuild.Fleet(pipelineStack, 'Fleet', {
    baseCapacity: 1,
    computeType: cbuild.FleetComputeType.SMALL,
    environmentType: cbuild.EnvironmentType.LINUX_CONTAINER,
  });

  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk-1', {
    synth: new CodeBuildStep('Synth', {
      env: {
        SOME_ENV_VAR: 'SomeValue',
      },
      installCommands: [
        'install1',
        'install2',
      ],
      commands: ['synth'],
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: 'cdk.out',
      buildEnvironment: {
        environmentVariables: {
          INNER_VAR: { value: 'InnerValue' },
        },
        privileged: true,
        dockerServer: {
          computeType: cbuild.DockerServerComputeType.SMALL,
          securityGroups: [securityGroup],
        },
        certificate: { bucket, objectKey: 'my-certificate' },
        fleet,
      },
    }),
  });

  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk-2', {
    synth: new cdkp.CodeBuildStep('Synth', {
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: '.',
      env: {
        SOME_ENV_VAR: 'SomeValue',
      },
      installCommands: [
        'install1',
        'install2',
      ],
      commands: ['synth'],
      buildEnvironment: {
        environmentVariables: {
          INNER_VAR: { value: 'InnerValue' },
        },
        privileged: true,
      },
    }),
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: Match.objectLike({
      PrivilegedMode: true,
      EnvironmentVariables: Match.arrayWith([
        {
          Name: 'INNER_VAR',
          Type: 'PLAINTEXT',
          Value: 'InnerValue',
        },
        {
          Name: 'SOME_ENV_VAR',
          Type: 'PLAINTEXT',
          Value: 'SomeValue',
        },
      ]),
      DockerServer: {
        ComputeType: 'BUILD_GENERAL1_SMALL',
        SecurityGroupIds: [{
          'Fn::GetAtt': ['SecurityGroupDD263621', 'GroupId'],
        }],
      },
      Certificate: 'arn:aws:s3:::this-particular-bucket/my-certificate',
      Fleet: {
        FleetArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('Fleet.*'), 'Arn'] },
      },
    }),
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['install1', 'install2'],
          },
          build: {
            commands: ['synth'],
          },
        },
      })),
    },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: Match.objectLike({
      PrivilegedMode: true,
      EnvironmentVariables: Match.arrayWith([
        {
          Name: 'INNER_VAR',
          Type: 'PLAINTEXT',
          Value: 'InnerValue',
        },
        {
          Name: 'SOME_ENV_VAR',
          Type: 'PLAINTEXT',
          Value: 'SomeValue',
        },
      ]),
    }),
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['install1', 'install2'],
          },
          build: {
            commands: ['synth'],
          },
        },
      })),
    },
  });
});

test('install command can be overridden/specified', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    installCommands: ['/bin/true'],
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['/bin/true'],
          },
        },
      })),
    },
  });
});

test('Synth can output additional artifacts', () => {
  // WHEN
  const synth = new cdkp.ShellStep('Synth', {
    input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
    commands: ['cdk synth'],
  });
  synth.addOutputDirectory('test');
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: synth,
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        artifacts: {
          'secondary-artifacts': {
            Synth_Output: {
              'base-directory': 'cdk.out',
              'files': '**/*',
            },
            Synth_test: {
              'base-directory': 'test',
              'files': '**/*',
            },
          },
        },
      })),
    },
  });
});

test('Synth can be made to run in a VPC', () => {
  const vpc: ec2.Vpc = new ec2.Vpc(pipelineStack, 'NpmSynthTestVpc');

  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    codeBuildDefaults: { vpc },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    VpcConfig: {
      SecurityGroupIds: [
        { 'Fn::GetAtt': ['CdkPipelineBuildSynthCdkBuildProjectSecurityGroupEA44D7C2', 'GroupId'] },
      ],
      Subnets: [
        { Ref: 'NpmSynthTestVpcPrivateSubnet1Subnet81E3AA56' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet2SubnetC1CA3EF0' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet3SubnetA04163EE' },
      ],
      VpcId: { Ref: 'NpmSynthTestVpc5E703F25' },
    },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    Roles: [
      { Ref: 'CdkPipelineBuildSynthCdkBuildProjectRole5E173C62' },
    ],
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: Match.arrayWith(['ec2:DescribeSecurityGroups']),
        Effect: 'Allow',
        Resource: '*',
      }]),
    },
  });
});

test('Modern, using the synthCodeBuildDefaults', () => {
  const vpc: ec2.Vpc = new ec2.Vpc(pipelineStack, 'NpmSynthTestVpc');
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synthCodeBuildDefaults: { vpc },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    VpcConfig: {
      SecurityGroupIds: [
        { 'Fn::GetAtt': ['CdkPipelineBuildSynthCdkBuildProjectSecurityGroupEA44D7C2', 'GroupId'] },
      ],
      Subnets: [
        { Ref: 'NpmSynthTestVpcPrivateSubnet1Subnet81E3AA56' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet2SubnetC1CA3EF0' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet3SubnetA04163EE' },
      ],
      VpcId: { Ref: 'NpmSynthTestVpc5E703F25' },
    },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    Roles: [
      { Ref: 'CdkPipelineBuildSynthCdkBuildProjectRole5E173C62' },
    ],
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: Match.arrayWith(['ec2:DescribeSecurityGroups']),
        Effect: 'Allow',
        Resource: '*',
      }]),
    },
  });
});

test('Modern, using CodeBuildStep', () => {
  const vpc: ec2.Vpc = new ec2.Vpc(pipelineStack, 'NpmSynthTestVpc');
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: new CodeBuildStep('Synth', {
      commands: ['asdf'],
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: 'cdk.out',
      buildEnvironment: {
        computeType: cbuild.ComputeType.LARGE,
      },
    }),
    codeBuildDefaults: { vpc },
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    VpcConfig: {
      SecurityGroupIds: [
        { 'Fn::GetAtt': ['CdkPipelineBuildSynthCdkBuildProjectSecurityGroupEA44D7C2', 'GroupId'] },
      ],
      Subnets: [
        { Ref: 'NpmSynthTestVpcPrivateSubnet1Subnet81E3AA56' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet2SubnetC1CA3EF0' },
        { Ref: 'NpmSynthTestVpcPrivateSubnet3SubnetA04163EE' },
      ],
      VpcId: { Ref: 'NpmSynthTestVpc5E703F25' },
    },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    Roles: [
      { Ref: 'CdkPipelineBuildSynthCdkBuildProjectRole5E173C62' },
    ],
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: Match.arrayWith(['ec2:DescribeSecurityGroups']),
        Effect: 'Allow',
        Resource: '*',
      }]),
    },
  });
});

test('Pipeline action contains a hash that changes as the buildspec changes', () => {
  const hash1 = modernSynthWithAction(() => ({ commands: ['asdf'] }));

  // To make sure the hash is not just random :)
  const hash1prime = modernSynthWithAction(() => ({ commands: ['asdf'] }));

  const hash2 = modernSynthWithAction(() => ({
    installCommands: ['do install'],
  }));
  const hash3 = modernSynthWithAction(() => ({
    synth: new CodeBuildStep('Synth', {
      commands: ['asdf'],
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: 'cdk.out',
      buildEnvironment: {
        computeType: cbuild.ComputeType.LARGE,
      },
    }),
  }));

  const hash4 = modernSynthWithAction(() => ({
    env: {
      xyz: 'SOME-VALUE',
    },
  }));

  expect(hash1).toEqual(hash1prime);

  expect(hash1).not.toEqual(hash2);
  expect(hash1).not.toEqual(hash3);
  expect(hash1).not.toEqual(hash4);
  expect(hash2).not.toEqual(hash3);
  expect(hash2).not.toEqual(hash4);
  expect(hash3).not.toEqual(hash4);
});

function modernSynthWithAction(cb: () => ModernTestGitHubNpmPipelineProps) {
  const _app = new TestApp({ outdir: OUTDIR });
  const _pipelineStack = new Stack(_app, 'PipelineStack', { env: PIPELINE_ENV });

  new ModernTestGitHubNpmPipeline(_pipelineStack, 'Cdk', cb());

  return captureProjectConfigHash(_pipelineStack);
}

function captureProjectConfigHash(_pipelineStack: Stack) {
  const theHash = new Capture();
  Template.fromStack(_pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Build',
      Actions: [
        Match.objectLike({
          Name: 'Synth',
          Configuration: Match.objectLike({
            EnvironmentVariables: Match.serializedJson([
              {
                name: '_PROJECT_CONFIG_HASH',
                type: 'PLAINTEXT',
                value: theHash,
              },
            ]),
          }),
        }),
      ],
    }]),
  });

  return theHash.asString();
}

test('Synth CodeBuild project role can be granted permissions', () => {
  const bucket: s3.IBucket = s3.Bucket.fromBucketArn(pipelineStack, 'Bucket', 'arn:aws:s3:::this-particular-bucket');

  // GIVEN
  const pipe = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
  pipe.buildPipeline();

  // WHEN
  bucket.grantRead(pipe.synthProject);

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([Match.objectLike({
        Action: ['s3:GetObject*', 's3:GetBucket*', 's3:List*'],
        Resource: ['arn:aws:s3:::this-particular-bucket', 'arn:aws:s3:::this-particular-bucket/*'],
      })]),
    },
  });
});

test('Synth can reference an imported ECR repo', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: new cdkp.CodeBuildStep('Synth', {
      commands: ['build'],
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: 'cdk.out',
      buildEnvironment: {
        buildImage: cbuild.LinuxBuildImage.fromEcrRepository(
          ecr.Repository.fromRepositoryName(pipelineStack, 'ECRImage', 'my-repo-name'),
        ),
      },
    }),
  });

  // THEN -- no exception (necessary for linter)
  expect(true).toBeTruthy();
});

test('CodeBuild: Can specify additional policy statements', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: new cdkp.CodeBuildStep('Synth', {
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      primaryOutputDirectory: '.',
      commands: ['synth'],
      rolePolicyStatements: [
        new iam.PolicyStatement({
          actions: ['codeartifact:*', 'sts:GetServiceBearerToken'],
          resources: ['arn:aws:codeartifact:us-east-1:123456789012:repository/my-domain/my-repo'],
        }),
      ],
    }),
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([Match.objectLike({
        Action: [
          'codeartifact:*',
          'sts:GetServiceBearerToken',
        ],
        Resource: 'arn:aws:codeartifact:us-east-1:123456789012:repository/my-domain/my-repo',
      })]),
    },
  });
});

test('Multiple input sources in side-by-side directories', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: new cdkp.ShellStep('Synth', {
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      commands: ['false'],
      additionalInputs: {
        '../sibling': cdkp.CodePipelineSource.gitHub('foo/bar', 'main'),
        'sub': new cdkp.ShellStep('Prebuild', {
          input: cdkp.CodePipelineSource.gitHub('pre/build', 'main'),
          commands: ['true'],
          primaryOutputDirectory: 'built',
        }),
      },
    }),
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([
      {
        Name: 'Source',
        Actions: [
          Match.objectLike({ Configuration: Match.objectLike({ Repo: 'bar' }) }),
          Match.objectLike({ Configuration: Match.objectLike({ Repo: 'build' }) }),
          Match.objectLike({ Configuration: Match.objectLike({ Repo: 'test' }) }),
        ],
      },
      {
        Name: 'Build',
        Actions: [
          Match.objectLike({ Name: 'Prebuild', RunOrder: 1 }),
          Match.objectLike({
            Name: 'Synth',
            RunOrder: 2,
            InputArtifacts: [
              // 3 input artifacts
              Match.anyValue(),
              Match.anyValue(),
              Match.anyValue(),
            ],
          }),
        ],
      },
    ]),
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: [
              '[ ! -d "../sibling" ] || { echo \'additionalInputs: "../sibling" must not exist yet. If you want to merge multiple artifacts, use a "cp" command.\'; exit 1; } && ln -s -- "$CODEBUILD_SRC_DIR_foo_bar_Source" "../sibling"',
              '[ ! -d "sub" ] || { echo \'additionalInputs: "sub" must not exist yet. If you want to merge multiple artifacts, use a "cp" command.\'; exit 1; } && ln -s -- "$CODEBUILD_SRC_DIR_Prebuild_Output" "sub"',
            ],
          },
          build: {
            commands: [
              'false',
            ],
          },
        },
      })),
    },
  });
});

test('Can easily switch on privileged mode for synth', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    dockerEnabledForSynth: true,
    commands: ['LookAtMe'],
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: Match.objectLike({
      PrivilegedMode: true,
    }),
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          build: {
            commands: [
              'LookAtMe',
            ],
          },
        },
      })),
    },
  });
});

test('can provide custom BuildSpec that is merged with generated one', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    synth: new cdkp.CodeBuildStep('Synth', {
      input: cdkp.CodePipelineSource.gitHub('test/test', 'main'),
      env: {
        SOME_ENV_VAR: 'SomeValue',
      },
      buildEnvironment: {
        environmentVariables: {
          INNER_VAR: { value: 'InnerValue' },
        },
        privileged: true,
      },
      installCommands: [
        'install1',
        'install2',
      ],
      commands: ['synth'],
      partialBuildSpec: cbuild.BuildSpec.fromObject({
        env: {
          variables: {
            FOO: 'bar',
          },
        },
        phases: {
          pre_build: {
            commands: ['installCustom'],
          },
        },
        cache: {
          paths: ['node_modules'],
        },
      }),
    }),
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: Match.objectLike({
      PrivilegedMode: true,
      EnvironmentVariables: Match.arrayWith([
        {
          Name: 'INNER_VAR',
          Type: 'PLAINTEXT',
          Value: 'InnerValue',
        },
      ]),
    }),
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        env: {
          variables: {
            FOO: 'bar',
          },
        },
        phases: {
          pre_build: {
            commands: Match.arrayWith(['installCustom']),
          },
          build: {
            commands: ['synth'],
          },
        },
        cache: {
          paths: ['node_modules'],
        },
      })),
    },
  });
});

test('stacks synthesized for pipeline will be checked during synth', () => {
  let stage: OneStackApp = new OneStackApp(pipelineStack, 'MyApp');

  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    installCommands: ['install1', 'install2'],
    commands: ['build1', 'build2', 'test1', 'test2', 'cdk synth'],
  });
  pipeline.addStage(stage);

  // All stacks in the ASM have been synthesized with 'validateOnSynth: true'
  const asm = stage.synth();
  for (const stack of asm.stacks) {
    expect(stack.validateOnSynth).toEqual(true);
  }
});
