import * as fs from 'fs';
import * as path from 'path';
import { Capture, Match, Template } from '../../../assertions';
import * as cb from '../../../aws-codebuild';
import * as ec2 from '../../../aws-ec2';
import { Stack, Stage } from '../../../core';
import { CDKP_DEFAULT_CODEBUILD_IMAGE } from '../../lib/private/default-codebuild-image';
import { PIPELINE_ENV, TestApp, ModernTestGitHubNpmPipeline, FileAssetApp, MegaAssetsApp, MultipleFileAssetsApp, DockerAssetApp, PlainStackApp, stringLike } from '../testhelpers';

const FILE_ASSET_SOURCE_HASH = '8289faf53c7da377bb2b90615999171adef5e1d8f6b88810e5fef75e6ca09ba5';
const FILE_ASSET_SOURCE_HASH2 = 'ac76997971c3f6ddf37120660003f1ced72b4fc58c498dfd99c78fa77e721e0e';

const FILE_PUBLISHING_ROLE = 'arn:${AWS::Partition}:iam::${AWS::AccountId}:role/cdk-hnb659fds-file-publishing-role-${AWS::AccountId}-${AWS::Region}';
const IMAGE_PUBLISHING_ROLE = 'arn:${AWS::Partition}:iam::${AWS::AccountId}:role/cdk-hnb659fds-image-publishing-role-${AWS::AccountId}-${AWS::Region}';

let app: TestApp;
let pipelineStack: Stack;

function expectedAssetRolePolicy(assumeRolePattern: string | string[], attachedRole: string) {
  if (typeof assumeRolePattern === 'string') { assumeRolePattern = [assumeRolePattern]; }

  return {
    PolicyDocument: {
      Statement: [{
        Action: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
        Effect: 'Allow',
        Resource: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            `:logs:${PIPELINE_ENV.region}:${PIPELINE_ENV.account}:log-group:/aws/codebuild/*`,
          ]],
        },
      },
      {
        Action: ['codebuild:CreateReportGroup', 'codebuild:CreateReport', 'codebuild:UpdateReport', 'codebuild:BatchPutTestCases', 'codebuild:BatchPutCodeCoverages'],
        Effect: 'Allow',
        Resource: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            `:codebuild:${PIPELINE_ENV.region}:${PIPELINE_ENV.account}:report-group/*`,
          ]],
        },
      },
      {
        Action: ['codebuild:BatchGetBuilds', 'codebuild:StartBuild', 'codebuild:StopBuild'],
        Effect: 'Allow',
        Resource: '*',
      },
      {
        Action: 'sts:AssumeRole',
        Effect: 'Allow',
        Resource: unsingleton(assumeRolePattern.map(arn => { return { 'Fn::Sub': arn }; })),
      },
      {
        Action: ['s3:GetObject*', 's3:GetBucket*', 's3:List*'],
        Effect: 'Allow',
        Resource: [
          { 'Fn::GetAtt': ['CdkPipelineArtifactsBucket7B46C7BF', 'Arn'] },
          { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['CdkPipelineArtifactsBucket7B46C7BF', 'Arn'] }, '/*']] },
        ],
      },
      {
        Action: ['kms:Decrypt', 'kms:DescribeKey'],
        Effect: 'Allow',
        Resource: { 'Fn::GetAtt': ['CdkPipelineArtifactsBucketEncryptionKeyDDD3258C', 'Arn'] },
      }],
    },
    Roles: [{ Ref: attachedRole }],
  };
}

beforeEach(() => {
  app = new TestApp();
  pipelineStack = new Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
});

afterEach(() => {
  app.cleanup();
});

describe('basic pipeline', () => {
  test('no assets stage if the application has no assets', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new PlainStackApp(app, 'App'));

    // THEN
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.not(Match.arrayWith([Match.objectLike({
        Name: 'Assets',
      })])),
    });
  });

  test('assets stage comes before any user-defined stages', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new FileAssetApp(app, 'App'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: [
        Match.objectLike({ Name: 'Source' }),
        Match.objectLike({ Name: 'Build' }),
        Match.objectLike({ Name: 'UpdatePipeline' }),
        Match.objectLike({ Name: 'Assets' }),
        Match.objectLike({ Name: 'App' }),
      ],
    });
  });

  test('up to 50 assets fit in a single stage', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new MegaAssetsApp(app, 'App', { numAssets: 50 }));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: [
        Match.objectLike({ Name: 'Source' }),
        Match.objectLike({ Name: 'Build' }),
        Match.objectLike({ Name: 'UpdatePipeline' }),
        Match.objectLike({ Name: 'Assets' }),
        Match.objectLike({ Name: 'App' }),
      ],
    });
  });

  describe('asset stage placement', () => {
    test('51 assets triggers a second stage', () => {
      // WHEN
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
      pipeline.addStage(new MegaAssetsApp(app, 'App', { numAssets: 51 }));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        Stages: [
          Match.objectLike({ Name: 'Source' }),
          Match.objectLike({ Name: 'Build' }),
          Match.objectLike({ Name: 'UpdatePipeline' }),
          Match.objectLike({ Name: stringLike('Assets*') }),
          Match.objectLike({ Name: stringLike('Assets*2') }),
          Match.objectLike({ Name: 'App' }),
        ],
      });
    },
    );

    test('101 assets triggers a third stage', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
      pipeline.addStage(new MegaAssetsApp(app, 'App', { numAssets: 101 }));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        Stages: [
          Match.objectLike({ Name: 'Source' }),
          Match.objectLike({ Name: 'Build' }),
          Match.objectLike({ Name: 'UpdatePipeline' }),
          Match.objectLike({ Name: stringLike('Assets*') }), // 'Assets' vs 'Assets.1'
          Match.objectLike({ Name: stringLike('Assets*2') }),
          Match.objectLike({ Name: stringLike('Assets*3') }),
          Match.objectLike({ Name: 'App' }),
        ],
      });
    },
    );
  });

  test('command line properly locates assets in subassembly', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new FileAssetApp(app, 'FileAssetApp'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      },
      Source: {
        BuildSpec: Match.serializedJson(Match.objectLike({
          phases: {
            build: {
              commands: Match.arrayWith([`cdk-assets --path "assembly-FileAssetApp/FileAssetAppStackEADD68C5.assets.json" --verbose publish "${FILE_ASSET_SOURCE_HASH}:current_account-current_region-3d50c90e"`]),
            },
          },
        })),
      },
    });
  });

  test('multiple assets are published in parallel', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new MultipleFileAssetsApp(app, 'FileAssetApp', { n: 2 }));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.arrayWith([{
        Name: 'Assets',
        Actions: [
          Match.objectLike({ RunOrder: 1 }),
          Match.objectLike({ RunOrder: 1 }),
        ],
      }]),
    });
  });

  test('file image asset publishers do not use privilegedmode', () => {
    // WHEN
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new FileAssetApp(app, 'FileAssetApp'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        BuildSpec: Match.serializedJson(Match.objectLike({
          phases: {
            build: {
              commands: Match.arrayWith([stringLike('cdk-assets *')]),
            },
          },
        })),
      },
      Environment: Match.objectLike({
        PrivilegedMode: false,
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      }),
    });
  });

  test('docker image asset publishers use privilegedmode', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
    pipeline.addStage(new DockerAssetApp(app, 'DockerAssetApp'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        BuildSpec: Match.serializedJson(Match.objectLike({
          phases: {
            build: {
              commands: Match.arrayWith([stringLike('cdk-assets *')]),
            },
          },
        })),
      },
      Environment: Match.objectLike({
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
        PrivilegedMode: true,
      }),
    });
  });

  test('can control fix/CLI version used in asset publishing', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      cliVersion: '1.2.3',
    });
    pipeline.addStage(new FileAssetApp(pipelineStack, 'FileAssetApp'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      },
      Source: {
        BuildSpec: Match.serializedJson(Match.objectLike({
          phases: {
            install: {
              commands: ['npm install -g aws-cdk@1.2.3'],
            },
          },
        })),
      },
    });
  });

  describe('asset roles and policies', () => {
    test('includes file publishing assets role for apps with file assets', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
        // Expectation expects to see KMS key policy permissions
        crossAccountKeys: true,
      });
      pipeline.addStage(new FileAssetApp(app, 'App1'));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'codebuild.amazonaws.com',
              },
            },
          ],
        },
      });
      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy(FILE_PUBLISHING_ROLE, 'CdkAssetsFileRole6BE17A07'));
    },
    );

    test('publishing assets role may assume roles from multiple environments', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
        // Expectation expects to see KMS key policy permissions
        crossAccountKeys: true,
      });

      pipeline.addStage(new FileAssetApp(app, 'App1'));
      pipeline.addStage(new FileAssetApp(app, 'App2', {
        env: {
          account: '012345678901',
          region: 'eu-west-1',
        },
      }));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy([FILE_PUBLISHING_ROLE, 'arn:${AWS::Partition}:iam::012345678901:role/cdk-hnb659fds-file-publishing-role-012345678901-eu-west-1'],
          'CdkAssetsFileRole6BE17A07'));
    },
    );

    test('publishing assets role de-dupes assumed roles', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
        // Expectation expects to see KMS key policy permissions
        crossAccountKeys: true,
      });
      pipeline.addStage(new FileAssetApp(app, 'App1'));
      pipeline.addStage(new FileAssetApp(app, 'App2'));
      pipeline.addStage(new FileAssetApp(app, 'App3'));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy(FILE_PUBLISHING_ROLE, 'CdkAssetsFileRole6BE17A07'));
    },
    );

    test('includes image publishing assets role for apps with Docker assets', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
        // Expectation expects to see KMS key policy permissions
        crossAccountKeys: true,
      });
      pipeline.addStage(new DockerAssetApp(app, 'App1'));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'codebuild.amazonaws.com',
              },
            },
          ],
        },
      });
      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy(IMAGE_PUBLISHING_ROLE, 'CdkAssetsDockerRole484B6DD3'));
    },
    );

    test('includes both roles for apps with both file and Docker assets', () => {
      const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
        // Expectation expects to see KMS key policy permissions
        crossAccountKeys: true,
      });
      pipeline.addStage(new FileAssetApp(app, 'App1'));
      pipeline.addStage(new DockerAssetApp(app, 'App2'));

      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy(FILE_PUBLISHING_ROLE, 'CdkAssetsFileRole6BE17A07'));
      Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy',
        expectedAssetRolePolicy(IMAGE_PUBLISHING_ROLE, 'CdkAssetsDockerRole484B6DD3'));
    },
    );
  });
});

test('can supply pre-install scripts to asset upload', () => {
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    assetPublishingCodeBuildDefaults: {
      partialBuildSpec: cb.BuildSpec.fromObject({
        version: '0.2',
        phases: {
          install: {
            commands: [
              'npm config set registry https://registry.com',
            ],
          },
        },
      }),
    },
  });
  pipeline.addStage(new FileAssetApp(app, 'FileAssetApp'));

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['npm config set registry https://registry.com', 'npm install -g cdk-assets@latest'],
          },
        },
      })),
    },
  });
});

describe('pipeline with VPC', () => {
  let vpc: ec2.Vpc;
  beforeEach(() => {
    vpc = new ec2.Vpc(pipelineStack, 'Vpc');
  });

  test('asset CodeBuild Project uses VPC subnets', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      codeBuildDefaults: { vpc },
    });
    pipeline.addStage(new DockerAssetApp(app, 'DockerAssetApp'));

    // THEN
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      VpcConfig: Match.objectLike({
        SecurityGroupIds: [
          { 'Fn::GetAtt': ['CdkAssetsDockerAsset1SecurityGroup078F5C66', 'GroupId'] },
        ],
        Subnets: [
          { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
          { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
          { Ref: 'VpcPrivateSubnet3SubnetF258B56E' },
        ],
        VpcId: { Ref: 'Vpc8378EB38' },
      }),
    });
  });

  test('Pipeline-generated CodeBuild Projects have appropriate execution role permissions', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      codeBuildDefaults: { vpc },
    });
    pipeline.addStage(new DockerAssetApp(app, 'DockerAssetApp'));
    // Assets Project
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
      Roles: [
        { Ref: 'CdkAssetsDockerRole484B6DD3' },
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

  test('Asset publishing CodeBuild Projects have correct VPC permissions', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      codeBuildDefaults: { vpc },
    });
    pipeline.addStage(new DockerAssetApp(app, 'DockerAssetApp'));
    // Assets Project
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          Match.objectLike({
            Resource: '*',
            Action: [
              'ec2:CreateNetworkInterface',
              'ec2:DescribeNetworkInterfaces',
              'ec2:DeleteNetworkInterface',
              'ec2:DescribeSubnets',
              'ec2:DescribeSecurityGroups',
              'ec2:DescribeDhcpOptions',
              'ec2:DescribeVpcs',
            ],
          }),
        ],
      },
      Roles: [{ Ref: 'CdkAssetsDockerRole484B6DD3' }],
    });
    Template.fromStack(pipelineStack).hasResource('AWS::CodeBuild::Project', {
      Properties: {
        ServiceRole: { 'Fn::GetAtt': ['CdkAssetsDockerRole484B6DD3', 'Arn'] },
      },
      DependsOn: [
        'CdkAssetsDockerAsset1PolicyDocument8DA96A22',
      ],
    });
  });
});

test('adding environment variable to assets job adds SecretsManager permissions', () => {
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Pipeline', {
    assetPublishingCodeBuildDefaults: {
      buildEnvironment: {
        environmentVariables: {
          FOOBAR: {
            value: 'FoobarSecret',
            type: cb.BuildEnvironmentVariableType.SECRETS_MANAGER,
          },
        },
      },
    },
  });
  pipeline.addStage(new FileAssetApp(pipelineStack, 'MyApp'));

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'secretsmanager:GetSecretValue',
          Effect: 'Allow',
          Resource: {
            'Fn::Join': ['', [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':secretsmanager:us-pipeline:123456789012:secret:FoobarSecret-??????',
            ]],
          },
        }),
      ]),
    },
  });
});

describe('pipeline with single asset publisher', () => {
  test('other pipeline writes to separate assets build spec file', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk-1', {
      publishAssetsInParallel: false,
    });
    pipeline.addStage(new MultipleFileAssetsApp(app, 'FileAssetApp', { n: 2 }));

    const pipelineStack2 = new Stack(app, 'PipelineStack2', { env: PIPELINE_ENV });
    const otherPipeline = new ModernTestGitHubNpmPipeline(pipelineStack2, 'Cdk-2', {
      publishAssetsInParallel: false,
    });
    otherPipeline.addStage(new MultipleFileAssetsApp(app, 'OtherFileAssetApp', { n: 2 }));
    // THEN
    const buildSpecName1 = new Capture(stringLike('buildspec-*.yaml'));
    const buildSpecName2 = new Capture(stringLike('buildspec-*.yaml'));
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        BuildSpec: buildSpecName1,
      },
    });
    Template.fromStack(pipelineStack2).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        BuildSpec: buildSpecName2,
      },
    });

    expect(buildSpecName1.asString()).not.toEqual(buildSpecName2.asString());
  });

  test('necessary secrets manager permissions get added to asset roles', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Pipeline', {
      assetPublishingCodeBuildDefaults: {
        buildEnvironment: {
          environmentVariables: {
            FOOBAR: {
              value: 'FoobarSecret',
              type: cb.BuildEnvironmentVariableType.SECRETS_MANAGER,
            },
          },
        },
      },
    });
    pipeline.addStage(new FileAssetApp(pipelineStack, 'MyApp'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([{
          Action: 'secretsmanager:GetSecretValue',
          Effect: 'Allow',
          Resource: {
            'Fn::Join': [
              '',
              [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':secretsmanager:us-pipeline:123456789012:secret:FoobarSecret-??????',
              ],
            ],
          },
        }]),
      },
      Roles: [
        { Ref: 'PipelineAssetsFileRole59943A77' },
      ],
    });
  });

  test('multiple assets are using the same job in singlePublisherMode', () => {
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      publishAssetsInParallel: false,
    });
    pipeline.addStage(new MultipleFileAssetsApp(app, 'FileAssetApp', { n: 2 }));

    // THEN
    const buildSpecName = new Capture(stringLike('buildspec-*.yaml'));
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.arrayWith([{
        Name: 'Assets',
        Actions: [
          // Only one file asset action
          Match.objectLike({ RunOrder: 1, Name: 'FileAssets' }),
        ],
      }]),
    });
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      },
      Source: {
        BuildSpec: buildSpecName,
      },
    });
    const assembly = synthesize(pipelineStack);

    const actualFileName = buildSpecName.asString();

    const buildSpec = JSON.parse(fs.readFileSync(path.join(assembly.directory, actualFileName), { encoding: 'utf-8' }));
    expect(buildSpec.phases.build.commands).toContain(`cdk-assets --path "assembly-FileAssetApp/FileAssetAppStackEADD68C5.assets.json" --verbose publish "${FILE_ASSET_SOURCE_HASH}:current_account-current_region-3d50c90e"`);
    expect(buildSpec.phases.build.commands).toContain(`cdk-assets --path "assembly-FileAssetApp/FileAssetAppStackEADD68C5.assets.json" --verbose publish "${FILE_ASSET_SOURCE_HASH2}:current_account-current_region-8b0ef02d"`);
  });
});

describe('pipeline with custom asset publisher BuildSpec', () => {
  test('custom buildspec is merged correctly', () => {
    // WHEN
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      publishAssetsInParallel: false,
      assetPublishingCodeBuildDefaults: {
        partialBuildSpec: cb.BuildSpec.fromObject({
          phases: {
            pre_install: {
              commands: 'preinstall',
            },
          },
          cache: {
            paths: 'node_modules',
          },
        }),
      },
    });
    pipeline.addStage(new MultipleFileAssetsApp(app, 'FileAssetApp', { n: 2 }));

    const buildSpecName = new Capture(stringLike('buildspec-*'));

    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.arrayWith([{
        Name: 'Assets',
        Actions: [
          // Only one file asset action
          Match.objectLike({ RunOrder: 1, Name: 'FileAssets' }),
        ],
      }]),
    });
    Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      },
      Source: {
        BuildSpec: buildSpecName,
      },
    });

    const assembly = synthesize(pipelineStack);
    const buildSpec = JSON.parse(fs.readFileSync(path.join(assembly.directory, buildSpecName.asString())).toString());
    expect(buildSpec.phases.build.commands).toContain(`cdk-assets --path "assembly-FileAssetApp/FileAssetAppStackEADD68C5.assets.json" --verbose publish "${FILE_ASSET_SOURCE_HASH}:current_account-current_region-3d50c90e"`);
    expect(buildSpec.phases.build.commands).toContain(`cdk-assets --path "assembly-FileAssetApp/FileAssetAppStackEADD68C5.assets.json" --verbose publish "${FILE_ASSET_SOURCE_HASH2}:current_account-current_region-8b0ef02d"`);
    expect(buildSpec.phases.pre_install.commands).toContain('preinstall');
    expect(buildSpec.cache.paths).toContain('node_modules');
  });
});

function synthesize(stack: Stack) {
  const root = Stage.of(stack);
  if (!Stage.isStage(root)) {
    throw new Error('unexpected: all stacks must be part of a Stage');
  }

  return root.synth({ skipValidation: true });
}

function unsingleton<A>(xs: A[]): A | A[] {
  if (xs.length === 1) {
    return xs[0];
  }
  return xs;
}
