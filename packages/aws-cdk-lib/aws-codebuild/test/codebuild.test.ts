import { Match, Template } from '../../assertions';
import * as codecommit from '../../aws-codecommit';
import * as ec2 from '../../aws-ec2';
import * as kms from '../../aws-kms';
import * as s3 from '../../aws-s3';
import * as cdk from '../../core';
import * as codebuild from '../lib';
import { CodePipelineSource } from '../lib/codepipeline-source';
import { NoSource } from '../lib/no-source';

/* eslint-disable @stylistic/quote-props */

describe('default properties', () => {
  test('with CodePipeline source', () => {
    const stack = new cdk.Stack();

    new codebuild.PipelineProject(stack, 'MyProject');

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProjectRole9BBE5233': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'codebuild.amazonaws.com',
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'MyProjectRoleDefaultPolicyB19B7C29': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    'logs:CreateLogGroup',
                    'logs:CreateLogStream',
                    'logs:PutLogEvents',
                  ],
                  'Effect': 'Allow',
                  'Resource': [
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                        ],
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                          ':*',
                        ],
                      ],
                    },
                  ],
                },
                {
                  'Action': [
                    'codebuild:CreateReportGroup',
                    'codebuild:CreateReport',
                    'codebuild:UpdateReport',
                    'codebuild:BatchPutTestCases',
                    'codebuild:BatchPutCodeCoverages',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::Join': ['', [
                      'arn:',
                      { 'Ref': 'AWS::Partition' },
                      ':codebuild:',
                      { 'Ref': 'AWS::Region' },
                      ':',
                      { 'Ref': 'AWS::AccountId' },
                      ':report-group/',
                      { 'Ref': 'MyProject39F7B0AE' },
                      '-*',
                    ]],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'MyProjectRoleDefaultPolicyB19B7C29',
            'Roles': [
              {
                'Ref': 'MyProjectRole9BBE5233',
              },
            ],
          },
        },
        'MyProject39F7B0AE': {
          'Type': 'AWS::CodeBuild::Project',
          'Properties': {
            'Source': {
              'Type': 'CODEPIPELINE',
            },
            'Artifacts': {
              'Type': 'CODEPIPELINE',
            },
            'ServiceRole': {
              'Fn::GetAtt': [
                'MyProjectRole9BBE5233',
                'Arn',
              ],
            },
            'Environment': {
              'Type': 'LINUX_CONTAINER',
              'PrivilegedMode': false,
              'Image': 'aws/codebuild/standard:7.0',
              'ImagePullCredentialsType': 'CODEBUILD',
              'ComputeType': 'BUILD_GENERAL1_SMALL',
            },
            'EncryptionKey': 'alias/aws/s3',
            'Cache': {
              'Type': 'NO_CACHE',
            },
          },
        },
      },
    });
  });

  test('with CodeCommit source', () => {
    const stack = new cdk.Stack();

    const repo = new codecommit.Repository(stack, 'MyRepo', {
      repositoryName: 'hello-cdk',
    });

    const source = codebuild.Source.codeCommit({ repository: repo, cloneDepth: 2 });

    new codebuild.Project(stack, 'MyProject', {
      source,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyRepoF4F48043': {
          'Type': 'AWS::CodeCommit::Repository',
          'Properties': {
            'RepositoryName': 'hello-cdk',
          },
        },
        'MyProjectRole9BBE5233': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'codebuild.amazonaws.com',
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'MyProjectRoleDefaultPolicyB19B7C29': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': 'codecommit:GitPull',
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::GetAtt': [
                      'MyRepoF4F48043',
                      'Arn',
                    ],
                  },
                },
                {
                  'Action': [
                    'logs:CreateLogGroup',
                    'logs:CreateLogStream',
                    'logs:PutLogEvents',
                  ],
                  'Effect': 'Allow',
                  'Resource': [
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                        ],
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                          ':*',
                        ],
                      ],
                    },
                  ],
                },
                {
                  'Action': [
                    'codebuild:CreateReportGroup',
                    'codebuild:CreateReport',
                    'codebuild:UpdateReport',
                    'codebuild:BatchPutTestCases',
                    'codebuild:BatchPutCodeCoverages',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::Join': ['', [
                      'arn:',
                      { 'Ref': 'AWS::Partition' },
                      ':codebuild:',
                      { 'Ref': 'AWS::Region' },
                      ':',
                      { 'Ref': 'AWS::AccountId' },
                      ':report-group/',
                      { 'Ref': 'MyProject39F7B0AE' },
                      '-*',
                    ]],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'MyProjectRoleDefaultPolicyB19B7C29',
            'Roles': [
              {
                'Ref': 'MyProjectRole9BBE5233',
              },
            ],
          },
        },
        'MyProject39F7B0AE': {
          'Type': 'AWS::CodeBuild::Project',
          'Properties': {
            'Artifacts': {
              'Type': 'NO_ARTIFACTS',
            },
            'Environment': {
              'ComputeType': 'BUILD_GENERAL1_SMALL',
              'Image': 'aws/codebuild/standard:7.0',
              'ImagePullCredentialsType': 'CODEBUILD',
              'PrivilegedMode': false,
              'Type': 'LINUX_CONTAINER',
            },
            'ServiceRole': {
              'Fn::GetAtt': [
                'MyProjectRole9BBE5233',
                'Arn',
              ],
            },
            'Source': {
              'Location': {
                'Fn::GetAtt': [
                  'MyRepoF4F48043',
                  'CloneUrlHttp',
                ],
              },
              'GitCloneDepth': 2,
              'Type': 'CODECOMMIT',
            },
            'EncryptionKey': 'alias/aws/s3',
            'Cache': {
              'Type': 'NO_CACHE',
            },
          },
        },
      },
    });
  });

  test('with S3Bucket source', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');

    new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path/to/source.zip',
      }),
      environment: {
        buildImage: codebuild.WindowsBuildImage.WINDOWS_BASE_2_0,
      },
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'MyProjectRole9BBE5233': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'codebuild.amazonaws.com',
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'MyProjectRoleDefaultPolicyB19B7C29': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    's3:GetObject*',
                    's3:GetBucket*',
                    's3:List*',
                  ],
                  'Effect': 'Allow',
                  'Resource': [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
                        'Arn',
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          {
                            'Fn::GetAtt': [
                              'MyBucketF68F3FF0',
                              'Arn',
                            ],
                          },
                          '/path/to/source.zip',
                        ],
                      ],
                    },
                  ],
                },
                {
                  'Action': [
                    'logs:CreateLogGroup',
                    'logs:CreateLogStream',
                    'logs:PutLogEvents',
                  ],
                  'Effect': 'Allow',
                  'Resource': [
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                        ],
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':logs:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':log-group:/aws/codebuild/',
                          {
                            'Ref': 'MyProject39F7B0AE',
                          },
                          ':*',
                        ],
                      ],
                    },
                  ],
                },
                {
                  'Action': [
                    'codebuild:CreateReportGroup',
                    'codebuild:CreateReport',
                    'codebuild:UpdateReport',
                    'codebuild:BatchPutTestCases',
                    'codebuild:BatchPutCodeCoverages',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::Join': ['', [
                      'arn:',
                      { 'Ref': 'AWS::Partition' },
                      ':codebuild:',
                      { 'Ref': 'AWS::Region' },
                      ':',
                      { 'Ref': 'AWS::AccountId' },
                      ':report-group/',
                      { 'Ref': 'MyProject39F7B0AE' },
                      '-*',
                    ]],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'MyProjectRoleDefaultPolicyB19B7C29',
            'Roles': [
              {
                'Ref': 'MyProjectRole9BBE5233',
              },
            ],
          },
        },
        'MyProject39F7B0AE': {
          'Type': 'AWS::CodeBuild::Project',
          'Properties': {
            'Artifacts': {
              'Type': 'NO_ARTIFACTS',
            },
            'Environment': {
              'ComputeType': 'BUILD_GENERAL1_MEDIUM',
              'Image': 'aws/codebuild/windows-base:2.0',
              'ImagePullCredentialsType': 'CODEBUILD',
              'PrivilegedMode': false,
              'Type': 'WINDOWS_CONTAINER',
            },
            'ServiceRole': {
              'Fn::GetAtt': [
                'MyProjectRole9BBE5233',
                'Arn',
              ],
            },
            'Source': {
              'Location': {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Ref': 'MyBucketF68F3FF0',
                    },
                    '/path/to/source.zip',
                  ],
                ],
              },
              'Type': 'S3',
            },
            'EncryptionKey': 'alias/aws/s3',
            'Cache': {
              'Type': 'NO_CACHE',
            },
          },
        },
      },
    });
  });

  test('with GitHub source', () => {
    const stack = new cdk.Stack();

    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        cloneDepth: 3,
        fetchSubmodules: true,
        webhook: true,
        reportBuildStatus: false,
        webhookFilters: [
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH).andTagIsNot('stable'),
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.PULL_REQUEST_REOPENED).andBaseBranchIs('main'),
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.RELEASED).andBaseBranchIs('main'),
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.PRERELEASED).andBaseBranchIs('main'),
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.WORKFLOW_JOB_QUEUED).andBaseBranchIs('main'),
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.PULL_REQUEST_CLOSED).andBaseBranchIs('main'),
        ],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'GITHUB',
        Location: 'https://github.com/testowner/testrepo.git',
        ReportBuildStatus: false,
        GitCloneDepth: 3,
        GitSubmodulesConfig: {
          FetchSubmodules: true,
        },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Triggers: {
        Webhook: true,
        FilterGroups: [
          [
            { Type: 'EVENT', Pattern: 'PUSH' },
            { Type: 'HEAD_REF', Pattern: 'refs/tags/stable', ExcludeMatchedPattern: true },
          ],
          [
            { Type: 'EVENT', Pattern: 'PULL_REQUEST_REOPENED' },
            { Type: 'BASE_REF', Pattern: 'refs/heads/main' },
          ],
          [
            { Type: 'EVENT', Pattern: 'RELEASED' },
            { Type: 'BASE_REF', Pattern: 'refs/heads/main' },
          ],
          [
            { Type: 'EVENT', Pattern: 'PRERELEASED' },
            { Type: 'BASE_REF', Pattern: 'refs/heads/main' },
          ],
          [
            { Type: 'EVENT', Pattern: 'WORKFLOW_JOB_QUEUED' },
            { Type: 'BASE_REF', Pattern: 'refs/heads/main' },
          ],
          [
            { Type: 'EVENT', Pattern: 'PULL_REQUEST_CLOSED' },
            { Type: 'BASE_REF', Pattern: 'refs/heads/main' },
          ],
        ],
      },
    });
  });

  test('with GitHubEnterprise source', () => {
    const stack = new cdk.Stack();

    const pushFilterGroup = codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH);
    new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.gitHubEnterprise({
        httpsCloneUrl: 'https://github.testcompany.com/testowner/testrepo',
        ignoreSslErrors: true,
        cloneDepth: 4,
        webhook: true,
        reportBuildStatus: false,
        webhookFilters: [
          pushFilterGroup.andBranchIs('main'),
          pushFilterGroup.andBranchIs('develop'),
          pushFilterGroup.andFilePathIs('ReadMe.md'),
        ],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'GITHUB_ENTERPRISE',
        InsecureSsl: true,
        GitCloneDepth: 4,
        ReportBuildStatus: false,
        Location: 'https://github.testcompany.com/testowner/testrepo',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Triggers: {
        Webhook: true,
        FilterGroups: [
          [
            { Type: 'EVENT', Pattern: 'PUSH' },
            { Type: 'HEAD_REF', Pattern: 'refs/heads/main' },
          ],
          [
            { Type: 'EVENT', Pattern: 'PUSH' },
            { Type: 'HEAD_REF', Pattern: 'refs/heads/develop' },
          ],
          [
            { Type: 'EVENT', Pattern: 'PUSH' },
            { Type: 'FILE_PATH', Pattern: 'ReadMe.md' },
          ],
        ],
      },
    });
  });

  test('with Bitbucket source', () => {
    const stack = new cdk.Stack();

    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.bitBucket({
        owner: 'testowner',
        repo: 'testrepo',
        cloneDepth: 5,
        reportBuildStatus: false,
        webhookFilters: [
          codebuild.FilterGroup.inEventOf(
            codebuild.EventAction.PULL_REQUEST_CREATED,
            codebuild.EventAction.PULL_REQUEST_UPDATED,
            codebuild.EventAction.PULL_REQUEST_MERGED,
            codebuild.EventAction.PULL_REQUEST_CLOSED,
          ).andTagIs('v.*'),
          // duplicate event actions are fine
          codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH, codebuild.EventAction.PUSH).andActorAccountIsNot('aws-cdk-dev'),
        ],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'BITBUCKET',
        Location: 'https://bitbucket.org/testowner/testrepo.git',
        GitCloneDepth: 5,
        ReportBuildStatus: false,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Triggers: {
        Webhook: true,
        FilterGroups: [
          [
            { Type: 'EVENT', Pattern: 'PULL_REQUEST_CREATED, PULL_REQUEST_UPDATED, PULL_REQUEST_MERGED, PULL_REQUEST_CLOSED' },
            { Type: 'HEAD_REF', Pattern: 'refs/tags/v.*' },
          ],
          [
            { Type: 'EVENT', Pattern: 'PUSH' },
            { Type: 'ACTOR_ACCOUNT_ID', Pattern: 'aws-cdk-dev', ExcludeMatchedPattern: true },
          ],
        ],
      },
    });
  });

  test('with webhookTriggersBatchBuild option', () => {
    const stack = new cdk.Stack();

    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        webhook: true,
        webhookTriggersBatchBuild: true,
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Triggers: {
        Webhook: true,
        BuildType: 'BUILD_BATCH',
      },
      BuildBatchConfig: {
        ServiceRole: {
          'Fn::GetAtt': [
            'ProjectBatchServiceRoleF97A1CFB',
            'Arn',
          ],
        },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
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
        Version: '2012-10-17',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'codebuild:StartBuild',
              'codebuild:StopBuild',
              'codebuild:RetryBuild',
            ],
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'ProjectC78D97AD',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('fail creating a Project when webhook false and webhookTriggersBatchBuild option', () => {
    [false, undefined].forEach((webhook) => {
      const stack = new cdk.Stack();

      expect(() => {
        new codebuild.Project(stack, 'Project', {
          source: codebuild.Source.gitHub({
            owner: 'testowner',
            repo: 'testrepo',
            webhook,
            webhookTriggersBatchBuild: true,
          }),
        });
      }).toThrow(/`webhookTriggersBatchBuild` cannot be used when `webhook` is `false`/);
    });
  });

  test('fail creating a Project when no build spec is given', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'MyProject', {
      });
    }).toThrow(/buildSpec/);
  });

  test('with VPC configuration', () => {
    const stack = new cdk.Stack();

    const bucket = new s3.Bucket(stack, 'MyBucket');
    const vpc = new ec2.Vpc(stack, 'MyVPC');
    const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
      securityGroupName: 'Bob',
      vpc,
      allowAllOutbound: true,
      description: 'Example',
    });
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path/to/source.zip',
      }),
      vpc,
      securityGroups: [securityGroup],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'VpcConfig': {
        'SecurityGroupIds': [
          {
            'Fn::GetAtt': [
              'SecurityGroup1F554B36F',
              'GroupId',
            ],
          },
        ],
        'Subnets': [
          {
            'Ref': 'MyVPCPrivateSubnet1Subnet641543F4',
          },
          {
            'Ref': 'MyVPCPrivateSubnet2SubnetA420D3F0',
          },
        ],
        'VpcId': {
          'Ref': 'MyVPCAFB07A31',
        },
      },
    });

    expect(project.connections).toBeDefined();
  });

  test('without VPC configuration but security group identified', () => {
    const stack = new cdk.Stack();

    const bucket = new s3.Bucket(stack, 'MyBucket');
    const vpc = new ec2.Vpc(stack, 'MyVPC');
    const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
      securityGroupName: 'Bob',
      vpc,
      allowAllOutbound: true,
      description: 'Example',
    });

    expect(() =>
      new codebuild.Project(stack, 'MyProject', {
        source: codebuild.Source.s3({
          bucket,
          path: 'path/to/source.zip',
        }),
        securityGroups: [securityGroup],
      }),
    ).toThrow(/Cannot configure 'securityGroup' or 'allowAllOutbound' without configuring a VPC/);
  });

  test('with VPC configuration but allowAllOutbound identified', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const vpc = new ec2.Vpc(stack, 'MyVPC');
    const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
      securityGroupName: 'Bob',
      vpc,
      allowAllOutbound: true,
      description: 'Example',
    });

    expect(() =>
      new codebuild.Project(stack, 'MyProject', {
        source: codebuild.Source.s3({
          bucket,
          path: 'path/to/source.zip',
        }),
        vpc,
        allowAllOutbound: true,
        securityGroups: [securityGroup],
      }),
    ).toThrow(/Configure 'allowAllOutbound' directly on the supplied SecurityGroup/);
  });

  test('without passing a VPC cannot access the connections property', () => {
    const stack = new cdk.Stack();

    const project = new codebuild.PipelineProject(stack, 'MyProject');

    expect(() => project.connections).toThrow(
      /Only VPC-associated Projects have security groups to manage. Supply the "vpc" parameter when creating the Project/);
  });

  test('no KMS Key defaults to default S3 managed key', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.PipelineProject(stack, 'MyProject');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      EncryptionKey: 'alias/aws/s3',
    });
  });

  test('with a KMS Key adds decrypt permissions to the CodeBuild Role', () => {
    const stack = new cdk.Stack();

    const key = new kms.Key(stack, 'MyKey');

    new codebuild.PipelineProject(stack, 'MyProject', {
      encryptionKey: key,
      grantReportGroupPermissions: false,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {}, // CloudWatch logs
          {
            'Action': [
              'kms:Decrypt',
              'kms:Encrypt',
              'kms:ReEncrypt*',
              'kms:GenerateDataKey*',
            ],
            'Effect': 'Allow',
            'Resource': {
              'Fn::GetAtt': [
                'MyKey6AB29FA6',
                'Arn',
              ],
            },
          },
        ],
      },
      'Roles': [
        {
          'Ref': 'MyProjectRole9BBE5233',
        },
      ],
    });
  });
});

test('using timeout and path in S3 artifacts sets it correctly', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Bucket');
  new codebuild.Project(stack, 'Project', {
    buildSpec: codebuild.BuildSpec.fromObject({
      version: '0.2',
    }),
    artifacts: codebuild.Artifacts.s3({
      path: 'some/path',
      name: 'some_name',
      bucket,
    }),
    timeout: cdk.Duration.minutes(123),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    'Artifacts': {
      'Path': 'some/path',
      'Name': 'some_name',
      'Type': 'S3',
    },
    'TimeoutInMinutes': 123,
  });
});

describe('secondary sources', () => {
  test('require providing an identifier when creating a Project', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'MyProject', {
        buildSpec: codebuild.BuildSpec.fromObject({
          version: '0.2',
        }),
        secondarySources: [
          codebuild.Source.s3({
            bucket: new s3.Bucket(stack, 'MyBucket'),
            path: 'path',
          }),
        ],
      });
    }).toThrow(/identifier/);
  });

  test('are not allowed for a Project with CodePipeline as Source', () => {
    const stack = new cdk.Stack();
    const project = new codebuild.PipelineProject(stack, 'MyProject');

    project.addSecondarySource(codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'MyBucket'),
      path: 'some/path',
      identifier: 'id',
    }));

    expect(() => Template.fromStack(stack)).toThrow(/secondary sources/);
  });

  test('added with an identifer after the Project has been created are rendered in the template', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'some/path',
      }),
    });

    project.addSecondarySource(codebuild.Source.s3({
      bucket,
      path: 'another/path',
      identifier: 'source1',
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'SecondarySources': [
        {
          'SourceIdentifier': 'source1',
          'Type': 'S3',
        },
      ],
    });
  });
});

describe('sources with customised build status configuration', () => {
  test('GitHub', () => {
    const context = 'My custom CodeBuild worker!';
    const stack = new cdk.Stack();
    const source = codebuild.Source.gitHub({
      owner: 'awslabs',
      repo: 'aws-cdk',
      buildStatusContext: context,
    });

    new codebuild.Project(stack, 'MyProject', { source });
    Template.fromStack(stack).findParameters('AWS::CodeBuild::Project', {
      Source: {
        buildStatusConfig: {
          context: context,
        },
      },
    });
  });

  test('GitHub Enterprise', () => {
    const context = 'My custom CodeBuild worker!';
    const stack = new cdk.Stack();
    const source = codebuild.Source.gitHubEnterprise({
      httpsCloneUrl: 'url',
      buildStatusContext: context,
    });
    new codebuild.Project(stack, 'MyProject', { source });
    Template.fromStack(stack).findParameters('AWS::CodeBuild::Project', {
      Source: {
        buildStatusConfig: {
          context: context,
        },
      },
    });
  });

  test('BitBucket', () => {
    const context = 'My custom CodeBuild worker!';
    const stack = new cdk.Stack();
    const source = codebuild.Source.bitBucket({ owner: 'awslabs', repo: 'aws-cdk' });
    new codebuild.Project(stack, 'MyProject', { source });
    Template.fromStack(stack).findParameters('AWS::CodeBuild::Project', {
      Source: {
        buildStatusConfig: {
          context: context,
        },
      },
    });
  });
});

describe('sources with customised build status configuration', () => {
  test('GitHub with targetUrl', () => {
    const targetUrl = 'https://example.com';
    const stack = new cdk.Stack();
    const source = codebuild.Source.gitHub({
      owner: 'awslabs',
      repo: 'aws-cdk',
      buildStatusUrl: targetUrl,
    });
    new codebuild.Project(stack, 'MyProject', { source });
    Template.fromStack(stack).findParameters('AWS::CodeBuild::Project', {
      Source: {
        buildStatusConfig: {
          targetUrl: targetUrl,
        },
      },
    });
  });
});

describe('secondary source versions', () => {
  test('allow secondary source versions', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'some/path',
      }),
    });

    project.addSecondarySource(codebuild.Source.s3({
      bucket,
      path: 'another/path',
      identifier: 'source1',
      version: 'someversion',
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'SecondarySources': [
        {
          'SourceIdentifier': 'source1',
          'Type': 'S3',
        },
      ],
      'SecondarySourceVersions': [
        {
          'SourceIdentifier': 'source1',
          'SourceVersion': 'someversion',
        },
      ],
    });
  });

  test('allow not to specify secondary source versions', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'some/path',
      }),
    });

    project.addSecondarySource(codebuild.Source.s3({
      bucket,
      path: 'another/path',
      identifier: 'source1',
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'SecondarySources': [
        {
          'SourceIdentifier': 'source1',
          'Type': 'S3',
        },
      ],
    });
  });
});

describe('fileSystemLocations', () => {
  test('create fileSystemLocation and validate attributes', () => {
    const stack = new cdk.Stack();
    new codebuild.Project(stack, 'MyProject', {
      buildSpec: codebuild.BuildSpec.fromObject({
        version: '0.2',
      }),
      fileSystemLocations: [codebuild.FileSystemLocation.efs({
        identifier: 'myidentifier2',
        location: 'myclodation.mydnsroot.com:/loc',
        mountPoint: '/media',
        mountOptions: 'opts',
      })],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'FileSystemLocations': [
        {
          'Identifier': 'myidentifier2',
          'MountPoint': '/media',
          'MountOptions': 'opts',
          'Location': 'myclodation.mydnsroot.com:/loc',
          'Type': 'EFS',
        },
      ],
    });
  });

  test('Multiple fileSystemLocation created', () => {
    const stack = new cdk.Stack();
    const project = new codebuild.Project(stack, 'MyProject', {
      buildSpec: codebuild.BuildSpec.fromObject({
        version: '0.2',
      }),
    });
    project.addFileSystemLocation(codebuild.FileSystemLocation.efs({
      identifier: 'myidentifier3',
      location: 'myclodation.mydnsroot.com:/loc',
      mountPoint: '/media',
      mountOptions: 'opts',
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'FileSystemLocations': [
        {
          'Identifier': 'myidentifier3',
          'MountPoint': '/media',
          'MountOptions': 'opts',
          'Location': 'myclodation.mydnsroot.com:/loc',
          'Type': 'EFS',
        },
      ],
    });
  });
});

describe('secondary artifacts', () => {
  test('require providing an identifier when creating a Project', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'MyProject', {
        buildSpec: codebuild.BuildSpec.fromObject({
          version: '0.2',
        }),
        secondaryArtifacts: [
          codebuild.Artifacts.s3({
            bucket: new s3.Bucket(stack, 'MyBucket'),
            path: 'some/path',
            name: 'name',
          }),
        ],
      });
    }).toThrow(/identifier/);
  });

  test('are not allowed for a Project with CodePipeline as Source', () => {
    const stack = new cdk.Stack();
    const project = new codebuild.PipelineProject(stack, 'MyProject');

    project.addSecondaryArtifact(codebuild.Artifacts.s3({
      bucket: new s3.Bucket(stack, 'MyBucket'),
      path: 'some/path',
      name: 'name',
      identifier: 'id',
    }));

    expect(() => Template.fromStack(stack)).toThrow(/secondary artifacts/);
  });

  test('added with an identifier after the Project has been created are rendered in the template', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'some/path',
      }),
    });

    project.addSecondaryArtifact(codebuild.Artifacts.s3({
      bucket,
      path: 'another/path',
      name: 'name',
      identifier: 'artifact1',
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'SecondaryArtifacts': [
        {
          'ArtifactIdentifier': 'artifact1',
          'Type': 'S3',
        },
      ],
    });
  });

  test('disabledEncryption is set', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const project = new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket,
        path: 'some/path',
      }),
    });

    project.addSecondaryArtifact(codebuild.Artifacts.s3({
      bucket,
      path: 'another/path',
      name: 'name',
      identifier: 'artifact1',
      encryption: false,
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      'SecondaryArtifacts': [
        {
          'ArtifactIdentifier': 'artifact1',
          'EncryptionDisabled': true,
        },
      ],
    });
  });
});

describe('artifacts', () => {
  describe('CodePipeline', () => {
    test('both source and artifacs are set to CodePipeline', () => {
      const stack = new cdk.Stack();

      new codebuild.PipelineProject(stack, 'MyProject');

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Source': {
          'Type': 'CODEPIPELINE',
        },
        'Artifacts': {
          'Type': 'CODEPIPELINE',
        },
        'ServiceRole': {
          'Fn::GetAtt': [
            'MyProjectRole9BBE5233',
            'Arn',
          ],
        },
        'Environment': {
          'Type': 'LINUX_CONTAINER',
          'PrivilegedMode': false,
          'Image': 'aws/codebuild/standard:7.0',
          'ImagePullCredentialsType': 'CODEBUILD',
          'ComputeType': 'BUILD_GENERAL1_SMALL',
        },
      });
    });
  });

  describe('S3', () => {
    test('name is not set so use buildspec', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      new codebuild.Project(stack, 'MyProject', {
        source: codebuild.Source.s3({
          bucket,
          path: 'some/path',
        }),
        artifacts: codebuild.Artifacts.s3({
          bucket,
          path: 'another/path',
          identifier: 'artifact1',
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Artifacts': {
          'Name': Match.absent(),
          'ArtifactIdentifier': 'artifact1',
          'OverrideArtifactName': true,
        },
      });
    });

    test('name is set so use it', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      new codebuild.Project(stack, 'MyProject', {
        source: codebuild.Source.s3({
          bucket,
          path: 'some/path',
        }),
        artifacts: codebuild.Artifacts.s3({
          bucket,
          path: 'another/path',
          name: 'specificname',
          identifier: 'artifact1',
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Artifacts': {
          'ArtifactIdentifier': 'artifact1',
          'Name': 'specificname',
          'OverrideArtifactName': Match.absent(),
        },
      });
    });
  });
});

test('events', () => {
  const stack = new cdk.Stack();
  const project = new codebuild.Project(stack, 'MyProject', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'MyBucket'),
      path: 'path',
    }),
  });

  project.onBuildFailed('OnBuildFailed', { target: { bind: () => ({ arn: 'ARN', id: 'ID' }) } });
  project.onBuildSucceeded('OnBuildSucceeded', { target: { bind: () => ({ arn: 'ARN', id: 'ID' }) } });
  project.onPhaseChange('OnPhaseChange', { target: { bind: () => ({ arn: 'ARN', id: 'ID' }) } });
  project.onStateChange('OnStateChange', { target: { bind: () => ({ arn: 'ARN', id: 'ID' }) } });
  project.onBuildStarted('OnBuildStarted', { target: { bind: () => ({ arn: 'ARN', id: 'ID' }) } });

  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    'EventPattern': {
      'source': [
        'aws.codebuild',
      ],
      'detail-type': [
        'CodeBuild Build State Change',
      ],
      'detail': {
        'project-name': [
          {
            'Ref': 'MyProject39F7B0AE',
          },
        ],
        'build-status': [
          'FAILED',
        ],
      },
    },
    'State': 'ENABLED',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    'EventPattern': {
      'source': [
        'aws.codebuild',
      ],
      'detail-type': [
        'CodeBuild Build State Change',
      ],
      'detail': {
        'project-name': [
          {
            'Ref': 'MyProject39F7B0AE',
          },
        ],
        'build-status': [
          'SUCCEEDED',
        ],
      },
    },
    'State': 'ENABLED',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    'EventPattern': {
      'source': [
        'aws.codebuild',
      ],
      'detail-type': [
        'CodeBuild Build Phase Change',
      ],
      'detail': {
        'project-name': [
          {
            'Ref': 'MyProject39F7B0AE',
          },
        ],
      },
    },
    'State': 'ENABLED',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    'EventPattern': {
      'source': [
        'aws.codebuild',
      ],
      'detail-type': [
        'CodeBuild Build State Change',
      ],
      'detail': {
        'project-name': [
          {
            'Ref': 'MyProject39F7B0AE',
          },
        ],
      },
    },
    'State': 'ENABLED',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    'EventPattern': {
      'source': [
        'aws.codebuild',
      ],
      'detail-type': [
        'CodeBuild Build State Change',
      ],
      'detail': {
        'project-name': [
          {
            'Ref': 'MyProject39F7B0AE',
          },
        ],
        'build-status': [
          'IN_PROGRESS',
        ],
      },
    },
    'State': 'ENABLED',
  });
});

test('environment variables can be overridden at the project level', () => {
  const stack = new cdk.Stack();

  new codebuild.PipelineProject(stack, 'Project', {
    environment: {
      environmentVariables: {
        FOO: { value: '1234' },
        BAR: { value: `111${cdk.Token.asString({ twotwotwo: '222' })}`, type: codebuild.BuildEnvironmentVariableType.PARAMETER_STORE },
      },
    },
    environmentVariables: {
      GOO: { value: 'ABC' },
      FOO: { value: 'OVERRIDE!' },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    'Source': {
      'Type': 'CODEPIPELINE',
    },
    'Artifacts': {
      'Type': 'CODEPIPELINE',
    },
    'ServiceRole': {
      'Fn::GetAtt': [
        'ProjectRole4CCB274E',
        'Arn',
      ],
    },
    'Environment': {
      'Type': 'LINUX_CONTAINER',
      'EnvironmentVariables': [
        {
          'Type': 'PLAINTEXT',
          'Value': 'OVERRIDE!',
          'Name': 'FOO',
        },
        {
          'Type': 'PARAMETER_STORE',
          'Value': {
            'Fn::Join': [
              '',
              [
                '111',
                { twotwotwo: '222' },
              ],
            ],
          },
          'Name': 'BAR',
        },
        {
          'Type': 'PLAINTEXT',
          'Value': 'ABC',
          'Name': 'GOO',
        },
      ],
      'PrivilegedMode': false,
      'Image': 'aws/codebuild/standard:7.0',
      'ImagePullCredentialsType': 'CODEBUILD',
      'ComputeType': 'BUILD_GENERAL1_SMALL',
    },
  });
});

test('.metricXxx() methods can be used to obtain Metrics for CodeBuild projects', () => {
  const stack = new cdk.Stack();

  const project = new codebuild.Project(stack, 'MyBuildProject', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'MyBucket'),
      path: 'path',
    }),
  });

  const metricBuilds = project.metricBuilds();
  expect(metricBuilds.dimensions!.ProjectName).toEqual(project.projectName);
  expect(metricBuilds.namespace).toEqual('AWS/CodeBuild');
  expect(metricBuilds.statistic).toEqual('Sum');
  expect(metricBuilds.metricName).toEqual('Builds');

  const metricDuration = project.metricDuration({ label: 'hello' });

  expect(metricDuration.metricName).toEqual('Duration');
  expect(metricDuration.label).toEqual('hello');

  expect(project.metricFailedBuilds().metricName).toEqual('FailedBuilds');
  expect(project.metricSucceededBuilds().metricName).toEqual('SucceededBuilds');
});

test('using ComputeType.Small with a Windows image fails validation', () => {
  const stack = new cdk.Stack();
  const invalidEnvironment: codebuild.BuildEnvironment = {
    buildImage: codebuild.WindowsBuildImage.WINDOWS_BASE_2_0,
    computeType: codebuild.ComputeType.SMALL,
  };

  expect(() => {
    new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'MyBucket'),
        path: 'path',
      }),
      environment: invalidEnvironment,
    });
  }).toThrow(/Windows images do not support the 'BUILD_GENERAL1_SMALL' compute type/);
});

test('using ComputeType.XLarge with a Windows image fails validation', () => {
  const stack = new cdk.Stack();
  const invalidEnvironment: codebuild.BuildEnvironment = {
    buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE,
    computeType: codebuild.ComputeType.X_LARGE,
  };

  expect(() => {
    new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'MyBucket'),
        path: 'path',
      }),
      environment: invalidEnvironment,
    });
  }).toThrow(/Windows images do not support the 'BUILD_GENERAL1_XLARGE' compute type/);
});

test('using ComputeType.X2Large with a Windows image fails validation', () => {
  const stack = new cdk.Stack();
  const invalidEnvironment: codebuild.BuildEnvironment = {
    buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE,
    computeType: codebuild.ComputeType.X2_LARGE,
  };

  expect(() => {
    new codebuild.Project(stack, 'MyProject', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'MyBucket'),
        path: 'path',
      }),
      environment: invalidEnvironment,
    });
  }).toThrow(/Windows images do not support the 'BUILD_GENERAL1_2XLARGE' compute type/);
});

test('fromCodebuildImage', () => {
  const stack = new cdk.Stack();
  new codebuild.PipelineProject(stack, 'Project', {
    environment: {
      buildImage: codebuild.LinuxBuildImage.fromCodeBuildImageId('aws/codebuild/standard:4.0'),
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    'Environment': {
      'Image': 'aws/codebuild/standard:4.0',
    },
  });
});

describe('Windows2019 image', () => {
  describe('WIN_SERVER_CORE_2019_BASE', () => {
    test('has type WINDOWS_SERVER_2019_CONTAINER and default ComputeType MEDIUM', () => {
      const stack = new cdk.Stack();
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'WINDOWS_SERVER_2019_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_MEDIUM',
        },
      });
    });

    test('cannot be used in conjunction with ComputeType LAMBDA_1GB', () => {
      const stack = new cdk.Stack();

      expect(() => {
        new codebuild.PipelineProject(stack, 'Project', {
          environment: {
            buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE,
            computeType: codebuild.ComputeType.LAMBDA_1GB,
          },
        });
      }).toThrow(/Invalid CodeBuild environment: Windows images do not support Lambda compute types/);
    });

    test('cannot be used in conjunction with privileged property', () => {
      const stack = new cdk.Stack();

      expect(() => {
        new codebuild.PipelineProject(stack, 'Project', {
          environment: {
            buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE,
            privileged: true,
          },
        });
      }).toThrow(/Invalid CodeBuild environment: Windows images do not support privileged mode/);
    });
  });
});

describe('Linux x86-64 Image', () => {
  test('cannot be used in conjunction with ComputeType LAMBDA_1GB', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
          computeType: codebuild.ComputeType.LAMBDA_1GB,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: x86-64 images do not support Lambda compute type/);
  });
});

describe('ARM image', () => {
  describe('AMAZON_LINUX_2_ARM_3', () => {
    test('has type ARM_CONTAINER and default ComputeType LARGE', () => {
      const stack = new cdk.Stack();
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_LARGE',
        },
      });
    });

    test('can be used with ComputeType SMALL', () => {
      const stack = new cdk.Stack();
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          computeType: codebuild.ComputeType.SMALL,
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_SMALL',
        },
      });
    });

    test('can be used with ComputeType MEDIUM', () => {
      const stack = new cdk.Stack();

      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
          computeType: codebuild.ComputeType.MEDIUM,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_MEDIUM',
        },
      });
    });

    test('can be used with ComputeType LARGE', () => {
      const stack = new cdk.Stack();
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          computeType: codebuild.ComputeType.LARGE,
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_LARGE',
        },
      });
    });

    test('can be used with ComputeType X_LARGE', () => {
      const stack = new cdk.Stack();
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          computeType: codebuild.ComputeType.X_LARGE,
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_XLARGE',
        },
      });
    });

    test('can be used with ComputeType X2_LARGE', () => {
      const stack = new cdk.Stack();

      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
          computeType: codebuild.ComputeType.X2_LARGE,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'Type': 'ARM_CONTAINER',
          'ComputeType': 'BUILD_GENERAL1_2XLARGE',
        },
      });
    });

    test('cannot be used in conjunction with ComputeType LAMBDA_1GB', () => {
      const stack = new cdk.Stack();

      expect(() => {
        new codebuild.PipelineProject(stack, 'Project', {
          environment: {
            buildImage: codebuild.LinuxBuildImage.AMAZON_LINUX_2_ARM_3,
            computeType: codebuild.ComputeType.LAMBDA_1GB,
          },
        });
      }).toThrow('Invalid CodeBuild environment: ARM images do not support Lambda ComputeTypes, got BUILD_LAMBDA_1GB');
    });
  });
});

test('badge support test', () => {
  const stack = new cdk.Stack();

  interface BadgeValidationTestCase {
    source: codebuild.Source;
    allowsBadge: boolean;
  }

  const repo = new codecommit.Repository(stack, 'MyRepo', {
    repositoryName: 'hello-cdk',
  });
  const bucket = new s3.Bucket(stack, 'MyBucket');

  const cases: BadgeValidationTestCase[] = [
    { source: new NoSource(), allowsBadge: false },
    { source: new CodePipelineSource(), allowsBadge: false },
    { source: codebuild.Source.codeCommit({ repository: repo }), allowsBadge: true },
    { source: codebuild.Source.s3({ bucket, path: 'path/to/source.zip' }), allowsBadge: false },
    { source: codebuild.Source.gitHub({ owner: 'awslabs', repo: 'aws-cdk' }), allowsBadge: true },
    { source: codebuild.Source.gitHubEnterprise({ httpsCloneUrl: 'url' }), allowsBadge: true },
    { source: codebuild.Source.bitBucket({ owner: 'awslabs', repo: 'aws-cdk' }), allowsBadge: true },
  ];

  cases.forEach(testCase => {
    const source = testCase.source;
    const validationBlock = () => { new codebuild.Project(stack, `MyProject-${source.type}`, { source, badge: true }); };
    if (testCase.allowsBadge) {
      expect(validationBlock).not.toThrow();
    } else {
      expect(validationBlock).toThrow(/Badge is not supported for source type /);
    }
  });
});

describe('webhook Filters', () => {
  test('a Group cannot be created with an empty set of event actions', () => {
    expect(() => {
      codebuild.FilterGroup.inEventOf();
    }).toThrow(/A filter group must contain at least one event action/);
  });

  test('cannot have base ref conditions if the Group contains the PUSH action', () => {
    const filterGroup = codebuild.FilterGroup.inEventOf(codebuild.EventAction.PULL_REQUEST_CREATED,
      codebuild.EventAction.PUSH);

    expect(() => {
      filterGroup.andBaseRefIs('.*');
    }).toThrow(/A base reference condition cannot be added if a Group contains a PUSH event action/);
  });

  test('cannot be used when webhook is false', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.bitBucket({
          owner: 'owner',
          repo: 'repo',
          webhook: false,
          webhookFilters: [
            codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH),
          ],
        }),
      });
    }).toThrow(/`webhookFilters` cannot be used when `webhook` is `false`/);
  });

  test('can have FILE_PATH filters if the Group contains PUSH and PR_CREATED events', () => {
    codebuild.FilterGroup.inEventOf(
      codebuild.EventAction.PULL_REQUEST_CREATED,
      codebuild.EventAction.PUSH)
      .andFilePathIsNot('.*\\.java');
  });

  test('BitBucket sources do not support the PULL_REQUEST_REOPENED event action', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.bitBucket({
          owner: 'owner',
          repo: 'repo',
          webhookFilters: [
            codebuild.FilterGroup.inEventOf(codebuild.EventAction.PULL_REQUEST_REOPENED),
          ],
        }),
      });
    }).toThrow(/BitBucket sources do not support the PULL_REQUEST_REOPENED webhook event action/);
  });

  test('BitBucket sources support file path conditions', () => {
    const stack = new cdk.Stack();
    const filterGroup = codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH).andFilePathIs('.*');

    expect(() => {
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.bitBucket({
          owner: 'owner',
          repo: 'repo',
          webhookFilters: [filterGroup],
        }),
      });
    }).not.toThrow();
  });

  test('GitHub Enterprise Server sources do not support FILE_PATH filters on PR events', () => {
    const stack = new cdk.Stack();
    const pullFilterGroup = codebuild.FilterGroup.inEventOf(
      codebuild.EventAction.PULL_REQUEST_CREATED,
      codebuild.EventAction.PULL_REQUEST_MERGED,
      codebuild.EventAction.PULL_REQUEST_REOPENED,
      codebuild.EventAction.PULL_REQUEST_UPDATED,
    );

    expect(() => {
      new codebuild.Project(stack, 'MyFilePathProject', {
        source: codebuild.Source.gitHubEnterprise({
          httpsCloneUrl: 'https://github.testcompany.com/testowner/testrepo',
          webhookFilters: [
            pullFilterGroup.andFilePathIs('ReadMe.md'),
          ],
        }),
      });
    }).toThrow(/FILE_PATH filters cannot be used with GitHub Enterprise Server pull request events/);
  });

  describe('COMMIT_MESSAGE Filter', () => {
    test('GitHub Enterprise Server sources do not support COMMIT_MESSAGE filters on PR events', () => {
      const stack = new cdk.Stack();
      const pullFilterGroup = codebuild.FilterGroup.inEventOf(
        codebuild.EventAction.PULL_REQUEST_CREATED,
        codebuild.EventAction.PULL_REQUEST_MERGED,
        codebuild.EventAction.PULL_REQUEST_REOPENED,
        codebuild.EventAction.PULL_REQUEST_UPDATED,
      );

      expect(() => {
        new codebuild.Project(stack, 'MyProject', {
          source: codebuild.Source.gitHubEnterprise({
            httpsCloneUrl: 'https://github.testcompany.com/testowner/testrepo',
            webhookFilters: [
              pullFilterGroup.andCommitMessageIs('the commit message'),
            ],
          }),
        });
      }).toThrow(/COMMIT_MESSAGE filters cannot be used with GitHub Enterprise Server pull request events/);
    });

    test('GitHub Enterprise Server sources support COMMIT_MESSAGE filters on PUSH events', () => {
      const stack = new cdk.Stack();
      const pushFilterGroup = codebuild.FilterGroup.inEventOf(codebuild.EventAction.PUSH);

      expect(() => {
        new codebuild.Project(stack, 'MyProject', {
          source: codebuild.Source.gitHubEnterprise({
            httpsCloneUrl: 'https://github.testcompany.com/testowner/testrepo',
            webhookFilters: [
              pushFilterGroup.andCommitMessageIs('the commit message'),
            ],
          }),
        });
      }).not.toThrow();
    });

    test('BitBucket and GitHub sources support a COMMIT_MESSAGE filter', () => {
      const stack = new cdk.Stack();
      const filterGroup = codebuild
        .FilterGroup
        .inEventOf(codebuild.EventAction.PUSH, codebuild.EventAction.PULL_REQUEST_CREATED)
        .andCommitMessageIs('the commit message');

      expect(() => {
        new codebuild.Project(stack, 'BitBucket Project', {
          source: codebuild.Source.bitBucket({
            owner: 'owner',
            repo: 'repo',
            webhookFilters: [filterGroup],
          }),
        });
        new codebuild.Project(stack, 'GitHub Project', {
          source: codebuild.Source.gitHub({
            owner: 'owner',
            repo: 'repo',
            webhookFilters: [filterGroup],
          }),
        });
      }).not.toThrow();
    });
  });
});

test('enableBatchBuilds()', () => {
  const stack = new cdk.Stack();

  const project = new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHub({
      owner: 'testowner',
      repo: 'testrepo',
    }),
  });

  const returnVal = project.enableBatchBuilds();
  if (!returnVal?.role) {
    throw new Error('Expecting return value with role');
  }

  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    BuildBatchConfig: {
      ServiceRole: {
        'Fn::GetAtt': [
          'ProjectBatchServiceRoleF97A1CFB',
          'Arn',
        ],
      },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
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
      Version: '2012-10-17',
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: [
            'codebuild:StartBuild',
            'codebuild:StopBuild',
            'codebuild:RetryBuild',
          ],
          Effect: 'Allow',
          Resource: {
            'Fn::GetAtt': [
              'ProjectC78D97AD',
              'Arn',
            ],
          },
        },
      ],
      Version: '2012-10-17',
    },
  });
});
