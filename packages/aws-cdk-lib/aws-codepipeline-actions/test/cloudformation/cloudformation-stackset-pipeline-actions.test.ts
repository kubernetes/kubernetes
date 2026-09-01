import { TestFixture } from './test-fixture';
import { Match, Template } from '../../../assertions';
import * as iam from '../../../aws-iam';
import * as cdk from '../../../core';
import * as cpactions from '../../lib';
/* eslint-disable @stylistic/quote-props */

let stack: TestFixture;
let importedAdminRole: iam.IRole;
beforeEach(() => {
  stack = new TestFixture({
    env: {
      account: '111111111111',
      region: 'us-east-1',
    },
  });
  importedAdminRole = iam.Role.fromRoleArn(stack, 'ChangeSetRole', 'arn:aws:iam::123456789012:role/ImportedAdminRole');
});

describe('StackSetAction', () => {
  function defaultOpts() {
    return {
      actionName: 'StackSetUpdate',
      description: 'desc',
      stackSetName: 'MyStack',
      cfnCapabilities: [cdk.CfnCapabilities.NAMED_IAM],
      failureTolerancePercentage: 50,
      maxAccountConcurrencyPercentage: 25,
      template: cpactions.StackSetTemplate.fromArtifactPath(stack.sourceOutput.atPath('template.yaml')),
      parameters: cpactions.StackSetParameters.fromArtifactPath(stack.sourceOutput.atPath('parameters.json')),
    };
  }

  describe('self-managed mode', () => {
    test('creates admin role if not specified', () => {
      stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
        ...defaultOpts(),
        stackInstances: cpactions.StackInstances.fromArtifactPath(
          stack.sourceOutput.atPath('accounts.json'),
          ['us-east-1', 'us-west-1', 'ca-central-1'],
        ),
        deploymentModel: cpactions.StackSetDeploymentModel.selfManaged({
          executionRoleName: 'Exec',
        }),
      }));

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([
            {
              'Action': [
                'cloudformation:CreateStackInstances',
                'cloudformation:CreateStackSet',
                'cloudformation:DescribeStackSet',
                'cloudformation:DescribeStackSetOperation',
                'cloudformation:ListStackInstances',
                'cloudformation:UpdateStackSet',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::Join': ['', [
                  'arn:',
                  { 'Ref': 'AWS::Partition' },
                  ':cloudformation:us-east-1:111111111111:stackset/MyStack:*',
                ]],
              },
            },
            {
              'Action': 'iam:PassRole',
              'Effect': 'Allow',
              'Resource': { 'Fn::GetAtt': ['PipelineDeployStackSetUpdateStackSetAdministrationRole183434B0', 'Arn'] },
            },
          ]),
        },
        'Roles': [
          { 'Ref': 'PipelineDeployStackSetUpdateCodePipelineActionRole3EDBB32C' },
        ],
      });

      template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          { 'Name': 'Source' /* don't care about the rest */ },
          {
            'Name': 'Deploy',
            'Actions': [
              {
                'Configuration': {
                  'StackSetName': 'MyStack',
                  'Description': 'desc',
                  'TemplatePath': 'SourceArtifact::template.yaml',
                  'Parameters': 'SourceArtifact::parameters.json',
                  'PermissionModel': 'SELF_MANAGED',
                  'AdministrationRoleArn': { 'Fn::GetAtt': ['PipelineDeployStackSetUpdateStackSetAdministrationRole183434B0', 'Arn'] },
                  'ExecutionRoleName': 'Exec',
                  'Capabilities': 'CAPABILITY_NAMED_IAM',
                  'DeploymentTargets': 'SourceArtifact::accounts.json',
                  'FailureTolerancePercentage': 50,
                  'MaxConcurrentPercentage': 25,
                  'Regions': 'us-east-1,us-west-1,ca-central-1',
                },
                'Name': 'StackSetUpdate',
              },
            ],
          },
        ],
      });

      template.hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              'Effect': 'Allow',
              'Action': 'sts:AssumeRole',
              'Resource': { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::*:role/Exec']] },
            },
          ],
        },
      });
    });

    test('passes admin role if specified', () => {
      stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
        ...defaultOpts(),
        stackInstances: cpactions.StackInstances.fromArtifactPath(
          stack.sourceOutput.atPath('accounts.json'),
          ['us-east-1', 'us-west-1', 'ca-central-1'],
        ),
        deploymentModel: cpactions.StackSetDeploymentModel.selfManaged({
          executionRoleName: 'Exec',
          administrationRole: importedAdminRole,
        }),
      }));

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              'Action': [
                'cloudformation:CreateStackInstances',
                'cloudformation:CreateStackSet',
                'cloudformation:DescribeStackSet',
                'cloudformation:DescribeStackSetOperation',
                'cloudformation:ListStackInstances',
                'cloudformation:UpdateStackSet',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { 'Ref': 'AWS::Partition' },
                    ':cloudformation:us-east-1:111111111111:stackset/MyStack:*',
                  ],
                ],
              },
            },
            {
              'Action': 'iam:PassRole',
              'Effect': 'Allow',
              'Resource': 'arn:aws:iam::123456789012:role/ImportedAdminRole',
            },
            {
              'Action': [
                's3:GetObject*',
                's3:GetBucket*',
                's3:List*',
              ],
              'Effect': 'Allow',
              'Resource': [
                { 'Fn::GetAtt': ['PipelineArtifactsBucket22248F97', 'Arn'] },
                {
                  'Fn::Join': ['', [
                    { 'Fn::GetAtt': ['PipelineArtifactsBucket22248F97', 'Arn'] },
                    '/*',
                  ]],
                },
              ],
            },
            {
              'Action': [
                'kms:Decrypt',
                'kms:DescribeKey',
              ],
              'Effect': 'Allow',
              'Resource': { 'Fn::GetAtt': ['PipelineArtifactsBucketEncryptionKey01D58D69', 'Arn'] },
            },
          ],
        },
        'Roles': [{ 'Ref': 'PipelineDeployStackSetUpdateCodePipelineActionRole3EDBB32C' }],
      });
    });
  });

  test('creates correct resources in organizations mode', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
      ...defaultOpts(),
      deploymentModel: cpactions.StackSetDeploymentModel.organizations(),
      stackInstances: cpactions.StackInstances.fromArtifactPath(
        stack.sourceOutput.atPath('accounts.json'),
        ['us-east-1', 'us-west-1', 'ca-central-1'],
      ),
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': Match.arrayWith([
          {
            'Action': [
              'cloudformation:CreateStackInstances',
              'cloudformation:CreateStackSet',
              'cloudformation:DescribeStackSet',
              'cloudformation:DescribeStackSetOperation',
              'cloudformation:ListStackInstances',
              'cloudformation:UpdateStackSet',
            ],
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { 'Ref': 'AWS::Partition' },
                  ':cloudformation:us-east-1:111111111111:stackset/MyStack:*',
                ],
              ],
            },
          },
        ]),
      },
      'Roles': [
        { 'Ref': 'PipelineDeployStackSetUpdateCodePipelineActionRole3EDBB32C' },
      ],
    });
  });

  test('creates correct pipeline resource with target list', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
      ...defaultOpts(),
      stackInstances: cpactions.StackInstances.inAccounts(
        ['111111111111', '222222222222'],
        ['us-east-1', 'us-west-1', 'ca-central-1'],
      ),
      deploymentModel: cpactions.StackSetDeploymentModel.selfManaged({
        administrationRole: importedAdminRole,
        executionRoleName: 'Exec',
      }),
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      'Stages': [
        { 'Name': 'Source' /* don't care about the rest */ },
        {
          'Name': 'Deploy',
          'Actions': [
            {
              'Configuration': {
                'StackSetName': 'MyStack',
                'Description': 'desc',
                'TemplatePath': 'SourceArtifact::template.yaml',
                'Parameters': 'SourceArtifact::parameters.json',
                'Capabilities': 'CAPABILITY_NAMED_IAM',
                'DeploymentTargets': '111111111111,222222222222',
                'FailureTolerancePercentage': 50,
                'MaxConcurrentPercentage': 25,
                'Regions': 'us-east-1,us-west-1,ca-central-1',
              },
              'InputArtifacts': [{ 'Name': 'SourceArtifact' }],
              'Name': 'StackSetUpdate',
            },
          ],
        },
      ],
    });
  });

  test('creates correct pipeline resource with parameter list', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
      ...defaultOpts(),
      parameters: cpactions.StackSetParameters.fromLiteral({
        key0: 'val0',
        key1: 'val1',
      }, ['key2', 'key3']),
      stackInstances: cpactions.StackInstances.fromArtifactPath(
        stack.sourceOutput.atPath('accounts.json'),
        ['us-east-1', 'us-west-1', 'ca-central-1'],
      ),
      deploymentModel: cpactions.StackSetDeploymentModel.selfManaged({
        administrationRole: importedAdminRole,
        executionRoleName: 'Exec',
      }),
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      'Stages': [
        { 'Name': 'Source' /* don't care about the rest */ },
        {
          'Name': 'Deploy',
          'Actions': [
            {
              'Configuration': {
                'StackSetName': 'MyStack',
                'Description': 'desc',
                'TemplatePath': 'SourceArtifact::template.yaml',
                'Parameters': 'ParameterKey=key0,ParameterValue=val0 ParameterKey=key1,ParameterValue=val1 ParameterKey=key2,UsePreviousValue=true ParameterKey=key3,UsePreviousValue=true',
                'Capabilities': 'CAPABILITY_NAMED_IAM',
                'DeploymentTargets': 'SourceArtifact::accounts.json',
                'FailureTolerancePercentage': 50,
                'MaxConcurrentPercentage': 25,
                'Regions': 'us-east-1,us-west-1,ca-central-1',
              },
              'InputArtifacts': [{ 'Name': 'SourceArtifact' }],
              'Name': 'StackSetUpdate',
            },
          ],
        },
      ],
    });
  });

  test('correctly passes region', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackSetAction({
      ...defaultOpts(),
      stackSetRegion: 'us-banana-2',
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      'Stages': [
        { 'Name': 'Source' /* don't care about the rest */ },
        {
          'Name': 'Deploy',
          'Actions': [
            {
              'Region': 'us-banana-2',
            },
          ],
        },
      ],
    });
  });
});

describe('StackInstancesAction', () => {
  function defaultOpts() {
    return {
      actionName: 'StackInstances',
      stackSetName: 'MyStack',
      failureTolerancePercentage: 50,
      maxAccountConcurrencyPercentage: 25,
    };
  }

  test('simple', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackInstancesAction({
      ...defaultOpts(),
      stackInstances: cpactions.StackInstances.inAccounts(
        ['1234', '5678'],
        ['us-east-1', 'us-west-1'],
      ),
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      'Stages': [
        { 'Name': 'Source' /* don't care about the rest */ },
        {
          'Name': 'Deploy',
          'Actions': [
            {
              'ActionTypeId': {
                'Category': 'Deploy',
                'Owner': 'AWS',
                'Provider': 'CloudFormationStackInstances',
                'Version': '1',
              },
              'Configuration': {
                'StackSetName': 'MyStack',
                'FailureTolerancePercentage': 50,
                'MaxConcurrentPercentage': 25,
                'DeploymentTargets': '1234,5678',
                'Regions': 'us-east-1,us-west-1',
              },
              'Name': 'StackInstances',
            },
          ],
        },
      ],
    });
  });

  test('correctly passes region', () => {
    stack.deployStage.addAction(new cpactions.CloudFormationDeployStackInstancesAction({
      ...defaultOpts(),
      stackSetRegion: 'us-banana-2',
      stackInstances: cpactions.StackInstances.inAccounts(['1'], ['us-east-1']),
    }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      'Stages': [
        { 'Name': 'Source' /* don't care about the rest */ },
        {
          'Name': 'Deploy',
          'Actions': [
            {
              'Region': 'us-banana-2',
            },
          ],
        },
      ],
    });
  });
});
