import type { Construct } from 'constructs';
import { Template, Annotations, Match } from '../../../assertions';
import * as ccommit from '../../../aws-codecommit';
import { Pipeline, PipelineType } from '../../../aws-codepipeline';
import * as iam from '../../../aws-iam';
import * as s3 from '../../../aws-s3';
import * as sns from '../../../aws-sns';
import * as sqs from '../../../aws-sqs';
import * as cdk from '../../../core';
import { Stack } from '../../../core';
import * as cxapi from '../../../cx-api';
import * as cdkp from '../../lib';
import { CodePipeline } from '../../lib';
import { PIPELINE_ENV, TestApp, ModernTestGitHubNpmPipeline, FileAssetApp, TwoStackApp, StageWithStackOutput, MultipleFileAssetsApp } from '../testhelpers';

let app: TestApp;

beforeEach(() => {
  app = new TestApp();
});

afterEach(() => {
  app.cleanup();
});

const testStageEnv: Required<cdk.Environment> = {
  account: '123456789012',
  region: 'us-east-1',
};

describe('CodePipeline support stack reuse', () => {
  test('CodePipeline generates the same support stack containing the replication Bucket without the need to bootstrap in that environment for multiple pipelines', () => {
    new ReuseCodePipelineStack(app, 'PipelineStackA', {
      env: PIPELINE_ENV,
    });
    new ReuseCodePipelineStack(app, 'PipelineStackB', {
      env: PIPELINE_ENV,
    });
    const assembly = app.synth();
    // 2 Pipeline Stacks and 1 support stack for both pipeline stacks.
    expect(assembly.stacks.length).toEqual(3);
    assembly.getStackByName(`PipelineStackA-support-${testStageEnv.region}`);
    expect(() => {
      assembly.getStackByName(`PipelineStackB-support-${testStageEnv.region}`);
    }).toThrow(/Unable to find stack with stack name/);
  });

  test('CodePipeline generates the unique support stack containing the replication Bucket without the need to bootstrap in that environment for multiple pipelines', () => {
    new ReuseCodePipelineStack(app, 'PipelineStackA', {
      env: PIPELINE_ENV,
      reuseCrossRegionSupportStacks: false,
    });
    new ReuseCodePipelineStack(app, 'PipelineStackB', {
      env: PIPELINE_ENV,
      reuseCrossRegionSupportStacks: false,
    });
    const assembly = app.synth();
    // 2 Pipeline Stacks and 1 support stack for each pipeline stack.
    expect(assembly.stacks.length).toEqual(4);
    const supportStackAArtifact = assembly.getStackByName(`PipelineStackA-support-${testStageEnv.region}`);
    const supportStackBArtifact = assembly.getStackByName(`PipelineStackB-support-${testStageEnv.region}`);

    const supportStackATemplate = Template.fromJSON(supportStackAArtifact.template);
    supportStackATemplate.hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'pipelinestacka-support-useplicationbucket80db37539447715cac4f',
    });
    supportStackATemplate.hasResourceProperties('AWS::KMS::Alias', {
      AliasName: 'alias/pport-ustencryptionalias5cad4575e12193eb33f2',
    });

    const supportStackBTemplate = Template.fromJSON(supportStackBArtifact.template);
    supportStackBTemplate.hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'pipelinestackb-support-useplicationbucket1d556ec771e60c10c683',
    });
    supportStackBTemplate.hasResourceProperties('AWS::KMS::Alias', {
      AliasName: 'alias/pport-ustencryptionalias668c7ffd2deea6ebc8a4',
    });
  });
});

describe('Providing codePipeline parameter and prop(s) of codePipeline parameter to CodePipeline constructor should throw error', () => {
  test('Providing codePipeline parameter and pipelineName parameter should throw error', () => {
    expect(() => new CodePipelinePropsCheckTest(app, 'CodePipeline', {
      pipelineName: 'randomName',
    }).create()).toThrow('Cannot set \'pipelineName\' if an existing CodePipeline is given using \'codePipeline\'');
  });
  test('Providing codePipeline parameter and enableKeyRotation parameter should throw error', () => {
    expect(() => new CodePipelinePropsCheckTest(app, 'CodePipeline', {
      enableKeyRotation: true,
    }).create()).toThrow('Cannot set \'enableKeyRotation\' if an existing CodePipeline is given using \'codePipeline\'');
  });
  test('Providing codePipeline parameter and crossAccountKeys parameter should throw error', () => {
    expect(() => new CodePipelinePropsCheckTest(app, 'CodePipeline', {
      crossAccountKeys: true,
    }).create()).toThrow('Cannot set \'crossAccountKeys\' if an existing CodePipeline is given using \'codePipeline\'');
  });
  test('Providing codePipeline parameter and reuseCrossRegionSupportStacks parameter should throw error', () => {
    expect(() => new CodePipelinePropsCheckTest(app, 'CodePipeline', {
      reuseCrossRegionSupportStacks: true,
    }).create()).toThrow('Cannot set \'reuseCrossRegionSupportStacks\' if an existing CodePipeline is given using \'codePipeline\'');
  });
  test('Providing codePipeline parameter and role parameter should throw error', () => {
    const stack = new Stack(app, 'Stack');

    expect(() => new CodePipelinePropsCheckTest(stack, 'CodePipeline', {
      role: new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
      }),
    }).create()).toThrow('Cannot set \'role\' if an existing CodePipeline is given using \'codePipeline\'');
  });
});

test('Policy sizes do not exceed the maximum size', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  pipelineStack.node.setContext('@aws-cdk/aws-iam:minimizePolicies', true);
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
  });

  // WHEN
  const regions = ['us-east-1', 'us-east-2', 'eu-west-1', 'eu-west-2', 'somethingelse1', 'somethingelse-2', 'yapregion', 'more-region'];
  for (let i = 0; i < 70; i++) {
    pipeline.addStage(new FileAssetApp(pipelineStack, `App${i}`, {
      env: {
        account: `${i}`.padStart(12, '0'),
        region: regions[i % regions.length],
      },
    }), {
      post: [
        new cdkp.ShellStep('DoAThing', { commands: ['true'] }),
        new cdkp.ShellStep('DoASecondThing', { commands: ['false'] }),
      ],
    });
  }

  // THEN
  const template = Template.fromStack(pipelineStack);

  // Collect policies by role
  const rolePolicies: Record<string, any[]> = {};
  for (const pol of Object.values(template.findResources('AWS::IAM::Policy'))) {
    for (const roleName of pol.Properties?.Roles ?? []) {
      const roleLogicalId = roleName.Ref; // Roles: [ { Ref: MyRole } ]
      if (!roleLogicalId) { continue; }

      if (!rolePolicies[roleLogicalId]) {
        rolePolicies[roleLogicalId] = [];
      }

      rolePolicies[roleLogicalId].push(pol.Properties.PolicyDocument);
    }
  }

  // Validate sizes
  //
  // Not entirely accurate, because our "Ref"s and "Fn::GetAtt"s actually need to be evaluated
  // to ARNs... but it gives an order-of-magnitude indication.
  // 10% of margin for CFN intrinsics like { Fn::Join } and { Ref: 'AWS::Partition' } which don't contribute to
  // the ACTUAL size, but do contribute to the measured size here.
  const cfnOverheadMargin = 1.10;

  for (const [logId, poldoc] of Object.entries(rolePolicies)) {
    const totalJson = JSON.stringify(poldoc);
    if (totalJson.length > 10000 * cfnOverheadMargin) {
      throw new Error(`Policy for Role ${logId} is too large (${totalJson.length} bytes): ${JSON.stringify(poldoc, undefined, 2)}`);
    }
  }

  for (const [logId, poldoc] of Object.entries(template.findResources('AWS::IAM::ManagedPolicy'))) {
    const totalJson = JSON.stringify(poldoc);
    if (totalJson.length > 6000 * cfnOverheadMargin) {
      throw new Error(`Managed Policy ${logId} is too large (${totalJson.length} bytes): ${JSON.stringify(poldoc, undefined, 2)}`);
    }
  }

  // expect template size warning
  const annotations = Annotations.fromStack(pipelineStack);
  annotations.hasWarning('*', Match.stringLikeRegexp('^Template size is approaching limit'));
  const warnings = annotations.findWarning('*', Match.anyValue());
  expect(warnings.length).toEqual(2);
});

test.each([
  [PipelineType.V1, 'V1'],
  [PipelineType.V2, 'V2'],
  [undefined, 'V1'],
])('can specify pipeline type %s when feature flag is not set', (type, expected) => {
  const stack = new cdk.Stack();
  const repo = new ccommit.Repository(stack, 'Repo', {
    repositoryName: 'MyRepo',
  });
  const cdkInput = cdkp.CodePipelineSource.codeCommit(
    repo,
    'main',
  );
  new CodePipeline(stack, 'Pipeline', {
    synth: new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    }),
    pipelineType: type,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    PipelineType: expected,
  });
});

test.each([
  [undefined, 'latest'],
  ['9.9.9', '9.9.9'],
])('When I request cdk-assets version %p I get %p', (requested, expected) => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipe = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    cdkAssetsCliVersion: requested,
  });

  pipe.addStage(new FileAssetApp(pipelineStack, 'App', {}));

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: [
              `npm install -g cdk-assets@${expected}`,
            ],
          },
        },
      })),
    },
  });
});

test('CodeBuild action role has the right AssumeRolePolicyDocument', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');

  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: [
        {
          Action: 'sts:AssumeRole',
          Principal: {
            AWS: { 'Fn::GetAtt': ['CdkPipelineRoleC09C4D44', 'Arn'] },
          },
        },
      ],
    },
  });
});

test('CodeBuild asset role has the right Principal with the feature enabled', () => {
  const stack = new cdk.Stack();
  stack.node.setContext(cxapi.PIPELINE_REDUCE_ASSET_ROLE_TRUST_SCOPE, true);
  const pipelineStack = new cdk.Stack(stack, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
  pipeline.addStage(new FileAssetApp(pipelineStack, 'App', {}));
  const template = Template.fromStack(pipelineStack);
  const assetRole = template.toJSON().Resources.CdkAssetsFileRole6BE17A07;
  const statementLength = assetRole.Properties.AssumeRolePolicyDocument.Statement;
  expect(statementLength).toStrictEqual(
    [
      {
        Action: 'sts:AssumeRole',
        Effect: 'Allow',
        Principal: {
          Service: 'codebuild.amazonaws.com',
        },
      },
    ],
  );
});

test.each([
  [undefined, '2'],
  ['9.9.9', '9.9.9'],
])('When I request CLI version version %p I get %p', (requested, expected) => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipe = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    cliVersion: requested,
  });

  pipe.addStage(new FileAssetApp(pipelineStack, 'App', {}));

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: [
              `npm install -g aws-cdk@${expected}`,
            ],
          },
        },
      })),
    },
  });
});

test('CodeBuild asset role has the right Principal with the feature disabled', () => {
  const stack = new cdk.Stack();
  stack.node.setContext(cxapi.PIPELINE_REDUCE_ASSET_ROLE_TRUST_SCOPE, false);
  const pipelineStack = new cdk.Stack(stack, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
  pipeline.addStage(new FileAssetApp(pipelineStack, 'App', {}));
  const template = Template.fromStack(pipelineStack);
  const assetRole = template.toJSON().Resources.CdkAssetsFileRole6BE17A07;
  const statementLength = assetRole.Properties.AssumeRolePolicyDocument.Statement;
  expect(statementLength).toStrictEqual(
    [
      {
        Action: 'sts:AssumeRole',
        Effect: 'Allow',
        Principal: {
          Service: 'codebuild.amazonaws.com',
        },
      },
      {
        Action: 'sts:AssumeRole',
        Effect: 'Allow',
        Principal: {
          AWS: {
            'Fn::Join': ['', [
              'arn:', { Ref: 'AWS::Partition' }, `:iam::${PIPELINE_ENV.account}:root`,
            ]],
          },
        },
      },
    ],
  );
});

test('CodePipeline throws when key rotation is enabled without enabling cross account keys', ()=>{
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const repo = new ccommit.Repository(pipelineStack, 'Repo', {
    repositoryName: 'MyRepo',
  });
  const cdkInput = cdkp.CodePipelineSource.codeCommit(
    repo,
    'main',
  );

  expect(() => new CodePipeline(pipelineStack, 'Pipeline', {
    enableKeyRotation: true,
    synth: new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    }),
  }).buildPipeline()).toThrow('Setting \'enableKeyRotation\' to true also requires \'crossAccountKeys\' to be enabled');
});

test('CodePipeline enables key rotation on cross account keys', ()=>{
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const repo = new ccommit.Repository(pipelineStack, 'Repo', {
    repositoryName: 'MyRepo',
  });
  const cdkInput = cdkp.CodePipelineSource.codeCommit(
    repo,
    'main',
  );

  new CodePipeline(pipelineStack, 'Pipeline', {
    enableKeyRotation: true,
    crossAccountKeys: true, // requirement of key rotation
    synth: new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    }),
  });

  const template = Template.fromStack(pipelineStack);

  template.hasResourceProperties('AWS::KMS::Key', {
    EnableKeyRotation: true,
  });
});

test('CodePipeline supports use of existing role', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const repo = new ccommit.Repository(pipelineStack, 'Repo', {
    repositoryName: 'MyRepo',
  });
  const cdkInput = cdkp.CodePipelineSource.codeCommit(
    repo,
    'main',
  );

  new CodePipeline(pipelineStack, 'Pipeline', {
    synth: new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    }),
    role: new iam.Role(pipelineStack, 'CustomRole', {
      assumedBy: new iam.ServicePrincipal('codepipeline.amazonaws.com'),
      roleName: 'MyCustomPipelineRole',
    }),
  });

  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: [
        {
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: {
            Service: 'codepipeline.amazonaws.com',
          },
        },
      ],
    },
    RoleName: 'MyCustomPipelineRole',
  });
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    RoleArn: { 'Fn::GetAtt': ['CustomRole6D8E6809', 'Arn'] },
  });
});

test('CodePipeline supports use of service role for actions', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const repo = new ccommit.Repository(pipelineStack, 'Repo', {
    repositoryName: 'MyRepo',
  });
  const cdkInput = cdkp.CodePipelineSource.codeCommit(
    repo,
    'main',
  );

  const pipeline = new CodePipeline(pipelineStack, 'Pipeline', {
    synth: new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    }),
    role: new iam.Role(pipelineStack, 'CustomRole', {
      assumedBy: new iam.ServicePrincipal('codepipeline.amazonaws.com'),
      roleName: 'MyCustomPipelineRole',
    }),
    usePipelineRoleForActions: true,
  });

  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: [
        {
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: {
            Service: 'codepipeline.amazonaws.com',
          },
        },
      ],
    },
    RoleName: 'MyCustomPipelineRole',
  });
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    RoleArn: { 'Fn::GetAtt': ['CustomRole6D8E6809', 'Arn'] },
  });

  pipeline.pipeline.stages.forEach(s =>{
    const actions = s.actions;
    actions.forEach(a => {
      expect(a.actionProperties.role).toBeUndefined();
    });
  });
});

describe('deployment of stack', () => {
  test('is done with Prepare and Deploy step by default', () => {
    const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      crossAccountKeys: true,
    });
    pipeline.addStage(new FileAssetApp(pipelineStack, 'App', {}));

    // THEN
    const template = Template.fromStack(pipelineStack);

    // There should be Prepare step in piepline
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.arrayWith([{
        Actions: Match.arrayWith([
          Match.objectLike({
            Configuration: Match.objectLike({
              ActionMode: 'CHANGE_SET_REPLACE',
            }),
            Name: 'Prepare',
          }),
          Match.objectLike({
            Configuration: Match.objectLike({
              ActionMode: 'CHANGE_SET_EXECUTE',
            }),
            Name: 'Deploy',
          }),
        ]),
        Name: 'App',
      }]),
    });
  });

  test('can be done with single step', () => {
    const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
    const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
      crossAccountKeys: true,
      useChangeSets: false,
    });
    pipeline.addStage(new FileAssetApp(pipelineStack, 'App', {}));

    // THEN
    const template = Template.fromStack(pipelineStack);

    // There should be Prepare step in piepline
    template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
      Stages: Match.arrayWith([{
        Actions: Match.arrayWith([
          Match.objectLike({
            Configuration: Match.objectLike({
              ActionMode: 'CREATE_UPDATE',
            }),
            Name: 'Deploy',
          }),
        ]),
        Name: 'App',
      }]),
    });
  });
});

test('display name can contain illegal characters which are sanitized for the pipeline', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
    useChangeSets: false,
  });
  pipeline.addStage(new FileAssetApp(pipelineStack, 'App', {
    displayName: 'asdf & also qwerty!',
  }));

  // THEN
  const template = Template.fromStack(pipelineStack);

  // We expect `asdf` and `asdf2` Actions in the pipeline
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Assets',
      Actions: Match.arrayWith([
        Match.objectLike({
          Name: 'asdf_also_qwerty_',
        }),
      ]),
    }]),
  });
});

test.each([2, 3])('%p assets can have same display name, which are reflected in the pipeline', (n) => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
    useChangeSets: false,
  });
  pipeline.addStage(new MultipleFileAssetsApp(pipelineStack, 'App', {
    n,
    displayNames: Array.from({ length: n }, () => 'asdf'),
  }));

  // THEN
  const template = Template.fromStack(pipelineStack);

  // We expect at least `asdf` and `asdf2` Actions in the pipeline,
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Assets',
      Actions: Match.arrayWith([
        Match.objectLike({ Name: 'asdf' }),
        Match.objectLike({ Name: 'asdf2' }),
        ...(n == 3 ? [Match.objectLike({ Name: 'asdf3' })] : []),
      ]),
    }]),
  });
});

test('assets can have display names that conflict with calculated action names', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
    useChangeSets: false,
  });
  pipeline.addStage(new MultipleFileAssetsApp(pipelineStack, 'App', {
    n: 3,
    displayNames: ['asdf', 'asdf', 'asdf2'], // asdf2 will conflict with the second generated name which will also be asdf2
  }));

  // THEN
  const template = Template.fromStack(pipelineStack);

  // We expect `asdf`, `asdf2` and `asdf22` Actions in the pipeline
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Assets',
      Actions: Match.arrayWith([
        Match.objectLike({ Name: 'asdf' }),
        Match.objectLike({ Name: 'asdf2' }),
        Match.objectLike({ Name: 'asdf22' }),
      ]),
    }]),
  });
});

test('action name does not reflect display names if publishAssetsInParallel is false', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    publishAssetsInParallel: false,
  });
  pipeline.addStage(new MultipleFileAssetsApp(pipelineStack, 'App', {
    n: 2,
    displayNames: ['asdf1', 'asdf2'],
  }));

  // THEN
  const template = Template.fromStack(pipelineStack);

  // We expect a single `FileAsset` action in the pipeline
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'Assets',
      Actions: [
        Match.objectLike({
          Name: 'FileAssets',
        }),
      ],
    }]),
  });
});

test('action name is calculated properly if it has cross-stack dependencies', () => {
  // GIVEN
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
  });

  // WHEN
  const s1step = new cdkp.ManualApprovalStep('S1');
  const s2step = new cdkp.ManualApprovalStep('S2');
  s1step.addStepDependency(s2step);

  // The issue we were diagnosing only manifests if the stacks don't have
  // a dependency on each other
  const stage = new TwoStackApp(app, 'TheApp', { withDependency: false });
  pipeline.addStage(stage, {
    stackSteps: [
      { stack: stage.stack1, post: [s1step] },
      { stack: stage.stack2, post: [s2step] },
    ],
  });

  // THEN
  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'TheApp',
      Actions: Match.arrayWith([
        Match.objectLike({ Name: 'Stack2.S2', RunOrder: 3 }),
        Match.objectLike({ Name: 'Stack1.S1', RunOrder: 4 }),
      ]),
    }]),
  });
});

test('synths with change set approvers', () => {
  // GIVEN
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');

  // WHEN
  const csApproval = new cdkp.ManualApprovalStep('ChangeSetApproval');

  // The issue we were diagnosing only manifests if the stacks don't have
  // a dependency on each other
  const stage = new TwoStackApp(app, 'TheApp', { withDependency: false });
  pipeline.addStage(stage, {
    stackSteps: [
      { stack: stage.stack1, changeSet: [csApproval] },
      { stack: stage.stack2, changeSet: [csApproval] },
    ],
  });

  // THEN
  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'TheApp',
      Actions: Match.arrayWith([
        Match.objectLike({ Name: 'Stack1.Prepare', RunOrder: 1 }),
        Match.objectLike({ Name: 'Stack2.Prepare', RunOrder: 1 }),
        Match.objectLike({ Name: 'Stack1.ChangeSetApproval', RunOrder: 2 }),
        Match.objectLike({ Name: 'Stack1.Deploy', RunOrder: 3 }),
        Match.objectLike({ Name: 'Stack2.Deploy', RunOrder: 3 }),
      ]),
    }]),
  });
});

test('selfMutationProject can be accessed after buildPipeline', () => {
  // GIVEN
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');
  pipeline.addStage(new StageWithStackOutput(pipelineStack, 'Stage'));

  // WHEN
  pipeline.buildPipeline();

  // THEN
  expect(pipeline.selfMutationProject).toBeTruthy();
});

test('selfMutationProject is undefined if switched off', () => {
  // GIVEN
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    selfMutation: false,
  });
  pipeline.addStage(new StageWithStackOutput(pipelineStack, 'Stage'));

  // WHEN
  pipeline.buildPipeline();

  // THEN
  expect(() => pipeline.selfMutationProject).toThrow(/No selfMutationProject/);
});

test('artifactBucket can be overridden', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    artifactBucket: new s3.Bucket(pipelineStack, 'CustomArtifact', {
      bucketName: 'my-custom-artifact-bucket',
    }),
  });
  // THEN
  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::S3::Bucket', {
    BucketName: 'my-custom-artifact-bucket',
  });
});

test('throws when deploy role session tags are used', () => {
  const synthesizer = new cdk.DefaultStackSynthesizer({
    deployRoleAdditionalOptions: {
      Tags: [{ Key: 'Departement', Value: 'Enginerring' }],
    },
  });

  expect(() => {
    new ReuseCodePipelineStack(app, 'PipelineStackA', {
      env: PIPELINE_ENV,
      reuseStageProps: {
        reuseStackProps: {
          synthesizer,
        },
      },
    });
  }).toThrow('Deployment of stack SampleStage-123456789012-us-east-1-SampleStack requires assuming the role arn:${AWS::Partition}:iam::123456789012:role/cdk-hnb659fds-deploy-role-123456789012-us-east-1 with session tags, but assuming roles with session tags is not supported by CodePipeline.');
});

test('test add external link for manual approval', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });

  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
  });

  const stage = new TwoStackApp(app, 'TheApp', { withDependency: false });

  const approval = new cdkp.ManualApprovalStep('Approval', {
    comment: 'Please approve',
    reviewUrl: 'https://approve-confirm.com',
  });

  pipeline.addStage(stage, {
    pre: [approval],
  });

  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'TheApp',
      Actions: Match.arrayWith([
        Match.objectLike({
          Name: 'Approval',
          RunOrder: 1,
          Configuration: Match.objectLike({
            ExternalEntityLink: 'https://approve-confirm.com',
          }),
        }),
      ]),
    }]),
  });
});

test('sns topic for manual approval', () => {
  const pipelineStack = new cdk.Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
  const pipeline = new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    crossAccountKeys: true,
  });
  const topic = new sns.Topic(pipelineStack, 'Topic', {
    topicName: 'MyTopic',
  });
  const approval = new cdkp.ManualApprovalStep('Approval', {
    comment: 'Please approve',
    notificationTopic: topic,
  });
  const stage = new TwoStackApp(app, 'TheApp', { withDependency: false });
  pipeline.addStage(stage, {
    pre: [approval],
  });
  const template = Template.fromStack(pipelineStack);
  template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'TheApp',
      Actions: Match.arrayWith([
        Match.objectLike({
          Name: 'Approval',
          RunOrder: 1,
          Configuration: Match.objectLike({
            NotificationArn: { Ref: 'TopicBFC7AF6E' },
          }),
        }),
      ]),
    }]),
  });
  template.hasResourceProperties('AWS::SNS::Topic', {
    TopicName: 'MyTopic',
  });
});

interface ReuseCodePipelineStackProps extends cdk.StackProps {
  reuseCrossRegionSupportStacks?: boolean;
  reuseStageProps?: ReuseStageProps;
}
class ReuseCodePipelineStack extends cdk.Stack {
  public constructor(scope: Construct, id: string, props: ReuseCodePipelineStackProps ) {
    super(scope, id, props);
    const repo = new ccommit.Repository(this, 'Repo', {
      repositoryName: 'MyRepo',
    });

    const cdkInput = cdkp.CodePipelineSource.codeCommit(
      repo,
      'main',
    );

    const synthStep = new cdkp.ShellStep('Synth', {
      input: cdkInput,
      installCommands: ['npm ci'],
      commands: [
        'npm run build',
        'npx cdk synth',
      ],
    });

    const pipeline = new cdkp.CodePipeline(this, 'Pipeline', {
      synth: synthStep,
      selfMutation: true,
      crossAccountKeys: true,
      reuseCrossRegionSupportStacks: props.reuseCrossRegionSupportStacks,
    });

    const stage = new ReuseStage(
      this,
      `SampleStage-${testStageEnv.account}-${testStageEnv.region}`,
      {
        env: testStageEnv,
        ...(props.reuseStageProps ?? {}),
      },
    );
    pipeline.addStage(stage);
  }
}

interface ReuseStageProps extends cdk.StageProps {
  readonly reuseStackProps?: cdk.StackProps;
}

class ReuseStage extends cdk.Stage {
  public constructor(scope: Construct, id: string, props: ReuseStageProps) {
    super(scope, id, props);
    new ReuseStack(this, 'SampleStack', props.reuseStackProps ?? {});
  }
}

class ReuseStack extends cdk.Stack {
  public constructor(scope: Construct, id: string, props: cdk.StackProps) {
    super(scope, id, props);
    new sqs.Queue(this, 'Queue');
  }
}

interface CodePipelineStackProps extends cdk.StackProps {
  pipelineName?: string;
  crossAccountKeys?: boolean;
  enableKeyRotation?: boolean;
  reuseCrossRegionSupportStacks?: boolean;
  role?: iam.IRole;
}

class CodePipelinePropsCheckTest extends cdk.Stack {
  cProps: CodePipelineStackProps;
  public constructor(scope: Construct, id: string, props: CodePipelineStackProps) {
    super(scope, id, props);
    this.cProps = props;
  }
  public create() {
    if (this.cProps.pipelineName !== undefined) {
      new cdkp.CodePipeline(this, 'CodePipeline1', {
        pipelineName: this.cProps.pipelineName,
        codePipeline: new Pipeline(this, 'Pipeline1'),
        synth: new cdkp.ShellStep('Synth', { commands: ['ls'] }),
      }).buildPipeline();
    }
    if (this.cProps.crossAccountKeys !== undefined) {
      new cdkp.CodePipeline(this, 'CodePipeline2', {
        crossAccountKeys: this.cProps.crossAccountKeys,
        codePipeline: new Pipeline(this, 'Pipeline2'),
        synth: new cdkp.ShellStep('Synth', { commands: ['ls'] }),
      }).buildPipeline();
    }
    if (this.cProps.enableKeyRotation !== undefined) {
      new cdkp.CodePipeline(this, 'CodePipeline3', {
        enableKeyRotation: this.cProps.enableKeyRotation,
        codePipeline: new Pipeline(this, 'Pipeline3'),
        synth: new cdkp.ShellStep('Synth', { commands: ['ls'] }),
      }).buildPipeline();
    }
    if (this.cProps.reuseCrossRegionSupportStacks !== undefined) {
      new cdkp.CodePipeline(this, 'CodePipeline4', {
        reuseCrossRegionSupportStacks: this.cProps.reuseCrossRegionSupportStacks,
        codePipeline: new Pipeline(this, 'Pipeline4'),
        synth: new cdkp.ShellStep('Synth', { commands: ['ls'] }),
      }).buildPipeline();
    }
    if (this.cProps.role !== undefined) {
      new cdkp.CodePipeline(this, 'CodePipeline5', {
        role: this.cProps.role,
        codePipeline: new Pipeline(this, 'Pipeline5'),
        synth: new cdkp.ShellStep('Synth', { commands: ['ls'] }),
      }).buildPipeline();
    }
  }
}
