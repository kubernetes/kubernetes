import { Match, Template } from '../../assertions';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as secretsmanager from '../../aws-secretsmanager';
import * as cdk from '../../core';
import * as codebuild from '../lib';

/* eslint-disable @stylistic/quote-props */

test('can use filename as buildspec', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'Bucket'),
      path: 'path',
    }),
    buildSpec: codebuild.BuildSpec.fromSourceFilename('hello.yml'),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: 'hello.yml',
    },
  });
});

test('can use buildspec literal', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new codebuild.Project(stack, 'Project', {
    buildSpec: codebuild.BuildSpec.fromObject({ phases: ['say hi'] }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: '{\n  "phases": [\n    "say hi"\n  ]\n}',
    },
  });
});

test('can use yamlbuildspec literal', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new codebuild.Project(stack, 'Project', {
    buildSpec: codebuild.BuildSpec.fromObjectToYaml({
      text: 'text',
      decimal: 10,
      list: ['say hi'],
      obj: {
        text: 'text',
        decimal: 10,
        list: ['say hi'],
      },
    }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    Source: {
      BuildSpec: 'text: text\ndecimal: 10\nlist:\n  - say hi\nobj:\n  text: text\n  decimal: 10\n  list:\n    - say hi\n',
    },
  });
});

test('must supply buildspec when using nosource', () => {
  // GIVEN
  const stack = new cdk.Stack();

  expect(() => {
    new codebuild.Project(stack, 'Project', {
    });
  }).toThrow(/you need to provide a concrete buildSpec/);
});

test('must supply literal buildspec when using nosource', () => {
  // GIVEN
  const stack = new cdk.Stack();

  expect(() => {
    new codebuild.Project(stack, 'Project', {
      buildSpec: codebuild.BuildSpec.fromSourceFilename('bla.yml'),
    });
  }).toThrow(/you need to provide a concrete buildSpec/);
});

describe('GitHub source', () => {
  test('has reportBuildStatus on by default', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        cloneDepth: 3,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'GITHUB',
        Location: 'https://github.com/testowner/testrepo.git',
        ReportBuildStatus: true,
        GitCloneDepth: 3,
      },
    });
  });

  test('can set a branch as the SourceVersion', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        branchOrRef: 'testbranch',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      SourceVersion: 'testbranch',
    });
  });

  test('can explicitly set reportBuildStatus to false', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        reportBuildStatus: false,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        ReportBuildStatus: false,
      },
    });
  });

  test('can explicitly set webhook to true', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
        webhook: true,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Triggers: {
        Webhook: true,
      },
    });
  });

  test('can create organizational webhook', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'GITHUB',
        Location: 'CODEBUILD_DEFAULT_WEBHOOK_SOURCE_LOCATION',
      },
      Triggers: {
        ScopeConfiguration: {
          Name: 'testowner',
        },
        FilterGroups: [
          [
            {
              Type: 'EVENT',
              Pattern: 'WORKFLOW_JOB_QUEUED',
            },
          ],
        ],
      },
    });
  });

  test('can create organizational webhook with filters', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const filter = codebuild.FilterGroup.inEventOf(codebuild.EventAction.WORKFLOW_JOB_QUEUED).andRepositoryNameIs('testrepo');
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        webhookFilters: [filter],
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Source: {
        Type: 'GITHUB',
        Location: 'CODEBUILD_DEFAULT_WEBHOOK_SOURCE_LOCATION',
      },
      Triggers: {
        ScopeConfiguration: {
          Name: 'testowner',
        },
        FilterGroups: [
          [
            {
              Type: 'EVENT',
              Pattern: 'WORKFLOW_JOB_QUEUED',
            },
            {
              Type: 'REPOSITORY_NAME',
              Pattern: 'testrepo',
            },
          ],
        ],
      },
    });
  });

  test('can be added to a CodePipeline', () => {
    const stack = new cdk.Stack();
    const project = new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHub({
        owner: 'testowner',
        repo: 'testrepo',
      }),
    });

    expect(() => {
      project.bindToCodePipeline(project, {
        artifactBucket: new s3.Bucket(stack, 'Bucket'),
      });
    }).not.toThrow(); // no exception
  });

  test('can provide credentials to use with the source', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.GitHubSourceCredentials(stack, 'GitHubSourceCredentials', {
      accessToken: cdk.SecretValue.unsafePlainText('my-access-token'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::SourceCredential', {
      'ServerType': 'GITHUB',
      'AuthType': 'PERSONAL_ACCESS_TOKEN',
      'Token': 'my-access-token',
    });
  });
});

describe('GitHub Enterprise source', () => {
  test('can use branchOrRef to set the source version', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.gitHubEnterprise({
        httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
        branchOrRef: 'testbranch',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      SourceVersion: 'testbranch',
    });
  });

  test('can provide credentials to use with the source', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.GitHubEnterpriseSourceCredentials(stack, 'GitHubEnterpriseSourceCredentials', {
      accessToken: cdk.SecretValue.unsafePlainText('my-access-token'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::SourceCredential', {
      'ServerType': 'GITHUB_ENTERPRISE',
      'AuthType': 'PERSONAL_ACCESS_TOKEN',
      'Token': 'my-access-token',
    });
  });
});

describe('BitBucket source', () => {
  test('can use branchOrRef to set the source version', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.bitBucket({
        owner: 'testowner',
        repo: 'testrepo',
        branchOrRef: 'testbranch',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      SourceVersion: 'testbranch',
    });
  });

  test('can provide credentials to use with the source', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.BitBucketSourceCredentials(stack, 'BitBucketSourceCredentials', {
      username: cdk.SecretValue.unsafePlainText('my-username'),
      password: cdk.SecretValue.unsafePlainText('password'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::SourceCredential', {
      'ServerType': 'BITBUCKET',
      'AuthType': 'BASIC_AUTH',
      'Username': 'my-username',
      'Token': 'password',
    });
  });
});

describe('caching', () => {
  test('using Cache.none() results in NO_CACHE in the template', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.PipelineProject(stack, 'Project', {
      cache: codebuild.Cache.none(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Cache: {
        Type: 'NO_CACHE',
        Location: Match.absent(),
      },
    });
  });

  test('project with s3 cache bucket', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'SourceBucket'),
        path: 'path',
      }),
      cache: codebuild.Cache.bucket(new s3.Bucket(stack, 'Bucket'), {
        prefix: 'cache-prefix',
        cacheNamespace: 'namespace',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Cache: {
        Type: 'S3',
        Location: {
          'Fn::Join': [
            '/',
            [
              {
                'Ref': 'Bucket83908E77',
              },
              'cache-prefix',
            ],
          ],
        },
        CacheNamespace: 'namespace',
      },
    });
  });

  test('s3 codebuild project with sourceVersion', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
        version: 's3version',
      }),
      cache: codebuild.Cache.local(codebuild.LocalCacheMode.CUSTOM, codebuild.LocalCacheMode.DOCKER_LAYER,
        codebuild.LocalCacheMode.SOURCE),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      SourceVersion: 's3version',
    });
  });

  test('project with local cache modes', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      cache: codebuild.Cache.local(codebuild.LocalCacheMode.CUSTOM, codebuild.LocalCacheMode.DOCKER_LAYER,
        codebuild.LocalCacheMode.SOURCE),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Cache: {
        Type: 'LOCAL',
        Modes: [
          'LOCAL_CUSTOM_CACHE',
          'LOCAL_DOCKER_LAYER_CACHE',
          'LOCAL_SOURCE_CACHE',
        ],
      },
    });
  });

  test('project by default has cache type set to NO_CACHE', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Cache: {
        Type: 'NO_CACHE',
        Location: Match.absent(),
      },
    });
  });
});

test('if a role is shared between projects in a VPC, the VPC Policy is only attached once', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
  });
  const source = codebuild.Source.gitHubEnterprise({
    httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
  });

  // WHEN
  new codebuild.Project(stack, 'Project1', { source, role, vpc, projectName: 'P1' });
  new codebuild.Project(stack, 'Project2', { source, role, vpc, projectName: 'P2' });

  // THEN
  // - 1 is for `ec2:CreateNetworkInterfacePermission`, deduplicated as they're part of a single policy
  // - 1 is for `ec2:CreateNetworkInterface`, this is the separate Policy we're deduplicating
  // We would have found 3 if the deduplication didn't work.
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 2);

  // THEN - both Projects have a DependsOn on the same policy
  Template.fromStack(stack).hasResource('AWS::CodeBuild::Project', {
    Properties: { Name: 'P1' },
    DependsOn: ['Project1PolicyDocumentF9761562'],
  });

  Template.fromStack(stack).hasResource('AWS::CodeBuild::Project', {
    Properties: { Name: 'P1' },
    DependsOn: ['Project1PolicyDocumentF9761562'],
  });
});

test('can use an imported Role for a Project within a VPC', () => {
  const stack = new cdk.Stack();

  const importedRole = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/service-role/codebuild-bruiser-service-role');
  const vpc = new ec2.Vpc(stack, 'Vpc');

  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHubEnterprise({
      httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
    }),
    role: importedRole,
    vpc,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    // no need to do any assertions
  });
});

test('can use an imported Role with mutable = false for a Project within a VPC', () => {
  const stack = new cdk.Stack();

  const importedRole = iam.Role.fromRoleArn(stack, 'Role',
    'arn:aws:iam::123456789012:role/service-role/codebuild-bruiser-service-role', {
      mutable: false,
    });
  const vpc = new ec2.Vpc(stack, 'Vpc');

  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHubEnterprise({
      httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
    }),
    role: importedRole,
    vpc,
  });

  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);

  // Check that the CodeBuild project does not have a DependsOn
  Template.fromStack(stack).hasResource('AWS::CodeBuild::Project', (res: any) => {
    if (res.DependsOn && res.DependsOn.length > 0) {
      throw new Error(`CodeBuild project should have no DependsOn, but got: ${JSON.stringify(res, undefined, 2)}`);
    }
    return true;
  });
});

test('can use an ImmutableRole for a Project within a VPC', () => {
  const stack = new cdk.Stack();

  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
  });

  const vpc = new ec2.Vpc(stack, 'Vpc');

  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHubEnterprise({
      httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
    }),
    role: role.withoutPolicyUpdates(),
    vpc,
  });

  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);

  // Check that the CodeBuild project does not have a DependsOn
  Template.fromStack(stack).hasResource('AWS::CodeBuild::Project', (res: any) => {
    if (res.DependsOn && res.DependsOn.length > 0) {
      throw new Error(`CodeBuild project should have no DependsOn, but got: ${JSON.stringify(res, undefined, 2)}`);
    }
    return true;
  });
});

test('metric method generates a valid CloudWatch metric', () => {
  const stack = new cdk.Stack();

  const project = new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHubEnterprise({
      httpsCloneUrl: 'https://mygithub-enterprise.com/myuser/myrepo',
    }),
  });

  const metric = project.metric('Builds');
  expect(metric.metricName).toEqual('Builds');
  expect(metric.period.toSeconds()).toEqual(cdk.Duration.minutes(5).toSeconds());
  expect(metric.statistic).toEqual('Average');
});

describe('CodeBuild test reports group', () => {
  test('adds the appropriate permissions when reportGroup.grantWrite() is called', () => {
    const stack = new cdk.Stack();

    const reportGroup = new codebuild.ReportGroup(stack, 'ReportGroup');

    const project = new codebuild.Project(stack, 'Project', {
      buildSpec: codebuild.BuildSpec.fromObject({
        version: '0.2',
        reports: {
          [reportGroup.reportGroupArn]: {
            files: '**/*',
          },
        },
      }),
      grantReportGroupPermissions: false,
    });
    reportGroup.grantWrite(project);

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {},
          {
            'Action': [
              'codebuild:CreateReport',
              'codebuild:UpdateReport',
              'codebuild:BatchPutTestCases',
            ],
            'Resource': {
              'Fn::GetAtt': [
                'ReportGroup8A84C76D',
                'Arn',
              ],
            },
          },
        ],
      },
    });
  });
});

describe('Environment', () => {
  test('build image - can use secret to access build image', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const secret = new secretsmanager.Secret(stack, 'Secret');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      environment: {
        buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('myimage', { secretsManagerCredentials: secret }),
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        RegistryCredential: {
          CredentialProvider: 'SECRETS_MANAGER',
          Credential: { 'Ref': 'SecretA720EF05' },
        },
      }),
    });
  });

  test('build image - can use secret to access build image', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const secret = new secretsmanager.Secret(stack, 'Secret');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.MEDIUM,
      environmentType: codebuild.EnvironmentType.MAC_ARM,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      environment: {
        buildImage: codebuild.MacBuildImage.fromDockerRegistry('myimage', { secretsManagerCredentials: secret }),
        fleet: fleet,
        computeType: codebuild.ComputeType.MEDIUM,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        RegistryCredential: {
          CredentialProvider: 'SECRETS_MANAGER',
          Credential: { 'Ref': 'SecretA720EF05' },
        },
      }),
    });
  });

  test('build image - can use imported secret by name', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const secret = secretsmanager.Secret.fromSecretNameV2(stack, 'Secret', 'MySecretName');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      environment: {
        buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('myimage', { secretsManagerCredentials: secret }),
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        RegistryCredential: {
          CredentialProvider: 'SECRETS_MANAGER',
          Credential: 'MySecretName',
        },
      }),
    });
  });

  test('build image - can use imported secret by name', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const secret = secretsmanager.Secret.fromSecretNameV2(stack, 'Secret', 'MySecretName');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.MEDIUM,
      environmentType: codebuild.EnvironmentType.MAC_ARM,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      environment: {
        buildImage: codebuild.MacBuildImage.fromDockerRegistry('myimage', { secretsManagerCredentials: secret }),
        fleet: fleet,
        computeType: codebuild.ComputeType.MEDIUM,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        RegistryCredential: {
          CredentialProvider: 'SECRETS_MANAGER',
          Credential: 'MySecretName',
        },
      }),
    });
  });

  test('logs config - cloudWatch', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const logGroup = logs.LogGroup.fromLogGroupName(stack, 'LogGroup', 'MyLogGroupName');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        cloudWatch: {
          logGroup,
          prefix: '/my-logs',
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        CloudWatchLogs: {
          GroupName: 'MyLogGroupName',
          Status: 'ENABLED',
          StreamName: '/my-logs',
        },
      }),
    });
  });

  test('logs config - cloudWatch disabled', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        cloudWatch: {
          enabled: false,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        CloudWatchLogs: {
          Status: 'DISABLED',
        },
      }),
    });
  });

  test('logs config - s3', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'LogBucket', 'mybucketname');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        s3: {
          bucket,
          prefix: 'my-logs',
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        S3Logs: {
          Location: 'mybucketname/my-logs',
          Status: 'ENABLED',
        },
      }),
    });
  });

  test('logs config - s3 with encrypted true sets EncryptionDisabled to false', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'LogBucket', 'mybucketname');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        s3: {
          bucket,
          encrypted: true,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        S3Logs: {
          Status: 'ENABLED',
          EncryptionDisabled: false,
        },
      }),
    });
  });

  test('logs config - s3 with encrypted false sets EncryptionDisabled to true', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'LogBucket', 'mybucketname');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        s3: {
          bucket,
          encrypted: false,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        S3Logs: {
          Status: 'ENABLED',
          EncryptionDisabled: true,
        },
      }),
    });
  });

  test('logs config - s3 with encrypted unset does not set EncryptionDisabled', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'LogBucket', 'mybucketname');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        s3: {
          bucket,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        S3Logs: {
          Status: 'ENABLED',
          EncryptionDisabled: Match.absent(),
        },
      }),
    });
  });

  test('logs config - cloudWatch and s3', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'LogBucket2', 'mybucketname');
    const logGroup = logs.LogGroup.fromLogGroupName(stack, 'LogGroup2', 'MyLogGroupName');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      logging: {
        cloudWatch: {
          logGroup,
        },
        s3: {
          bucket,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      LogsConfig: Match.objectLike({
        CloudWatchLogs: {
          GroupName: 'MyLogGroupName',
          Status: 'ENABLED',
        },
        S3Logs: {
          Location: 'mybucketname',
          Status: 'ENABLED',
        },
      }),
    });
  });

  test('certificate arn', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        certificate: {
          bucket,
          objectKey: 'path',
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        Certificate: {
          'Fn::Join': ['', [
            'arn:',
            { 'Ref': 'AWS::Partition' },
            ':s3:::my-bucket/path',
          ]],
        },
      }),
    });
  });

  test.each([
    ['Standard 7.0', codebuild.LinuxBuildImage.STANDARD_7_0, 'aws/codebuild/standard:7.0'],
    ['Standard 6.0', codebuild.LinuxBuildImage.STANDARD_6_0, 'aws/codebuild/standard:6.0'],
    ['Amazon Linux 4.0', codebuild.LinuxBuildImage.AMAZON_LINUX_2_4, 'aws/codebuild/amazonlinux2-x86_64-standard:4.0'],
    ['Amazon Linux 5.0', codebuild.LinuxBuildImage.AMAZON_LINUX_2_5, 'aws/codebuild/amazonlinux2-x86_64-standard:5.0'],
    ['Windows Server Core 2019 1.0', codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE, 'aws/codebuild/windows-base:2019-1.0'],
    ['Windows Server Core 2019 2.0', codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE_2_0, 'aws/codebuild/windows-base:2019-2.0'],
    ['Windows Server Core 2019 3.0', codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE_3_0, 'aws/codebuild/windows-base:2019-3.0'],
    ['Windows Server Core 2022 3.0', codebuild.WindowsBuildImage.WIN_SERVER_CORE_2022_BASE_3_0, 'aws/codebuild/windows-base:2022-1.0'],
  ])('has build image for %s', (_, buildImage, expected) => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        buildImage: buildImage,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        Image: expected,
      }),
    });
  });

  test.each([
    ['Base 14', codebuild.MacBuildImage.BASE_14, 'aws/codebuild/macos-arm-base:14'],
    ['Base 15', codebuild.MacBuildImage.BASE_15, 'aws/codebuild/macos-arm-base:15'],
    ['Base 26', codebuild.MacBuildImage.BASE_26, 'aws/codebuild/macos-arm-base:26'],
  ])('has build image for %s', (_, buildImage, expected) => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.MEDIUM,
      environmentType: codebuild.EnvironmentType.MAC_ARM,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        buildImage: buildImage,
        fleet: fleet,
        computeType: codebuild.ComputeType.MEDIUM,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        Image: expected,
      }),
    });
  });

  test('can set fleet', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.SMALL,
      environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        fleet,
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        ComputeType: 'BUILD_GENERAL1_SMALL',
        Image: 'aws/codebuild/standard:7.0',
        Type: 'LINUX_CONTAINER',
        Fleet: {
          FleetArn: { 'Fn::GetAtt': ['Fleet30813DF3', 'Arn'] },
        },
      }),
    });
  });

  test.each([
    ['BASE_14', codebuild.MacBuildImage.BASE_14, 'aws/codebuild/macos-arm-base:14'],
    ['BASE_15', codebuild.MacBuildImage.BASE_15, 'aws/codebuild/macos-arm-base:15'],
    ['BASE_26', codebuild.MacBuildImage.BASE_26, 'aws/codebuild/macos-arm-base:26'],
  ])('can set macOS fleet with %s', (_, buildImage, expectedImage) => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.MEDIUM,
      environmentType: codebuild.EnvironmentType.MAC_ARM,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        fleet,
        buildImage,
        computeType: codebuild.ComputeType.MEDIUM,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        ComputeType: 'BUILD_GENERAL1_MEDIUM',
        Image: expectedImage,
        Type: 'MAC_ARM',
        Fleet: {
          FleetArn: { 'Fn::GetAtt': ['Fleet30813DF3', 'Arn'] },
        },
      }),
    });
  });

  test('can set imported fleet', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');
    const fleet = codebuild.Fleet.fromFleetArn(stack, 'Fleet', 'arn:aws:codebuild:us-east-1:123456789012:fleet/MyFleet:uuid');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        fleet,
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        ComputeType: 'BUILD_GENERAL1_SMALL',
        Image: 'aws/codebuild/standard:7.0',
        Type: 'LINUX_CONTAINER',
        Fleet: {
          FleetArn: 'arn:aws:codebuild:us-east-1:123456789012:fleet/MyFleet:uuid',
        },
      }),
    });
  });

  test.each([
    ['BASE_14', codebuild.MacBuildImage.BASE_14, 'aws/codebuild/macos-arm-base:14'],
    ['BASE_15', codebuild.MacBuildImage.BASE_15, 'aws/codebuild/macos-arm-base:15'],
    ['BASE_26', codebuild.MacBuildImage.BASE_26, 'aws/codebuild/macos-arm-base:26'],
  ])('can set imported macOS fleet with %s', (_, buildImage, expectedImage) => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket');
    const fleet = codebuild.Fleet.fromFleetArn(stack, 'Fleet', 'arn:aws:codebuild:us-east-1:123456789012:fleet/MyFleet:uuid');

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        fleet,
        buildImage,
        computeType: codebuild.ComputeType.MEDIUM,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        ComputeType: 'BUILD_GENERAL1_MEDIUM',
        Image: expectedImage,
        Type: 'MAC_ARM',
        Fleet: {
          FleetArn: 'arn:aws:codebuild:us-east-1:123456789012:fleet/MyFleet:uuid',
        },
      }),
    });
  });

  test('Can use Windows 2022 build image with a fleet', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.MEDIUM,
      environmentType: codebuild.EnvironmentType.WINDOWS_SERVER_2022_CONTAINER,
    });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket,
        path: 'path',
      }),
      environment: {
        fleet,
        buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2022_BASE_3_0,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        ComputeType: 'BUILD_GENERAL1_MEDIUM',
        Image: 'aws/codebuild/windows-base:2022-1.0',
        Type: 'WINDOWS_SERVER_2022_CONTAINER',
        Fleet: {
          FleetArn: { 'Fn::GetAtt': ['Fleet30813DF3', 'Arn'] },
        },
      }),
    });
  });

  test('throws when fleet environmentType does not match the buildImage', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'my-bucket'); // (stack, 'Bucket');
    const fleet = new codebuild.Fleet(stack, 'Fleet', {
      fleetName: 'MyFleet',
      baseCapacity: 1,
      computeType: codebuild.FleetComputeType.SMALL,
      environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
    });

    // THEN
    expect(() => {
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.s3({
          bucket,
          path: 'path',
        }),
        environment: {
          fleet,
          buildImage: codebuild.WindowsBuildImage.WIN_SERVER_CORE_2019_BASE_2_0,
        },
      });
    }).toThrow('The environment type of the fleet (LINUX_CONTAINER) must match the environment type of the build image (WINDOWS_SERVER_2019_CONTAINER)');
  });

  test('can enable docker server by setting docker server compute type', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc });

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      environment: {
        dockerServer: {
          computeType: codebuild.DockerServerComputeType.SMALL,
          securityGroups: [securityGroup],
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: Match.objectLike({
        DockerServer: {
          ComputeType: 'BUILD_GENERAL1_SMALL',
          SecurityGroupIds: [{
            'Fn::GetAtt': ['SecurityGroupDD263621', 'GroupId'],
          }],
        },
      }),
    });
  });
});

describe('EnvironmentVariables', () => {
  describe('from SSM', () => {
    test('can use environment variables', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.s3({
          bucket: new s3.Bucket(stack, 'Bucket'),
          path: 'path',
        }),
        environment: {
          buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('myimage'),
        },
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.PARAMETER_STORE,
            value: '/params/param1',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        Environment: Match.objectLike({
          EnvironmentVariables: [{
            Name: 'ENV_VAR1',
            Type: 'PARAMETER_STORE',
            Value: '/params/param1',
          }],
        }),
      });
    });

    test('can use environment variables', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fleet = new codebuild.Fleet(stack, 'Fleet', {
        fleetName: 'MyFleet',
        baseCapacity: 1,
        computeType: codebuild.FleetComputeType.MEDIUM,
        environmentType: codebuild.EnvironmentType.MAC_ARM,
      });

      // WHEN
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.s3({
          bucket: new s3.Bucket(stack, 'Bucket'),
          path: 'path',
        }),
        environment: {
          buildImage: codebuild.MacBuildImage.fromDockerRegistry('myimage'),
          computeType: codebuild.ComputeType.MEDIUM,
          fleet: fleet,
        },
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.PARAMETER_STORE,
            value: '/params/param1',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        Environment: Match.objectLike({
          EnvironmentVariables: [{
            Name: 'ENV_VAR1',
            Type: 'PARAMETER_STORE',
            Value: '/params/param1',
          }],
        }),
      });
    });

    test('grants the correct read permissions', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.s3({
          bucket: new s3.Bucket(stack, 'Bucket'),
          path: 'path',
        }),
        environment: {
          buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('myimage'),
        },
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.PARAMETER_STORE,
            value: '/params/param1',
          },
          'ENV_VAR2': {
            type: codebuild.BuildEnvironmentVariableType.PARAMETER_STORE,
            value: 'params/param2',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([Match.objectLike({
            'Action': 'ssm:GetParameters',
            'Effect': 'Allow',
            'Resource': [{
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':ssm:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':parameter/params/param1',
                ],
              ],
            },
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':ssm:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':parameter/params/param2',
                ],
              ],
            }],
          })]),
        },
      });
    });

    test('does not grant read permissions when variables are not from parameter store', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.Project(stack, 'Project', {
        source: codebuild.Source.s3({
          bucket: new s3.Bucket(stack, 'Bucket'),
          path: 'path',
        }),
        environment: {
          buildImage: codebuild.LinuxBuildImage.fromDockerRegistry('myimage'),
        },
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.PLAINTEXT,
            value: 'var1-value',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', Match.not({
        'PolicyDocument': {
          'Statement': Match.arrayWith([Match.objectLike({
            'Action': 'ssm:GetParameters',
            'Effect': 'Allow',
          })]),
        },
      }));
    });
  });

  describe('from SecretsManager', () => {
    test('can be provided as a verbatim secret name', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'my-secret',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'my-secret',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':secretsmanager:',
                { Ref: 'AWS::Region' },
                ':',
                { Ref: 'AWS::AccountId' },
                ':secret:my-secret-??????',
              ]],
            },
          }]),
        },
      });
    });

    test('can be provided as a verbatim secret name followed by a JSON key', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'my-secret:json-key',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'my-secret:json-key',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':secretsmanager:',
                { Ref: 'AWS::Region' },
                ':',
                { Ref: 'AWS::AccountId' },
                ':secret:my-secret-??????',
              ]],
            },
          }]),
        },
      });
    });

    test('can be provided as a verbatim full secret ARN followed by a JSON key', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'arn:aws:secretsmanager:us-west-2:123456789012:secret:my-secret-123456:json-key',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:my-secret-123456:json-key',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:my-secret-123456*',
          }]),
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', Match.not({
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': 'arn:aws:kms:us-west-2:123456789012:key/*',
          }]),
        },
      }));
    });

    test('can be provided as a verbatim partial secret ARN', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret*',
          }]),
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', Match.not({
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': 'arn:aws:kms:us-west-2:123456789012:key/*',
          }]),
        },
      }));
    });

    test("when provided as a verbatim partial secret ARN from another account, adds permission to decrypt keys in the Secret's account", () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'ProjectStack', {
        env: { account: '123456789012' },
      });

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'arn:aws:secretsmanager:us-west-2:901234567890:secret:mysecret',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': 'arn:aws:kms:us-west-2:901234567890:key/*',
          }]),
        },
      });
    });

    test('when two secrets from another account are provided as verbatim partial secret ARNs, adds only one permission for decrypting', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'ProjectStack', {
        env: { account: '123456789012' },
      });

      // WHEN
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'arn:aws:secretsmanager:us-west-2:901234567890:secret:mysecret',
          },
          'ENV_VAR2': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: 'arn:aws:secretsmanager:us-west-2:901234567890:secret:other-secret',
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': 'arn:aws:kms:us-west-2:901234567890:key/*',
          }]),
        },
      });
    });

    test('can be provided as the ARN attribute of a new Secret', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = new secretsmanager.Secret(stack, 'Secret');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: secret.secretArn,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': { 'Ref': 'SecretA720EF05' },
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': { 'Ref': 'SecretA720EF05' },
          }]),
        },
      });
    });

    test('when the same new secret is provided with different JSON keys, only adds the resource once', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = new secretsmanager.Secret(stack, 'Secret');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key1`,
          },
          'ENV_VAR2': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key2`,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': { 'Fn::Join': ['', [{ 'Ref': 'SecretA720EF05' }, ':json-key1']] },
            },
            {
              'Name': 'ENV_VAR2',
              'Type': 'SECRETS_MANAGER',
              'Value': { 'Fn::Join': ['', [{ 'Ref': 'SecretA720EF05' }, ':json-key2']] },
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': { 'Ref': 'SecretA720EF05' },
          }]),
        },
      });
    });

    test('can be provided as the ARN attribute of a new Secret, followed by a JSON key', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = new secretsmanager.Secret(stack, 'Secret');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key:version-stage`,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': {
                'Fn::Join': ['', [
                  { 'Ref': 'SecretA720EF05' },
                  ':json-key:version-stage',
                ]],
              },
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': { 'Ref': 'SecretA720EF05' },
          }]),
        },
      });
    });

    test('can be provided as the name attribute of a Secret imported by name', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = secretsmanager.Secret.fromSecretNameV2(stack, 'Secret', 'mysecret');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: secret.secretName,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'mysecret',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':secretsmanager:',
                { 'Ref': 'AWS::Region' },
                ':',
                { 'Ref': 'AWS::AccountId' },
                ':secret:mysecret-??????',
              ]],
            },
          }]),
        },
      });
    });

    test('can be provided as the ARN attribute of a Secret imported by partial ARN, followed by a JSON key', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = secretsmanager.Secret.fromSecretPartialArn(stack, 'Secret',
        'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key`,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret:json-key',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret*',
          }]),
        },
      });
    });

    test('can be provided as the ARN attribute of a Secret imported by complete ARN, followed by a JSON key', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const secret = secretsmanager.Secret.fromSecretCompleteArn(stack, 'Secret',
        'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret-123456');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key`,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret-123456:json-key',
            },
          ],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mysecret-123456*',
          }]),
        },
      });
    });

    test('can be provided as a SecretArn of a new Secret, with its physical name set, created in a different account', () => {
      // GIVEN
      const app = new cdk.App();
      const secretStack = new cdk.Stack(app, 'SecretStack', {
        env: { account: '012345678912' },
      });
      const stack = new cdk.Stack(app, 'ProjectStack', {
        env: { account: '123456789012' },
      });

      // WHEN
      const secret = new secretsmanager.Secret(secretStack, 'Secret', { secretName: 'secret-name' });
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: secret.secretArn,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': {
                'Fn::Join': ['', [
                  'arn:',
                  { 'Ref': 'AWS::Partition' },
                  ':secretsmanager:',
                  { 'Ref': 'AWS::Region' },
                  ':012345678912:secret:secret-name',
                ]],
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':secretsmanager:',
                { 'Ref': 'AWS::Region' },
                ':012345678912:secret:secret-name-??????',
              ]],
            },
          }]),
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':kms:',
                { 'Ref': 'AWS::Region' },
                ':012345678912:key/*',
              ]],
            },
          }]),
        },
      });
    });

    test('can be provided as a SecretArn of a Secret imported by name in a different account', () => {
      // GIVEN
      const app = new cdk.App();
      const secretStack = new cdk.Stack(app, 'SecretStack', {
        env: { account: '012345678912' },
      });
      const stack = new cdk.Stack(app, 'ProjectStack', {
        env: { account: '123456789012' },
      });

      // WHEN
      const secret = secretsmanager.Secret.fromSecretNameV2(secretStack, 'Secret', 'secret-name');
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: `${secret.secretArn}:json-key`,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': {
                'Fn::Join': ['', [
                  'arn:',
                  { 'Ref': 'AWS::Partition' },
                  ':secretsmanager:',
                  { 'Ref': 'AWS::Region' },
                  ':012345678912:secret:secret-name:json-key',
                ]],
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':secretsmanager:',
                { 'Ref': 'AWS::Region' },
                ':012345678912:secret:secret-name*',
              ]],
            },
          }]),
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':kms:',
                { 'Ref': 'AWS::Region' },
                ':012345678912:key/*',
              ]],
            },
          }]),
        },
      });
    });

    test('can be provided as a SecretArn of a Secret imported by complete ARN from a different account', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'ProjectStack', {
        env: { account: '123456789012' },
      });
      const secretArn = 'arn:aws:secretsmanager:us-west-2:901234567890:secret:mysecret-123456';

      // WHEN
      const secret = secretsmanager.Secret.fromSecretCompleteArn(stack, 'Secret', secretArn);
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'ENV_VAR1': {
            type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
            value: secret.secretArn,
          },
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
        'Environment': {
          'EnvironmentVariables': [
            {
              'Name': 'ENV_VAR1',
              'Type': 'SECRETS_MANAGER',
              'Value': secretArn,
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'secretsmanager:GetSecretValue',
            'Effect': 'Allow',
            'Resource': `${secretArn}*`,
          }]),
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([{
            'Action': 'kms:Decrypt',
            'Effect': 'Allow',
            'Resource': 'arn:aws:kms:us-west-2:901234567890:key/*',
          }]),
        },
      });
    });

    test('should fail when the parsed Arn does not contain a secret name', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      expect(() => {
        new codebuild.PipelineProject(stack, 'Project', {
          environmentVariables: {
            'ENV_VAR1': {
              type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER,
              value: 'arn:aws:secretsmanager:us-west-2:123456789012:secret',
            },
          },
        });
      }).toThrow(/SecretManager ARN is missing the name of the secret:/);
    });
  });

  test('should fail creating when using a secret value in a plaintext variable', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // THEN
    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'a': {
            value: `a_${cdk.SecretValue.secretsManager('my-secret')}_b`,
          },
        },
      });
    }).toThrow(/Plaintext environment variable 'a' contains a secret value!/);
  });

  test("should allow opting out of the 'secret value in a plaintext variable' validation", () => {
    // GIVEN
    const stack = new cdk.Stack();

    // THEN
    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        environmentVariables: {
          'b': {
            value: cdk.SecretValue.secretsManager('my-secret'),
          },
        },
        checkSecretsInPlainTextEnvVariables: false,
      });
    }).not.toThrow();
  });
});

describe('Timeouts', () => {
  test('can add queued timeout', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      queuedTimeout: cdk.Duration.minutes(30),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      QueuedTimeoutInMinutes: 30,
    });
  });

  test('can override build timeout', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      timeout: cdk.Duration.minutes(30),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      TimeoutInMinutes: 30,
    });
  });
});

describe('Maximum concurrency', () => {
  test('can limit maximum concurrency', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      concurrentBuildLimit: 1,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      ConcurrentBuildLimit: 1,
    });
  });
});

test('can automatically add ssm permissions', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'Bucket'),
      path: 'path',
    }),
    ssmSessionPermissions: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: Match.arrayWith([
            'ssmmessages:CreateControlChannel',
            'ssmmessages:CreateDataChannel',
          ]),
        }),
      ]),
    },
  });
});

test('can Setting Visibility', () => {
  // GIVEN
  const stackPublic = new cdk.Stack();
  const stackPrivate = new cdk.Stack();

  // WHEN
  new codebuild.Project(stackPublic, 'ProjectPublic', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stackPublic, 'Bucket-ProjectPublic'),
      path: 'path',
    }),
    visibility: codebuild.ProjectVisibility.PUBLIC_READ,
  });
  new codebuild.Project(stackPrivate, 'ProjectPrivate', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stackPrivate, 'Bucket-ProjectPrivate'),
      path: 'path',
    }),
    visibility: codebuild.ProjectVisibility.PRIVATE,
  });

  // THEN
  Template.fromStack(stackPublic).hasResourceProperties('AWS::CodeBuild::Project', {
    Visibility: 'PUBLIC_READ',
  });
  Template.fromStack(stackPrivate).hasResourceProperties('AWS::CodeBuild::Project', {
    Visibility: 'PRIVATE',
  });
});

describe('can be imported', () => {
  test('by ARN', () => {
    const stack = new cdk.Stack();
    const project = codebuild.Project.fromProjectArn(stack, 'Project',
      'arn:aws:codebuild:us-west-2:123456789012:project/My-Project');

    expect(project.projectName).toEqual('My-Project');
    expect(project.env.account).toEqual('123456789012');
    expect(project.env.region).toEqual('us-west-2');
  });
});

test('can set autoRetryLimit', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.s3({
      bucket: new s3.Bucket(stack, 'Bucket'),
      path: 'path',
    }),
    buildSpec: codebuild.BuildSpec.fromSourceFilename('hello.yml'),
    autoRetryLimit: 2,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
    AutoRetryLimit: 2,
  });
});

test.each([-1, 15])('throws when autoRetryLimit is invalid', (autoRetryLimit) => {
  const stack = new cdk.Stack();

  expect(() => {
    new codebuild.Project(stack, 'Project', {
      source: codebuild.Source.s3({
        bucket: new s3.Bucket(stack, 'Bucket'),
        path: 'path',
      }),
      buildSpec: codebuild.BuildSpec.fromSourceFilename('hello.yml'),
      autoRetryLimit,
    });
  }).toThrow(`autoRetryLimit must be a value between 0 and 10, got ${autoRetryLimit}.`);
});
