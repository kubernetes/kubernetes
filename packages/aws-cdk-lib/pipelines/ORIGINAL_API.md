# CDK Pipelines, original API

This document describes the API the CDK Pipelines library originally went into
Developer Preview with. The API has since been reworked, but the original one
left in place because of popular adoption. The original API still works and is
still supported, but the revised one is preferred for future projects as it
is more flexible and abstracts more unnecessary details from the user.

## Migrating from the original to the modern API

It's possible to migrate a pipeline in-place from the original to the modern API.
The changes necessary are the following:

### The Pipeline

Replace `new CdkPipeline` with `new CodePipeline`. Some
configuration properties have been changed:

| Old API                        | New API                                                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `cloudAssemblyArtifact`        | removed                                                                                        |
| `sourceAction`                 | removed                                                                                        |
| `synthAction`                  | `synth`                                                                                        |
| `crossAccountKeys`             | new default is `false`; specify `crossAccountKeys: true` if you need cross-account deployments |
| `cdkCliVersion`                | `cliVersion`                                                                                   |
| `selfMutating`                 | `selfMutation`                                                                                 |
| `vpc`, `subnetSelection`       | `codeBuildDefaults.vpc`, `codeBuildDefaults.subnetSelection`                                   |
| `selfMutationBuildSpec`        | `selfMutationCodeBuildDefaults.partialBuildSpec`                                               |
| `assetBuildSpec`               | `assetPublishingCodeBuildDefaults.partialBuildSpec`                                            |
| `assetPreinstallCommands`      | use `assetPublishingCodeBuildDefaults.partialBuildSpec` instead                                |
| `singlePublisherPerType: true` | `publishAssetsInParallel: false`                                                               |
| `supportDockerAssets`          | `dockerEnabledForSelfMutation`                                                                 |

### The synth

As the argument to `synth`, use `new ShellStep` or `new CodeBuildStep`,
depending on whether or not you want to customize the AWS CodeBuild Project that gets generated.

Contrary to `SimpleSynthAction.standardNpmSynth`, you need to specify
all commands necessary to do a full CDK build and synth, so do include
installing dependencies and running the CDK CLI. For example, the old API:

```ts
const sourceArtifact = new codepipeline.Artifact();
const cloudAssemblyArtifact = new codepipeline.Artifact();
pipelines.SimpleSynthAction.standardNpmSynth({
  sourceArtifact,
  cloudAssemblyArtifact,

  // Use this if you need a build step (if you're not using ts-node
  // or if you have TypeScript Lambdas that need to be compiled).
  buildCommand: 'npm run build',
}),
```

Becomes:

```ts
new pipelines.ShellStep('Synth', {
  input: pipelines.CodePipelineSource.connection('my-org/my-app', 'main', {
    connectionArn: 'arn:aws:codestar-connections:us-east-1:222222222222:connection/7d2469ff-514a-4e4f-9003-5ca4a43cdc41', // Created using the AWS console * });',
  }),
  commands: [
    'npm ci',
    'npm run build',
    'npx cdk synth',
  ],
});
```

Instead of specifying the pipeline source with the `sourceAction` property to
the pipeline, specify it as the `input` property to the `ShellStep` instead.
You can use any of the factory functions on `CodePipelineSource`.

For example, for a GitHub source, the following old API:

```ts
sourceAction: new cpactions.GitHubSourceAction({
  actionName: 'GitHub',
  output: sourceArtifact,
  // Replace these with your actual GitHub project name
  owner: 'OWNER',
  repo: 'REPO',
  branch: 'main', // default: 'master'
}),
```

Translates into:

```ts
input: pipelines.CodePipelineSource.gitHub('OWNER/REPO', 'main', {
  authentication: cdk.SecretValue.secretsManager('GITHUB_TOKEN_NAME'),
}),
```

### Deployments

Adding CDK Stages to deploy is done by calling `addStage()`, or
potentially `addWave().addStage()`. All stages inside a wave are
deployed in parallel, which was not a capability of the original API.

| Old API                       | New API                                                                                                                       |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `addApplicationStage()`       | `addStage()`                                                                                                                  |
| `addStage().addApplication()` | `addStage()`. Adding multiple CDK Stages into a single Pipeline stage is not supported, add multiple Pipeline stages instead. |

### Approvals

Approvals are added by adding `pre` and `post` options to `addStage()`, with
steps to execute before and after the deployments, respectively. We recommend
putting manual approvals in `pre` steps, and automated approvals in `post` steps.

#### Manual approvals

For example, specifying a manual approval on a stage deployment in old API:

```ts
declare const pipeline: pipelines.CdkPipeline;
const stage = pipeline.addApplicationStage(...);
stage.addAction(new pipelines.ManualApprovalAction({
  actionName: 'ManualApproval',
  runOrder: testingStage.nextSequentialRunOrder(),
}));
```

Becomes:

```ts
const stage = new MyApplicationStage(this, 'MyApplication');
pipeline.addStage(stage, {
  pre: [
    new pipelines.ManualApprovalStep('ManualApproval'),
  ],
});
```

Note that this we've used `pre` to put the manual approval *before* a Stage
deployment (this was not possible in the old API). Be sure to put the manual
approval in the `pre` steps list of the *next* Stage to keep
it in the same location in the pipeline.

#### Automated approvals

For example, specifying an automated approval after a stage is deployed in the following old API:

```ts
const stage = pipeline.addApplicationStage(...);
stage.addActions(new pipelines.ShellScriptAction({
  actionName: 'MyValidation',
  commands: ['curl -Ssf $VAR'],
  useOutputs: {
    VAR: pipeline.stackOutput(stage.cfnOutput),
  },
  // Optionally specify a BuildEnvironment
  environment: { ... },
}));
```

Becomes:

```ts
const stage = new MyApplicationStage(this, 'MyApplication');
pipeline.addStage(stage, {
  post: [
    new pipelines.CodeBuildStep('MyValidation', {
      commands: ['curl -Ssf $VAR'],
      envFromCfnOutput: {
        VAR: stage.cfnOutput,
      },
      // Optionally specify a BuildEnvironment
      buildEnvironment: { ... },
    }),
  ],
});
```

You can also use `ShellStep` if you don't need any of the CodeBuild Project
customizations (like `buildEnvironment`).

#### Change set approvals

In the old API, there were two properties that were used to add actions to the pipeline
in between the `CreateChangeSet` and `ExecuteChangeSet` actions: `manualApprovals` and `extraRunOrderSpace`.
This can be achieved in the modern API via the `stackSteps` property, which allows steps to be added
at the stack level:

```ts
const stage = new MyApplicationStage(this, 'MyApplication');
pipeline.addStage(stage, {
  stackSteps: [{
    stack: stage.stack1,
    changeSet: [new pipelines.ManualApprovalStep('ChangeSet Approval')],
  }],
});
```

### Custom CodePipeline Actions

See the section [**Arbitrary CodePipeline actions** in the
main `README`](https://github.com/aws/aws-cdk/blob/main/packages/@aws-cdk/pipelines/README.md#arbitrary-codepipeline-actions) for an example of how to inject arbitrary
CodeBuild Actions.

## Defining the pipeline

In the original API, you have to import the `aws-codepipeline` construct
library and create `Artifact` objects for the source and Cloud Assembly
artifacts:

```ts
import { Construct, Stage, Stack, StackProps, StageProps } from 'aws-cdk-lib';
import * as codepipeline from 'aws-cdk-lib/aws-codepipeline';

/**
 * Stack to hold the pipeline
 */
class MyPipelineStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const sourceArtifact = new codepipeline.Artifact();
    const cloudAssemblyArtifact = new codepipeline.Artifact();

    const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
      cloudAssemblyArtifact,

      sourceAction: new cpactions.GitHubSourceAction({
        actionName: 'GitHub',
        output: sourceArtifact,
        oauthToken: cdk.SecretValue.secretsManager('GITHUB_TOKEN_NAME'),
        // Replace these with your actual GitHub project name
        owner: 'OWNER',
        repo: 'REPO',
        branch: 'main', // default: 'master'
      }),

      synthAction: pipelines.SimpleSynthAction.standardNpmSynth({
        sourceArtifact,
        cloudAssemblyArtifact,

        // Use this if you need a build step (if you're not using ts-node
        // or if you have TypeScript Lambdas that need to be compiled).
        buildCommand: 'npm run build',
      }),
    });

    // Do this as many times as necessary with any account and region
    // Account and region may different from the pipeline's.
    pipeline.addApplicationStage(new MyApplication(this, 'Prod', {
      env: {
        account: '123456789012',
        region: 'eu-west-1',
      }
    }));
  }
}
```

### A note on cost

By default, the `CdkPipeline` construct creates an AWS Key Management Service
(AWS KMS) Customer Master Key (CMK) for you to encrypt the artifacts in the
artifact bucket, which incurs a cost of
**$1/month**. This default configuration is necessary to allow cross-account
deployments.

If you do not intend to perform cross-account deployments, you can disable
the creation of the Customer Master Keys by passing `crossAccountKeys: false`
when defining the Pipeline:

```ts
const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
  crossAccountKeys: false,

  // ...
});
```

### Defining the Pipeline (Source and Synth)

The pipeline is defined by instantiating `CdkPipeline` in a Stack. This defines the
source location for the pipeline as well as the build commands. For example, the following
defines a pipeline whose source is stored in a GitHub repository, and uses NPM
to build. The Pipeline will be provisioned in account `111111111111` and region
`eu-west-1`:

```ts
class MyPipelineStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const sourceArtifact = new codepipeline.Artifact();
    const cloudAssemblyArtifact = new codepipeline.Artifact();

    const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
      pipelineName: 'MyAppPipeline',
      cloudAssemblyArtifact,

      sourceAction: new cpactions.GitHubSourceAction({
        actionName: 'GitHub',
        output: sourceArtifact,
        oauthToken: cdk.SecretValue.secretsManager('GITHUB_TOKEN_NAME'),
        // Replace these with your actual GitHub project name
        owner: 'OWNER',
        repo: 'REPO',
        branch: 'main', // default: 'master'
      }),

      synthAction: pipelines.SimpleSynthAction.standardNpmSynth({
        sourceArtifact,
        cloudAssemblyArtifact,

        // Optionally specify a VPC in which the action runs
        vpc: new ec2.Vpc(this, 'NpmSynthVpc'),

        // Use this if you need a build step (if you're not using ts-node
        // or if you have TypeScript Lambdas that need to be compiled).
        buildCommand: 'npm run build',
      }),
    });
  }
}

const app = new App();
new MyPipelineStack(app, 'PipelineStack', {
  env: {
    account: '111111111111',
    region: 'eu-west-1',
  }
});
```

If you prefer more control over the underlying CodePipeline object, you can
create one yourself, including custom Source and Build stages:

```ts
const codePipeline = new codepipeline.Pipeline(pipelineStack, 'CodePipeline', {
  stages: [
    {
      stageName: 'CustomSource',
      actions: [...],
    },
    {
      stageName: 'CustomBuild',
      actions: [...],
    },
  ],
});

const app = new App();
const cdkPipeline = new pipelines.CdkPipeline(app, 'CdkPipeline', {
  codePipeline,
  cloudAssemblyArtifact,
});
```

If you use assets for files or Docker images, every asset will get its own upload action during the asset stage.
By setting the value `singlePublisherPerType` to `true`, only one action for files and one action for
Docker images is created that handles all assets of the respective type.

If you need to run commands to setup proxies, mirrors, etc you can supply them using the `assetPreInstallCommands`.

#### Sources

Any of the regular sources from the [`aws-cdk-lib/aws-codepipeline-actions`](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-codepipeline-actions-readme.html#github) module can be used.

#### Synths

You define how to build and synth the project by specifying a `synthAction`.
This can be any CodePipeline action that produces an artifact with a CDK
Cloud Assembly in it (the contents of the `cdk.out` directory created when
`cdk synth` is called). Pass the output artifact of the synth in the
Pipeline's `cloudAssemblyArtifact` property.

`SimpleSynthAction` is available for synths that can be performed by running a couple
of simple shell commands (install, build, and synth) using AWS CodeBuild. When
using these, the source repository does not need to have a `buildspec.yml`. An example
of using `SimpleSynthAction` to run a Maven build followed by a CDK synth:

```ts
const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
  // ...
  synthAction: new pipelines.SimpleSynthAction({
    sourceArtifact,
    cloudAssemblyArtifact,
    installCommands: ['npm install -g aws-cdk'],
    buildCommands: ['mvn package'],
    synthCommand: 'cdk synth',
  })
});
```

Available as factory functions on `SimpleSynthAction` are some common
convention-based synth:

* `SimpleSynthAction.standardNpmSynth()`: build using NPM conventions. Expects a `package-lock.json`,
  a `cdk.json`, and expects the CLI to be a versioned dependency in `package.json`. Does
  not perform a build step by default.
* `CdkSynth.standardYarnSynth()`: build using Yarn conventions. Expects a `yarn.lock`
  a `cdk.json`, and expects the CLI to be a versioned dependency in `package.json`. Does
  not perform a build step by default.

If you need a custom build/synth step that is not covered by `SimpleSynthAction`, you can
always add a custom CodeBuild project and pass a corresponding `CodeBuildAction` to the
pipeline.

#### Add Additional permissions to the CodeBuild Project Role for building and synthesizing

You can customize the role permissions used by the CodeBuild project so it has access to
the needed resources. eg: Adding CodeArtifact repo permissions so we pull npm packages
from the CA repo instead of NPM.

```ts
class MyPipelineStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    ...
    const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
      ...
      synthAction: pipelines.SimpleSynthAction.standardNpmSynth({
        sourceArtifact,
        cloudAssemblyArtifact,

        // Use this to customize and a permissions required for the build
        // and synth
        rolePolicyStatements: [
          new iam.PolicyStatement({
            actions: ['codeartifact:*', 'sts:GetServiceBearerToken'],
            resources: ['arn:codeartifact:repo:arn'],
          }),
        ],

        // Then you can login to codeartifact repository
        // and npm will now pull packages from your repository
        // Note the codeartifact login command requires more params to work.
        buildCommands: [
          'aws codeartifact login --tool npm',
          'npm run build',
        ],
      }),
    });
  }
}
```

### Adding Application Stages

To define an application that can be added to the pipeline integrally, define a subclass
of `Stage`. The `Stage` can contain one or more stack which make up your application. If
there are dependencies between the stacks, the stacks will automatically be added to the
pipeline in the right order. Stacks that don't depend on each other will be deployed in
parallel. You can add a dependency relationship between stacks by calling
`stack1.addStackDependency(stack2)`.

Stages take a default `env` argument which the Stacks inside the Stage will fall back to
if no `env` is defined for them.

An application is added to the pipeline by calling `addApplicationStage()` with instances
of the Stage. The same class can be instantiated and added to the pipeline multiple times
to define different stages of your DTAP or multi-region application pipeline:

```ts
// Testing stage
pipeline.addApplicationStage(new MyApplication(this, 'Testing', {
  env: { account: '111111111111', region: 'eu-west-1' }
}));

// Acceptance stage
pipeline.addApplicationStage(new MyApplication(this, 'Acceptance', {
  env: { account: '222222222222', region: 'eu-west-1' }
}));

// Production stage
pipeline.addApplicationStage(new MyApplication(this, 'Production', {
  env: { account: '333333333333', region: 'eu-west-1' }
}));
```

> Be aware that adding new stages via `addApplicationStage()` will
> automatically add them to the pipeline and deploy the new stacks, but
> *removing* them from the pipeline or deleting the pipeline stack will not
> automatically delete deployed application stacks. You must delete those
> stacks by hand using the AWS CloudFormation console or the AWS CLI.

### More Control

Every *Application Stage* added by `addApplicationStage()` will lead to the addition of
an individual *Pipeline Stage*, which is subsequently returned. You can add more
actions to the stage by calling `addAction()` on it. For example:

```ts
const testingStage = pipeline.addApplicationStage(new MyApplication(this, 'Testing', {
  env: { account: '111111111111', region: 'eu-west-1' }
}));

// Add a action -- in this case, a Manual Approval action
// (for illustration purposes: testingStage.addManualApprovalAction() is a
// convenience shorthand that does the same)
testingStage.addAction(new pipelines.ManualApprovalAction({
  actionName: 'ManualApproval',
  runOrder: testingStage.nextSequentialRunOrder(),
}));
```

You can also add more than one *Application Stage* to one *Pipeline Stage*. For example:

```ts
// Create an empty pipeline stage
const testingStage = pipeline.addStage('Testing');

// Add two application stages to the same pipeline stage
testingStage.addApplication(new MyApplication1(this, 'MyApp1', {
  env: { account: '111111111111', region: 'eu-west-1' }
}));
testingStage.addApplication(new MyApplication2(this, 'MyApp2', {
  env: { account: '111111111111', region: 'eu-west-1' }
}));
```

Even more, adding a manual approval action or reserving space for some extra sequential actions
between 'Prepare' and 'Execute' ChangeSet actions is possible.

```ts
  pipeline.addApplicationStage(new MyApplication(this, 'Production'), {
    manualApprovals: true,
    extraRunOrderSpace: 1,
  });
```

### Adding validations to the pipeline

You can add any type of CodePipeline Action to the pipeline in order to validate
the deployments you are performing.

The CDK Pipelines construct library comes with a `ShellScriptAction` which uses AWS CodeBuild
to run a set of shell commands (potentially running a test set that comes with your application,
using stack outputs of the deployed stacks).

In its simplest form, adding validation actions looks like this:

```ts
const stage = pipeline.addApplicationStage(new MyApplication(/* ... */));

stage.addActions(new pipelines.ShellScriptAction({
  actionName: 'MyValidation',
  commands: ['curl -Ssf https://my.webservice.com/'],
  // Optionally specify a VPC if, for example, the service is deployed with a private load balancer
  vpc,
  // Optionally specify SecurityGroups
  securityGroups,
  // Optionally specify a BuildEnvironment
  environment,
}));
```

#### Using CloudFormation Stack Outputs in ShellScriptAction

Because many CloudFormation deployments result in the generation of resources with unpredictable
names, validations have support for reading back CloudFormation Outputs after a deployment. This
makes it possible to pass (for example) the generated URL of a load balancer to the test set.

To use Stack Outputs, expose the `CfnOutput` object you're interested in, and
call `pipeline.stackOutput()` on it:

```ts
class MyLbApplication extends Stage {
  public readonly loadBalancerAddress: CfnOutput;

  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);

    const lbStack = new LoadBalancerStack(this, 'Stack');

    // Or create this in `LoadBalancerStack` directly
    this.loadBalancerAddress = new CfnOutput(lbStack, 'LbAddress', {
      value: `https://${lbStack.loadBalancer.loadBalancerDnsName}/`
    });
  }
}

const lbApp = new MyLbApplication(this, 'MyApp', {
  env: { /* ... */ }
});
const stage = pipeline.addApplicationStage(lbApp);
stage.addActions(new pipelines.ShellScriptAction({
  // ...
  useOutputs: {
    // When the test is executed, this will make $URL contain the
    // load balancer address.
    URL: pipeline.stackOutput(lbApp.loadBalancerAddress),
  }
});
```

#### Using additional files in Shell Script Actions

As part of a validation, you probably want to run a test suite that's more
elaborate than what can be expressed in a couple of lines of shell script.
You can bring additional files into the shell script validation by supplying
the `additionalArtifacts` property.

Here are some typical examples for how you might want to bring in additional
files from several sources:

* Directory from the source repository
* Additional compiled artifacts from the synth step

#### Controlling IAM permissions

IAM permissions can be added to the execution role of a `ShellScriptAction` in
two ways.

Either pass additional policy statements in the `rolePolicyStatements` property:

```ts
new pipelines.ShellScriptAction({
  // ...
  rolePolicyStatements: [
    new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: ['*'],
    }),
  ],
}));
```

The Action can also be used as a Grantable after having been added to a Pipeline:

```ts
const action = new pipelines.ShellScriptAction({ /* ... */ });
pipeline.addStage('Test').addActions(action);

bucket.grants.read(action);
```

#### Additional files from the source repository

Bringing in additional files from the source repository is appropriate if the
files in the source repository are directly usable in the test (for example,
if they are executable shell scripts themselves). Pass the `sourceArtifact`:

```ts
const sourceArtifact = new codepipeline.Artifact();

const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
  // ...
});

const validationAction = new pipelines.ShellScriptAction({
  actionName: 'TestUsingSourceArtifact',
  additionalArtifacts: [sourceArtifact],

  // 'test.sh' comes from the source repository
  commands: ['./test.sh'],
});
```

#### Additional files from the synth step

Getting the additional files from the synth step is appropriate if your
tests need the compilation step that is done as part of synthesis.

On the synthesis step, specify `additionalArtifacts` to package
additional subdirectories into artifacts, and use the same artifact
in the `ShellScriptAction`'s `additionalArtifacts`:

```ts
// If you are using additional output artifacts from the synth step,
// they must be named.
const cloudAssemblyArtifact = new codepipeline.Artifact('CloudAsm');
const integTestsArtifact = new codepipeline.Artifact('IntegTests');

const pipeline = new pipelines.CdkPipeline(this, 'Pipeline', {
  synthAction: pipelines.SimpleSynthAction.standardNpmSynth({
    sourceArtifact,
    cloudAssemblyArtifact,
    buildCommands: ['npm run build'],
    additionalArtifacts: [
      {
        directory: 'test',
        artifact: integTestsArtifact,
      }
    ],
  }),
  // ...
});

const validationAction = new pipelines.ShellScriptAction({
  actionName: 'TestUsingBuildArtifact',
  additionalArtifacts: [integTestsArtifact],
  // 'test.js' was produced from 'test/test.ts' during the synth step
  commands: ['node ./test.js'],
});
```

### Confirm permissions broadening

To keep tabs on the security impact of changes going out through your pipeline,
you can insert a security check before any stage deployment. This security check
will check if the upcoming deployment would add any new IAM permissions or
security group rules, and if so pause the pipeline and require you to confirm
the changes.

The security check will appear as two distinct actions in your pipeline: first
a CodeBuild project that runs `cdk diff` on the stage that's about to be deployed,
followed by a Manual Approval action that pauses the pipeline. If it so happens
that there no new IAM permissions or security group rules will be added by the deployment,
the manual approval step is automatically satisfied. The pipeline will look like this:

```txt
Pipeline
├── ...
├── MyApplicationStage
│    ├── MyApplicationSecurityCheck       // Security Diff Action
│    ├── MyApplicationManualApproval      // Manual Approval Action
│    ├── Stack.Prepare
│    └── Stack.Deploy
└── ...
```

You can enable the security check by passing `confirmBroadeningPermissions` to
`addApplicationStage`:

```ts
const stage = pipeline.addApplicationStage(new MyApplication(this, 'PreProd'), {
  confirmBroadeningPermissions: true,
});
```

To get notified when there is a change that needs your manual approval,
create an SNS Topic, subscribe your own email address, and pass it in via
`securityNotificationTopic`:

```ts
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';

const topic = new sns.Topic(this, 'SecurityChangesTopic');
topic.addSubscription(new subscriptions.EmailSubscription('test@email.com'));

const pipeline = new pipelines.CdkPipeline(app, 'Pipeline', { /* ... */ });
const stage = pipeline.addApplicationStage(new MyApplication(this, 'PreProd'), {
  confirmBroadeningPermissions: true,
  securityNotificationTopic: topic,
});
```

**Note**: Manual Approvals notifications only apply when an application has security
check enabled.
