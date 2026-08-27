# AWS CodeBuild Construct Library


AWS CodeBuild is a fully managed continuous integration service that compiles
source code, runs tests, and produces software packages that are ready to
deploy. With CodeBuild, you don’t need to provision, manage, and scale your own
build servers. CodeBuild scales continuously and processes multiple builds
concurrently, so your builds are not left waiting in a queue. You can get
started quickly by using prepackaged build environments, or you can create
custom build environments that use your own build tools. With CodeBuild, you are
charged by the minute for the compute resources you use.

## Source

Build projects are usually associated with a _source_, which is specified via
the `source` property which accepts a class that extends the `Source`
abstract base class.
The default is to have no source associated with the build project;
the `buildSpec` option is required in that case.

Here's a CodeBuild project with no source which simply prints `Hello,
CodeBuild!`:

[Minimal Example](./test/integ.defaults.lit.ts)

### `CodeCommitSource`

Use an AWS CodeCommit repository as the source of this build:

```ts
import * as codecommit from 'aws-cdk-lib/aws-codecommit';

const repository = new codecommit.Repository(this, 'MyRepo', { repositoryName: 'foo' });
new codebuild.Project(this, 'MyFirstCodeCommitProject', {
  source: codebuild.Source.codeCommit({ repository }),
});
```

### `S3Source`

Create a CodeBuild project with an S3 bucket as the source:

```ts
const bucket = new s3.Bucket(this, 'MyBucket');

new codebuild.Project(this, 'MyProject', {
  source: codebuild.Source.s3({
    bucket: bucket,
    path: 'path/to/file.zip',
  }),
});
```

The CodeBuild role will be granted to read just the given path from the given `bucket`.

### `GitHubSource` and `GitHubEnterpriseSource`

These source types can be used to build code from a GitHub repository.
Example:

```ts
const gitHubSource = codebuild.Source.gitHub({
  owner: 'awslabs',
  repo: 'aws-cdk', // optional, default: undefined if unspecified will create organization webhook
  webhook: true, // optional, default: true if `webhookFilters` were provided, false otherwise
  webhookTriggersBatchBuild: true, // optional, default is false
  webhookFilters: [
    codebuild.FilterGroup
      .inEventOf(codebuild.EventAction.PUSH)
      .andBranchIs('main')
      .andCommitMessageIs('the commit message'),
    codebuild.FilterGroup
      .inEventOf(codebuild.EventAction.RELEASED)
      .andBranchIs('main')
  ], // optional, by default all pushes and Pull Requests will trigger a build
});
```

The `GitHubSource` is also able to trigger all repos in GitHub Organizations
Example:
```ts
const gitHubSource = codebuild.Source.gitHub({
  owner: 'aws',
  webhookTriggersBatchBuild: true, // optional, default is false
  webhookFilters: [
    codebuild.FilterGroup
      .inEventOf(codebuild.EventAction.WORKFLOW_JOB_QUEUED)
      .andRepositoryNameIs("aws-.*")
      .andRepositoryNameIsNot("aws-cdk-lib"),
  ], // optional, by default all pushes and Pull Requests will trigger a build
});
```

To provide GitHub credentials, please either go to AWS CodeBuild Console to connect
or call `ImportSourceCredentials` to persist your personal access token.
Example:

```console
aws codebuild import-source-credentials --server-type GITHUB --auth-type PERSONAL_ACCESS_TOKEN --token <token_value>
```

### `BitBucketSource`

This source type can be used to build code from a BitBucket repository.

```ts
const bbSource = codebuild.Source.bitBucket({
  owner: 'owner',
  repo: 'repo',
});
```

### For all Git sources

For all Git sources, you can fetch submodules while cloning git repo.

```ts
const gitHubSource = codebuild.Source.gitHub({
  owner: 'awslabs',
  repo: 'aws-cdk',
  fetchSubmodules: true,
});
```

## BuildSpec

The build spec can be provided from a number of different sources

### File path relative to the root of the source

You can specify a specific filename that exists within the project's source artifact to use as the buildspec.

```ts
const project = new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromSourceFilename('my-buildspec.yml'),
  source: codebuild.Source.gitHub({
    owner: 'awslabs',
    repo: 'aws-cdk',
  })
});
```

This will use `my-buildspec.yml` file within the `awslabs/aws-cdk` repository as the build spec.

### File within the CDK project codebuild

You can also specify a file within your cdk project directory to use as the buildspec.

```ts
const project = new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromAsset('my-buildspec.yml'),
});
```

This file will be uploaded to S3 and referenced from the codebuild project.

### Inline object

```ts
const project = new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
  }),
});
```

This will result in the buildspec being rendered as JSON within the codebuild project, if you prefer it to be rendered as YAML, use `fromObjectToYaml`.

```ts
const project = new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromObjectToYaml({
    version: '0.2',
  }),
});
```

## Artifacts

CodeBuild Projects can produce Artifacts and upload them to S3. For example:

```ts
declare const bucket: s3.Bucket;

const project = new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
  }),
  artifacts: codebuild.Artifacts.s3({
      bucket,
      includeBuildId: false,
      packageZip: true,
      path: 'another/path',
      identifier: 'AddArtifact1',
    }),
});
```

Because we've not set the `name` property, this example will set the
`overrideArtifactName` parameter, and produce an artifact named as defined in
the Buildspec file, uploaded to an S3 bucket (`bucket`). The path will be
`another/path` and the artifact will be a zipfile.

## CodePipeline

To add a CodeBuild Project as an Action to CodePipeline,
use the `PipelineProject` class instead of `Project`.
It's a simple class that doesn't allow you to specify `sources`,
`secondarySources`, `artifacts` or `secondaryArtifacts`,
as these are handled by setting input and output CodePipeline `Artifact` instances on the Action,
instead of setting them on the Project.

```ts
const project = new codebuild.PipelineProject(this, 'Project', {
  // properties as above...
})
```

For more details, see the readme of the `@aws-cdk/aws-codepipeline-actions` package.

## Caching

You can save time when your project builds by using a cache. A cache can store reusable pieces of your build environment and use them across multiple builds. Your build project can use one of two types of caching: Amazon S3 or local. In general, S3 caching is a good option for small and intermediate build artifacts that are more expensive to build than to download. Local caching is a good option for large intermediate build artifacts because the cache is immediately available on the build host.

### S3 Caching

With S3 caching, the cache is stored in an S3 bucket which is available
regardless from what CodeBuild instance gets selected to run your CodeBuild job
on. When using S3 caching, you must also add in a `cache` section to your
buildspec which indicates the files to be cached:

```ts
declare const myCachingBucket: s3.Bucket;

new codebuild.Project(this, 'Project', {
  source: codebuild.Source.bitBucket({
    owner: 'awslabs',
    repo: 'aws-cdk',
  }),

  cache: codebuild.Cache.bucket(myCachingBucket),

  // BuildSpec with a 'cache' section necessary for S3 caching. This can
  // also come from 'buildspec.yml' in your source.
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: ['...'],
      },
    },
    cache: {
      paths: [
        // The '**/*' is required to indicate all files in this directory
        '/root/cachedir/**/*',
      ],
    },
  }),
});
```

If you want to [share the same cache between multiple projects](https://docs.aws.amazon.com/codebuild/latest/userguide/caching-s3.html#caching-s3-sharing), you must must do the following:

- Use the same `cacheNamespace`.
- Specify the same cache key.
- Define identical cache paths.
- Use the same Amazon S3 buckets and `pathPrefix` if set.

```ts
declare const sourceBucket: s3.Bucket;
declare const myCachingBucket: s3.Bucket;

new codebuild.Project(this, 'ProjectA', {
  source: codebuild.Source.s3({
    bucket: sourceBucket,
    path: 'path/to/source-a.zip',
  }),
  // configure the same bucket and path prefix
  cache: codebuild.Cache.bucket(myCachingBucket, {
    prefix: 'cache',
    // use the same cache namespace
    cacheNamespace: 'cache-namespace',
  }),
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: ['...'],
      },
    },
    // specify the same cache key and paths
    cache: {
      key: 'unique-key',
      paths: [
        '/root/cachedir/**/*',
      ],
    },
  }),
});

new codebuild.Project(this, 'ProjectB', {
  source: codebuild.Source.s3({
    bucket: sourceBucket,
    path: 'path/to/source-b.zip',
  }),
  // configure the same bucket and path prefix
  cache: codebuild.Cache.bucket(myCachingBucket, {
    prefix: 'cache',
    // use the same cache namespace
    cacheNamespace: 'cache-namespace',
  }),
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: ['...'],
      },
    },
    // specify the same cache key and paths
    cache: {
      key: 'unique-key',
      paths: [
        '/root/cachedir/**/*',
      ],
    },
  }),
});
```

### Local Caching

With local caching, the cache is stored on the codebuild instance itself. This
is simple, cheap and fast, but CodeBuild cannot guarantee a reuse of instance
and hence cannot guarantee cache hits. For example, when a build starts and
caches files locally, if two subsequent builds start at the same time afterwards
only one of those builds would get the cache. Three different cache modes are
supported, which can be turned on individually.

* `LocalCacheMode.SOURCE` caches Git metadata for primary and secondary sources.
* `LocalCacheMode.DOCKER_LAYER` caches existing Docker layers.
* `LocalCacheMode.CUSTOM` caches directories you specify in the buildspec file.

```ts
new codebuild.Project(this, 'Project', {
  source: codebuild.Source.gitHubEnterprise({
    httpsCloneUrl: 'https://my-github-enterprise.com/owner/repo',
  }),

  // Enable Docker AND custom caching
  cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER, codebuild.LocalCacheMode.CUSTOM),

  // BuildSpec with a 'cache' section necessary for 'CUSTOM' caching. This can
  // also come from 'buildspec.yml' in your source.
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: ['...'],
      },
    },
    cache: {
      paths: [
        // The '**/*' is required to indicate all files in this directory
        '/root/cachedir/**/*',
      ],
    },
  }),
});
```

## Environment

By default, projects use a small instance with an Ubuntu 18.04 image. You
can use the `environment` property to customize the build environment:

* `buildImage` defines the Docker image used. See [Images](#images) below for
  details on how to define build images.
* `certificate` defines the location of a PEM encoded certificate to import.
* `computeType` defines the instance type used for the build.
* `dockerServer` defines the docker server used for the build.
* `privileged` can be set to `true` to allow privileged access.
* `environmentVariables` can be set at this level (and also at the project
  level).

Finally, you can also set the build environment `fleet` property to create
a reserved capacity project. See [Fleet](#fleet) for more information.

## Images

The CodeBuild library supports Linux, Windows, and Mac images via the
`LinuxBuildImage` (or `LinuxArmBuildImage`), `WindowsBuildImage`, and `MacBuildImage` classes, respectively.
With the introduction of Lambda compute support, the `LinuxLambdaBuildImage ` (or `LinuxArmLambdaBuildImage`) class
is available for specifying Lambda-compatible images.

You can specify one of the predefined Windows/Linux images by using one
of the constants such as `WindowsBuildImage.WIN_SERVER_CORE_2019_BASE`,
`WindowsBuildImage.WINDOWS_BASE_2_0`, `LinuxBuildImage.STANDARD_2_0`,
`LinuxBuildImage.AMAZON_LINUX_2_5`, `MacBuildImage.BASE_14`, `MacBuildImage.BASE_15`, `MacBuildImage.BASE_26`, `LinuxArmBuildImage.AMAZON_LINUX_2_ARM`,
`LinuxLambdaBuildImage.AMAZON_LINUX_2_NODE_18`, `LinuxLambdaBuildImage.AMAZON_LINUX_2023_NODE_24`, `LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18` or `LinuxArmLambdaBuildImage.AMAZON_LINUX_2023_NODE_24`.

Alternatively, you can specify a custom image using one of the static methods on
`LinuxBuildImage`:

* `LinuxBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }])` to reference an image in any public or private Docker registry.
* `LinuxBuildImage.fromEcrRepository(repo[, tag])` to reference an image available in an
  ECR repository.
* `LinuxBuildImage.fromAsset(parent, id, props)` to use an image created from a
  local asset.
* `LinuxBuildImage.fromCodeBuildImageId(id)` to reference a pre-defined, CodeBuild-provided Docker image.

or one of the corresponding methods on `WindowsBuildImage`:

* `WindowsBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }, imageType])`
* `WindowsBuildImage.fromEcrRepository(repo[, tag, imageType])`
* `WindowsBuildImage.fromAsset(parent, id, props, [, imageType])`

or one of the corresponding methods on `MacBuildImage`:

* `MacBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }, imageType])`
* `MacBuildImage.fromEcrRepository(repo[, tag, imageType])`
* `MacBuildImage.fromAsset(parent, id, props, [, imageType])`

or one of the corresponding methods on `LinuxArmBuildImage`:

* `LinuxArmBuildImage.fromDockerRegistry(image[, { secretsManagerCredentials }])`
* `LinuxArmBuildImage.fromEcrRepository(repo[, tag])`

**Note:** You cannot specify custom images on `LinuxLambdaBuildImage` or `LinuxArmLambdaBuildImage` images.

Note that the `WindowsBuildImage` version of the static methods accepts an optional parameter of type `WindowsImageType`,
which can be either `WindowsImageType.STANDARD`, the default, or `WindowsImageType.SERVER_2019`:

```ts
declare const ecrRepository: ecr.Repository;

new codebuild.Project(this, 'Project', {
  environment: {
    buildImage: codebuild.WindowsBuildImage.fromEcrRepository(ecrRepository, 'v1.0', codebuild.WindowsImageType.SERVER_2019),
    // optional certificate to include in the build image
    certificate: {
      bucket: s3.Bucket.fromBucketName(this, 'Bucket', 'amzn-s3-demo-bucket'),
      objectKey: 'path/to/cert.pem',
    },
  },
  // ...
})
```

The following example shows how to define an image from a Docker asset:

[Docker asset example](./test/integ.docker-asset.lit.ts)

The following example shows how to define an image from an ECR repository:

[ECR example](./test/integ.ecr.lit.ts)

The following example shows how to define an image from a private docker registry:

[Docker Registry example](./test/integ.docker-registry.lit.ts)

### GPU images

The class `LinuxGpuBuildImage` contains constants for working with
[AWS Deep Learning Container images](https://aws.amazon.com/releasenotes/available-deep-learning-containers-images):


```ts
new codebuild.Project(this, 'Project', {
  environment: {
    buildImage: codebuild.LinuxGpuBuildImage.DLC_TENSORFLOW_2_1_0_INFERENCE,
  },
  // ...
})
```

One complication is that the repositories for the DLC images are in
different accounts in different AWS regions.
In most cases, the CDK will handle providing the correct account for you;
in rare cases (for example, deploying to new regions)
where our information might be out of date,
you can always specify the account
(along with the repository name and tag)
explicitly using the `awsDeepLearningContainersImage` method:

```ts
new codebuild.Project(this, 'Project', {
  environment: {
    buildImage: codebuild.LinuxGpuBuildImage.awsDeepLearningContainersImage(
      'tensorflow-inference', '2.1.0-gpu-py36-cu101-ubuntu18.04', '123456789012'),
  },
  // ...
})
```

Alternatively, you can reference an image available in an ECR repository using the `LinuxGpuBuildImage.fromEcrRepository(repo[, tag])` method.

### Lambda images

The `LinuxLambdaBuildImage` (or `LinuxArmLambdaBuildImage`) class contains constants for working with
Lambda compute:

```ts
new codebuild.Project(this, 'Project', {
  environment: {
    buildImage: codebuild.LinuxLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
  },
  // ...
})
```

> Visit [AWS Lambda compute in AWS CodeBuild](https://docs.aws.amazon.com/codebuild/latest/userguide/lambda.html) for more details.

## Fleet

By default, a CodeBuild project will request on-demand compute resources
to process your build requests. While being able to scale and handle high load,
on-demand resources can also be slow to provision.

Reserved capacity fleets are an alternative to on-demand.
Dedicated instances, maintained by CodeBuild,
will be ready to fulfill your build requests immediately.
Skipping the provisioning step in your project will reduce your build time,
at the cost of paying for these reserved instances, even when idling, until they are released.

For more information, see [Working with reserved capacity in AWS CodeBuild](https://docs.aws.amazon.com/codebuild/latest/userguide/fleets.html) in the CodeBuild documentation.

```ts
const fleet = new codebuild.Fleet(this, 'Fleet', {
    computeType: codebuild.FleetComputeType.MEDIUM,
    environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
    baseCapacity: 1,
});

new codebuild.Project(this, 'Project', {
  environment: {
    fleet,
    buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
  },
  // ...
})
```

You can also import an existing fleet to share its resources
among several projects across multiple stacks:

```ts
new codebuild.Project(this, 'Project', {
  environment: {
    fleet: codebuild.Fleet.fromFleetArn(
      this, 'SharedFleet',
      'arn:aws:codebuild:us-east-1:123456789012:fleet/MyFleet:ed0d0823-e38a-4c10-90a1-1bf25f50fa76',
    ),
    buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
  },
  // ...
})
```

### Attribute-based compute

You can use [attribute-based compute](https://docs.aws.amazon.com/codebuild/latest/userguide/fleets.html#fleets.attribute-compute) for your fleet by setting the `computeType` to `ATTRIBUTE_BASED`.
This allows you to specify the attributes in `computeConfiguration` such as vCPUs, memory, disk space, and the machineType.
After specifying some or all of the available attributes, CodeBuild will select the cheapest compute type from available instance types as that at least matches all given criteria.

```ts
import { Size } from 'aws-cdk-lib';

const fleet = new codebuild.Fleet(this, 'MyFleet', {
  baseCapacity: 1,
  computeType: codebuild.FleetComputeType.ATTRIBUTE_BASED,
  environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
  computeConfiguration: {
    vCpu: 2,
    memory: Size.gibibytes(4),
    disk: Size.gibibytes(10),
    machineType: codebuild.MachineType.GENERAL,
  },
});
```

### Custom instance types
You can use [specific EC2 instance
types](https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html#environment-reserved-capacity.instance-types)
for your fleet by setting the `computeType` to `CUSTOM_INSTANCE_TYPE`.  This
allows you to specify the `instanceType` in `computeConfiguration`. Only certain
EC2 instance types are supported; see the linked documentation for details.

```ts
import { Size } from 'aws-cdk-lib';

const fleet = new codebuild.Fleet(this, 'MyFleet', {
  baseCapacity: 1,
  computeType: codebuild.FleetComputeType.CUSTOM_INSTANCE_TYPE,
  environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
  computeConfiguration: {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
    // By default, 64 GiB of disk space is included. Any value optionally
    // specified here is _incremental_ on top of the included disk space.
    disk: Size.gibibytes(10),
  },
});
```

### Fleet overflow behavior

When your builds exceed the capacity of your fleet, you can specify how CodeBuild should handle the overflow builds by setting the `overflowBehavior` property:

```ts
const fleet = new codebuild.Fleet(this, 'Fleet', {
  computeType: codebuild.FleetComputeType.MEDIUM,
  environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
  baseCapacity: 1,
  overflowBehavior: codebuild.FleetOverflowBehavior.ON_DEMAND,
});
```

The available overflow behaviors are:

- `QUEUE` (default): Overflow builds wait for existing fleet instances to become available
- `ON_DEMAND`: Overflow builds run on CodeBuild on-demand instances

Note: If you set overflow behavior to `ON_DEMAND` for a VPC-connected fleet, ensure your VPC settings allow access to public AWS services.

### VPCs
The same considerations that apply to [Project
VPCs](#definition-of-vpc-configuration-in-codebuild-project) also apply to Fleet
VPCs.  When using a Fleet in a CodeBuild Project, it is an error to configure a
VPC on the Project. Configure a VPC on the fleet instead.

```ts
declare const loadBalancer: elbv2.ApplicationLoadBalancer;

const vpc = new ec2.Vpc(this, 'MyVPC');
const fleet = new codebuild.Fleet(this, 'MyProject', {
  computeType: codebuild.FleetComputeType.MEDIUM,
  environmentType: codebuild.EnvironmentType.LINUX_CONTAINER,
  baseCapacity: 1,
  vpc,
});

fleet.connections.allowTo(loadBalancer, ec2.Port.tcp(443));

const project = new codebuild.Project(this, 'MyProject', {
  environment: {
    fleet,
  },
  buildSpec: codebuild.BuildSpec.fromObject({
    // ...
  }),
  // Trying to configure a project-level VPC is an error, because this project
  // runs on the Fleet created above.
  // vpc,
});
```

## Logs

CodeBuild lets you specify an S3 Bucket, CloudWatch Log Group or both to receive logs from your projects.

By default, logs will go to cloudwatch.

### CloudWatch Logs Example

```ts
new codebuild.Project(this, 'Project', {
  logging: {
    cloudWatch: {
      logGroup: new logs.LogGroup(this, `MyLogGroup`),
    }
  },
})
```

### S3 Logs Example

```ts
new codebuild.Project(this, 'Project', {
  logging: {
    s3: {
      bucket: new s3.Bucket(this, `LogBucket`)
    }
  },
})
```

## Debugging builds interactively using SSM Session Manager

Integration with SSM Session Manager makes it possible to add breakpoints to your
build commands, pause the build there and log into the container to interactively
debug the environment.

To do so, you need to:

* Create the build with `ssmSessionPermissions: true`.
* Use a build image with SSM agent installed and configured (default CodeBuild images come with the image preinstalled).
* Start the build with [debugSessionEnabled](https://docs.aws.amazon.com/codebuild/latest/APIReference/API_StartBuild.html#CodeBuild-StartBuild-request-debugSessionEnabled) set to true.

If these conditions are met, execution of the command `codebuild-breakpoint`
will suspend your build and allow you to attach a Session Manager session from
the CodeBuild console.

For more information, see [View a running build in Session
Manager](https://docs.aws.amazon.com/codebuild/latest/userguide/session-manager.html)
in the CodeBuild documentation.

Example:

```ts
new codebuild.Project(this, 'Project', {
  environment: {
    buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
  },
  ssmSessionPermissions: true,
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: [
          // Pause the build container if possible
          'codebuild-breakpoint',
          // Regular build in a script in the repository
          './my-build.sh',
        ],
      },
    },
  }),
})
```

## Credentials

CodeBuild allows you to store credentials used when communicating with various sources,
like GitHub:

```ts
new codebuild.GitHubSourceCredentials(this, 'CodeBuildGitHubCreds', {
  accessToken: SecretValue.secretsManager('my-token'),
});
// GitHub Enterprise is almost the same,
// except the class is called GitHubEnterpriseSourceCredentials
```

and BitBucket:

```ts
new codebuild.BitBucketSourceCredentials(this, 'CodeBuildBitBucketCreds', {
  username: SecretValue.secretsManager('my-bitbucket-creds', { jsonField: 'username' }),
  password: SecretValue.secretsManager('my-bitbucket-creds', { jsonField: 'password' }),
});
```

**Note**: the credentials are global to a given account in a given region -
they are not defined per CodeBuild project.
CodeBuild only allows storing a single credential of a given type
(GitHub, GitHub Enterprise or BitBucket)
in a given account in a given region -
any attempt to save more than one will result in an error.
You can use the [`list-source-credentials` AWS CLI operation](https://docs.aws.amazon.com/cli/latest/reference/codebuild/list-source-credentials.html)
to inspect what credentials are stored in your account.

## Test reports

You can specify a test report in your buildspec:

```ts
const project = new codebuild.Project(this, 'Project', {
  buildSpec: codebuild.BuildSpec.fromObject({
    // ...
    reports: {
      myReport: {
        files: '**/*',
        'base-directory': 'build/test-results',
      },
    },
  }),
});
```

This will create a new test report group,
with the name `<ProjectName>-myReport`.

The project's role in the CDK will always be granted permissions to create and use report groups
with names starting with the project's name;
if you'd rather not have those permissions added,
you can opt out of it when creating the project:

```ts
declare const source: codebuild.Source;

const project = new codebuild.Project(this, 'Project', {
  source,
  grantReportGroupPermissions: false,
});
```

Alternatively, you can specify an ARN of an existing resource group,
instead of a simple name, in your buildspec:

```ts
declare const source: codebuild.Source;

// create a new ReportGroup
const reportGroup = new codebuild.ReportGroup(this, 'ReportGroup');

const project = new codebuild.Project(this, 'Project', {
  source,
  buildSpec: codebuild.BuildSpec.fromObject({
    // ...
    reports: {
      [reportGroup.reportGroupArn]: {
        files: '**/*',
        'base-directory': 'build/test-results',
      },
    },
  }),
});
```

For a code coverage report, you can specify a report group with the code coverage report group type.

```ts
declare const source: codebuild.Source;

// create a new ReportGroup
const reportGroup = new codebuild.ReportGroup(this, 'ReportGroup', {
    type: codebuild.ReportGroupType.CODE_COVERAGE
});

const project = new codebuild.Project(this, 'Project', {
  source,
  buildSpec: codebuild.BuildSpec.fromObject({
    // ...
    reports: {
      [reportGroup.reportGroupArn]: {
        files: '**/*',
        'base-directory': 'build/coverage-report.xml',
        'file-format': 'JACOCOXML'
      },
    },
  }),
});
```

If you specify a report group, you need to grant the project's role permissions to write reports to that report group:

```ts
declare const project: codebuild.Project;
declare const reportGroup: codebuild.ReportGroup;

reportGroup.grantWrite(project);
```

The created policy will adjust to the report group type. If no type is specified when creating the report group the created policy will contain the action for the test report group type.

For more information on the test reports feature,
see the [AWS CodeBuild documentation](https://docs.aws.amazon.com/codebuild/latest/userguide/test-reporting.html).

### Report group deletion

When a report group is removed from a stack (or the stack is deleted), the report
group will be removed according to its removal policy (which by default will
simply orphan the report group and leave it in your AWS account). If the removal
policy is set to `RemovalPolicy.DESTROY`, the report group will be deleted as long
as it does not contain any reports.

To override this and force all reports to get deleted during report group deletion,
enable the `deleteReports` option as well as setting the removal policy to
`RemovalPolicy.DESTROY`.

```ts
import * as cdk from 'aws-cdk-lib';

const reportGroup = new codebuild.ReportGroup(this, 'ReportGroup', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      deleteReports: true,
});
```

## Events

CodeBuild projects can be used either as a source for events or be triggered
by events via an event rule.

### Using Project as an event target

The `aws-cdk-lib/aws-events-targets.CodeBuildProject` allows using an AWS CodeBuild
project as a AWS CloudWatch event rule target:

```ts
// start build when a commit is pushed
import * as codecommit from 'aws-cdk-lib/aws-codecommit';
import * as targets from 'aws-cdk-lib/aws-events-targets';

declare const codeCommitRepository: codecommit.Repository;
declare const project: codebuild.Project;

codeCommitRepository.onCommit('OnCommit', {
  target: new targets.CodeBuildProject(project),
});
```

### Using Project as an event source

To define Amazon CloudWatch event rules for build projects, use one of the `onXxx`
methods:

```ts
import * as targets from 'aws-cdk-lib/aws-events-targets';
declare const fn: lambda.Function;
declare const project: codebuild.Project;

const rule = project.onStateChange('BuildStateChange', {
  target: new targets.LambdaFunction(fn)
});
```

## CodeStar Notifications

To define CodeStar Notification rules for Projects, use one of the `notifyOnXxx()` methods.
They are very similar to `onXxx()` methods for CloudWatch events:

```ts
import * as chatbot from 'aws-cdk-lib/aws-chatbot';

declare const project: codebuild.Project;

const target = new chatbot.SlackChannelConfiguration(this, 'MySlackChannel', {
  slackChannelConfigurationName: 'YOUR_CHANNEL_NAME',
  slackWorkspaceId: 'YOUR_SLACK_WORKSPACE_ID',
  slackChannelId: 'YOUR_SLACK_CHANNEL_ID',
});

const rule = project.notifyOnBuildSucceeded('NotifyOnBuildSucceeded', target);
```

## Secondary sources and artifacts

CodeBuild Projects can get their sources from multiple places, and produce
multiple outputs. For example:

```ts
import * as codecommit from 'aws-cdk-lib/aws-codecommit';
declare const repo: codecommit.Repository;
declare const bucket: s3.Bucket;

const project = new codebuild.Project(this, 'MyProject', {
  secondarySources: [
    codebuild.Source.codeCommit({
      identifier: 'source2',
      repository: repo,
    }),
  ],
  secondaryArtifacts: [
    codebuild.Artifacts.s3({
      identifier: 'artifact2',
      bucket: bucket,
      path: 'some/path',
      name: 'file.zip',
    }),
  ],
  // ...
});
```

Note that the `identifier` property is required for both secondary sources and
artifacts.

The contents of the secondary source is available to the build under the
directory specified by the `CODEBUILD_SRC_DIR_<identifier>` environment variable
(so, `CODEBUILD_SRC_DIR_source2` in the above case).

The secondary artifacts have their own section in the buildspec, under the
regular `artifacts` one. Each secondary artifact has its own section, beginning
with their identifier.

So, a buildspec for the above Project could look something like this:

```ts
const project = new codebuild.Project(this, 'MyProject', {
  // secondary sources and artifacts as above...
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
    phases: {
      build: {
        commands: [
          'cd $CODEBUILD_SRC_DIR_source2',
          'touch output2.txt',
        ],
      },
    },
    artifacts: {
      'secondary-artifacts': {
        'artifact2': {
          'base-directory': '$CODEBUILD_SRC_DIR_source2',
          'files': [
            'output2.txt',
          ],
        },
      },
    },
  }),
});
```

### Definition of VPC configuration in CodeBuild Project

Typically, resources in an VPC are not accessible by AWS CodeBuild. To enable
access, you must provide additional VPC-specific configuration information as
part of your CodeBuild project configuration. This includes the VPC ID, the
VPC subnet IDs, and the VPC security group IDs. VPC-enabled builds are then
able to access resources inside your VPC.

For further Information see https://docs.aws.amazon.com/codebuild/latest/userguide/vpc-support.html

**Use Cases**
VPC connectivity from AWS CodeBuild builds makes it possible to:

* Run integration tests from your build against data in an Amazon RDS database that's isolated on a private subnet.
* Query data in an Amazon ElastiCache cluster directly from tests.
* Interact with internal web services hosted on Amazon EC2, Amazon ECS, or services that use internal Elastic Load Balancing.
* Retrieve dependencies from self-hosted, internal artifact repositories, such as PyPI for Python, Maven for Java, and npm for Node.js.
* Access objects in an Amazon S3 bucket configured to allow access through an Amazon VPC endpoint only.
* Query external web services that require fixed IP addresses through the Elastic IP address of the NAT gateway or NAT instance associated with your subnet(s).

Your builds can access any resource that's hosted in your VPC.

**Enable Amazon VPC Access in your CodeBuild Projects**

Pass the VPC when defining your Project, then make sure to
give the CodeBuild's security group the right permissions
to access the resources that it needs by using the
`connections` object.

For example:

```ts
declare const loadBalancer: elbv2.ApplicationLoadBalancer;

const vpc = new ec2.Vpc(this, 'MyVPC');
const project = new codebuild.Project(this, 'MyProject', {
  vpc: vpc,
  buildSpec: codebuild.BuildSpec.fromObject({
    // ...
  }),
});

project.connections.allowTo(loadBalancer, ec2.Port.tcp(443));
```

## Project File System Location EFS

Add support for CodeBuild to build on AWS EFS file system mounts using
the new ProjectFileSystemLocation.
The `fileSystemLocations` property which accepts a list `ProjectFileSystemLocation`
as represented by the interface `IFileSystemLocations`.
The only supported file system type is `EFS`.

For example:

```ts
new codebuild.Project(this, 'MyProject', {
  buildSpec: codebuild.BuildSpec.fromObject({
    version: '0.2',
  }),
  fileSystemLocations: [
    codebuild.FileSystemLocation.efs({
      identifier: "myidentifier2",
      location: "myclodation.mydnsroot.com:/loc",
      mountPoint: "/media",
      mountOptions: "opts"
    })
  ]
});
```

Here's a CodeBuild project with a simple example that creates a project mounted on AWS EFS:

[Minimal Example](./test/integ.project-file-system-location.ts)

## Batch builds

To enable batch builds you should call `enableBatchBuilds()` on the project instance.

It returns an object containing the batch service role that was created,
or `undefined` if batch builds could not be enabled, for example if the project was imported.

```ts
declare const source: codebuild.Source;

const project = new codebuild.Project(this, 'MyProject', { source, });

if (project.enableBatchBuilds()) {
  console.log('Batch builds were enabled');
}
```

## Timeouts

There are two types of timeouts that can be set when creating your Project.
The `timeout` property can be used to set an upper limit on how long your Project is able to run without being marked as completed.
The default is 60 minutes.
An example of overriding the default follows.

```ts
new codebuild.Project(this, 'MyProject', {
  timeout: Duration.minutes(90)
});
```

The `queuedTimeout` property can be used to set an upper limit on how your Project remains queued to run.
There is no default value for this property.
As an example, to allow your Project to queue for up to thirty (30) minutes before the build fails,
use the following code.

```ts
new codebuild.Project(this, 'MyProject', {
  queuedTimeout: Duration.minutes(30)
});
```

## Limiting concurrency

By default if a new build is triggered it will be run even if there is a previous build already in progress.
It is possible to limit the maximum concurrent builds to value between 1 and the account specific maximum limit.
By default there is no explicit limit.

```ts
new codebuild.Project(this, 'MyProject', {
  concurrentBuildLimit: 1
});
```

## Visibility
When you can specify the visibility of the project builds. This setting controls whether the builds are publicly readable or remain private.

Visibility options:
- `PUBLIC_READ`: The project builds are visible to the public.
- `PRIVATE`: The project builds are not visible to the public.

Examples:

```ts
new codebuild.Project(this, 'MyProject', {
  visibility: codebuild.ProjectVisibility.PUBLIC_READ,
});
```

## Auto retry limit
You can automatically retry your builds in AWS CodeBuild by setting `autoRetryLimit` property.

With auto-retry enabled, CodeBuild will automatically call RetryBuild using the project's service role after a failed build up to a specified limit.

For example, if the auto-retry limit is set to two, CodeBuild will call the RetryBuild API to automatically retry your build for up to two additional times.

```ts
new codebuild.Project(this, 'MyProject', {
  autoRetryLimit: 2,
});
```
