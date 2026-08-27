import { App, Stack } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as path from 'path';
import { Alias, Code, DockerImageCode, DockerImageFunction, Function, Runtime, SnapStartConf } from 'aws-cdk-lib/aws-lambda';
import { Platform } from 'aws-cdk-lib/aws-ecr-assets';

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
  },
});

const stack = new Stack(app, 'aws-cdk-lambda-runtime-management');

new Function(stack, 'JavaSnapstartLambda', {
  code: Code.fromAsset(path.join(__dirname, 'handler-snapstart.zip')),
  handler: 'Handler',
  runtime: Runtime.JAVA_11,
  snapStart: SnapStartConf.ON_PUBLISHED_VERSIONS,
});

new Function(stack, 'Python312SnapstartLambda', {
  code: Code.fromInline('pass'),
  handler: 'Handler',
  runtime: Runtime.PYTHON_3_12,
  snapStart: SnapStartConf.ON_PUBLISHED_VERSIONS,
});

new Function(stack, 'Python313SnapstartLambda', {
  code: Code.fromInline('pass'),
  handler: 'Handler',
  runtime: Runtime.PYTHON_3_13,
  snapStart: SnapStartConf.ON_PUBLISHED_VERSIONS,
});

new Function(stack, 'DotnetSnapstartLambda', {
  code: Code.fromAsset(path.join(__dirname, 'dotnet-handler')),
  handler: 'Handler',
  runtime: Runtime.DOTNET_8,
  snapStart: SnapStartConf.ON_PUBLISHED_VERSIONS,
});

const containerImageFn = new DockerImageFunction(stack, 'ContainerImageSnapstartLambda', {
  code: DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-lambda-handler'), {
    platform: Platform.LINUX_AMD64,
  }),
  snapStart: SnapStartConf.ON_PUBLISHED_VERSIONS,
});

new Alias(stack, 'ContainerImageSnapstartAlias', {
  aliasName: 'live',
  version: containerImageFn.currentVersion,
});

new integ.IntegTest(app, 'lambda-runtime-management', {
  testCases: [stack],
});
