import { Template } from '../../assertions';
import * as codecommit from '../../aws-codecommit';
import * as cdk from '../../core';
import * as codebuild from '../lib';

describe('Linux ARM Lambda build image', () => {
  test('has type ARM_LAMBDA_CONTAINER and default ComputeType LAMBDA_1GB', () => {
    const stack = new cdk.Stack();
    new codebuild.PipelineProject(stack, 'Project', {
      environment: {
        buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Type: 'ARM_LAMBDA_CONTAINER',
        ComputeType: 'BUILD_LAMBDA_1GB',
      },
    });
  });

  test('can be used with ComputeType LAMBDA_2GB', () => {
    const stack = new cdk.Stack();
    new codebuild.PipelineProject(stack, 'Project', {
      environment: {
        computeType: codebuild.ComputeType.LAMBDA_2GB,
        buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Type: 'ARM_LAMBDA_CONTAINER',
        ComputeType: 'BUILD_LAMBDA_2GB',
      },
    });
  });

  test('can be used with ComputeType LAMBDA_4GB', () => {
    const stack = new cdk.Stack();
    new codebuild.PipelineProject(stack, 'Project', {
      environment: {
        computeType: codebuild.ComputeType.LAMBDA_4GB,
        buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Type: 'ARM_LAMBDA_CONTAINER',
        ComputeType: 'BUILD_LAMBDA_4GB',
      },
    });
  });

  test('can be used with ComputeType LAMBDA_8GB', () => {
    const stack = new cdk.Stack();
    new codebuild.PipelineProject(stack, 'Project', {
      environment: {
        computeType: codebuild.ComputeType.LAMBDA_8GB,
        buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Type: 'ARM_LAMBDA_CONTAINER',
        ComputeType: 'BUILD_LAMBDA_8GB',
      },
    });
  });

  test('can be used with ComputeType LAMBDA_10GB', () => {
    const stack = new cdk.Stack();
    new codebuild.PipelineProject(stack, 'Project', {
      environment: {
        computeType: codebuild.ComputeType.LAMBDA_10GB,
        buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeBuild::Project', {
      Environment: {
        Type: 'ARM_LAMBDA_CONTAINER',
        ComputeType: 'BUILD_LAMBDA_10GB',
      },
    });
  });

  test('cannot be used in conjunction with ComputeType SMALL', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          computeType: codebuild.ComputeType.SMALL,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Lambda images only support Lambda ComputeTypes between 'BUILD_LAMBDA_1GB' and 'BUILD_LAMBDA_10GB', got 'BUILD_GENERAL1_SMALL'/);
  });

  test('cannot be used in conjunction with timeoutInMinutes property', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        timeout: cdk.Duration.minutes(10),
        environment: {
          computeType: codebuild.ComputeType.LAMBDA_10GB,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Cannot specify timeout for Lambda compute/);
  });

  test('AMAZON_LINUX_2023_NODE_24 has correct image id', () => {
    expect(codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2023_NODE_24.imageId).toEqual('aws/codebuild/amazonlinux-aarch64-lambda-standard:nodejs24');
  });

  test('cannot be used in conjunction with privileged property', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        environment: {
          privileged: true,
          computeType: codebuild.ComputeType.LAMBDA_1GB,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Lambda compute type does not support privileged mode/);
  });

  test('cannot be used in conjunction with queuedTimeout property', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        queuedTimeout: cdk.Duration.minutes(10),
        environment: {
          computeType: codebuild.ComputeType.LAMBDA_1GB,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Cannot specify queuedTimeout for Lambda compute/);
  });

  test('cannot be used in conjunction with cache property', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.PipelineProject(stack, 'Project', {
        cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER),
        environment: {
          computeType: codebuild.ComputeType.LAMBDA_1GB,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Cannot specify cache for Lambda compute/);
  });

  test('cannot be used in conjunction with badge property', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new codebuild.Project(stack, 'Project', {
        badge: true,
        source: codebuild.Source.codeCommit({
          repository: codecommit.Repository.fromRepositoryName(stack, 'Repo', 'repo-name'),
        }),
        environment: {
          computeType: codebuild.ComputeType.LAMBDA_1GB,
          buildImage: codebuild.LinuxArmLambdaBuildImage.AMAZON_LINUX_2_NODE_18,
        },
      });
    }).toThrow(/Invalid CodeBuild environment: Cannot enable badge for Lambda compute/);
  });
});
