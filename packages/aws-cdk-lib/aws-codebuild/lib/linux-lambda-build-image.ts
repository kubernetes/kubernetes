import type { BuildSpec } from './build-spec';
import { ComputeType } from './compute-type';
import { isLambdaComputeType } from './is-lambda-compute-type';
import { runScriptLinuxBuildSpec } from './private/run-script-linux-build-spec';
import type { BuildEnvironment, IBuildImage } from './project';

/**
 * Construction properties of `LinuxLambdaBuildImage`.
 * Module-private, as the constructor of `LinuxLambdaBuildImage` is private.
 */
interface LinuxLambdaBuildImageProps {
  readonly imageId: string;
}

/**
 * A CodeBuild image running x86-64 Lambda.
 *
 * This class has a bunch of public constants that represent the CodeBuild Lambda x86-64 images.
 *
 * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
 */
export class LinuxLambdaBuildImage implements IBuildImage {
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs18` build image. */
  public static readonly AMAZON_LINUX_2_NODE_18 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs18');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs20` build image. */
  public static readonly AMAZON_LINUX_2023_NODE_20 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs20');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs22` build image. */
  public static readonly AMAZON_LINUX_2023_NODE_22 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs22');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs24` build image. */
  public static readonly AMAZON_LINUX_2023_NODE_24 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs24');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.11` build image. */
  public static readonly AMAZON_LINUX_2_PYTHON_3_11 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.11');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.12` build image. */
  public static readonly AMAZON_LINUX_2023_PYTHON_3_12 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.12');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.13` build image. */
  public static readonly AMAZON_LINUX_2023_PYTHON_3_13 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:python3.13');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:ruby3.2` build image. */
  public static readonly AMAZON_LINUX_2_RUBY_3_2 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:ruby3.2');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:ruby3.4` build image. */
  public static readonly AMAZON_LINUX_2023_RUBY_3_4 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:ruby3.4');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto21` build image. */
  public static readonly AMAZON_LINUX_2023_CORRETTO_21 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto21');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto17` build image. */
  public static readonly AMAZON_LINUX_2_CORRETTO_17 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto17');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto11` build image. */
  public static readonly AMAZON_LINUX_2_CORRETTO_11 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:corretto11');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:go1.21` build image. */
  public static readonly AMAZON_LINUX_2_GO_1_21 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:go1.21');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:go1.24` build image. */
  public static readonly AMAZON_LINUX_2023_GO_1_24 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:go1.24');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:dotnet6` build image. */
  public static readonly AMAZON_LINUX_2_DOTNET_6 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:dotnet6');
  /** The `aws/codebuild/amazonlinux-x86_64-lambda-standard:dotnet8` build image. */
  public static readonly AMAZON_LINUX_2023_DOTNET_8 = LinuxLambdaBuildImage.fromCodeBuildImageId('aws/codebuild/amazonlinux-x86_64-lambda-standard:dotnet8');

  /**
   * Uses a Docker image provided by CodeBuild.
   *
   * NOTE: In Lambda compute, since only specified images can be used, this method is set to private.
   *
   * @see https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
   *
   * @param id The image identifier
   * @example 'aws/codebuild/amazonlinux-x86_64-lambda-standard:nodejs18'
   * @returns A Docker image provided by CodeBuild.
   */
  private static fromCodeBuildImageId(id: string): IBuildImage {
    return new LinuxLambdaBuildImage({
      imageId: id,
    });
  }

  public readonly type = 'LINUX_LAMBDA_CONTAINER';
  public readonly defaultComputeType = ComputeType.LAMBDA_1GB;
  public readonly imageId: string;

  private constructor(props: LinuxLambdaBuildImageProps) {
    this.imageId = props.imageId;
  }

  public validate(buildEnvironment: BuildEnvironment): string[] {
    const errors = [];

    if (buildEnvironment.privileged) {
      errors.push('Lambda compute type does not support privileged mode');
    }

    if (buildEnvironment.computeType && !isLambdaComputeType(buildEnvironment.computeType)) {
      errors.push([
        'Lambda images only support Lambda ComputeTypes between',
        `'${ComputeType.LAMBDA_1GB}'`,
        'and',
        `'${ComputeType.LAMBDA_10GB}',`,
        `got '${buildEnvironment.computeType}'`,
      ].join(' '));
    }

    return errors;
  }

  public runScriptBuildspec(entrypoint: string): BuildSpec {
    return runScriptLinuxBuildSpec(entrypoint);
  }
}
