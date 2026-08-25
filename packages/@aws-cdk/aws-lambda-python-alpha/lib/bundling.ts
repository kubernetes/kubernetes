import * as path from 'path';
import type { AssetCode, Runtime } from 'aws-cdk-lib/aws-lambda';
import { Architecture, Code } from 'aws-cdk-lib/aws-lambda';
import type { BundlingFileAccess, BundlingOptions as CdkBundlingOptions, DockerVolume } from 'aws-cdk-lib/core';
import { AssetStaging, DockerImage, PERF_BUNDLING_SRC_SYM } from 'aws-cdk-lib/core';
import { UnscopedValidationError } from 'aws-cdk-lib/core/lib/errors';
import { lit, posixShellEscape, profileSpan } from 'aws-cdk-lib/core/lib/helpers-internal';
import { Packaging, DependenciesFile } from './packaging';
import type { BundlingOptions, ICommandHooks } from './types';

/**
 * Dependency files to exclude from the asset hash.
 */
export const DEPENDENCY_EXCLUDES = ['*.pyc'];

/**
 * The location in the image that the bundler image caches dependencies.
 */
export const BUNDLER_DEPENDENCIES_CACHE = '/var/dependencies';

/**
 * Options for bundling
 */
export interface BundlingProps extends BundlingOptions {
  /**
   * Entry path
   */
  readonly entry: string;

  /**
   * The runtime environment.
   */
  readonly runtime: Runtime;

  /**
   * The system architecture of the lambda function
   *
   * @default Architecture.X86_64
   */
  readonly architecture?: Architecture;

  /**
   * Whether or not the bundling process should be skipped
   *
   * @default - Does not skip bundling
   */
  readonly skip?: boolean;

  /**
   * Which option to use to copy the source files to the docker container and output files back
   * @default - BundlingFileAccess.BIND_MOUNT
   */
  bundlingFileAccess?: BundlingFileAccess;
}

/**
 * Produce bundled Lambda asset code
 */
export class Bundling implements CdkBundlingOptions {
  public static bundle(options: BundlingProps): AssetCode {
    validateOutputPathSuffix(options.outputPathSuffix ?? '');

    return Code.fromAsset(options.entry, {
      assetHash: options.assetHash,
      assetHashType: options.assetHashType,
      exclude: DEPENDENCY_EXCLUDES,
      bundling: options.skip ? undefined : new Bundling(options),
    });
  }

  public readonly [PERF_BUNDLING_SRC_SYM] = 'PythonFunction';
  public readonly image: DockerImage;
  public readonly entrypoint?: string[];
  public readonly command: string[];
  public readonly volumes?: DockerVolume[];
  public readonly volumesFrom?: string[];
  public readonly environment?: { [key: string]: string };
  public readonly workingDirectory?: string;
  public readonly user?: string;
  public readonly securityOpt?: string;
  public readonly network?: string;
  public readonly bundlingFileAccess?: BundlingFileAccess;

  constructor(props: BundlingProps) {
    const {
      entry,
      runtime,
      architecture = Architecture.X86_64,
      outputPathSuffix = '',
      image,
      poetryIncludeHashes,
      poetryWithoutUrls,
      commandHooks,
      assetExcludes = [],
    } = props;

    const outputPath = path.posix.join(AssetStaging.BUNDLING_OUTPUT_DIR, outputPathSuffix);

    const bundlingCommands = this.createBundlingCommand({
      entry,
      inputDir: AssetStaging.BUNDLING_INPUT_DIR,
      outputDir: outputPath,
      poetryIncludeHashes,
      poetryWithoutUrls,
      commandHooks,
      assetExcludes,
    });

    if (image) {
      // Use the user's image
      this.image = image;
    } else {
      // Build our own image to do the build in in. We do some counter trickery here: we do want to count
      // the time spent here as part of 'bundle:PythonFunction', but by default only the RUNNING of the Docker
      // image would count as that. So we add an additional timer span just for the building of the runner image.
      using _span = profileSpan(`bundle:${this[PERF_BUNDLING_SRC_SYM]}`, { telemetry: true, skipCount: true });
      this.image = DockerImage.fromBuild(path.join(__dirname, 'docker'), {
        buildArgs: {
          ...props.buildArgs,
          IMAGE: runtime.bundlingImage.image,
        },
        platform: architecture.dockerPlatform,
        network: props.network,
      });
    }

    this.command = props.command ?? ['bash', '-c', chain(bundlingCommands)];
    this.entrypoint = props.entrypoint;
    this.volumes = props.volumes;
    this.volumesFrom = props.volumesFrom;
    this.environment = props.environment;
    this.workingDirectory = props.workingDirectory;
    this.user = props.user;
    this.securityOpt = props.securityOpt;
    this.network = props.network;
    this.bundlingFileAccess = props.bundlingFileAccess;
  }

  private createBundlingCommand(options: BundlingCommandOptions): string[] {
    const packaging = Packaging.fromEntry(options.entry, options.poetryIncludeHashes, options.poetryWithoutUrls);
    let bundlingCommands: string[] = [];
    bundlingCommands.push(...options.commandHooks?.beforeBundling(options.inputDir, options.outputDir) ?? []);

    const excludes = [...(options.assetExcludes ?? [])];
    if (packaging.dependenciesFile == DependenciesFile.UV && !excludes.includes('.python-version')) {
      excludes.push('.python-version');
    }

    const exclusionStr = excludes.map(item => `--exclude=${posixShellEscape(item)}`).join(' ');
    bundlingCommands.push([
      'rsync', '-rLv', exclusionStr ?? '', posixShellEscape(`${options.inputDir}/`), posixShellEscape(options.outputDir),
    ].filter(item => item).join(' '));
    bundlingCommands.push(`cd ${posixShellEscape(options.outputDir)}`);
    bundlingCommands.push(packaging.exportCommand ?? '');

    if (packaging.dependenciesFile == DependenciesFile.UV) {
      bundlingCommands.push(`uv pip install -r ${DependenciesFile.PIP} --target ${posixShellEscape(options.outputDir)}`);
    } else if (packaging.dependenciesFile) {
      bundlingCommands.push(`python -m pip install -r ${DependenciesFile.PIP} -t ${posixShellEscape(options.outputDir)}`);
    }

    bundlingCommands.push(...options.commandHooks?.afterBundling(options.inputDir, options.outputDir) ?? []);
    return bundlingCommands;
  }
}

interface BundlingCommandOptions {
  readonly entry: string;
  readonly inputDir: string;
  readonly outputDir: string;
  readonly assetExcludes?: string[];
  readonly poetryIncludeHashes?: boolean;
  readonly poetryWithoutUrls?: boolean;
  readonly commandHooks?: ICommandHooks;
}

/**
 * Chain commands
 */
function chain(commands: string[]): string {
  return commands.filter(c => !!c).join(' && ');
}

function validateOutputPathSuffix(outputPathSuffix: string): void {
  const outputPath = path.posix.join(AssetStaging.BUNDLING_OUTPUT_DIR, outputPathSuffix);
  if (pathEscapesRoot(path.posix.relative(AssetStaging.BUNDLING_OUTPUT_DIR, outputPath))) {
    throw new UnscopedValidationError(
      lit`OutputPathSuffixEscapesOutputDir`,
      `outputPathSuffix (${outputPathSuffix}) should not escape ${AssetStaging.BUNDLING_OUTPUT_DIR}`,
    );
  }
}

function pathEscapesRoot(relativePath: string): boolean {
  return path.posix.isAbsolute(relativePath) || relativePath === '..' || relativePath.startsWith('../');
}
