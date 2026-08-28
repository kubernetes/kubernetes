import * as crypto from 'crypto';
import * as path from 'path';
import { Construct } from 'constructs';
import * as fs from 'fs-extra';
import type { AssetOptions } from './assets';
import { AssetHashType, FileAssetPackaging } from './assets';
import type { BundlingOptions } from './bundling';
import { BundlingFileAccess, BundlingOutput, PERF_BUNDLING_SRC_SYM } from './bundling';
import { AssumptionError, ValidationError } from './errors';
import type { FingerprintOptions } from './fs';
import { FileSystem, SymlinkFollowMode, IgnoreStrategy } from './fs';
import { clearLargeFileFingerprintCache } from './fs/fingerprint';
import { Names } from './names';
import { AssetBundlingVolumeCopy, AssetBundlingBindMount } from './private/asset-staging';
import { Cache } from './private/cache';
import { stackOf, stageOf } from './private/core-construct-finders';
import { lit } from './private/literal-string';
import { profileSpan } from './private/perf';
import type { Stack } from './stack';
import * as cxapi from '../../cx-api';
import { isInternalPath, resolveLinkTarget } from './fs/utils';

const ARCHIVE_EXTENSIONS = ['.tar.gz', '.zip', '.jar', '.tar', '.tgz'];

const ASSET_SALT_CONTEXT_KEY = '@aws-cdk/core:assetHashSalt';

/**
 * A previously staged asset
 */
interface StagedAsset {
  /**
   * The path where we wrote this asset previously
   */
  readonly stagedPath: string;

  /**
   * The hash we used previously
   */
  readonly assetHash: string;

  /**
   * The packaging of the asset
   */
  readonly packaging: FileAssetPackaging;

  /**
   * Whether this asset is an archive
   */
  readonly isArchive: boolean;
}

/**
 * Initialization properties for `AssetStaging`.
 */
export interface AssetStagingProps extends FingerprintOptions, AssetOptions {
  /**
   * The source file or directory to copy from.
   */
  readonly sourcePath: string;
}

/**
 * Stages a file or directory from a location on the file system into a staging
 * directory.
 *
 * This is controlled by the context key 'aws:cdk:asset-staging' and enabled
 * by the CLI by default in order to ensure that when the CDK app exists, all
 * assets are available for deployment. Otherwise, if an app references assets
 * in temporary locations, those will not be available when it exists (see
 * https://github.com/aws/aws-cdk/issues/1716).
 *
 * The `stagedPath` property is a stringified token that represents the location
 * of the file or directory after staging. It will be resolved only during the
 * "prepare" stage and may be either the original path or the staged path
 * depending on the context setting.
 *
 * The file/directory are staged based on their content hash (fingerprint). This
 * means that only if content was changed, copy will happen.
 */
export class AssetStaging extends Construct {
  /**
   * The directory inside the bundling container into which the asset sources will be mounted.
   */
  public static readonly BUNDLING_INPUT_DIR = '/asset-input';

  /**
   * The directory inside the bundling container into which the bundled output should be written.
   */
  public static readonly BUNDLING_OUTPUT_DIR = '/asset-output';

  /**
   * Clears the asset hash cache
   */
  public static clearAssetHashCache() {
    this.assetCache.clear();
    clearLargeFileFingerprintCache();
  }

  /**
   * Cache of asset hashes based on asset configuration to avoid repeated file
   * system and bundling operations.
   */
  private static assetCache = new Cache<StagedAsset>();

  /**
   * Absolute path to the asset data.
   *
   * If asset staging is disabled, this will just be the source path or
   * a temporary directory used for bundling.
   *
   * If asset staging is enabled it will be the staged path.
   *
   * IMPORTANT: If you are going to call `addFileAsset()`, use
   * `relativeStagedPath()` instead.
   *
   * @deprecated - Use `absoluteStagedPath` instead.
   */
  public readonly stagedPath: string;

  /**
   * Absolute path to the asset data.
   *
   * If asset staging is disabled, this will just be the source path or
   * a temporary directory used for bundling.
   *
   * If asset staging is enabled it will be the staged path.
   *
   * IMPORTANT: If you are going to call `addFileAsset()`, use
   * `relativeStagedPath()` instead.
   */
  public readonly absoluteStagedPath: string;

  /**
   * The absolute path of the asset as it was referenced by the user.
   */
  public readonly sourcePath: string;

  /**
   * A cryptographic hash of the asset.
   */
  public readonly assetHash: string;

  /**
   * How this asset should be packaged.
   */
  public readonly packaging: FileAssetPackaging;

  /**
   * Whether this asset is an archive (zip or jar).
   */
  public readonly isArchive: boolean;

  private readonly fingerprintOptions: FingerprintOptions;

  private readonly hashType: AssetHashType;
  private readonly assetOutdir: string;

  /**
   * A custom source fingerprint given by the user
   *
   * Will not be used literally, always hashed later on.
   */
  private customSourceFingerprint?: string;

  private readonly cacheKey: string;

  private readonly _sourceStats?: fs.Stats;

  constructor(scope: Construct, id: string, props: AssetStagingProps) {
    super(scope, id);

    const salt = this.node.tryGetContext(ASSET_SALT_CONTEXT_KEY);

    this.sourcePath = path.resolve(props.sourcePath);
    this.fingerprintOptions = {
      ...props,
      exclude: ['.is_custom_resource', ...props.exclude ?? []],
      extraHash: props.extraHash || salt ? `${props.extraHash ?? ''}${salt ?? ''}` : undefined,
    };

    if (!fs.existsSync(this.sourcePath)) {
      throw new ValidationError(lit`CannotFindAsset`, `Cannot find asset at ${this.sourcePath}`, this);
    }

    const ignoreStrategy = IgnoreStrategy.fromCopyOptions(props, this.sourcePath);
    // look for invalid (external) symlinks
    if (props.follow == SymlinkFollowMode.BLOCK_EXTERNAL && fs.statSync(this.sourcePath).isDirectory()) {
      validateInternalSymlinks(this.sourcePath, scope, props.follow, ignoreStrategy);
    }

    this._sourceStats = fs.statSync(this.sourcePath);

    const outdir = stageOf(this)?.assetOutdir;
    if (!outdir) {
      throw new ValidationError(lit`UnableToDetermineCloudAssembly`, 'unable to determine cloud assembly asset output directory. Assets must be defined indirectly within a "Stage" or an "App" scope', this);
    }
    this.assetOutdir = outdir;

    // Determine the hash type based on the props as props.assetHashType is
    // optional from a caller perspective.
    this.customSourceFingerprint = props.assetHash;
    this.hashType = determineHashType(this, props.assetHashType, this.customSourceFingerprint);

    // Decide what we're going to do, without actually doing it yet
    let stageThisAsset: () => StagedAsset;
    let skip = false;
    if (props.bundling) {
      // Check if we actually have to bundle for this stack
      skip = !stackOf(this).bundlingRequired;
      const bundling = props.bundling;
      stageThisAsset = () => this.stageByBundling(bundling, skip, props);
    } else {
      stageThisAsset = () => this.stageByCopying();
    }

    // Calculate a cache key from the props. This way we can check if we already
    // staged this asset and reuse the result (e.g. the same asset with the same
    // configuration is used in multiple stacks). In this case we can completely
    // skip file system and bundling operations.
    //
    // The output directory and whether this asset is skipped or not should also be
    // part of the cache key to make sure we don't accidentally return the wrong
    // staged asset from the cache.
    this.cacheKey = calculateCacheKey({
      outdir: this.assetOutdir,
      sourcePath: path.resolve(props.sourcePath),
      bundling: props.bundling,
      assetHashType: this.hashType,
      customFingerprint: this.customSourceFingerprint,
      extraHash: props.extraHash,
      exclude: props.exclude,
      ignoreMode: props.ignoreMode,
      skip,
    });

    const staged = AssetStaging.assetCache.obtain(this.cacheKey, stageThisAsset);
    this.stagedPath = staged.stagedPath;
    this.absoluteStagedPath = staged.stagedPath;
    this.assetHash = staged.assetHash;
    this.packaging = staged.packaging;
    this.isArchive = staged.isArchive;

    // Memory optimization: this._sourceStats is used as a field to covertly pass
    // arguments between functions in the constructor, but the size of that object is 1.8kB
    //
    // That's holding on to a lot of unnecessary memory if there are a lot of assets (think 100k+).
    //
    // Release the object here, we don't need it again.
    this._sourceStats = undefined;
  }

  private get sourceStats(): fs.Stats {
    if (!this._sourceStats) {
      throw new AssumptionError(lit`SourceStatusUnset`, '_sourceStats has been unset');
    }
    return this._sourceStats;
  }

  /**
   * A cryptographic hash of the asset.
   *
   * @deprecated see `assetHash`.
   */
  public get sourceHash(): string {
    return this.assetHash;
  }

  /**
   * Return the path to the staged asset, relative to the Cloud Assembly (manifest) directory of the given stack
   *
   * Only returns a relative path if the asset was staged, returns an absolute path if
   * it was not staged.
   *
   * A bundled asset might end up in the outDir and still not count as
   * "staged"; if asset staging is disabled we're technically expected to
   * reference source directories, but we don't have a source directory for the
   * bundled outputs (as the bundle output is written to a temporary
   * directory). Nevertheless, we will still return an absolute path.
   *
   * A non-obvious directory layout may look like this:
   *
   * ```
   *   CLOUD ASSEMBLY ROOT
   *     +-- asset.12345abcdef/
   *     +-- assembly-Stage
   *           +-- MyStack.template.json
   *           +-- MyStack.assets.json <- will contain { "path": "../asset.12345abcdef" }
   * ```
   */
  public relativeStagedPath(stack: Stack) {
    const asmManifestDir = stageOf(stack)?.outdir;
    if (!asmManifestDir) { return this.stagedPath; }

    const isOutsideAssetDir = path.relative(this.assetOutdir, this.stagedPath).startsWith('..');
    if (isOutsideAssetDir || this.stagingDisabled) {
      return this.stagedPath;
    }

    return path.relative(asmManifestDir, this.stagedPath);
  }

  /**
   * Stage the source to the target by copying
   *
   * Optionally skip if staging is disabled, in which case we pretend we did something but we don't really.
   */
  private stageByCopying(): StagedAsset {
    const assetHash = this.calculateHash(this.hashType);
    const targetPath = this.stagingDisabled
      ? this.sourcePath
      : path.resolve(this.assetOutdir, renderAssetFilename(assetHash, getExtension(this.sourcePath)));
    const stagedPath = this.renderStagedPath(this.sourcePath, targetPath);

    if (!this.sourceStats.isDirectory() && !this.sourceStats.isFile()) {
      throw new ValidationError(lit`AssetExpectedDirectoryOrFile`, `Asset ${this.sourcePath} is expected to be either a directory or a regular file`, this);
    }

    this.stageAsset(this.sourcePath, stagedPath, 'copy');

    return {
      assetHash,
      stagedPath,
      packaging: this.sourceStats.isDirectory() ? FileAssetPackaging.ZIP_DIRECTORY : FileAssetPackaging.FILE,
      isArchive: this.sourceStats.isDirectory() || ARCHIVE_EXTENSIONS.includes(getExtension(this.sourcePath).toLowerCase()),
    };
  }

  /**
   * Stage the source to the target by bundling
   *
   * Optionally skip, in which case we pretend we did something but we don't really.
   */
  private stageByBundling(bundling: BundlingOptions, skip: boolean, props: AssetStagingProps): StagedAsset {
    if (!this.sourceStats.isDirectory()) {
      throw new ValidationError(lit`AssetExpectedDirectoryForBundling`, `Asset ${this.sourcePath} is expected to be a directory when bundling`, this);
    }

    if (skip) {
      // We should have bundled, but didn't to save time. Still pretend to have a hash.
      // If the asset uses OUTPUT or BUNDLE, we use a CUSTOM hash to avoid fingerprinting
      // a potentially very large source directory. Other hash types are kept the same.
      let hashType = this.hashType;
      if (hashType === AssetHashType.OUTPUT || hashType === AssetHashType.BUNDLE) {
        this.customSourceFingerprint = Names.uniqueId(this);
        hashType = AssetHashType.CUSTOM;
      }
      return {
        assetHash: this.calculateHash(hashType, bundling),
        stagedPath: this.sourcePath,
        packaging: FileAssetPackaging.ZIP_DIRECTORY,
        isArchive: true,
      };
    }

    // Try to calculate assetHash beforehand (if we can)
    let assetHash = this.hashType === AssetHashType.SOURCE || this.hashType === AssetHashType.CUSTOM
      ? this.calculateHash(this.hashType, bundling)
      : undefined;

    const bundleDir = this.determineBundleDir(this.assetOutdir, assetHash);
    this.bundle(bundling, bundleDir);

    // Check bundling output content and determine if we will need to archive
    const bundlingOutputType = bundling.outputType ?? BundlingOutput.AUTO_DISCOVER;
    const ignore = IgnoreStrategy.fromCopyOptions(props, bundleDir);
    const bundledAsset = determineBundledAsset(this, bundleDir, bundlingOutputType, ignore, props.follow);

    // Calculate assetHash afterwards if we still must
    assetHash = assetHash ?? this.calculateHash(this.hashType, bundling, bundledAsset.path);

    const stagedPath = this.renderStagedPath(
      bundledAsset.path,
      path.resolve(this.assetOutdir, renderAssetFilename(assetHash, bundledAsset.extension)),
    );

    this.stageAsset(bundledAsset.path, stagedPath, 'move');

    // If bundling produced a single archive file we "touch" this file in the bundling
    // directory after it has been moved to the staging directory if the hash is known before bundling. This way if bundling
    // is skipped because the bundling directory already exists we can still determine
    // the correct packaging type.
    // If the hash is calculated after bundling we remove the temporary directory now.
    if (bundledAsset.packaging === FileAssetPackaging.FILE) {
      if (this.hashType === AssetHashType.OUTPUT || this.hashType === AssetHashType.BUNDLE) {
        fs.removeSync(path.dirname(bundledAsset.path));
      } else {
        fs.closeSync(fs.openSync(bundledAsset.path, 'w'));
      }
    }

    return {
      assetHash,
      stagedPath,
      packaging: bundledAsset.packaging,
      isArchive: bundlingOutputType !== BundlingOutput.SINGLE_FILE,
    };
  }

  /**
   * Whether staging has been disabled
   */
  private get stagingDisabled() {
    return !!this.node.tryGetContext(cxapi.DISABLE_ASSET_STAGING_CONTEXT);
  }

  /**
   * Copies or moves the files from sourcePath to targetPath.
   *
   * Moving implies the source directory is temporary and can be trashed.
   *
   * Will not do anything if source and target are the same.
   */
  private stageAsset(sourcePath: string, targetPath: string, style: 'move' | 'copy') {
    // Is the work already done?
    const isAlreadyStaged = fs.existsSync(targetPath);
    if (isAlreadyStaged) {
      if (style === 'move' && sourcePath !== targetPath) {
        fs.removeSync(sourcePath);
      }
      return;
    }

    // Moving can be done quickly
    if (style === 'move') {
      fs.renameSync(sourcePath, targetPath);
      return;
    }

    // Copy file/directory to staging directory
    if (this.sourceStats.isFile()) {
      fs.copyFileSync(sourcePath, targetPath);
    } else if (this.sourceStats.isDirectory()) {
      fs.mkdirSync(targetPath);
      FileSystem.copyDirectory(sourcePath, targetPath, this.fingerprintOptions);
    } else {
      throw new ValidationError(lit`UnknownFileType`, `Unknown file type: ${sourcePath}`, this);
    }
  }

  /**
   * Determine the directory where we're going to write the bundling output
   *
   * This is the target directory where we're going to write the staged output
   * files if we can (if the hash is fully known), or a temporary directory
   * otherwise.
   */
  private determineBundleDir(outdir: string, sourceHash?: string) {
    if (sourceHash) {
      return path.resolve(outdir, renderAssetFilename(sourceHash));
    }

    // When the asset hash isn't known in advance, bundler outputs to an
    // intermediate directory named after the asset's cache key
    return path.resolve(outdir, `bundling-temp-${this.cacheKey}`);
  }

  /**
   * Bundles an asset to the given directory
   *
   * If the given directory already exists, assume that everything's already
   * in order and don't do anything.
   *
   * @param options Bundling options
   * @param bundleDir Where to create the bundle directory
   * @returns The fully resolved bundle output directory.
   */
  private bundle(options: BundlingOptions, bundleDir: string) {
    if (fs.existsSync(bundleDir)) { return; }

    const tempDir = `${bundleDir}-building`;
    // Remove the tempDir if it exists, then recreate it
    fs.rmSync(tempDir, { recursive: true, force: true });

    fs.ensureDirSync(tempDir);
    // Chmod the bundleDir to full access.
    fs.chmodSync(tempDir, 0o777);

    let localBundling: boolean | undefined;
    try {
      process.stderr.write(`Bundling asset ${this.node.path}...\n`);

      using _span = timerSpanFromOptions(options);

      localBundling = options.local?.tryBundle(tempDir, options);
      if (!localBundling) {
        const assetStagingOptions = {
          sourcePath: this.sourcePath,
          bundleDir: tempDir,
          ...options,
        };

        switch (options.bundlingFileAccess) {
          case BundlingFileAccess.VOLUME_COPY:
            new AssetBundlingVolumeCopy(assetStagingOptions).run();
            break;
          case BundlingFileAccess.BIND_MOUNT:
          default:
            new AssetBundlingBindMount(assetStagingOptions).run();
            break;
        }
      }

      // Success, rename the tempDir into place
      fs.renameSync(tempDir, bundleDir);
    } catch (err) {
      throw new ValidationError(lit`FailedToBundleAsset`, `Failed to bundle asset ${this.node.path}, bundle output is located at ${tempDir}: ${err}`, this);
    }

    if (FileSystem.isEmpty(bundleDir)) {
      const outputDir = localBundling ? bundleDir : AssetStaging.BUNDLING_OUTPUT_DIR;
      throw new ValidationError(lit`BundlingProducedNoOutput`, `Bundling did not produce any output. Check that content is written to ${outputDir}.`, this);
    }
  }

  private calculateHash(hashType: AssetHashType, bundling?: BundlingOptions, outputDir?: string): string {
    // When bundling a CUSTOM or SOURCE asset hash type, we want the hash to include
    // the bundling configuration. We handle CUSTOM and bundled SOURCE hash types
    // as a special case to preserve existing user asset hashes in all other cases.
    if (hashType == AssetHashType.CUSTOM || (hashType == AssetHashType.SOURCE && bundling)) {
      const hash = crypto.createHash('sha256');

      // if asset hash is provided by user, use it, otherwise fingerprint the source.
      hash.update(this.customSourceFingerprint ?? FileSystem.fingerprint(this.sourcePath, this.fingerprintOptions));

      // If we're bundling an asset, include the bundling configuration in the hash
      if (bundling) {
        hash.update(JSON.stringify(bundling, sanitizeHashValue));
      }

      return hash.digest('hex');
    }

    switch (hashType) {
      case AssetHashType.SOURCE:
        return FileSystem.fingerprint(this.sourcePath, this.fingerprintOptions);
      case AssetHashType.BUNDLE:
      case AssetHashType.OUTPUT:
        if (!outputDir) {
          throw new ValidationError(lit`CannotUseHashTypeWithoutBundling`, `Cannot use \`${hashType}\` hash type when \`bundling\` is not specified.`, this);
        }
        return FileSystem.fingerprint(outputDir, this.fingerprintOptions);
      default:
        throw new ValidationError(lit`UnknownAssetHashType`, 'Unknown asset hash type.', this);
    }
  }

  private renderStagedPath(sourcePath: string, targetPath: string): string {
    // Add a suffix to the asset file name
    // because when a file without extension is specified, the source directory name is the same as the staged asset file name.
    // But when the hashType is `AssetHashType.OUTPUT`, the source directory name begins with `bundling-temp-` and the staged asset file name is different.
    // We only need to add a suffix when the hashType is not `AssetHashType.OUTPUT`.
    if (this.hashType !== AssetHashType.OUTPUT && path.dirname(sourcePath) === targetPath) {
      targetPath = targetPath + '_noext';
    }
    return targetPath;
  }
}

function renderAssetFilename(assetHash: string, extension = '') {
  return `asset.${assetHash}${extension}`;
}

/**
 * Determines the hash type from user-given prop values.
 *
 * @param assetHashType Asset hash type construct prop
 * @param customSourceFingerprint Asset hash seed given in the construct props
 */
function determineHashType(scope: Construct, assetHashType?: AssetHashType, customSourceFingerprint?: string) {
  const hashType = customSourceFingerprint
    ? (assetHashType ?? AssetHashType.CUSTOM)
    : (assetHashType ?? AssetHashType.SOURCE);

  if (customSourceFingerprint && hashType !== AssetHashType.CUSTOM) {
    throw new ValidationError(lit`CannotSpecifyAssetHashTypeWithAssetHash`, `Cannot specify \`${assetHashType}\` for \`assetHashType\` when \`assetHash\` is specified. Use \`CUSTOM\` or leave \`undefined\`.`, scope);
  }
  if (hashType === AssetHashType.CUSTOM && !customSourceFingerprint) {
    throw new ValidationError(lit`MustBeSpecified`, '`assetHash` must be specified when `assetHashType` is set to `AssetHashType.CUSTOM`.', scope);
  }

  return hashType;
}

/**
 * Walk the directory tree, throw if we find external symlinks
 * @param root true root of the directory
 * @param subRoot used for walking subdirectories
 */
function validateInternalSymlinks(
  root: string,
  scope: Construct,
  followMode: SymlinkFollowMode,
  ignoreStrat: IgnoreStrategy,
  subRoot: string = root,
) {
  const entries = fs.readdirSync(subRoot, { withFileTypes: true });
  for (const entry of entries) {
    const childPath = path.join(subRoot, entry.name);
    if (entry.isDirectory()) {
      if (ignoreStrat.completelyIgnores(childPath)) {
        continue;
      }
      validateInternalSymlinks(root, scope, followMode, ignoreStrat, childPath);
    } else if (!entry.isSymbolicLink()) {
      continue;
    } else { // we have a symlink
      if (ignoreStrat.completelyIgnores(childPath)) {
        continue;
      }
      // check whether this is internal or external
      const linkPath = fs.readlinkSync(childPath);
      const resolvedPath = resolveLinkTarget(childPath, linkPath);
      if (!isInternalPath(root, resolvedPath)) {
        throw new ValidationError(
          lit`BundlingFileSymlinkForbidden`,
          `The file ${resolvedPath} is an external symbolic link which is forbidden due to follow mode ${followMode}. Set \`follow\` to a mode that will follow symlinks (ALWAYS or EXTERNAL) or emit a regular file`,
          scope,
        );
      }
    }
  }
}

/**
 * Calculates a cache key from the props. Normalize by sorting keys.
 */
function calculateCacheKey<A extends object>(props: A): string {
  return crypto.createHash('sha256')
    .update(JSON.stringify(sortObject(props), sanitizeHashValue))
    .digest('hex');
}

/**
 * Recursively sort object keys
 */
function sortObject(object: { [key: string]: any }): { [key: string]: any } {
  if (typeof object !== 'object' || object instanceof Array) {
    return object;
  }
  const ret: { [key: string]: any } = {};
  for (const key of Object.keys(object).sort()) {
    ret[key] = sortObject(object[key]);
  }
  return ret;
}

/**
 * Removes the auth token from pip URLs if present to prevent an unnecessary
 * rebuild.
 *
 * @see https://github.com/aws/aws-cdk/issues/27331
 */
function sanitizeHashValue(key: string, value: any): any {
  if (key === 'PIP_INDEX_URL' || key === 'PIP_EXTRA_INDEX_URL') {
    try {
      let url = new URL(value);
      if (url.password) {
        url.password = '';
        return url.toString();
      }
    } catch (e: any) {
      if (e.name === 'TypeError') {
        throw new AssumptionError(lit`MustBeValid`, `${key} must be a valid URL, got ${value}.`);
      }
      throw e;
    }
  }
  return value;
}

/**
 * Returns the single archive file of a directory or undefined
 */
function findSingleFile(scope: Construct, directory: string, archiveOnly: boolean): string | undefined {
  if (!fs.existsSync(directory)) {
    throw new ValidationError(lit`DirectoryDoesNotExist`, `Directory ${directory} does not exist.`, scope);
  }

  if (!fs.statSync(directory).isDirectory()) {
    throw new ValidationError(lit`PathIsNotDirectory`, `${directory} is not a directory.`, scope);
  }

  const content = fs.readdirSync(directory);
  if (content.length === 1) {
    const file = path.join(directory, content[0]);
    const extension = getExtension(content[0]).toLowerCase();

    if (fs.statSync(file).isFile() && (!archiveOnly || ARCHIVE_EXTENSIONS.includes(extension))) {
      return file;
    }
  }

  return undefined;
}

interface BundledAsset {
  path: string;
  packaging: FileAssetPackaging;
  extension?: string;
}

/**
 * Returns the bundled asset to use based on the content of the bundle directory
 * and the type of output.
 */
function determineBundledAsset(
  scope: Construct,
  bundleDir: string,
  outputType: BundlingOutput,
  ignore: IgnoreStrategy,
  followMode?: SymlinkFollowMode,
): BundledAsset {
  const archiveFile = findSingleFile(scope, bundleDir, outputType !== BundlingOutput.SINGLE_FILE);

  // auto-discover means that if there is an archive file, we take it as the
  // bundle, otherwise, we will archive here.
  if (outputType === BundlingOutput.AUTO_DISCOVER) {
    outputType = archiveFile ? BundlingOutput.ARCHIVED : BundlingOutput.NOT_ARCHIVED;
  }

  switch (outputType) {
    case BundlingOutput.NOT_ARCHIVED:
      if (followMode == SymlinkFollowMode.BLOCK_EXTERNAL) {
        validateInternalSymlinks(bundleDir, scope, followMode, ignore);
      }
      return { path: bundleDir, packaging: FileAssetPackaging.ZIP_DIRECTORY };
    case BundlingOutput.ARCHIVED:
    case BundlingOutput.SINGLE_FILE:
      if (!archiveFile) {
        throw new ValidationError(lit`BundlingOutputDirectoryExpectedSingleFile`, 'Bundling output directory is expected to include only a single file when `output` is set to `ARCHIVED` or `SINGLE_FILE`', scope);
      } else if (fs.lstatSync(archiveFile).isSymbolicLink()) {
        throw new ValidationError(lit`SymlinkInBundlingOutput`, 'The output from bundling is not allowed to be a symlink.', scope);
      }
      return { path: archiveFile, packaging: FileAssetPackaging.FILE, extension: getExtension(archiveFile) };
  }
}

/**
 * Return the extension name of a source path
 *
 * Loop through ARCHIVE_EXTENSIONS for valid archive extensions.
 */
function getExtension(source: string): string {
  for ( const ext of ARCHIVE_EXTENSIONS ) {
    if (source.toLowerCase().endsWith(ext)) {
      return ext;
    }
  }

  return path.extname(source);
}

function timerSpanFromOptions(x: any): Disposable | undefined {
  const src = bundlingSourceFromOptions(x);
  return src ? profileSpan(`bundle:${src}`, { telemetry: true }) : undefined;
}

/**
 * Get the bundling source from the options object
 *
 * If this is a built-in CDK bundling source, it will have a value here we use to log a timer
 */
function bundlingSourceFromOptions(x: any): string | undefined {
  const value = x[PERF_BUNDLING_SRC_SYM];
  return typeof value === 'string' ? value : undefined;
}
