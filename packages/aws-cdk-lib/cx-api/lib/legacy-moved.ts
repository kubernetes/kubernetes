/**
 * Re-exports of members and types that historically used to live here, but have been moved to a more apt package.
 *
 * Re-exported here for backwards compatibility.
 *
 * To satisfy the (jsii) use of these types in the public API of this package, we do the following:
 *
 * - Re-declare the types that we use in this assembly here (`declare class ...`).
 * - Copy the implementation from the upstream library.
 *
 * That way we satisfy the jsii and backwards compatibility checker, while keeping the implementation
 * centralized in a single place.
 */
import {
  ASSET_PREFIX_SEPARATOR,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_ARGS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_CONTEXTS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_SECRETS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_SSH_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_TARGET_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_DISABLED_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_FROM_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_TO_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_OUTPUTS_KEY,
  ASSET_RESOURCE_METADATA_DOCKERFILE_PATH_KEY,
  ASSET_RESOURCE_METADATA_ENABLED_CONTEXT,
  ASSET_RESOURCE_METADATA_IS_BUNDLED_KEY,
  ASSET_RESOURCE_METADATA_PATH_KEY,
  ASSET_RESOURCE_METADATA_PROPERTY_KEY,
  AVAILABILITY_ZONE_FALLBACK_CONTEXT_KEY,
  AssetManifestArtifact as AssetManifestArtifact_,
  CloudArtifact as CloudArtifact_,
  CloudAssembly as CloudAssembly_,
  CloudAssemblyBuilder as CloudAssemblyBuilder_,
  CloudFormationStackArtifact as CloudFormationStackArtifact_,
  ENDPOINT_SERVICE_AVAILABILITY_ZONE_PROVIDER,
  EnvironmentUtils as EnvironmentUtils_,
  EnvironmentPlaceholders as EnvironmentPlaceholders_,
  NestedCloudAssemblyArtifact as NestedCloudAssemblyArtifact_,
  PATH_METADATA_KEY,
  PROVIDER_ERROR_KEY,
  SSMPARAM_NO_INVALIDATE,
  TreeCloudArtifact as TreeCloudArtifact_,
  UNKNOWN_ACCOUNT,
  UNKNOWN_REGION,
  type AmiContextResponse,
  type AvailabilityZonesContextResponse,
  type EndpointServiceAvailabilityZonesContextResponse,
  type StackMetadata,
} from '@aws-cdk/cloud-assembly-api';
import type * as cxschema from '@aws-cdk/cloud-assembly-schema';

export {
  ASSET_PREFIX_SEPARATOR,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_ARGS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_CONTEXTS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_SECRETS_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_SSH_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_BUILD_TARGET_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_DISABLED_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_FROM_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_CACHE_TO_KEY,
  ASSET_RESOURCE_METADATA_DOCKER_OUTPUTS_KEY,
  ASSET_RESOURCE_METADATA_DOCKERFILE_PATH_KEY,
  ASSET_RESOURCE_METADATA_ENABLED_CONTEXT,
  ASSET_RESOURCE_METADATA_IS_BUNDLED_KEY,
  ASSET_RESOURCE_METADATA_PATH_KEY,
  ASSET_RESOURCE_METADATA_PROPERTY_KEY,
  AVAILABILITY_ZONE_FALLBACK_CONTEXT_KEY,
  ENDPOINT_SERVICE_AVAILABILITY_ZONE_PROVIDER,
  PATH_METADATA_KEY,
  PROVIDER_ERROR_KEY,
  SSMPARAM_NO_INVALIDATE,
  UNKNOWN_ACCOUNT,
  UNKNOWN_REGION,
  AmiContextResponse,
  AvailabilityZonesContextResponse,
  EndpointServiceAvailabilityZonesContextResponse,
  StackMetadata,
};

// ----------------------------------------------------------------------
// Copy/paste of some types we use in the public API of `aws-cdk-lib`.
//
// jsii has a limitation/bug, in that:
//
// - it will correctly consider types *exported* here as seeming to originate
//   from the current jsii assembly; BUT
// - types *used* in the current assembly will be resolved all the way to their
//   source definition, which in this case is not a jsii assembly.
//
// So the types used in `aws-cdk-lib` must have their definition site here,
// and the only way to get that correct is:
//
// - For types: declaring them here and using TS to assert that the types are equivalent.
// - For classes: declaring them here and using runtime JS trickery to use the
//   underlying class type.
//
// This is an indication of a missing jsii feature, that we might or might not build.
// For now, this'll do (pig).

/**
 * @deprecated The official definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface SynthesisMessage {
  readonly level: SynthesisMessageLevel;
  readonly id: string;
  readonly entry: cxschema.MetadataEntry;
}

/**
 * @deprecated The official definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export enum SynthesisMessageLevel {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

/**
 * Properties to `stack.reportMissingContext().
 *
 * @deprecated Use `stack.reportMissingContextKey()` instead
 */
export interface MissingContext {
  /**
   * The missing context key.
   */
  readonly key: string;
  /**
   * The provider from which we expect this context key to be obtained.
   *
   * (This is the old untyped definition, which is necessary for backwards compatibility.
   * See cxschema for a type definition.)
   */
  readonly provider: string;
  /**
   * A set of provider-specific options.
   *
   * (This is the old untyped definition, which is necessary for backwards compatibility.
   * See cxschema for a type definition.)
   */
  readonly props: Record<string, any>;
}

/**
 * Represents a deployable cloud application.
 */
export declare class CloudAssembly implements cxschema.ICloudAssembly {
  /**
   * Return whether the given object is a CloudAssembly.
   *
   * We do attribute detection since we can't reliably use 'instanceof'.
   */
  static isCloudAssembly(x: any): x is CloudAssembly;
  /**
   * Cleans up any temporary assembly directories that got created in this process
   *
   * If a Cloud Assembly is emitted to a temporary directory, its directory gets
   * added to a list. This function iterates over that list and deletes each
   * directory in it, to free up disk space.
   *
   * This function will normally be called automatically during Node process
   * exit and so you don't need to call this. However, some test environments do
   * not properly trigger Node's `exit` event. Notably: Jest does not trigger
   * the `exit` event (<https://github.com/jestjs/jest/issues/10927>).
   *
   * ## Cleaning up temporary directories in jest
   *
   * For Jest, you have to make sure this function is called at the end of the
   * test suite instead:
   *
   * ```js
   * import { CloudAssembly } from 'aws-cdk-lib/cx-api';
   *
   * afterAll(CloudAssembly.cleanupTemporaryDirectories);
   * ```
   *
   * Alternatively, you can use the `setupFilesAfterEnv` feature and use a
   * provided helper script to automatically inject the above into every
   * test file, so you don't have to do it by hand.
   *
   * ```
   * $ npx jest --setupFilesAfterEnv aws-cdk-lib/testhelpers/jest-autoclean
   * ```
   *
   * Or put the following into `jest.config.js`:
   *
   * ```js
   * module.exports = {
   *   // ...
   *   setupFilesAfterEnv: ['aws-cdk-lib/testhelpers/jest-autoclean'],
   * };
   * ```
   */
  static cleanupTemporaryDirectories(): void;
  /**
   * The root directory of the cloud assembly.
   */
  readonly directory: string;
  /**
   * The schema version of the assembly manifest.
   */
  readonly version: string;
  /**
   * All artifacts included in this assembly.
   */
  readonly artifacts: CloudArtifact[];
  /**
   * Runtime information such as module versions used to synthesize this assembly.
   */
  readonly runtime: cxschema.RuntimeInfo;
  /**
   * The raw assembly manifest.
   */
  readonly manifest: cxschema.AssemblyManifest;
  /**
   * Reads a cloud assembly from the specified directory.
   * @param directory - The root directory of the assembly.
   */
  constructor(directory: string, loadOptions?: cxschema.LoadManifestOptions);
  /**
   * Attempts to find an artifact with a specific identity.
   * @returns A `CloudArtifact` object or `undefined` if the artifact does not exist in this assembly.
   * @param id - The artifact ID
   */
  tryGetArtifact(id: string): CloudArtifact | undefined;
  /**
   * Returns a CloudFormation stack artifact from this assembly.
   *
   * Will only search the current assembly.
   *
   * @param stackName - the name of the CloudFormation stack.
   * @throws if there is no stack artifact by that name
   * @throws if there is more than one stack with the same stack name. You can
   * use `getStackArtifact(stack.artifactId)` instead.
   * @returns a `CloudFormationStackArtifact` object.
   */
  getStackByName(stackName: string): CloudFormationStackArtifact;
  /**
   * Returns a CloudFormation stack artifact by name from this assembly.
   * @deprecated renamed to `getStackByName` (or `getStackArtifact(id)`)
   */
  getStack(stackName: string): CloudFormationStackArtifact;
  /**
   * Returns a CloudFormation stack artifact from this assembly.
   *
   * @param artifactId - the artifact id of the stack (can be obtained through `stack.artifactId`).
   * @throws if there is no stack artifact with that id
   * @returns a `CloudFormationStackArtifact` object.
   */
  getStackArtifact(artifactId: string): CloudFormationStackArtifact;
  private tryGetArtifactRecursively;
  /**
   * Returns all the stacks, including the ones in nested assemblies
   */
  get stacksRecursively(): CloudFormationStackArtifact[];
  /**
   * Returns a nested assembly artifact.
   *
   * @param artifactId - The artifact ID of the nested assembly
   */
  getNestedAssemblyArtifact(artifactId: string): NestedCloudAssemblyArtifact;
  /**
   * Returns a nested assembly.
   *
   * @param artifactId - The artifact ID of the nested assembly
   */
  getNestedAssembly(artifactId: string): CloudAssembly;
  /**
   * Returns the tree metadata artifact from this assembly.
   * @throws if there is no metadata artifact by that name
   * @returns a `TreeCloudArtifact` object if there is one defined in the manifest, `undefined` otherwise.
   */
  tree(): TreeCloudArtifact | undefined;
  /**
   * @returns all the CloudFormation stack artifacts that are included in this assembly.
   */
  get stacks(): CloudFormationStackArtifact[];
  /**
   * The nested assembly artifacts in this assembly
   */
  get nestedAssemblies(): NestedCloudAssemblyArtifact[];
  private validateDeps;
  private renderArtifacts;
}
exports.CloudAssembly = CloudAssembly_;

/**
 * Represents an artifact within a cloud assembly.
 */
export declare class CloudArtifact {
  readonly assembly: CloudAssembly;
  readonly id: string;
  /**
   * Returns a subclass of `CloudArtifact` based on the artifact type defined in the artifact manifest.
   *
   * @param assembly - The cloud assembly from which to load the artifact
   * @param id - The artifact ID
   * @param artifact - The artifact manifest
   * @returns the `CloudArtifact` that matches the artifact type or `undefined` if it's an artifact type that is unrecognized by this module.
   */
  static fromManifest(assembly: CloudAssembly, id: string, artifact: cxschema.ArtifactManifest): CloudArtifact | undefined;
  /**
   * The artifact's manifest
   */
  readonly manifest: cxschema.ArtifactManifest;
  /**
   * The set of messages extracted from the artifact's metadata.
   */
  readonly messages: SynthesisMessage[];
  /**
   * Cache of resolved dependencies.
   */
  private _deps?;
  protected constructor(assembly: CloudAssembly, id: string, manifest: cxschema.ArtifactManifest);
  /**
   * Returns all the artifacts that this artifact depends on.
   */
  get dependencies(): CloudArtifact[];
  /**
   * @returns all the metadata entries of a specific type in this artifact.
   */
  findMetadataByType(type: string): MetadataEntryResult[];
  private renderMessages;
  /**
   * An identifier that shows where this artifact is located in the tree
   * of nested assemblies, based on their manifests. Defaults to the normal
   * id. Should only be used in user interfaces.
   */
  get hierarchicalId(): string;

  /**
   * Returns the metadata associated with this Cloud Artifact
   */
  get metadata(): Record<string, cxschema.MetadataEntry[]>;
}
// Make these classes equal to the upstream versions
exports.CloudArtifact = CloudArtifact_;

export declare class CloudFormationStackArtifact extends CloudArtifact {
  /**
   * Checks if `art` is an instance of this class.
   *
   * Use this method instead of `instanceof` to properly detect `CloudFormationStackArtifact`
   * instances, even when the construct library is symlinked.
   *
   * Explanation: in JavaScript, multiple copies of the `cx-api` library on
   * disk are seen as independent, completely different libraries. As a
   * consequence, the class `CloudFormationStackArtifact` in each copy of the `cx-api` library
   * is seen as a different class, and an instance of one class will not test as
   * `instanceof` the other class. `npm install` will not create installations
   * like this, but users may manually symlink construct libraries together or
   * use a monorepo tool: in those cases, multiple copies of the `cx-api`
   * library can be accidentally installed, and `instanceof` will behave
   * unpredictably. It is safest to avoid using `instanceof`, and using
   * this type-testing method instead.
   */
  static isCloudFormationStackArtifact(art: any): art is CloudFormationStackArtifact;
  /**
   * The file name of the template.
   */
  readonly templateFile: string;
  /**
   * The original name as defined in the CDK app.
   */
  readonly originalName: string;
  /**
   * Any assets associated with this stack.
   */
  readonly assets: cxschema.AssetMetadataEntry[];
  /**
   * CloudFormation parameters to pass to the stack.
   */
  readonly parameters: {
    [id: string]: string;
  };
  /**
   * CloudFormation tags to pass to the stack.
   */
  readonly tags: {
    [id: string]: string;
  };
  /**
   * SNS Topics that will receive stack events.
   */
  readonly notificationArns?: string[];
  /**
   * The physical name of this stack.
   */
  readonly stackName: string;
  /**
   * A string that represents this stack. Should only be used in user
   * interfaces. If the stackName has not been set explicitly, or has been set
   * to artifactId, it will return the hierarchicalId of the stack. Otherwise,
   * it will return something like "<hierarchicalId> (<stackName>)"
   */
  readonly displayName: string;
  /**
   * The physical name of this stack.
   * @deprecated renamed to `stackName`
   */
  readonly name: string;
  /**
   * The environment into which to deploy this artifact.
   */
  readonly environment: Environment;
  /**
   * The role that needs to be assumed to deploy the stack
   *
   * @default - No role is assumed (current credentials are used)
   */
  readonly assumeRoleArn?: string;
  /**
   * External ID to use when assuming role for cloudformation deployments
   *
   * @default - No external ID
   */
  readonly assumeRoleExternalId?: string;
  /**
   * Additional options to pass to STS when assuming the role for cloudformation deployments.
   *
   * - `RoleArn` should not be used. Use the dedicated `assumeRoleArn` property instead.
   * - `ExternalId` should not be used. Use the dedicated `assumeRoleExternalId` instead.
   * - `TransitiveTagKeys` defaults to use all keys (if any) specified in `Tags`. E.g, all tags are transitive by default.
   *
   * @see https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/STS.html#assumeRole-property
   * @default - No additional options.
   */
  readonly assumeRoleAdditionalOptions?: {
    [key: string]: any;
  };
  /**
   * The role that is passed to CloudFormation to execute the change set
   *
   * @default - No role is passed (currently assumed role/credentials are used)
   */
  readonly cloudFormationExecutionRoleArn?: string;
  /**
   * The role to use to look up values from the target AWS account
   *
   * @default - No role is assumed (current credentials are used)
   */
  readonly lookupRole?: cxschema.BootstrapRole;
  /**
   * If the stack template has already been included in the asset manifest, its asset URL
   *
   * @default - Not uploaded yet, upload just before deploying
   */
  readonly stackTemplateAssetObjectUrl?: string;
  /**
   * Version of bootstrap stack required to deploy this stack
   *
   * @default - No bootstrap stack required
   */
  readonly requiresBootstrapStackVersion?: number;
  /**
   * Name of SSM parameter with bootstrap stack version
   *
   * @default - Discover SSM parameter by reading stack
   */
  readonly bootstrapStackVersionSsmParameter?: string;
  /**
   * Whether termination protection is enabled for this stack.
   */
  readonly terminationProtection?: boolean;
  /**
   * Whether this stack should be validated by the CLI after synthesis
   *
   * @default - false
   */
  readonly validateOnSynth?: boolean;
  private _template;
  constructor(assembly: CloudAssembly, artifactId: string, artifact: cxschema.ArtifactManifest);
  /**
   * Full path to the template file
   */
  get templateFullPath(): string;
  /**
   * The CloudFormation template for this stack.
   */
  get template(): any;
}
exports.CloudFormationStackArtifact = CloudFormationStackArtifact_;

/**
 * Asset manifest is a description of a set of assets which need to be built and published
 */
export declare class AssetManifestArtifact extends CloudArtifact {
  /**
   * Checks if `art` is an instance of this class.
   *
   * Use this method instead of `instanceof` to properly detect `AssetManifestArtifact`
   * instances, even when the construct library is symlinked.
   *
   * Explanation: in JavaScript, multiple copies of the `cx-api` library on
   * disk are seen as independent, completely different libraries. As a
   * consequence, the class `AssetManifestArtifact` in each copy of the `cx-api` library
   * is seen as a different class, and an instance of one class will not test as
   * `instanceof` the other class. `npm install` will not create installations
   * like this, but users may manually symlink construct libraries together or
   * use a monorepo tool: in those cases, multiple copies of the `cx-api`
   * library can be accidentally installed, and `instanceof` will behave
   * unpredictably. It is safest to avoid using `instanceof`, and using
   * this type-testing method instead.
   */
  static isAssetManifestArtifact(this: void, art: any): art is AssetManifestArtifact;
  /**
   * The file name of the asset manifest
   */
  readonly file: string;
  /**
   * Version of bootstrap stack required to deploy this stack
   */
  readonly requiresBootstrapStackVersion: number | undefined;
  /**
   * Name of SSM parameter with bootstrap stack version
   *
   * @default - Discover SSM parameter by reading stack
   */
  readonly bootstrapStackVersionSsmParameter?: string;
  private _contents?;
  constructor(assembly: CloudAssembly, name: string, artifact: cxschema.ArtifactManifest);
  /**
   * The Asset Manifest contents
   */
  get contents(): cxschema.AssetManifest;
}
exports.AssetManifestArtifact = AssetManifestArtifact_;

/**
 * Asset manifest is a description of a set of assets which need to be built and published
 */
export declare class NestedCloudAssemblyArtifact extends CloudArtifact {
  /**
   * Checks if `art` is an instance of this class.
   *
   * Use this method instead of `instanceof` to properly detect `NestedCloudAssemblyArtifact`
   * instances, even when the construct library is symlinked.
   *
   * Explanation: in JavaScript, multiple copies of the `cx-api` library on
   * disk are seen as independent, completely different libraries. As a
   * consequence, the class `NestedCloudAssemblyArtifact` in each copy of the `cx-api` library
   * is seen as a different class, and an instance of one class will not test as
   * `instanceof` the other class. `npm install` will not create installations
   * like this, but users may manually symlink construct libraries together or
   * use a monorepo tool: in those cases, multiple copies of the `cx-api`
   * library can be accidentally installed, and `instanceof` will behave
   * unpredictably. It is safest to avoid using `instanceof`, and using
   * this type-testing method instead.
   */
  static isNestedCloudAssemblyArtifact(art: any): art is NestedCloudAssemblyArtifact;
  /**
   * The relative directory name of the asset manifest
   */
  readonly directoryName: string;
  /**
   * Display name
   */
  readonly displayName: string;
  /**
   * The nested Assembly
   */
  readonly nestedAssembly: CloudAssembly;
  constructor(assembly: CloudAssembly, name: string, artifact: cxschema.ArtifactManifest);
  /**
   * Full path to the nested assembly directory
   */
  get fullPath(): string;
}
exports.NestedCloudAssemblyArtifact = NestedCloudAssemblyArtifact_;

export declare class TreeCloudArtifact extends CloudArtifact {
  /**
   * Checks if `art` is an instance of this class.
   *
   * Use this method instead of `instanceof` to properly detect `TreeCloudArtifact`
   * instances, even when the construct library is symlinked.
   *
   * Explanation: in JavaScript, multiple copies of the `cx-api` library on
   * disk are seen as independent, completely different libraries. As a
   * consequence, the class `TreeCloudArtifact` in each copy of the `cx-api` library
   * is seen as a different class, and an instance of one class will not test as
   * `instanceof` the other class. `npm install` will not create installations
   * like this, but users may manually symlink construct libraries together or
   * use a monorepo tool: in those cases, multiple copies of the `cx-api`
   * library can be accidentally installed, and `instanceof` will behave
   * unpredictably. It is safest to avoid using `instanceof`, and using
   * this type-testing method instead.
   */
  static isTreeCloudArtifact(art: any): art is TreeCloudArtifact;
  readonly file: string;
  constructor(assembly: CloudAssembly, name: string, artifact: cxschema.ArtifactManifest);
}
exports.TreeCloudArtifact = TreeCloudArtifact_;

/**
 * @deprecated The official definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface MetadataEntryResult extends cxschema.MetadataEntry {
  /**
   * The path in which this entry was defined.
   */
  readonly path: string;
}

/**
 * Models an AWS execution environment, for use within the CDK toolkit.
 *
 * @deprecated The official definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface Environment {
  /** The arbitrary name of this environment (user-set, or at least user-meaningful) */
  readonly name: string;
  /** The AWS account this environment deploys into */
  readonly account: string;
  /** The AWS region name where this environment deploys into */
  readonly region: string;
}

/**
 * Can be used to build a cloud assembly.
 */
export declare class CloudAssemblyBuilder {
  /**
   * The root directory of the resulting cloud assembly.
   */
  readonly outdir: string;
  /**
   * The directory where assets of this Cloud Assembly should be stored
   */
  readonly assetOutdir: string;
  private readonly artifacts;
  private readonly missing;
  private readonly parentBuilder?;
  /**
   * Initializes a cloud assembly builder.
   * @param outdir - The output directory, uses temporary directory if undefined
   */
  constructor(outdir?: string, props?: CloudAssemblyBuilderProps);
  /**
   * Adds an artifact into the cloud assembly.
   * @param id - The ID of the artifact.
   * @param manifest - The artifact manifest
   */
  addArtifact(id: string, manifest: cxschema.ArtifactManifest): void;
  /**
   * Reports that some context is missing in order for this cloud assembly to be fully synthesized.
   * @param missing - Missing context information.
   */
  addMissing(missing: cxschema.MissingContext): void;
  /**
   * Finalizes the cloud assembly into the output directory returns a
   * `CloudAssembly` object that can be used to inspect the assembly.
   */
  buildAssembly(options?: AssemblyBuildOptions): CloudAssembly;
  /**
   * Creates a nested cloud assembly
   */
  createNestedAssembly(artifactId: string, displayName: string): CloudAssemblyBuilder;
  /**
   * Delete the cloud assembly directory
   */
  delete(): void;
}
exports.CloudAssemblyBuilder = CloudAssemblyBuilder_;

/**
 * Convert one CloudAssembly type to another (public to private and vice-versa)
 *
 * @internal
 */
export function _convertCloudAssembly(x: CloudAssembly_): CloudAssembly;
export function _convertCloudAssembly(x: CloudAssembly): CloudAssembly_;
export function _convertCloudAssembly(x: any): any {
  return x;
}

/**
 * Convert one CloudAssemblyBuilder type to another (public to private and vice-versa)
 *
 * @internal
 */
export function _convertCloudAssemblyBuilder(x: CloudAssemblyBuilder_): CloudAssemblyBuilder;
export function _convertCloudAssemblyBuilder(x: CloudAssemblyBuilder): CloudAssemblyBuilder_;
export function _convertCloudAssemblyBuilder(x: any): any {
  return x;
}

/**
 * Construction properties for CloudAssemblyBuilder
 */
export interface CloudAssemblyBuilderProps {
  /**
   * Use the given asset output directory
   *
   * @default - Same as the manifest outdir
   */
  readonly assetOutdir?: string;
  /**
   * If this builder is for a nested assembly, the parent assembly builder
   *
   * @default - This is a root assembly
   */
  readonly parentBuilder?: CloudAssemblyBuilder;
}

export interface AssemblyBuildOptions {
  /**
   * Include the specified runtime information (module versions) in manifest.
   * @default - if this option is not specified, runtime info will not be included
   * @deprecated All template modifications that should result from this should
   * have already been inserted into the template.
   */
  readonly runtimeInfo?: RuntimeInfo;
}

/**
 * Backwards compatibility for when `RuntimeInfo`
 * was defined here. This is necessary because its used as an input in the stable
 * @aws-cdk/core library.
 *
 * @deprecated moved to package 'cloud-assembly-schema'
 * @see core.ConstructNode.synth
 */
export interface RuntimeInfo extends cxschema.RuntimeInfo {
}

/**
 * Placeholders which can be used manifests
 *
 * These can occur both in the Asset Manifest as well as the general
 * Cloud Assembly manifest.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export declare class EnvironmentPlaceholders {
  /**
   * Insert this into the destination fields to be replaced with the current region
   */
  static readonly CURRENT_REGION = '${AWS::Region}';
  /**
   * Insert this into the destination fields to be replaced with the current account
   */
  static readonly CURRENT_ACCOUNT = '${AWS::AccountId}';
  /**
   * Insert this into the destination fields to be replaced with the current partition
   */
  static readonly CURRENT_PARTITION = '${AWS::Partition}';
  /**
   * Replace the environment placeholders in all strings found in a complex object.
   *
   * Duplicated between cdk-assets and aws-cdk CLI because we don't have a good single place to put it
   * (they're nominally independent tools).
   */
  static replace(object: any, values: EnvironmentPlaceholderValues): any;
  /**
   * Like 'replace', but asynchronous
   */
  static replaceAsync(object: any, provider: IEnvironmentPlaceholderProvider): Promise<any>;
  private static recurse;
}
exports.EnvironmentPlaceholders = EnvironmentPlaceholders_;

/**
 * Return the appropriate values for the environment placeholders
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface EnvironmentPlaceholderValues {
  /**
   * Return the region
   */
  readonly region: string;
  /**
   * Return the account
   */
  readonly accountId: string;
  /**
   * Return the partition
   */
  readonly partition: string;
}
/**
 * Return the appropriate values for the environment placeholders
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface IEnvironmentPlaceholderProvider {
  /**
   * Return the region
   */
  region(): Promise<string>;
  /**
   * Return the account
   */
  accountId(): Promise<string>;
  /**
   * Return the partition
   */
  partition(): Promise<string>;
}

/**
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export declare class EnvironmentUtils {
  static parse(environment: string): Environment;
  /**
   * Build an environment object from an account and region
   */
  static make(account: string, region: string): Environment;
  /**
   * Format an environment string from an account and region
   */
  static format(account: string, region: string): string;
}
exports.EnvironmentUtils = EnvironmentUtils_;

/**
 * Properties of a discovered SecurityGroup.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface SecurityGroupContextResponse {
  /**
   * The security group's id.
   */
  readonly securityGroupId: string;
  /**
   * Whether the security group allows all outbound traffic. This will be true
   * when the security group has all-protocol egress permissions to access both
   * `0.0.0.0/0` and `::/0`.
   */
  readonly allowAllOutbound: boolean;
}

/**
 * The type of subnet group.
 * Same as SubnetType in the aws-cdk-lib/aws-ec2 package,
 * but we can't use that because of cyclical dependencies.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export enum VpcSubnetGroupType {
  /** Public subnet group type. */
  PUBLIC = 'Public',
  /** Private subnet group type. */
  PRIVATE = 'Private',
  /** Isolated subnet group type. */
  ISOLATED = 'Isolated',
}
/**
 * A subnet representation that the VPC provider uses.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface VpcSubnet {
  /** The identifier of the subnet. */
  readonly subnetId: string;
  /**
   * The code of the availability zone this subnet is in
   * (for example, 'us-west-2a').
   */
  readonly availabilityZone: string;
  /** The identifier of the route table for this subnet. */
  readonly routeTableId: string;
  /**
   * CIDR range of the subnet
   *
   * @default - CIDR information not available
   */
  readonly cidr?: string;
}
/**
 * A group of subnets returned by the VPC provider.
 * The included subnets do NOT have to be symmetric!
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface VpcSubnetGroup {
  /**
   * The name of the subnet group,
   * determined by looking at the tags of of the subnets
   * that belong to it.
   */
  readonly name: string;
  /** The type of the subnet group. */
  readonly type: VpcSubnetGroupType;
  /**
   * The subnets that are part of this group.
   * There is no condition that the subnets have to be symmetric
   * in the group.
   */
  readonly subnets: VpcSubnet[];
}
/**
 * Properties of a discovered VPC
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface VpcContextResponse {
  /**
   * VPC id
   */
  readonly vpcId: string;
  /**
   * VPC cidr
   *
   * @default - CIDR information not available
   */
  readonly vpcCidrBlock?: string;
  /**
   * AZs
   */
  readonly availabilityZones: string[];
  /**
   * IDs of all public subnets
   *
   * Element count: #(availabilityZones) · #(publicGroups)
   */
  readonly publicSubnetIds?: string[];
  /**
   * Name of public subnet groups
   *
   * Element count: #(publicGroups)
   */
  readonly publicSubnetNames?: string[];
  /**
   * Route Table IDs of public subnet groups.
   *
   * Element count: #(availabilityZones) · #(publicGroups)
   */
  readonly publicSubnetRouteTableIds?: string[];
  /**
   * IDs of all private subnets
   *
   * Element count: #(availabilityZones) · #(privateGroups)
   */
  readonly privateSubnetIds?: string[];
  /**
   * Name of private subnet groups
   *
   * Element count: #(privateGroups)
   */
  readonly privateSubnetNames?: string[];
  /**
   * Route Table IDs of private subnet groups.
   *
   * Element count: #(availabilityZones) · #(privateGroups)
   */
  readonly privateSubnetRouteTableIds?: string[];
  /**
   * IDs of all isolated subnets
   *
   * Element count: #(availabilityZones) · #(isolatedGroups)
   */
  readonly isolatedSubnetIds?: string[];
  /**
   * Name of isolated subnet groups
   *
   * Element count: #(isolatedGroups)
   */
  readonly isolatedSubnetNames?: string[];
  /**
   * Route Table IDs of isolated subnet groups.
   *
   * Element count: #(availabilityZones) · #(isolatedGroups)
   */
  readonly isolatedSubnetRouteTableIds?: string[];
  /**
   * The VPN gateway ID
   */
  readonly vpnGatewayId?: string;
  /**
   * The subnet groups discovered for the given VPC.
   * Unlike the above properties, this will include asymmetric subnets,
   * if the VPC has any.
   * This property will only be populated if `VpcContextQuery.returnAsymmetricSubnets`
   * is true.
   *
   * @default - no subnet groups will be returned unless `VpcContextQuery.returnAsymmetricSubnets` is true
   */
  readonly subnetGroups?: VpcSubnetGroup[];
  /**
   * The region in which the VPC is in.
   *
   * @default - Region of the parent stack
   */
  readonly region?: string;
  /**
   * The ID of the AWS account that owns the VPC.
   *
   * @default the account id of the parent stack
   */
  readonly ownerAccountId?: string;
}

/**
 * Load balancer ip address type.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export enum LoadBalancerIpAddressType {
  /**
   * IPV4 ip address
   */
  IPV4 = 'ipv4',
  /**
   * Dual stack address
   */
  DUAL_STACK = 'dualstack',
  /**
   * IPv6 only public addresses, with private IPv4 and IPv6 addresses
   */
  DUAL_STACK_WITHOUT_PUBLIC_IPV4 = 'dualstack-without-public-ipv4',
}
/**
 * Properties of a discovered load balancer
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface LoadBalancerContextResponse {
  /**
   * The ARN of the load balancer.
   */
  readonly loadBalancerArn: string;
  /**
   * The hosted zone ID of the load balancer's name.
   */
  readonly loadBalancerCanonicalHostedZoneId: string;
  /**
   * Load balancer's DNS name
   */
  readonly loadBalancerDnsName: string;
  /**
   * Type of IP address
   */
  readonly ipAddressType: LoadBalancerIpAddressType;
  /**
   * Load balancer's security groups
   */
  readonly securityGroupIds: string[];
  /**
   * Load balancer's VPC
   */
  readonly vpcId: string;
}
/**
 * Properties of a discovered load balancer listener.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface LoadBalancerListenerContextResponse {
  /**
   * The ARN of the listener.
   */
  readonly listenerArn: string;
  /**
   * The port the listener is listening on.
   */
  readonly listenerPort: number;
  /**
   * The security groups of the load balancer.
   */
  readonly securityGroupIds: string[];
}

/**
 * Properties of a discovered key
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface KeyContextResponse {
  /**
   * Id of the key
   */
  readonly keyId: string;
}

/**
 * Query to hosted zone context provider
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface EndpointServiceAvailabilityZonesContextQuery {
  /**
   * Query account
   */
  readonly account?: string;
  /**
   * Query region
   */
  readonly region?: string;
  /**
   * Query service name
   */
  readonly serviceName?: string;
}

/**
 * Backwards compatibility for when `MetadataEntry`
 * was defined here. This is necessary because its used as an input in the stable
 * @aws-cdk/core library.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 * @see core.ConstructNode.metadata
 */
export interface MetadataEntry extends cxschema.MetadataEntry {
}

/**
 * Artifact properties for CloudFormation stacks.
 *
 * @deprecated The definition of this type has moved to `@aws-cdk/cloud-assembly-api`.
 */
export interface AwsCloudFormationStackProperties {
  /**
   * A file relative to the assembly root which contains the CloudFormation template for this stack.
   */
  readonly templateFile: string;
  /**
   * Values for CloudFormation stack parameters that should be passed when the stack is deployed.
   */
  readonly parameters?: {
    [id: string]: string;
  };
  /**
   * The name to use for the CloudFormation stack.
   * @default - name derived from artifact ID
   */
  readonly stackName?: string;
  /**
   * Whether to enable termination protection for this stack.
   *
   * @default false
   */
  readonly terminationProtection?: boolean;
}
