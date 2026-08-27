import * as cxschema from '@aws-cdk/cloud-assembly-schema';
import { Annotations } from './annotations';
import type { CfnCondition } from './cfn-condition';
// import required to be here, otherwise causes a cycle when running the generated JavaScript
/* eslint-disable import/order */
import { CfnRefElement } from './cfn-element';
import type { CfnCreationPolicy, CfnUpdatePolicy } from './cfn-resource-policy';
import { CfnDeletionPolicy } from './cfn-resource-policy';
import type { Construct, Node } from 'constructs';
import { dispatchDependencyOperation } from './private/deps';
import { CfnReference } from './private/cfn-reference';
import type { Reference } from './reference';
import type { RemovalPolicyOptions } from './removal-policy';
import { RemovalPolicy } from './removal-policy';
import { debugModeEnabled } from './debug';
import { TagManager } from './tag-manager';
import { capitalizePropertyNames, ignoreEmpty, PostResolveToken } from './util';
import { FeatureFlags } from './feature-flags';
import type { ResolutionTypeHint } from './type-hints';
import * as cxapi from '../../cx-api';
import type { ReferenceStrength } from './cross-stack-reference-strength';
import { ValidationError } from './errors';
import { deepMerge } from './private/deep-merge';
import type { ResourceEnvironment } from './environment';
import { lit } from './private/literal-string';
import { captureStackTrace } from './private/stack-trace';
import { Stack } from './stack';
import { isCfnResource, STACK_TYPE } from './private/core-construct-finders';

export interface CfnResourceProps {
  /**
   * CloudFormation resource type (e.g. `AWS::S3::Bucket`).
   */
  readonly type: string;

  /**
   * Resource properties.
   *
   * @default - No resource properties.
   */
  readonly properties?: { [name: string]: any };
}

/**
 * Represents a CloudFormation resource.
 */
export class CfnResource extends CfnRefElement {
  /**
   * Check whether the given object is a CfnResource
   */
  public static isCfnResource(this: void, x: any): x is CfnResource {
    return isCfnResource(x);
  }

  // MAINTAINERS NOTE: this class serves as the base class for the generated L1
  // ("CFN") resources (such as `s3.CfnBucket`). These resources will have a
  // property for each CloudFormation property of the resource. This means that
  // if at some point in the future a property is introduced with a name similar
  // to one of the properties here, it will be "masked" by the derived class. To
  // that end, we prefix all properties in this class with `cfnXxx` with the
  // hope to avoid those conflicts in the future.

  /**
   * Options for this resource, such as condition, update policy etc.
   */
  public readonly cfnOptions: ICfnResourceOptions = {};

  /**
   * AWS resource type.
   */
  public readonly cfnResourceType: string;

  /**
   * AWS CloudFormation resource properties.
   *
   * This object is returned via cfnProperties
   * @internal
   */
  protected readonly _cfnProperties: any;

  /**
   * An object to be merged on top of the entire resource definition.
   */
  private readonly rawOverrides: any = Object.create(null); // Prevent prototype pollution

  /**
   * Logical IDs of dependencies.
   *
   * Is filled during prepare().
   */
  private dependsOn: Set<CfnResource> | undefined;

  private _crossStackReferenceStrength?: ReferenceStrength;

  protected readonly cfnPropertyNames: Record<string, string> = {};

  /**
   * Creates a resource construct.
   * @param cfnResourceType The CloudFormation type of this resource (e.g. AWS::DynamoDB::Table)
   */
  constructor(scope: Construct, id: string, props: CfnResourceProps) {
    super(scope, id);

    if (!props.type) {
      throw new ValidationError(lit`IsRequiredPropertyRequired`, 'The `type` property is required', this);
    }

    this.cfnResourceType = props.type;
    this._cfnProperties = props.properties || {};

    // if aws:cdk:enable-path-metadata is set, embed the current construct's
    // path in the CloudFormation template, so it will be possible to trace
    // back to the actual construct path.
    if (this.node.tryGetContext(cxapi.PATH_METADATA_ENABLE_CONTEXT)) {
      this.addMetadata(cxapi.PATH_METADATA_KEY, this.node.path);
    }
  }

  public get env(): ResourceEnvironment {
    return {
      account: this.stack.account,
      region: this.stack.region,
    };
  }

  /**
   * Sets the deletion policy of the resource based on the removal policy specified.
   *
   * The Removal Policy controls what happens to this resource when it stops
   * being managed by CloudFormation, either because you've removed it from the
   * CDK application or because you've made a change that requires the resource
   * to be replaced.
   *
   * The resource can be deleted (`RemovalPolicy.DESTROY`), or left in your AWS
   * account for data recovery and cleanup later (`RemovalPolicy.RETAIN`). In some
   * cases, a snapshot can be taken of the resource prior to deletion
   * (`RemovalPolicy.SNAPSHOT`). A list of resources that support this policy
   * can be found in the following link:
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-attribute-deletionpolicy.html#aws-attribute-deletionpolicy-options
   */
  public applyRemovalPolicy(policy: RemovalPolicy | undefined, options: RemovalPolicyOptions = {}) {
    policy = policy || options.default || RemovalPolicy.RETAIN;

    let deletionPolicy;
    let updateReplacePolicy;

    switch (policy) {
      case RemovalPolicy.DESTROY:
        deletionPolicy = CfnDeletionPolicy.DELETE;
        updateReplacePolicy = CfnDeletionPolicy.DELETE;
        break;

      case RemovalPolicy.RETAIN:
        deletionPolicy = CfnDeletionPolicy.RETAIN;
        updateReplacePolicy = CfnDeletionPolicy.RETAIN;
        break;

      case RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE:
        deletionPolicy = CfnDeletionPolicy.RETAIN_EXCEPT_ON_CREATE;
        updateReplacePolicy = CfnDeletionPolicy.RETAIN;
        break;

      case RemovalPolicy.SNAPSHOT:
        // https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-attribute-deletionpolicy.html
        const snapshottableResourceTypes = [
          'AWS::DocDB::DBCluster',
          'AWS::EC2::Volume',
          'AWS::ElastiCache::CacheCluster',
          'AWS::ElastiCache::ReplicationGroup',
          'AWS::Neptune::DBCluster',
          'AWS::RDS::DBCluster',
          'AWS::RDS::DBInstance',
          'AWS::Redshift::Cluster',
        ];

        // error if flag is set, warn if flag is not
        const problematicSnapshotPolicy = !snapshottableResourceTypes.includes(this.cfnResourceType);
        if (problematicSnapshotPolicy) {
          if (FeatureFlags.of(this).isEnabled(cxapi.VALIDATE_SNAPSHOT_REMOVAL_POLICY)) {
            throw new ValidationError(lit`SnapshotRemovalNotSupported`, `${this.cfnResourceType} does not support snapshot removal policy`, this);
          } else {
            Annotations.of(this).addWarningV2(`@aws-cdk/core:${this.cfnResourceType}SnapshotRemovalPolicyIgnored`, `${this.cfnResourceType} does not support snapshot removal policy. This policy will be ignored.`);
          }
        }

        deletionPolicy = CfnDeletionPolicy.SNAPSHOT;
        updateReplacePolicy = CfnDeletionPolicy.SNAPSHOT;
        break;

      default:
        throw new ValidationError(lit`InvalidRemovalPolicy`, `Invalid removal policy: ${policy}`, this);
    }

    this.cfnOptions.deletionPolicy = deletionPolicy;
    if (options.applyToUpdateReplacePolicy !== false) {
      this.cfnOptions.updateReplacePolicy = updateReplacePolicy;
    }
  }

  /**
   * Sets the cross-stack reference strength for this resource.
   *
   * When set, any cross-stack reference to this resource will use the specified
   * strength instead of the global default from the consuming stack's context.
   *
   * @param strength - The reference strength to use for this resource.
   */
  public applyCrossStackReferenceStrength(strength: ReferenceStrength): void {
    this._crossStackReferenceStrength = strength;
  }

  /**
   * @internal
   */
  public get _crossStackReferenceStrengthOverride(): ReferenceStrength | undefined {
    return this._crossStackReferenceStrength;
  }

  /**
   * Returns a token for an runtime attribute of this resource.
   * Ideally, use generated attribute accessors (e.g. `resource.arn`), but this can be used for future compatibility
   * in case there is no generated attribute.
   * @param attributeName The name of the attribute.
   */
  public getAtt(attributeName: string, typeHint?: ResolutionTypeHint): Reference {
    return CfnReference.for(this, attributeName, undefined, typeHint);
  }

  /**
   * Adds an override to the synthesized CloudFormation resource. To add a
   * property override, either use `addPropertyOverride` or prefix `path` with
   * "Properties." (i.e. `Properties.TopicName`).
   *
   * If the override is nested, separate each nested level using a dot (.) in the path parameter.
   * If there is an array as part of the nesting, specify the index in the path.
   *
   * To include a literal `.` in the property name, prefix with a `\`. In most
   * programming languages you will need to write this as `"\\."` because the
   * `\` itself will need to be escaped.
   *
   * For example,
   * ```typescript
   * cfnResource.addOverride('Properties.GlobalSecondaryIndexes.0.Projection.NonKeyAttributes', ['myattribute']);
   * cfnResource.addOverride('Properties.GlobalSecondaryIndexes.1.ProjectionType', 'INCLUDE');
   * ```
   * would add the overrides
   * ```json
   * "Properties": {
   *   "GlobalSecondaryIndexes": [
   *     {
   *       "Projection": {
   *         "NonKeyAttributes": [ "myattribute" ]
   *         ...
   *       }
   *       ...
   *     },
   *     {
   *       "ProjectionType": "INCLUDE"
   *       ...
   *     },
   *   ]
   *   ...
   * }
   * ```
   *
   * The `value` argument to `addOverride` will not be processed or translated
   * in any way. Pass raw JSON values in here with the correct capitalization
   * for CloudFormation. If you pass CDK classes or structs, they will be
   * rendered with lowercased key names, and CloudFormation will reject the
   * template.
   *
   * @param path - The path of the property, you can use dot notation to
   *        override values in complex types. Any intermediate keys
   *        will be created as needed.
   * @param value - The value. Could be primitive or complex.
   */
  public addOverride(path: string, value: any) {
    const parts = splitOnPeriods(path);
    let curr: any = this.rawOverrides;

    while (parts.length > 1) {
      const key = parts.shift()!;

      // if we can't recurse further or the previous value is not an
      // object overwrite it with an object.
      const isObject = curr[key] != null && typeof (curr[key]) === 'object' && !Array.isArray(curr[key]);
      if (!isObject) {
        curr[key] = Object.create(null); // Prevent prototype pollution
      }

      curr = curr[key];
    }

    const lastKey = parts.shift()!;
    curr[lastKey] = value;
  }

  /**
   * Syntactic sugar for `addOverride(path, undefined)`.
   * @param path The path of the value to delete
   */
  public addDeletionOverride(path: string) {
    this.addOverride(path, undefined);
  }

  /**
   * Adds an override to a resource property.
   *
   * Syntactic sugar for `addOverride("Properties.<...>", value)`.
   *
   * @param propertyPath The path of the property
   * @param value The value
   */
  public addPropertyOverride(propertyPath: string, value: any) {
    const parts = splitOnPeriods(propertyPath);
    traceProperty(this.node, parts[0]);
    this.addOverride(`Properties.${propertyPath}`, value);
  }

  /**
   * Adds an override that deletes the value of a property from the resource definition.
   * @param propertyPath The path to the property.
   */
  public addPropertyDeletionOverride(propertyPath: string) {
    this.addPropertyOverride(propertyPath, undefined);
  }

  public cfnPropertyName(cdkPropertyName: string): string | undefined {
    return this.cfnPropertyNames[cdkPropertyName];
  }

  /**
   * Indicates that this resource depends on another resource and cannot be
   * provisioned unless the other resource has been successfully provisioned.
   *
   * This can be used for resources across stacks (or nested stack) boundaries
   * and the dependency will automatically be transferred to the relevant scope.
   *
   * This method has been renamed to `addResourceDependency`, which makes it
   * more clear that this method operates at a different level from the
   * construct-level `construct.node.addDependency()` mechanism.
   *
   * @deprecated Use `addResourceDependency` instead.
   */
  public addDependsOn(target: CfnResource) {
    return this.addResourceDependency(target);
  }

  /**
   * Indicates that this resource depends on another resource and cannot be
   * provisioned unless the other resource has been successfully provisioned.
   *
   * This can be used for resources across stacks (or nested stack) boundaries
   * and the dependency will automatically be transferred to the relevant scope.
   *
   * This method only adds dependencies between L1 resources. If you are
   * looking for a generic construct-to-construct dependency mechanism that works
   * for all constructs including L2s, use `construct.node.addDependency` instead.
   */
  public addResourceDependency(target: CfnResource, reason?: string) {
    // skip this dependency if the target is not part of the output
    if (!target.shouldSynthesize()) {
      return;
    }

    dispatchDependencyOperation({
      kind: 'add',
      source: this,
      target,
      reason: reason ?? `<${this.node.path}>.addResourceDependency(<${target.node.path}>)`,
    });
  }

  /**
   * Indicates that this resource depends on another resource and cannot be
   * provisioned unless the other resource has been successfully provisioned.
   *
   * This method has been renamed to `addResourceDependency` to more clearly
   * set it apart from `construct.node.addDependency`. See the documentation
   * of that function for more details.
   *
   * @deprecated Use `addResourceDependency` instead.
   */
  public addDependency(target: CfnResource) {
    return this.addResourceDependency(target);
  }

  /**
   * Indicates that this resource no longer depends on another resource.
   *
   * This can be used for resources across stacks (including nested stacks)
   * and the dependency will automatically be removed from the relevant scope.
   */
  public removeResourceDependency(target: CfnResource): void {
    // skip this dependency if the target is not part of the output
    if (!target.shouldSynthesize()) {
      return;
    }

    dispatchDependencyOperation({
      kind: 'remove',
      source: this,
      target,
    });
  }

  /**
   * Indicates that this resource no longer depends on another resource.
   *
   * This can be used for resources across stacks (including nested stacks)
   * and the dependency will automatically be removed from the relevant scope.
   *
   * @deprecated Use `removeResourceDependency` instead
   */
  public removeDependency(target: CfnResource): void {
    return this.removeResourceDependency(target);
  }

  /**
   * Retrieves an array of resources and stacks this resource depends on.
   *
   * For resources depended on directly, returns the `CfnResource` object. For
   * dependencies on other stacks, returns the `Stack` object. The order of the
   * array is not guaranteed.
   */
  public obtainDependencies(): Array<CfnResource | Stack> {
    const ret: Array<CfnResource | Stack> = [];
    ret.push(...this._directResourceDependencies());

    let stack: Stack | undefined = Stack.of(this);
    while (stack) {
      ret.push(...stack._stackDependenciesCausedBy(this).filter((x) => STACK_TYPE.isMarked(x) || CfnResource.isCfnResource(x)));
      stack = stack.nestedStackParent;
    }

    return ret;
  }

  /**
   * Replaces one dependency with another.
   * @param target The dependency to replace
   * @param newTarget The new dependency to add
   */
  public replaceDependency(target: CfnResource, newTarget: CfnResource): void {
    if (this.obtainDependencies().includes(target)) {
      this.removeResourceDependency(target);
      this.addResourceDependency(newTarget);
    } else {
      throw new ValidationError(lit`CannotReplaceDependency`, `"${this.node.path}" does not depend on "${target.node.path}"`, this);
    }
  }

  /**
   * Add a value to the CloudFormation Resource Metadata
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/metadata-section-structure.html
   *
   * Note that this is a different set of metadata from CDK node metadata; this
   * metadata ends up in the stack template under the resource, whereas CDK
   * node metadata ends up in the Cloud Assembly.
   */
  public addMetadata(key: string, value: any) {
    if (!this.cfnOptions.metadata) {
      this.cfnOptions.metadata = {};
    }

    this.cfnOptions.metadata[key] = value;
  }

  /**
   * Retrieve a value value from the CloudFormation Resource Metadata
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/metadata-section-structure.html
   *
   * Note that this is a different set of metadata from CDK node metadata; this
   * metadata ends up in the stack template under the resource, whereas CDK
   * node metadata ends up in the Cloud Assembly.
   */
  public getMetadata(key: string): any {
    return this.cfnOptions.metadata?.[key];
  }

  /**
   * @returns a string representation of this resource
   */
  public toString() {
    return `${super.toString()} [${this.cfnResourceType}]`;
  }

  /**
   * Called by `dispatchDependencyOperation` to realize a dependency between two resources.
   *
   * All validation for appropriate scope has already been done, cycle detection has not been done yet.
   *
   * @internal
   */
  public _addResourceDependency(target: CfnResource) {
    if (!this.dependsOn) {
      this.dependsOn = new Set();
    }
    this.dependsOn.add(target);

    if (process.env.CDK_DEBUG_DEPS) {
      // eslint-disable-next-line no-console
      console.error(`[CDK_DEBUG_DEPS] resource "${this.node.path}" depends on "${target.node.path}"`);
    }
  }

  /**
   * Get a shallow copy of dependencies between this resource and other resources
   * in the same stack.
   *
   * @internal
   */
  public _directResourceDependencies() {
    return Array.from(this.dependsOn?.values() ?? []);
  }

  /**
   * Called by `dispatchDependencyOperation` to remove a dependency between two resources.
   *
   * All validation for appropriateness has already been done.
   *
   * @internal
   */
  public _removeResourceDependency(target: CfnResource) {
    this.dependsOn?.delete(target);

    if (process.env.CDK_DEBUG_DEPS) {
      // eslint-disable-next-line no-console
      console.error(`[CDK_DEBUG_DEPS] resource "${this.node.path}" no longer depends on "${target.node.path}"`);
    }
  }

  /**
   * Emits CloudFormation for this resource.
   * @internal
   */
  public _toCloudFormation(): object {
    if (!this.shouldSynthesize()) {
      return {};
    }

    try {
      const ret = {
        Resources: {
          // Post-Resolve operation since otherwise deepMerge is going to mix values into
          // the Token objects returned by ignoreEmpty.
          [this.logicalId]: new PostResolveToken({
            Type: this.cfnResourceType,
            Properties: ignoreEmpty(this.cfnProperties),
            DependsOn: ignoreEmpty(renderDependsOn(this.dependsOn)),
            CreationPolicy: capitalizePropertyNames(this, renderCreationPolicy(this.cfnOptions.creationPolicy)),
            UpdatePolicy: capitalizePropertyNames(this, this.cfnOptions.updatePolicy),
            UpdateReplacePolicy: capitalizePropertyNames(this, this.cfnOptions.updateReplacePolicy),
            DeletionPolicy: capitalizePropertyNames(this, this.cfnOptions.deletionPolicy),
            Version: this.cfnOptions.version,
            Description: this.cfnOptions.description,
            Metadata: ignoreEmpty(this.cfnOptions.metadata),
            Condition: this.cfnOptions.condition && this.cfnOptions.condition.logicalId,
          }, (resourceDef, context) => {
            const renderedProps = this.renderProperties(resourceDef.Properties || {});
            if (renderedProps) {
              const hasDefined = Object.values(renderedProps).find(v => v !== undefined);
              resourceDef.Properties = hasDefined !== undefined ? renderedProps : undefined;
            }
            const resolvedRawOverrides = context.resolve(this.rawOverrides, {
              // we need to preserve the empty elements here,
              // as that's how removing overrides are represented as
              removeEmpty: false,
            });
            return deepMerge(resourceDef, resolvedRawOverrides);
          }),
        },
      };
      return ret;
    } catch (e: any) {
      // Change message
      e.message = `While synthesizing ${this.node.path}: ${e.message}`;
      // Adjust stack trace (make it look like node built it, too...)
      const trace = this.creationStack;
      if (trace) {
        const creationStack = ['--- resource created at ---', ...trace].join('\n  at ');
        const problemTrace = e.stack.slice(e.stack.indexOf(e.message) + e.message.length);
        e.stack = `${e.message}\n  ${creationStack}\n  --- problem discovered at ---${problemTrace}`;
      }

      // Re-throw
      throw e;
    }

    // returns the set of logical ID (tokens) this resource depends on
    // sorted by construct paths to ensure test determinism
    function renderDependsOn(dependsOn: Iterable<CfnResource> | undefined) {
      if (!dependsOn) {
        return [];
      }

      return Array
        .from(dependsOn)
        .filter((r) => r.shouldSynthesize())
        .sort((x, y) => x.node.path.localeCompare(y.node.path))
        .map(r => r.logicalId);
    }

    function renderCreationPolicy(policy: CfnCreationPolicy | undefined): any {
      if (!policy) { return undefined; }
      const result: any = { ...policy };
      if (policy.resourceSignal && policy.resourceSignal.timeout) {
        result.resourceSignal = policy.resourceSignal;
      }
      return result;
    }
  }

  protected get cfnProperties(): { [key: string]: any } {
    const props = this._cfnProperties || {};
    const tagMgr = TagManager.of(this);
    if (tagMgr) {
      const tagsProp: { [key: string]: any } = {};
      // If this object has a TagManager, then render it out into the correct field. We assume there
      // is no shadow tags object, so we don't pass anything to renderTags().
      tagsProp[tagMgr.tagPropertyName] = tagMgr.renderTags();
      return deepMerge(props, tagsProp);
    }
    return props;
  }

  protected renderProperties(props: { [key: string]: any }): { [key: string]: any } {
    return props;
  }

  /**
   * Deprecated
   * @deprecated use `updatedProperties`
   *
   * Return properties modified after initiation
   *
   * Resources that expose mutable properties should override this function to
   * collect and return the properties object for this resource.
   */
  protected get updatedProperites(): { [key: string]: any } {
    return this.updatedProperties;
  }

  /**
   * Return properties modified after initiation
   *
   * Resources that expose mutable properties should override this function to
   * collect and return the properties object for this resource.
   */
  protected get updatedProperties(): { [key: string]: any } {
    return this._cfnProperties;
  }

  protected validateProperties(_properties: any) {
    // Nothing
  }

  /**
   * Can be overridden by subclasses to determine if this resource will be rendered
   * into the cloudformation template.
   *
   * @returns `true` if the resource should be included or `false` is the resource
   * should be omitted.
   */
  protected shouldSynthesize() {
    return true;
  }
}

export enum TagType {
  STANDARD = 'StandardTag',
  AUTOSCALING_GROUP = 'AutoScalingGroupTag',
  MAP = 'StringToStringMap',
  KEY_VALUE = 'KeyValue',
  NOT_TAGGABLE = 'NotTaggable',
}

export interface ICfnResourceOptions {
  /**
   * A condition to associate with this resource. This means that only if the condition evaluates to 'true' when the stack
   * is deployed, the resource will be included. This is provided to allow CDK projects to produce legacy templates, but normally
   * there is no need to use it in CDK projects.
   */
  condition?: CfnCondition;

  /**
   * Associate the CreationPolicy attribute with a resource to prevent its status from reaching create complete until
   * AWS CloudFormation receives a specified number of success signals or the timeout period is exceeded. To signal a
   * resource, you can use the cfn-signal helper script or SignalResource API. AWS CloudFormation publishes valid signals
   * to the stack events so that you track the number of signals sent.
   */
  creationPolicy?: CfnCreationPolicy;

  /**
   * With the DeletionPolicy attribute you can preserve or (in some cases) backup a resource when its stack is deleted.
   * You specify a DeletionPolicy attribute for each resource that you want to control. If a resource has no DeletionPolicy
   * attribute, AWS CloudFormation deletes the resource by default. Note that this capability also applies to update operations
   * that lead to resources being removed.
   */
  deletionPolicy?: CfnDeletionPolicy;

  /**
   * Use the UpdatePolicy attribute to specify how AWS CloudFormation handles updates to the AWS::AutoScaling::AutoScalingGroup
   * resource. AWS CloudFormation invokes one of three update policies depending on the type of change you make or whether a
   * scheduled action is associated with the Auto Scaling group.
   */
  updatePolicy?: CfnUpdatePolicy;

  /**
   * Use the UpdateReplacePolicy attribute to retain or (in some cases) backup the existing physical instance of a resource
   * when it is replaced during a stack update operation.
   */
  updateReplacePolicy?: CfnDeletionPolicy;

  /**
   * The version of this resource.
   * Used only for custom CloudFormation resources.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-cfn-customresource.html
   */
  version?: string;

  /**
   * The description of this resource.
   * Used for informational purposes only, is not processed in any way
   * (and stays with the CloudFormation template, is not passed to the underlying resource,
   * even if it does have a 'description' property).
   */
  description?: string;

  /**
   * Metadata associated with the CloudFormation resource. This is not the same as the construct metadata which can be added
   * using construct.addMetadata(), but would not appear in the CloudFormation template automatically.
   */
  metadata?: { [key: string]: any };
}

/**
 * Split on periods while processing escape characters \
 */
function splitOnPeriods(x: string): string[] {
  // Build this list in reverse because it's more convenient to get the "current"
  // item by doing ret[0] than by ret[ret.length - 1].
  const ret = [''];
  for (let i = 0; i < x.length; i++) {
    if (x[i] === '\\' && i + 1 < x.length) {
      ret[0] += x[i + 1];
      i++;
    } else if (x[i] === '.') {
      ret.unshift('');
    } else {
      ret[0] += x[i];
    }
  }

  ret.reverse();
  return ret;
}

/**
 * Records a metadata entry on a construct node to trace a property assignment.
 *
 * When debug mode is enabled (via the `CDK_DEBUG` environment variable),
 * this attaches `aws:cdk:propertyAssignment` metadata to the given node,
 * including a stack trace pointing back to the caller. This is useful for
 * diagnosing where a particular property value was set during synthesis.
 *
 * This is a no-op when debug mode is not enabled.
 *
 * @param node the construct node to attach the metadata to.
 * @param propertyName the name of the property being assigned.
 */
export function traceProperty(node: Node, propertyName: string) {
  if (debugModeEnabled()) {
    node.addMetadata(cxschema.ArtifactMetadataEntryType.PROPERTY_ASSIGNMENT, {
      propertyName,
      stackTrace: captureStackTrace(traceProperty),
    });
  }
}
