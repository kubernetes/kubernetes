import { CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS } from '../../../cx-api';
import type { CfnCondition } from '../cfn-condition';
import type { CfnElement } from '../cfn-element';
import { Fn } from '../cfn-fn';
import type { CfnMapping } from '../cfn-mapping';
import { Aws } from '../cfn-pseudo';
import type { CfnResource } from '../cfn-resource';
import type {
  CfnAutoScalingInstanceRefresh,
  CfnAutoScalingInstanceRefreshAlarmSpecification,
  CfnAutoScalingInstanceRefreshPreferences,
  CfnAutoScalingReplacingUpdate,
  CfnAutoScalingRollingUpdate,
  CfnAutoScalingScheduledAction,
  CfnCodeDeployLambdaAliasUpdate,
  CfnCreationPolicy,
  CfnResourceAutoScalingCreationPolicy,
  CfnResourceSignal,
  CfnUpdatePolicy,
} from '../cfn-resource-policy';
import { CfnDeletionPolicy } from '../cfn-resource-policy';
import type { CfnTag } from '../cfn-tag';
import { UnscopedValidationError } from '../errors';
import { FeatureFlags } from '../feature-flags';
import { Lazy } from '../lazy';
import { CfnReference, ReferenceRendering } from '../private/cfn-reference';
import { stackOf } from '../private/core-construct-finders';
import { lit } from '../private/literal-string';
import type { IResolvable } from '../resolvable';
import type { Validator } from '../runtime';
import type { Stack } from '../stack';
import { isResolvableObject, Token } from '../token';
import { undefinedIfAllValuesAreEmpty } from '../util';

/**
 * The class used as the intermediate result from the generated L1 methods
 * that convert from CloudFormation's UpperCase to CDK's lowerCase property names.
 * Saves any extra properties that were present in the argument object,
 * but that were not found in the CFN schema,
 * so that they're not lost from the final CDK-rendered template.
 */
export class FromCloudFormationResult<T> {
  public readonly value: T;
  public readonly extraProperties: { [key: string]: any };

  public constructor(value: T) {
    this.value = value;
    this.extraProperties = {};
  }

  public appendExtraProperties(prefix: string, properties: { [key: string]: any } | undefined): void {
    for (const [key, val] of Object.entries(properties ?? {})) {
      this.extraProperties[`${prefix}.${key}`] = val;
    }
  }
}

/**
 * A property object we will accumulate properties into
 */
export class FromCloudFormationPropertyObject<T extends Record<string, any>> extends FromCloudFormationResult<T> {
  private readonly recognizedProperties = new Set<string>();

  public constructor() {
    super({} as any); // We're still accumulating
  }

  /**
   * Add a parse result under a given key
   */
  public addPropertyResult(cdkPropName: keyof T, cfnPropName: string, result?: FromCloudFormationResult<any>): void {
    this.recognizedProperties.add(cfnPropName);
    if (!result) { return; }
    this.value[cdkPropName] = result.value;
    this.appendExtraProperties(cfnPropName, result.extraProperties);
  }

  public addUnrecognizedPropertiesAsExtra(properties: object): void {
    for (const [key, val] of Object.entries(properties)) {
      if (!this.recognizedProperties.has(key)) {
        this.extraProperties[key] = val;
      }
    }
  }
}

/**
 * This class contains static methods called when going from
 * translated values received from `CfnParser.parseValue`
 * to the actual L1 properties -
 * things like changing IResolvable to the appropriate type
 * (string, string array, or number), etc.
 *
 * While this file not exported from the module
 * (to not make it part of the public API),
 * it is directly referenced in the generated L1 code.
 *
 */
export class FromCloudFormation {
  // nothing to for any but return it
  public static getAny(value: any): FromCloudFormationResult<any> {
    return new FromCloudFormationResult(value);
  }

  public static getBoolean(this: void, value: any): FromCloudFormationResult<boolean | IResolvable> {
    if (typeof value === 'string') {
      // CloudFormation allows passing strings as boolean
      switch (value) {
        case 'true': return new FromCloudFormationResult(true);
        case 'false': return new FromCloudFormationResult(false);
        default: throw new UnscopedValidationError(lit`ExpectedTrueFalseBoolean`, `Expected 'true' or 'false' for boolean value, got: '${value}'`);
      }
    }

    // in all other cases, just return the value,
    // and let a validator handle if it's not a boolean
    return new FromCloudFormationResult(value);
  }

  public static getDate(value: any): FromCloudFormationResult<Date | IResolvable> {
    // if the date is a deploy-time value, just return it
    if (isResolvableObject(value)) {
      return new FromCloudFormationResult(value);
    }

    // if the date has been given as a string, convert it
    if (typeof value === 'string') {
      return new FromCloudFormationResult(new Date(value));
    }

    // all other cases - just return the value,
    // if it's not a Date, a validator should catch it
    return new FromCloudFormationResult(value);
  }

  // won't always return a string; if the input can't be resolved to a string,
  // the input will be returned.
  public static getString(this: void, value: any): FromCloudFormationResult<string> {
    // if the string is a deploy-time value, serialize it to a Token
    if (isResolvableObject(value)) {
      return new FromCloudFormationResult(value.toString());
    }

    // CloudFormation treats numbers and strings interchangeably;
    // so, if we get a number here, convert it to a string
    if (typeof value === 'number') {
      return new FromCloudFormationResult(value.toString());
    }

    // CloudFormation treats booleans and strings interchangeably;
    // so, if we get a boolean here, convert it to a string
    if (typeof value === 'boolean') {
      return new FromCloudFormationResult(value.toString());
    }

    // in all other cases, just return the input,
    // and let a validator handle it if it's not a string
    return new FromCloudFormationResult(value);
  }

  // won't always return a number; if the input can't be parsed to a number,
  // the input will be returned.
  public static getNumber(this: void, value: any): FromCloudFormationResult<number> {
    // if the string is a deploy-time value, serialize it to a Token
    if (isResolvableObject(value)) {
      return new FromCloudFormationResult(Token.asNumber(value));
    }

    // return a number, if the input can be parsed as one
    if (typeof value === 'string') {
      const parsedValue = parseFloat(value);
      if (!isNaN(parsedValue)) {
        return new FromCloudFormationResult(parsedValue);
      }
    }

    // otherwise return the input,
    // and let a validator handle it if it's not a number
    return new FromCloudFormationResult(value);
  }

  public static getStringArray(this: void, value: any): FromCloudFormationResult<string[]> {
    // if the array is a deploy-time value, serialize it to a Token
    if (isResolvableObject(value)) {
      return new FromCloudFormationResult(Token.asList(value));
    }

    // in all other cases, delegate to the standard mapping logic
    return FromCloudFormation.getArray(FromCloudFormation.getString)(value);
  }

  public static getArray<T>(this: void, mapper: (arg: any) => FromCloudFormationResult<T>): (x: any) => FromCloudFormationResult<T[]> {
    return (value: any) => {
      if (!Array.isArray(value)) {
        // break the type system, and just return the given value,
        // which hopefully will be reported as invalid by the validator
        // of the property we're transforming
        // (unless it's a deploy-time value,
        // which we can't map over at build time anyway)
        return new FromCloudFormationResult(value);
      }

      const values = new Array<any>();
      const ret = new FromCloudFormationResult(values);
      for (let i = 0; i < value.length; i++) {
        const result = mapper(value[i]);
        values.push(result.value);
        ret.appendExtraProperties(`${i}`, result.extraProperties);
      }
      return ret;
    };
  }

  public static getMap<T>(this: void, mapper: (arg: any) => FromCloudFormationResult<T>): (x: any) => FromCloudFormationResult<{ [key: string]: T }> {
    return (value: any) => {
      if (typeof value !== 'object') {
        // if the input is not a map (= object in JS land),
        // just return it, and let the validator of this property handle it
        // (unless it's a deploy-time value,
        // which we can't map over at build time anyway)
        return new FromCloudFormationResult(value);
      }

      const values: { [key: string]: T } = {};
      const ret = new FromCloudFormationResult(values);
      for (const [key, val] of Object.entries(value)) {
        const result = mapper(val);
        values[key] = result.value;
        ret.appendExtraProperties(key, result.extraProperties);
      }
      return ret;
    };
  }

  public static getCfnTag(this: void, tag: any): FromCloudFormationResult<CfnTag | IResolvable> {
    if (isResolvableObject(tag)) { return new FromCloudFormationResult(tag); }
    return tag == null
      ? new FromCloudFormationResult({ } as any) // break the type system - this should be detected at runtime by a tag validator
      : new FromCloudFormationResult({
        key: tag.Key,
        value: tag.Value,
      });
  }

  /**
   * Return a function that, when applied to a value, will return the first validly deserialized one
   */
  public static getTypeUnion(this: void, validators: Validator[], mappers: Array<(x: any) => FromCloudFormationResult<any>>):
  (x: any) => FromCloudFormationResult<any> {
    return (value: any) => {
      for (let i = 0; i < validators.length; i++) {
        const candidate = mappers[i](value);
        if (validators[i](candidate.value).isSuccess) {
          return candidate;
        }
      }

      // if nothing matches, just return the input unchanged, and let validators catch it
      return new FromCloudFormationResult(value);
    };
  }
}

/**
 * An interface that represents callbacks into a CloudFormation template.
 * Used by the fromCloudFormation methods in the generated L1 classes.
 */
export interface ICfnFinder {
  /**
   * Return the Condition with the given name from the template.
   * If there is no Condition with that name in the template,
   * returns undefined.
   */
  findCondition(conditionName: string): CfnCondition | undefined;

  /**
   * Return the Mapping with the given name from the template.
   * If there is no Mapping with that name in the template,
   * returns undefined.
   */
  findMapping(mappingName: string): CfnMapping | undefined;

  /**
   * Returns the element referenced using a Ref expression with the given name.
   * If there is no element with this name in the template,
   * return undefined.
   */
  findRefTarget(elementName: string): CfnElement | undefined;

  /**
   * Returns the resource with the given logical ID in the template.
   * If a resource with that logical ID was not found in the template,
   * returns undefined.
   */
  findResource(logicalId: string): CfnResource | undefined;
}

/**
 * The interface used as the last argument to the fromCloudFormation
 * static method of the generated L1 classes.
 */
export interface FromCloudFormationOptions {
  /**
   * The parser used to convert CloudFormation to values the CDK understands.
   */
  readonly parser: CfnParser;
}

/**
 * The context in which the parsing is taking place.
 *
 * Some fragments of CloudFormation templates behave differently than others
 * (for example, the 'Conditions' sections treats { "Condition": "NameOfCond" }
 * differently than the 'Resources' section).
 * This enum can be used to change the created `CfnParser` behavior,
 * based on the template context.
 */
export enum CfnParsingContext {
  /** We're currently parsing the 'Conditions' section. */
  CONDITIONS,

  /** We're currently parsing the 'Rules' section. */
  RULES,
}

/**
 * The options for `FromCloudFormation.parseValue`.
 */
export interface ParseCfnOptions {
  /**
   * The finder interface used to resolve references in the template.
   */
  readonly finder: ICfnFinder;

  /**
   * The context we're parsing the template in.
   *
   * @default - the default context (no special behavior)
   */
  readonly context?: CfnParsingContext;

  /**
   * Values provided here will replace references to parameters in the parsed template.
   */
  readonly parameters: { [parameterName: string]: any };
}

/**
 * This class contains methods for translating from a pure CFN value
 * (like a JS object { "Ref": "Bucket" })
 * to a form CDK understands
 * (like Fn.ref('Bucket')).
 *
 * While this file not exported from the module
 * (to not make it part of the public API),
 * it is directly referenced in the generated L1 code,
 * so any renames of it need to be reflected in codegen as well.
 *
 */
export class CfnParser {
  private readonly options: ParseCfnOptions;
  private stack?: Stack;

  constructor(options: ParseCfnOptions) {
    this.options = options;
  }

  public handleAttributes(resource: CfnResource, resourceAttributes: any, logicalId: string): void {
    const cfnOptions = resource.cfnOptions;
    this.stack = stackOf(resource);

    const creationPolicy = this.parseCreationPolicy(resourceAttributes.CreationPolicy, logicalId);
    const updatePolicy = this.parseUpdatePolicy(resourceAttributes.UpdatePolicy, logicalId);
    cfnOptions.creationPolicy = creationPolicy.value;
    cfnOptions.updatePolicy = updatePolicy.value;
    cfnOptions.deletionPolicy = this.parseDeletionPolicy(resourceAttributes.DeletionPolicy);
    cfnOptions.updateReplacePolicy = this.parseDeletionPolicy(resourceAttributes.UpdateReplacePolicy);
    cfnOptions.version = this.parseValue(resourceAttributes.Version);
    cfnOptions.description = this.parseValue(resourceAttributes.Description);
    cfnOptions.metadata = this.parseValue(resourceAttributes.Metadata);

    for (const [key, value] of Object.entries(creationPolicy.extraProperties)) {
      resource.addOverride(`CreationPolicy.${key}`, value);
    }
    for (const [key, value] of Object.entries(updatePolicy.extraProperties)) {
      resource.addOverride(`UpdatePolicy.${key}`, value);
    }

    // handle Condition
    if (resourceAttributes.Condition) {
      const condition = this.finder.findCondition(resourceAttributes.Condition);
      if (!condition) {
        throw new UnscopedValidationError(lit`ResourceUsesConditionDoesnT`, `Resource '${logicalId}' uses Condition '${resourceAttributes.Condition}' that doesn't exist`);
      }
      cfnOptions.condition = condition;
    }

    // handle DependsOn
    resourceAttributes.DependsOn = resourceAttributes.DependsOn ?? [];
    const dependencies: string[] = Array.isArray(resourceAttributes.DependsOn) ?
      resourceAttributes.DependsOn : [resourceAttributes.DependsOn];
    for (const dep of dependencies) {
      const depResource = this.finder.findResource(dep);
      if (!depResource) {
        throw new UnscopedValidationError(lit`ResourceDependsDoesnTExist`, `Resource '${logicalId}' depends on '${dep}' that doesn't exist`);
      }
      resource.node.addDependency(depResource);
    }
  }

  private parseCreationPolicy(policy: any, logicalId: string): FromCloudFormationResult<CfnCreationPolicy | undefined> {
    if (typeof policy !== 'object') { return new FromCloudFormationResult(undefined); }
    this.throwIfIsIntrinsic(policy, logicalId);
    const self = this;

    const creaPol = new ObjectParser<CfnCreationPolicy>(this.parseValue(policy));
    creaPol.parseCase('autoScalingCreationPolicy', parseAutoScalingCreationPolicy);
    creaPol.parseCase('resourceSignal', parseResourceSignal);
    return creaPol.toResult();

    function parseAutoScalingCreationPolicy(p: any): FromCloudFormationResult<CfnResourceAutoScalingCreationPolicy | undefined> {
      self.throwIfIsIntrinsic(p, logicalId);
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }

      const autoPol = new ObjectParser<CfnResourceAutoScalingCreationPolicy>(p);
      autoPol.parseCase('minSuccessfulInstancesPercent', FromCloudFormation.getNumber);
      return autoPol.toResult();
    }

    function parseResourceSignal(p: any): FromCloudFormationResult<CfnResourceSignal | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const sig = new ObjectParser<CfnResourceSignal>(p);
      sig.parseCase('count', FromCloudFormation.getNumber);
      sig.parseCase('timeout', FromCloudFormation.getString);
      return sig.toResult();
    }
  }

  private parseUpdatePolicy(policy: any, logicalId: string): FromCloudFormationResult<CfnUpdatePolicy | undefined> {
    if (typeof policy !== 'object') { return new FromCloudFormationResult(undefined); }
    this.throwIfIsIntrinsic(policy, logicalId);
    const self = this;

    // change simple JS values to their CDK equivalents
    const uppol = new ObjectParser<CfnUpdatePolicy>(this.parseValue(policy));
    uppol.parseCase('autoScalingReplacingUpdate', parseAutoScalingReplacingUpdate);
    uppol.parseCase('autoScalingRollingUpdate', parseAutoScalingRollingUpdate);
    uppol.parseCase('autoScalingInstanceRefresh', parseAutoScalingInstanceRefresh);
    uppol.parseCase('autoScalingScheduledAction', parseAutoScalingScheduledAction);
    uppol.parseCase('codeDeployLambdaAliasUpdate', parseCodeDeployLambdaAliasUpdate);
    uppol.parseCase('enableVersionUpgrade', (x) => FromCloudFormation.getBoolean(x) as any);
    uppol.parseCase('useOnlineResharding', (x) => FromCloudFormation.getBoolean(x) as any);
    return uppol.toResult();

    function parseAutoScalingReplacingUpdate(p: any): FromCloudFormationResult<CfnAutoScalingReplacingUpdate | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const repUp = new ObjectParser<CfnAutoScalingReplacingUpdate>(p);
      repUp.parseCase('willReplace', (x) => x);
      return repUp.toResult();
    }

    function parseAutoScalingRollingUpdate(p: any): FromCloudFormationResult<CfnAutoScalingRollingUpdate | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const rollUp = new ObjectParser<CfnAutoScalingRollingUpdate>(p);
      rollUp.parseCase('maxBatchSize', FromCloudFormation.getNumber);
      rollUp.parseCase('minInstancesInService', FromCloudFormation.getNumber);
      rollUp.parseCase('minSuccessfulInstancesPercent', FromCloudFormation.getNumber);
      rollUp.parseCase('minActiveInstancesPercent', FromCloudFormation.getNumber);
      rollUp.parseCase('pauseTime', FromCloudFormation.getString);
      rollUp.parseCase('suspendProcesses', FromCloudFormation.getStringArray);
      rollUp.parseCase('waitOnResourceSignals', FromCloudFormation.getBoolean);
      return rollUp.toResult();
    }

    function parseAutoScalingInstanceRefresh(p: any): FromCloudFormationResult<CfnAutoScalingInstanceRefresh | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const irUp = new ObjectParser<CfnAutoScalingInstanceRefresh>(p);
      irUp.parseCase('strategy', FromCloudFormation.getString);
      irUp.parseCase('preferences', parseInstanceRefreshPreferences);
      return irUp.toResult();
    }

    function parseInstanceRefreshPreferences(p: any): FromCloudFormationResult<CfnAutoScalingInstanceRefreshPreferences | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const prefs = new ObjectParser<CfnAutoScalingInstanceRefreshPreferences>(p);
      prefs.parseCase('minHealthyPercentage', FromCloudFormation.getNumber);
      prefs.parseCase('maxHealthyPercentage', FromCloudFormation.getNumber);
      prefs.parseCase('instanceWarmup', FromCloudFormation.getNumber);
      prefs.parseCase('skipMatching', FromCloudFormation.getBoolean);
      prefs.parseCase('checkpointPercentages', FromCloudFormation.getArray(FromCloudFormation.getNumber));
      prefs.parseCase('checkpointDelay', FromCloudFormation.getNumber);
      prefs.parseCase('bakeTime', FromCloudFormation.getNumber);
      prefs.parseCase('alarmSpecification', parseAlarmSpecification);
      prefs.parseCase('scaleInProtectedInstances', FromCloudFormation.getString);
      prefs.parseCase('standbyInstances', FromCloudFormation.getString);
      return prefs.toResult();
    }

    function parseAlarmSpecification(p: any): FromCloudFormationResult<CfnAutoScalingInstanceRefreshAlarmSpecification | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const alarm = new ObjectParser<CfnAutoScalingInstanceRefreshAlarmSpecification>(p);
      alarm.parseCase('alarms', FromCloudFormation.getStringArray);
      return alarm.toResult();
    }

    function parseCodeDeployLambdaAliasUpdate(p: any): FromCloudFormationResult<CfnCodeDeployLambdaAliasUpdate | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const cdUp = new ObjectParser<CfnCodeDeployLambdaAliasUpdate>(p);
      cdUp.parseCase('beforeAllowTrafficHook', FromCloudFormation.getString);
      cdUp.parseCase('afterAllowTrafficHook', FromCloudFormation.getString);
      cdUp.parseCase('applicationName', FromCloudFormation.getString);
      cdUp.parseCase('deploymentGroupName', FromCloudFormation.getString);
      return cdUp.toResult();
    }

    function parseAutoScalingScheduledAction(p: any): FromCloudFormationResult<CfnAutoScalingScheduledAction | undefined> {
      if (typeof p !== 'object') { return new FromCloudFormationResult(undefined); }
      self.throwIfIsIntrinsic(p, logicalId);

      const schedUp = new ObjectParser<CfnAutoScalingScheduledAction>(p);
      schedUp.parseCase('ignoreUnmodifiedGroupSizeProperties', FromCloudFormation.getBoolean);
      return schedUp.toResult();
    }
  }

  private parseDeletionPolicy(policy: any): CfnDeletionPolicy | undefined {
    if (policy === undefined || policy === null) {
      return undefined;
    }
    const isIntrinsic = this.looksLikeCfnIntrinsic(policy);
    switch (policy) {
      case 'Delete': return CfnDeletionPolicy.DELETE;
      case 'Retain': return CfnDeletionPolicy.RETAIN;
      case 'Snapshot': return CfnDeletionPolicy.SNAPSHOT;
      case 'RetainExceptOnCreate': return CfnDeletionPolicy.RETAIN_EXCEPT_ON_CREATE;
      default: if (isIntrinsic) {
        policy = this.parseValue(policy);
        return policy;
      } else {
        throw new UnscopedValidationError(lit`UnrecognizedDeletionpolicy`, `Unrecognized DeletionPolicy '${policy}'`);
      }
    }
  }

  public parseValue(cfnValue: any): any {
    // == null captures undefined as well
    if (cfnValue == null) {
      return undefined;
    }
    // if we have any late-bound values,
    // just return them
    if (isResolvableObject(cfnValue)) {
      return cfnValue;
    }
    if (Array.isArray(cfnValue)) {
      return cfnValue.map(el => this.parseValue(el));
    }
    if (typeof cfnValue === 'object') {
      // an object can be either a CFN intrinsic, or an actual object
      const cfnIntrinsic = this.parseIfCfnIntrinsic(cfnValue);
      if (cfnIntrinsic !== undefined) {
        return cfnIntrinsic;
      }
      const ret: any = {};
      for (const [key, val] of Object.entries(cfnValue)) {
        ret[key] = this.parseValue(val);
      }
      return ret;
    }
    // in all other cases, just return the input
    return cfnValue;
  }

  public get finder(): ICfnFinder {
    return this.options.finder;
  }

  private parseIfCfnIntrinsic(object: any): any {
    const key = this.looksLikeCfnIntrinsic(object);
    switch (key) {
      case undefined:
        return undefined;
      case 'Ref': {
        const refTarget = object[key];
        const specialRef = this.specialCaseRefs(refTarget);
        if (specialRef !== undefined) {
          return specialRef;
        } else {
          const refElement = this.finder.findRefTarget(refTarget);
          if (!refElement) {
            throw new UnscopedValidationError(lit`ElementUsedExpressionLogical`, `Element used in Ref expression with logical ID: '${refTarget}' not found`);
          }
          return CfnReference.for(refElement, 'Ref');
        }
      }
      case 'Fn::GetAtt': {
        const value = object[key];
        let logicalId: string, attributeName: string, stringForm: boolean;
        // Fn::GetAtt takes as arguments either a string...
        if (typeof value === 'string') {
          // ...in which case the logical ID and the attribute name are separated with '.'
          const dotIndex = value.indexOf('.');
          if (dotIndex === -1) {
            throw new UnscopedValidationError(lit`ShortFormFnGetattContain`, `Short-form Fn::GetAtt must contain a '.' in its string argument, got: '${value}'`);
          }
          logicalId = value.slice(0, dotIndex);
          attributeName = value.slice(dotIndex + 1); // the +1 is to skip the actual '.'
          stringForm = true;
        } else {
          // ...or a 2-element list
          logicalId = value[0];
          attributeName = value[1];
          stringForm = false;
        }
        const target = this.finder.findResource(logicalId);
        if (!target) {
          throw new UnscopedValidationError(lit`ResourceUsedGetattExpression`, `Resource used in GetAtt expression with logical ID: '${logicalId}' not found`);
        }
        return CfnReference.for(target, attributeName, stringForm ? ReferenceRendering.GET_ATT_STRING : undefined);
      }
      case 'Fn::Join': {
        // Fn::Join takes a 2-element list as its argument,
        // where the first element is the delimiter,
        // and the second is the list of elements to join
        const value = this.parseValue(object[key]);
        // wrap the array as a Token,
        // as otherwise Fn.join() will try to concatenate
        // the non-token parts,
        // causing a diff with the original template
        return Fn.join(value[0], Lazy.list({ produce: () => value[1] }));
      }
      case 'Fn::Cidr': {
        const value = this.parseValue(object[key]);
        return Fn.cidr(value[0], value[1], value[2]);
      }
      case 'Fn::FindInMap': {
        const value = this.parseValue(object[key]);
        // the first argument to FindInMap is the mapping name
        let mappingName: string;
        if (Token.isUnresolved(value[0])) {
          // the first argument can be a dynamic expression like Ref: Param;
          // if it is, we can't find the mapping in advance
          mappingName = value[0];
        } else {
          const mapping = this.finder.findMapping(value[0]);
          if (!mapping) {
            throw new UnscopedValidationError(lit`MappingUsedFindinmapExpression`, `Mapping used in FindInMap expression with name '${value[0]}' was not found in the template`);
          }
          mappingName = mapping.logicalId;
        }
        return Fn._findInMap(mappingName, value[1], value[2]);
      }
      case 'Fn::Select': {
        const value = this.parseValue(object[key]);
        return Fn.select(value[0], value[1]);
      }
      case 'Fn::GetAZs': {
        const value = this.parseValue(object[key]);
        return Fn.getAzs(value);
      }
      case 'Fn::ImportValue': {
        const value = this.parseValue(object[key]);
        return Fn.importValue(value);
      }
      case 'Fn::Split': {
        const value = this.parseValue(object[key]);
        return Fn.split(value[0], value[1]);
      }
      case 'Fn::Transform': {
        const value = this.parseValue(object[key]);
        return Fn.transform(value.Name, value.Parameters);
      }
      case 'Fn::Base64': {
        const value = this.parseValue(object[key]);
        return Fn.base64(value);
      }
      case 'Fn::If': {
        // Fn::If takes a 3-element list as its argument,
        // where the first element is the name of a Condition
        const value = this.parseValue(object[key]);
        const condition = this.finder.findCondition(value[0]);
        if (!condition) {
          throw new UnscopedValidationError(lit`ConditionUsedFnIfExpression`, `Condition '${value[0]}' used in an Fn::If expression does not exist in the template`);
        }
        return Fn.conditionIf(condition.logicalId, value[1], value[2]);
      }
      case 'Fn::Equals': {
        const value = this.parseValue(object[key]);
        return Fn.conditionEquals(value[0], value[1]);
      }
      case 'Fn::And': {
        const value = this.parseValue(object[key]);
        return Fn.conditionAnd(...value);
      }
      case 'Fn::Not': {
        const value = this.parseValue(object[key]);
        return Fn.conditionNot(value[0]);
      }
      case 'Fn::Or': {
        const value = this.parseValue(object[key]);
        return Fn.conditionOr(...value);
      }
      case 'Fn::Sub': {
        const value = this.parseValue(object[key]);
        let fnSubString: string;
        let map: { [key: string]: any } | undefined;
        if (typeof value === 'string') {
          fnSubString = value;
          map = undefined;
        } else {
          fnSubString = value[0];
          map = value[1];
        }

        return this.parseFnSubString(fnSubString, map);
      }
      case 'Condition': {
        // a reference to a Condition from another Condition
        const condition = this.finder.findCondition(object[key]);
        if (!condition) {
          throw new UnscopedValidationError(lit`ReferencedConditionNameFound`, `Referenced Condition with name '${object[key]}' was not found in the template`);
        }
        return { Condition: condition.logicalId };
      }
      default:
        if (this.options.context === CfnParsingContext.RULES) {
          return this.handleRulesIntrinsic(key, object);
        } else {
          throw new UnscopedValidationError(lit`UnsupportedCloudFormationFunction`, `Unsupported CloudFormation function '${key}'`);
        }
    }
  }

  private throwIfIsIntrinsic(object: object, logicalId: string): void {
    // Top-level parsing functions check before we call `parseValue`, which requires
    // calling `looksLikeCfnIntrinsic`. Helper parsing functions check after we call
    // `parseValue`, which requires calling `isResolvableObject`.
    if (!this.stack) {
      throw new UnscopedValidationError(lit`CannotCallMethodHandleAttributes`, 'cannot call this method before handleAttributes!');
    }
    if (FeatureFlags.of(this.stack).isEnabled(CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS)) {
      if (isResolvableObject(object ?? {}) || this.looksLikeCfnIntrinsic(object ?? {})) {
        throw new UnscopedValidationError(lit`CannotConvertResource`, `Cannot convert resource '${logicalId}' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify '${logicalId}' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output.`);
      }
    }
  }

  private looksLikeCfnIntrinsic(object: object): string | undefined {
    const objectKeys = Object.keys(object);
    // a CFN intrinsic is always an object with a single key
    if (objectKeys.length !== 1) {
      return undefined;
    }

    const key = objectKeys[0];
    return key === 'Ref' || key.startsWith('Fn::') ||
    // special intrinsic only available in the 'Conditions' section
    (this.options.context === CfnParsingContext.CONDITIONS && key === 'Condition')
      ? key
      : undefined;
  }

  private parseFnSubString(templateString: string, expressionMap: { [key: string]: any } | undefined): string {
    const map = expressionMap ?? {};
    const self = this;
    return Fn.sub(go(templateString), Object.keys(map).length === 0 ? expressionMap : map);

    function go(value: string): string {
      const leftBrace = value.indexOf('${');
      if (leftBrace === -1) {
        return value;
      }
      // search for the closing brace to the right of the opening '${'
      // (in theory, there could be other braces in the string,
      // for example if it represents a JSON object)
      const rightBrace = value.indexOf('}', leftBrace);
      if (rightBrace === -1) {
        return value;
      }

      const leftHalf = value.substring(0, leftBrace);
      const rightHalf = value.substring(rightBrace + 1);
      // don't include left and right braces when searching for the target of the reference
      const refTarget = value.substring(leftBrace + 2, rightBrace).trim();
      if (refTarget[0] === '!') {
        return value.substring(0, rightBrace + 1) + go(rightHalf);
      }

      // lookup in map
      if (refTarget in map) {
        return leftHalf + '${' + refTarget + '}' + go(rightHalf);
      }

      // since it's not in the map, check if it's a pseudo-parameter
      // (or a value to be substituted for a Parameter, provided by the customer)
      const specialRef = self.specialCaseSubRefs(refTarget);
      if (specialRef !== undefined) {
        if (Token.isUnresolved(specialRef)) {
          // specialRef can only be a Token if the value passed by the customer
          // for substituting a Parameter was a Token.
          // This is actually bad here,
          // because the Token can potentially be something that doesn't render
          // well inside an Fn::Sub template string, like a { Ref } object.
          // To handle this case,
          // instead of substituting the Parameter directly with the token in the template string,
          // add a new entry to the Fn::Sub map,
          // with key refTarget, and the token as the value.
          // This is safe, because this sort of shadowing is legal in CloudFormation,
          // and also because we're certain the Fn::Sub map doesn't contain an entry for refTarget
          // (as we check that condition in the code right above this).
          map[refTarget] = specialRef;
          return leftHalf + '${' + refTarget + '}' + go(rightHalf);
        } else {
          return leftHalf + specialRef + go(rightHalf);
        }
      }

      const dotIndex = refTarget.indexOf('.');
      const isRef = dotIndex === -1;
      if (isRef) {
        const refElement = self.finder.findRefTarget(refTarget);
        if (!refElement) {
          throw new UnscopedValidationError(lit`ElementReferencedFnSubExpression`, `Element referenced in Fn::Sub expression with logical ID: '${refTarget}' was not found in the template`);
        }
        return leftHalf + CfnReference.for(refElement, 'Ref', ReferenceRendering.FN_SUB).toString() + go(rightHalf);
      } else {
        const targetId = refTarget.substring(0, dotIndex);
        const refResource = self.finder.findResource(targetId);
        if (!refResource) {
          throw new UnscopedValidationError(lit`ResourceReferencedFnSubExpression`, `Resource referenced in Fn::Sub expression with logical ID: '${targetId}' was not found in the template`);
        }
        const attribute = refTarget.substring(dotIndex + 1);
        return leftHalf + CfnReference.for(refResource, attribute, ReferenceRendering.FN_SUB).toString() + go(rightHalf);
      }
    }
  }

  private handleRulesIntrinsic(key: string, object: any): any {
    // Rules have their own set of intrinsics:
    // https://docs.aws.amazon.com/servicecatalog/latest/adminguide/intrinsic-function-reference-rules.html
    switch (key) {
      case 'Fn::ValueOf': {
        // ValueOf is special,
        // as it takes the name of a Parameter as its first argument
        const value = this.parseValue(object[key]);
        const parameterName = value[0];
        if (parameterName in this.parameters) {
          // since ValueOf returns the value of a specific attribute,
          // fail here - this substitution is not allowed
          throw new UnscopedValidationError(lit`CannotSubstituteParameter`, `Cannot substitute parameter '${parameterName}' used in Fn::ValueOf expression with attribute '${value[1]}'`);
        }
        const param = this.finder.findRefTarget(parameterName);
        if (!param) {
          throw new UnscopedValidationError(lit`RuleReferencesParameterWhich`, `Rule references parameter '${parameterName}' which was not found in the template`);
        }
        // create an explicit IResolvable,
        // as Fn.valueOf() returns a string,
        // which is incorrect
        // (Fn::ValueOf can also return an array)
        return Lazy.any({ produce: () => ({ 'Fn::ValueOf': [param.logicalId, value[1]] }) });
      }
      default:
        // I don't want to hard-code the list of supported Rules-specific intrinsics in this function;
        // so, just return undefined here,
        // and they will be treated as a regular JSON object
        return undefined;
    }
  }

  private specialCaseRefs(value: any): any {
    if (value in this.parameters) {
      return this.parameters[value];
    }
    switch (value) {
      case 'AWS::AccountId': return Aws.ACCOUNT_ID;
      case 'AWS::Region': return Aws.REGION;
      case 'AWS::Partition': return Aws.PARTITION;
      case 'AWS::URLSuffix': return Aws.URL_SUFFIX;
      case 'AWS::NotificationARNs': return Aws.NOTIFICATION_ARNS;
      case 'AWS::StackId': return Aws.STACK_ID;
      case 'AWS::StackName': return Aws.STACK_NAME;
      // AWS.NO_VALUE is defined as a string, however NoValue should be able to use any type
      case 'AWS::NoValue': return Token.asAny(Aws.NO_VALUE);
      default: return undefined;
    }
  }

  private specialCaseSubRefs(value: string): string | undefined {
    if (value in this.parameters) {
      return this.parameters[value];
    }
    return value.indexOf('::') === -1 ? undefined: '${' + value + '}';
  }

  private get parameters(): { [parameterName: string]: any } {
    return this.options.parameters || {};
  }
}

class ObjectParser<T extends object> {
  private readonly parsed: Record<string, unknown> = {};
  private readonly unparsed: Record<string, unknown> = {};

  constructor(input: Record<string, unknown>) {
    this.unparsed = { ...input };
  }

  /**
   * Parse a single field from the object into the target object
   *
   * The source key will be assumed to be the exact same as the
   * target key, but with an uppercase first letter.
   */
  public parseCase<K extends keyof T>(targetKey: K, parser: (x: any) => T[K] | FromCloudFormationResult<T[K] | IResolvable>) {
    const sourceKey = ucfirst(String(targetKey));
    this.parse(targetKey, sourceKey, parser);
  }

  public parse<K extends keyof T>(targetKey: K, sourceKey: string, parser: (x: any) => T[K] | FromCloudFormationResult<T[K] | IResolvable>) {
    if (!(sourceKey in this.unparsed)) {
      return;
    }

    const value = parser(this.unparsed[sourceKey]);
    delete this.unparsed[sourceKey];

    if (value instanceof FromCloudFormationResult) {
      for (const [k, v] of Object.entries(value.extraProperties ?? {})) {
        this.unparsed[`${sourceKey}.${k}`] = v;
      }
      this.parsed[targetKey as any] = value.value;
    } else {
      this.parsed[targetKey as any] = value;
    }
  }

  public toResult(): FromCloudFormationResult<T | undefined> {
    const ret = new FromCloudFormationResult(undefinedIfAllValuesAreEmpty(this.parsed as any));
    for (const [k, v] of Object.entries(this.unparsedKeys)) {
      ret.extraProperties[k] = v;
    }
    return ret;
  }

  private get unparsedKeys(): Record<string, unknown> {
    const unparsed = { ...this.unparsed };

    for (const [k, v] of Object.entries(this.unparsed)) {
      if (v instanceof FromCloudFormationResult) {
        for (const [k2, v2] of Object.entries(v.extraProperties ?? {})) {
          unparsed[`${k}.${k2}`] = v2;
        }
      } else {
        unparsed[k] = v;
      }
    }

    return unparsed;
  }
}

function ucfirst(x: string) {
  return x.slice(0, 1).toUpperCase() + x.slice(1);
}

