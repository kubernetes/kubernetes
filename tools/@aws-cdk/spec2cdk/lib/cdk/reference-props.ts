import type { Resource } from '@aws-cdk/service-spec-types';
import type { Expression, PropertySpec } from '@cdklabs/typewriter';
import { $this, expr, Type } from '@cdklabs/typewriter';
import { propertyNameFromCloudFormation, referencePropertyName } from '../naming';
import { extractResourceVariablesFromArnFormat, findArnProperty, findNonIdentifierArnProperty } from './arn';
import { attributePropertyNames } from './attribute-name-conflict-resolutions';
import { CDK_CORE } from './cdk';

export interface ReferenceProp {
  readonly declaration: PropertySpec;
  readonly cfnValue: Expression;
}

/**
 * Calculations for to the resource reference interface.
 *
 * This  class is slightly complicated because it needs to account for the differences between CloudFormation
 * and CC-API identifiers.
 *
 * - In principle, the CC-API identifier is leading: it uniquely identifies a resource inside an environment.
 * - For backwards compatibility reasons, the CFN identifier (the value returned by `{ Ref }`) isn't necessarily
 *   always the same. If it isn't the same, there can be two reasons for them to diverge:
 *
 *    - SPECIFICITY: the CC-API identifier is more specific than the CFN
 *      identifier. For example, for `ApiGateway::Stage`, the unique identifier is
 *      `[ApiId, StageName]` but the value that `{ Ref }` returns is just
 *      `StageName`.
 *
 *      This distinction happens for subresources, and in those cases the
 *      primary resource will be a required input property. For maximum
 *      flexibility we generate the interface according to the CC-API
 *      identifier, and get values from the CFN identifier, attributes and input
 *      properties as necessary.
 *
 *    - ALIASING: the CC-API uses a different form of identifying the resource than the
 *      CFN identifier. For example, for `Batch::JobDefinition` the spec says the primary
 *      identifier is the `Name` but the actual value that `{ Ref }` returns is the
 *      the `Arn`. We will just use the CFN value as leading.
 *
 * We will identify the difference between these 2 cases by the length of the primary
 * identifier: equal length = aliasing, different length = specificity.
 *
 * If available, we also add an ARN field into the reference interface.
 */
export class ResourceReference {
  public readonly resource: Resource;
  public readonly arnPropertyName?: string;

  public get referenceProps(): ReferenceProp[] {
    return [...this._referenceProps.values()];
  }

  public get arnVariables(): Record<string, string> | undefined {
    return this._arnVariables.length ? Object.fromEntries(this._arnVariables) : undefined;
  }

  public get hasArnGetter(): boolean {
    return this.arnPropertyName != null || this.arnVariables != null;
  }

  private _arnVariables: [string, string][] = [];
  private readonly _referenceProps = new FirstOccurrenceMap<string, ReferenceProp>();

  /** Computed from the same pure function the resource class uses, so the two always agree. */
  private readonly attributePropertyNames: Map<string, string>;

  public constructor(resource: Resource) {
    this.resource = resource;
    this.attributePropertyNames = attributePropertyNames(resource.cloudFormationType, Object.keys(resource.attributes));
    this.arnPropertyName = this.findArnPropertyName();
    this._referenceProps = new FirstOccurrenceMap<string, ReferenceProp>();
    this.collectReferencesProps();
  }

  /** A renamed attribute's preferred name belongs to a different attribute, so never derive it here. */
  private attributeGetter(attrName: string): Expression {
    return $this[this.attributePropertyNames.get(attrName)!];
  }

  private findArnPropertyName(): string | undefined {
    const arnProp = findArnProperty(this.resource);
    if (!arnProp) {
      return;
    }
    return referencePropertyName(arnProp, this.resource.name);
  }

  private collectReferencesProps() {
    // Reference fields
    for (const cfnName of this.referenceFields) {
      const name = referencePropertyName(cfnName, this.resource.name);
      this._referenceProps.setIfAbsent(name, {
        declaration: {
          name,
          type: Type.STRING,
          immutable: true,
          docs: {
            summary: `The ${cfnName} of the ${this.resource.name} resource.`,
          },
        },
        cfnValue: this.getStringValue(cfnName),
      });
    }

    // Arn identifier
    const arnProp = findNonIdentifierArnProperty(this.resource);
    if (arnProp) {
      const name = referencePropertyName(arnProp, this.resource.name);
      this._referenceProps.setIfAbsent(name, {
        declaration: {
          name,
          type: Type.STRING,
          immutable: true,
          docs: {
            summary: `The ARN of the ${this.resource.name} resource.`,
          },
        },
        cfnValue: this.attributeGetter(arnProp),
      });
    }

    // If we do not have an ARN prop, see if we can construct it from the arn template
    if (this.resource.arnTemplate && !this.arnPropertyName) {
      const variables = extractResourceVariablesFromArnFormat(this.resource.arnTemplate);
      const allResolved = variables.every(v => this.tryFindUsableValue(v));
      if (allResolved) {
        for (const variable of variables) {
          const access = this.tryFindUsableValue(variable)!;
          const refPropName = referencePropertyName(variable, this.resource.name);
          this._arnVariables.push([variable, refPropName]);
          this._referenceProps.setIfAbsent(refPropName, {
            declaration: {
              name: refPropName,
              type: Type.STRING,
              immutable: true,
              docs: {
                summary: `The ${variable} of the ${this.resource.name} resource.`,
              },
            },
            cfnValue: access,
          });
        }
      }
    }
  }

  public tryFindUsableValue(name: string): Expression| undefined {
    const refPropName = referencePropertyName(name, this.resource.name);

    // an already discovered reference prop
    if (this._referenceProps.has(refPropName)) {
      return this._referenceProps.get(refPropName)!.cfnValue;
    }
    // an attribute
    if (this.resource.attributes[name]) {
      return this.attributeGetter(name);
    }
    // a required prop
    if (this.resource.properties[name]?.required) {
      return $this[refPropName];
    }

    // we don't have a matching value
    return undefined;
  }

  /**
   * The actual reference fields
   *
   * The CFN values if present and the same length as the CC-API values, otherwise the CC-API values.
   *
   * For a CC-API identifier we filter out optional properties, such as for `ECS::Cluster`: the real
   * unique identifier includes `Cluster` but that is an optional property because the Service will fall
   * back to some implicit default Cluster that we can never replicate.
   */
  public get referenceFields(): string[] {
    if (this.resource.cfnRefIdentifier && this.resource.cfnRefIdentifier.length === this.resource.primaryIdentifier?.length) {
      return this.resource.cfnRefIdentifier;
    }

    // Filter out properties we can't find a value for (will only be optional properties)
    return (this.resource.primaryIdentifier ?? [])
      .filter(p => this.tryGetStringValue(p) !== undefined);
  }

  /**
   * What `{ Ref }` returns in CloudFormation
   */
  public get cfnRefComponents(): string[] {
    return this.resource.cfnRefIdentifier ?? this.resource.primaryIdentifier ?? [];
  }

  /**
   * Return an expression to return the given value from the { Ref } or any of the attributes or properties
   */
  private tryGetStringValue(name: string): Expression | undefined {
    for (const [i, field] of this.cfnRefComponents.entries()) {
      if (field === name) {
        // Return entire field or Split expression, depending on whether we need to split at all
        return this.cfnRefComponents.length > 1 ? splitSelect('|', i, $this.ref) : $this.ref;
      }
    }

    // Is it an attr?
    if (this.resource.attributes[name]) {
      return this.attributeGetter(name);
    }

    // A required prop?
    if (this.resource.properties[name]?.required) {
      return $this[propertyNameFromCloudFormation(name)];
    }

    return undefined;
  }

  /**
   * Return a value, failing if it doesn't exist.
   */
  private getStringValue(name: string): Expression {
    const ret = this.tryGetStringValue(name);
    if (ret) {
      return ret;
    }

    const attributeNames = Object.keys(this.resource.attributes);
    const requiredPropertyNames = Object.entries(this.resource.properties).filter(([_, p]) => p.required).map(([n, _]) => n);

    throw new Error(`Cannot find reference interface value name ${name} for resource ${this.resource.cloudFormationType} (Ref components: ${this.cfnRefComponents}, attributes: ${attributeNames}, requiredProps: ${requiredPropertyNames})`);
  }
}

class FirstOccurrenceMap<K, V> extends Map<K, V> {
  setIfAbsent(key: K, value: V): void {
    if (!this.has(key)) this.set(key, value);
  }
}

function splitSelect(sep: string, n: number, base: Expression) {
  return CDK_CORE.Fn.select(expr.lit(n), CDK_CORE.Fn.split(expr.lit(sep), base));
}

