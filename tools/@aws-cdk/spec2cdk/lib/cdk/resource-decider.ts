import type { Property, Resource, TagVariant } from '@aws-cdk/service-spec-types';
import { Deprecation, RichProperty } from '@aws-cdk/service-spec-types';
import type { Expression, PropertySpec } from '@cdklabs/typewriter';
import { $E, $T, $this, Type, expr } from '@cdklabs/typewriter';
import { attributePropertyNames } from './attribute-name-conflict-resolutions';
import { CDK_CORE } from './cdk';
import type { PropertyMapping } from './cloudformation-mapping';
import type { RelationshipDecider } from './relationship-decider';
import { ResolverBuilder } from './resolver-builder';
import type { TaggabilityStyle } from './tagging';
import { NON_RESOLVABLE_PROPERTY_NAMES, resourceTaggabilityStyle } from './tagging';
import type { TypeConverter } from './type-converter';
import { camelcasedResourceName, cloudFormationDocLink, propertyNameFromCloudFormation } from '../naming';
import { splitDocumentation } from '../util';
import { ResourceReference } from './reference-props';

/**
 * Decide how properties get mapped between model types, Typescript types, and CloudFormation
 */
export class ResourceDecider {
  public static taggabilityInterfaces(resource: Resource) {
    const taggability = resourceTaggabilityStyle(resource);
    return taggability?.style === 'legacy'
      ? [CDK_CORE.ITaggable]
      : taggability?.style === 'modern'
        ? [CDK_CORE.ITaggableV2]
        : [];
  }

  private readonly taggability?: TaggabilityStyle;
  private readonly resolverBuilder: ResolverBuilder;

  private readonly attributePropertyNames: Map<string, string>;

  public readonly resourceReference: ResourceReference;
  public readonly propsProperties = new Array<PropsProperty>();
  public readonly classProperties = new Array<ClassProperty>();
  public readonly classAttributeProperties = new Array<ClassAttributeProperty>();
  public readonly camelResourceName: string;

  constructor(
    private readonly resource: Resource,
    private readonly converter: TypeConverter,
    private readonly relationshipDecider: RelationshipDecider,
  ) {
    this.camelResourceName = camelcasedResourceName(resource);
    this.taggability = resourceTaggabilityStyle(this.resource);
    this.resolverBuilder = new ResolverBuilder(this.converter, this.relationshipDecider, this.converter.module);
    this.attributePropertyNames = attributePropertyNames(resource.cloudFormationType, Object.keys(resource.attributes));

    this.convertProperties();
    this.convertAttributes();

    this.resourceReference = new ResourceReference(this.resource);

    this.propsProperties.sort((p1, p2) => p1.propertySpec.name.localeCompare(p2.propertySpec.name));
    this.classProperties.sort((p1, p2) => p1.propertySpec.name.localeCompare(p2.propertySpec.name));
    this.classAttributeProperties.sort((p1, p2) => p1.propertySpec.name.localeCompare(p2.propertySpec.name));
  }

  private convertProperties() {
    for (const [name, prop] of Object.entries(this.resource.properties)) {
      if (name === this.taggability?.tagPropertyName) {
        switch (this.taggability?.style) {
          case 'legacy':
            this.handleTagPropertyLegacy(name, prop, this.taggability.variant);
            continue;
          case 'modern':
            this.handleTagPropertyModern(name, prop, this.taggability.variant);
            continue;
        }
      } else {
        this.handleTypeHistoryTypes(prop);
      }

      this.handlePropertyDefault(name, prop);
    }
  }

  /**
   * Default mapping for a property
   */
  private handlePropertyDefault(cfnName: string, prop: Property) {
    const optional = !prop.required;

    const resolverResult = this.resolverBuilder.buildResolver(prop, cfnName);

    this.propsProperties.push({
      propertySpec: {
        name: resolverResult.name,
        type: resolverResult.propType,
        optional,
        docs: this.defaultPropDocs(cfnName, prop),
      },
      validateRequiredInConstructor: !!prop.required,
      cfnMapping: {
        cfnName,
        propName: resolverResult.name,
        baseType: resolverResult.baseType,
        optional,
      },
    });
    this.classProperties.push({
      propertySpec: {
        name: resolverResult.name,
        type: resolverResult.resolvableType,
        optional,
        immutable: false,
        docs: this.defaultClassPropDocs(cfnName, prop),
      },
      cfnName,
      initializer: resolverResult.resolver,
      cfnValueToRender: { [resolverResult.name]: $this[`_${resolverResult.name}`] },
    });
  }

  /**
   * Emit unused types from type history
   *
   * We currently render all types into the spec and need to keep doing that for backwards compatibility.
   */
  private handleTypeHistoryTypes(prop: Property) {
    this.converter
      .typeHistoryFromProperty(prop)
      .slice(1)
      .map((t) => this.converter.typeFromSpecType(t));
  }

  /**
   * Emit legacy taggability
   *
   * This entails:
   *
   * - A props property named after the tags-holding property that is
   *   standardized: either a built-in CDK Tag type array, or a string map
   * - A class property named 'tags' that holds a TagManager and is initialized
   *   from the tags-holding property.
   *
   * We also add a mutable L1 property called '<tagsProperty>Raw' which can be used
   * to add tags apart from the TagManager.
   */
  private handleTagPropertyLegacy(cfnName: string, prop: Property, variant: TagVariant) {
    const originalName = propertyNameFromCloudFormation(cfnName);
    const rawTagsPropName = `${originalName}Raw`;

    const { type, baseType } = this.legacyCompatiblePropType(cfnName, prop);

    this.propsProperties.push({
      propertySpec: {
        name: originalName,
        type,
        optional: true, // Tags are never required
        docs: this.defaultPropDocs(cfnName, prop),
      },
      validateRequiredInConstructor: false, // Tags are never required
      cfnMapping: {
        cfnName,
        propName: originalName,
        baseType,
        optional: true,
      },
    });
    this.classProperties.push(
      {
        propertySpec: {
          // Must be called 'tags' to count as ITaggable
          name: 'tags',
          type: CDK_CORE.TagManager,
          immutable: true,
          docs: {
            summary: 'Tag Manager which manages the tags for this resource',
          },
        },
        cfnName,
        initializer: (props: Expression) =>
          new CDK_CORE.TagManager(
            this.tagManagerVariant(variant),
            expr.lit(this.resource.cloudFormationType),
            $E(props)[originalName],
            expr.object({ tagPropertyName: expr.lit(originalName) }),
          ),
        cfnValueToRender: {
          [originalName]: $this.tags.renderTags(),
        },
      },
      {
        propertySpec: {
          name: rawTagsPropName,
          type,
          optional: true, // Tags are never required
          docs: this.defaultClassPropDocs(cfnName, prop),
        },
        cfnName,
        initializer: (props: Expression) => $E(props)[originalName],
        cfnValueToRender: {}, // Gets rendered as part of the TagManager above
      },
    );
  }

  private handleTagPropertyModern(cfnName: string, prop: Property, variant: TagVariant) {
    const originalName = propertyNameFromCloudFormation(cfnName);
    const originalType = this.converter.typeFromPropertyForModernTags(prop);

    this.propsProperties.push({
      propertySpec: {
        name: originalName,
        type: originalType,
        optional: true, // Tags are never required
        docs: this.defaultPropDocs(cfnName, prop),
      },
      validateRequiredInConstructor: false, // Tags are never required
      cfnMapping: {
        cfnName,
        propName: originalName,
        baseType: originalType,
        optional: true,
      },
    });

    this.classProperties.push(
      {
        propertySpec: {
          // Must be called 'cdkTagManager' to count as ITaggableV2
          name: 'cdkTagManager',
          type: CDK_CORE.TagManager,
          immutable: true,
          docs: {
            summary: 'Tag Manager which manages the tags for this resource',
          },
        },
        cfnName,
        initializer: (_: Expression) =>
          new CDK_CORE.TagManager(
            this.tagManagerVariant(variant),
            expr.lit(this.resource.cloudFormationType),
            expr.UNDEFINED,
            expr.object({ tagPropertyName: expr.lit(originalName) }),
          ),
        cfnValueToRender: {
          [originalName]: $this.cdkTagManager.renderTags($this[`_${originalName}`]),
        },
      },
      {
        propertySpec: {
          name: originalName,
          type: originalType,
          optional: true, // Tags are never required
          docs: this.defaultClassPropDocs(cfnName, prop),
        },
        cfnName,
        initializer: (props: Expression) => $E(props)[originalName],
        cfnValueToRender: {}, // Gets rendered as part of the TagManager above
      },
    );
  }

  /**
   * Return the resolvable and base types for a given property
   *
   * Does type deducation compatibly with the old cfn2ts code base.
   *
   * - Returns the special Tag type if this property had the intrinstic 'Tag' type
   *   in the old spec, otherwise resolves the type as normal.
   * - Skips making the type resolvable if the property has one of the predefined tag
   *   property names.
   */
  private legacyCompatiblePropType(cfnName: string, prop: Property) {
    const baseType = this.converter.typeFromProperty(prop);

    // Whether or not a property is made `IResolvable` originally depended on
    // the name of the property. These conditions were probably expected to coincide
    // with it being a taggable type or not, but they don't always coincide.
    const type = cfnName in NON_RESOLVABLE_PROPERTY_NAMES ? baseType : this.converter.makeTypeResolvable(baseType);

    return { type, baseType };
  }

  private convertAttributes() {
    const $ResolutionTypeHint = $T(CDK_CORE.ResolutionTypeHint);

    for (const [attrName, attr] of Object.entries(this.resource.attributes)) {
      // Just use the oldest type for now
      const specType = new RichProperty(attr).types()[0];

      let type: Type;
      let initializer: Expression;

      if (specType.type === 'string') {
        type = Type.STRING;
        initializer = CDK_CORE.tokenAsString($this.getAtt(expr.lit(attrName), $ResolutionTypeHint.STRING));
      } else if (specType.type === 'integer') {
        type = Type.NUMBER;
        initializer = CDK_CORE.tokenAsNumber($this.getAtt(expr.lit(attrName), $ResolutionTypeHint.NUMBER));
      } else if (specType.type === 'number') {
        // COMPAT: Although numbers/doubles could be represented as numbers, historically in cfn2ts they were represented as IResolvable.
        type = CDK_CORE.IResolvable;
        initializer = $this.getAtt(expr.lit(attrName), $ResolutionTypeHint.NUMBER);
      } else if (specType.type === 'array' && specType.element.type === 'string') {
        type = Type.arrayOf(Type.STRING);
        initializer = CDK_CORE.tokenAsList($this.getAtt(expr.lit(attrName), $ResolutionTypeHint.STRING_LIST));
      } else {
        // This may reference a type we need to generate, so call this function because of its side effect
        this.converter.typeFromSpecType(specType);
        type = CDK_CORE.IResolvable;
        initializer = $this.getAtt(expr.lit(attrName));
      }

      this.classAttributeProperties.push({
        propertySpec: {
          name: this.attributePropertyNames.get(attrName)!,
          type,
          immutable: true,
          docs: {
            summary: attr.documentation,
            remarks: [`@cloudformationAttribute ${attrName}`].join('\n'),
          },
        },
        initializer,
      });
    }
  }

  private defaultPropDocs(cfnName: string, prop: Property) {
    return {
      ...splitDocumentation(prop.documentation),
      default: prop.defaultValue ?? undefined,
      see: cloudFormationDocLink({
        resourceType: this.resource.cloudFormationType,
        propName: cfnName,
      }),
      deprecated: deprecationMessage(prop),
    };
  }

  private defaultClassPropDocs(cfnName: string, prop: Property) {
    void cfnName;
    return {
      summary: splitDocumentation(prop.documentation).summary,
      deprecated: deprecationMessage(prop),
    };
  }

  /**
   * Translates a TagVariant to the core.TagType enum
   */
  private tagManagerVariant(variant: TagVariant) {
    switch (variant) {
      case 'standard':
        return CDK_CORE.TagType.STANDARD;
      case 'asg':
        return CDK_CORE.TagType.AUTOSCALING_GROUP;
      case 'map':
        return CDK_CORE.TagType.MAP;
      default:
        assertNever(variant);
    }
  }
}

/**
 * Utility function to ensure exhaustive checks for never type.
 */
function assertNever(x: never): never {
  throw new Error(`Unexpected object: ${x}`);
}

export interface PropsProperty {
  readonly propertySpec: PropertySpec;
  readonly validateRequiredInConstructor: boolean;
  readonly cfnMapping: PropertyMapping;
}

export interface ClassProperty {
  readonly propertySpec: PropertySpec;

  /** The original CloudFormation property name */
  readonly cfnName: string;

  /** Given the name of the props value, produce the member value */
  readonly initializer: (props: Expression) => Expression;

  /**
   * Lowercase property name(s) and expression(s) to render to get this property into CFN
   *
   * We will do a separate conversion of the casing of the props object, so don't do that here.
   */
  readonly cfnValueToRender: Record<string, Expression>;
}

export interface ClassAttributeProperty {
  readonly propertySpec: PropertySpec;

  /** Produce the initializer value for the member */
  readonly initializer: Expression;
}

export function deprecationMessage(property: Property): string | undefined {
  switch (property.deprecated) {
    case Deprecation.WARN:
      return 'this property has been deprecated';
    case Deprecation.IGNORE:
      return 'this property will be ignored';
  }

  return undefined;
}
