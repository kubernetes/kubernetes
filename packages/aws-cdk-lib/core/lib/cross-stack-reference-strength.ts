import type { IConstruct } from 'constructs';
import { CfnResource } from './cfn-resource';
import { ValidationError } from './errors';
import * as cxapi from '../../cx-api';
import { lit } from './private/literal-string';

/**
 * Controls how cross-stack references to a resource are resolved.
 */
export enum ReferenceStrength {
  /**
   * Strong reference: uses CloudFormation Export/Import (same region)
   * or ExportWriter/ExportReader custom resources (cross-region).
   *
   * The producing stack cannot be deleted while consumers exist.
   */
  STRONG = 'strong',

  /**
   * Weak reference: uses Fn::GetStackOutput to read an output directly
   * from the producing stack.
   *
   * The producing stack or resource can be deleted independently of consumers.
   * This will cause infrastructure in consuming stacks to temporarily reference a nonexistent
   * resource until the consumers are updated as well, causing any accesses in that time
   * frame to fail.
   *
   * Strong references prevent this.
   */
  WEAK = 'weak',

  /**
   * Both strong and weak mechanisms are created (transitional state).
   *
   * Use this when migrating from strong to weak. The producer keeps the
   * strong-side artifacts and also adds a plain Output. The consumer
   * switches to Fn::GetStackOutput.
   */
  BOTH = 'both',
}

/**
 * Ergonomic API for configuring cross-stack reference strength on a construct.
 */
export class CrossStackReferences {
  /**
   * Returns a `CrossStackReferences` configurator for the given construct.
   *
   * @param scope The construct to configure.
   */
  public static of(scope: IConstruct): CrossStackReferences {
    return new CrossStackReferences(scope);
  }

  private constructor(private readonly scope: IConstruct) {}

  /**
   * Set how this resource is referenced when consumed from another stack.
   *
   * This controls the producing side: any cross-stack reference pointing at
   * this resource will use the specified strength instead of the global default.
   *
   * Equivalent to `scope.applyCrossStackReferenceStrength(strength)`.
   *
   * @param strength - The reference strength to use.
   */
  public produce(strength: ReferenceStrength): void {
    const target = this.cfnTarget();
    target.applyCrossStackReferenceStrength(strength);
  }

  /**
   * Set the default reference strength used when this scope consumes references
   * from other stacks.
   *
   * This controls the consuming side: sets the context key that determines how
   * incoming cross-stack references are resolved for this scope and its descendants.
   *
   * Equivalent to `scope.node.setContext(DEFAULT_CROSS_STACK_REFERENCES, strength)`.
   *
   * @param strength - The reference strength to use.
   */
  public consume(strength: ReferenceStrength): void {
    this.scope.node.setContext(cxapi.DEFAULT_CROSS_STACK_REFERENCES, strength);
  }

  private cfnTarget(): CfnResource {
    const defaultChild = this.scope.node.defaultChild;
    if (defaultChild && CfnResource.isCfnResource(defaultChild)) {
      return defaultChild;
    }
    if (CfnResource.isCfnResource(this.scope)) {
      return this.scope;
    }
    throw new ValidationError(lit`CannotApplyXRefStrength`,
      'Cannot apply cross-stack reference strength: scope has no default child CfnResource and is not a CfnResource itself.',
      this.scope);
  }
}
