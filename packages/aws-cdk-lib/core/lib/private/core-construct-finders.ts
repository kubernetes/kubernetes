/**
 * Type testing routines for important constructs.
 *
 * These are in a separate file because we need them all over the place (for things like `Stack.isStack` and `Stack.of`)
 * and they commonly lead to circular dependencies between core files.
 *
 * Circular `import type` is fine, circular regular `import` is not.
 */
import { type IConstruct } from 'constructs';
import type { App } from '../app';
import type { CfnResource } from '../cfn-resource';
import { ValidationError } from '../errors';
import type { NestedStack } from '../nested-stack';
import type { Stack } from '../stack';
import type { Stage } from '../stage';
import { lit } from './literal-string';

class ConstructType<A> {
  constructor(private readonly sym: symbol) { }

  public mark(x: A) {
    Object.defineProperty(x, this.sym, { value: true });
  }

  public isMarked(x: any): x is A {
    return x !== null && typeof (x) === 'object' && this.sym in x;
  }
}

export const APP_TYPE = new ConstructType<App>(Symbol.for('@aws-cdk/core.App'));
export const STACK_TYPE = new ConstructType<Stack>(Symbol.for('@aws-cdk/core.Stack'));
export const STAGE_TYPE = new ConstructType<Stage>(Symbol.for('@aws-cdk/core.Stage'));
export const NESTED_STACK_TYPE = new ConstructType<NestedStack>(Symbol.for('@aws-cdk/core.NestedStack'));

const MY_STACK_CACHE = Symbol.for('@aws-cdk/core.Stack.myStack');

export function stackOf(construct: IConstruct): Stack {
  // we want this to be as cheap as possible. cache this result by mutating
  // the object. anecdotally, at the time of this writing, @aws-cdk/core unit
  // tests hit this cache 1,112 times, @aws-cdk/aws-cloudformation unit tests
  // hit this 2,435 times).
  const cache = (construct as any)[MY_STACK_CACHE] as Stack | undefined;
  if (cache) {
    return cache;
  } else {
    const value = _lookup(construct);
    Object.defineProperty(construct, MY_STACK_CACHE, {
      enumerable: false,
      writable: false,
      configurable: false,
      value,
    });
    return value;
  }

  function _lookup(c: IConstruct): Stack {
    if (STACK_TYPE.isMarked(c)) {
      return c;
    }

    const _scope = c.node.scope;
    if (STAGE_TYPE.isMarked(c) || !_scope) {
      throw new ValidationError(lit`ShouldBeCreatedInStackScope`, `${construct.constructor?.name ?? 'Construct'} at '${construct.node.path}' should be created in the scope of a Stack, but no Stack found`, c);
    }

    return _lookup(_scope);
  }
}

export function stageOf(construct: IConstruct): Stage | undefined {
  return construct.node.scopes.reverse().slice(1).find(STAGE_TYPE.isMarked.bind(STAGE_TYPE));
}

export function appOf(construct: IConstruct): App | undefined {
  const root = construct.node.root;
  return APP_TYPE.isMarked(root) ? root : undefined;
}

/**
 * Check whether the given object is a CfnResource
 */
export function isCfnResource(x: any): x is CfnResource {
  return x !== null && typeof (x) === 'object' && x.cfnResourceType !== undefined;
}
