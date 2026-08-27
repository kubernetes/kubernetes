import { Dependable, type IConstruct } from 'constructs';
import { UnscopedValidationError } from '../errors';
import type { NestedStack } from '../nested-stack';
import { iterateDfsPreorder, iterateStackCfnResources } from './construct-iteration';
import { NESTED_STACK_TYPE, STACK_TYPE, STAGE_TYPE } from './core-construct-finders';
import { lit } from '../private/literal-string';
import type { Stack } from '../stack';
import type { Stage } from '../stage';

/**
 * Turn all dependencies added via `source.node.addDependency(target)` into concrete dependencies.
 *
 * The semantics are that every construct underneath the source construct
 * depends on every construct underneath the target construct, but we reify this by
 * only applying the dependency between closest "source container" and "target container",
 * which may be for example stacks, and then we can get away with adding a single stack dependency.
 *
 * These will either turn into:
 *
 * - `stack.addStackDependency()` calls
 * - `resource.addResourceDependency()` calls
 *
 * As is appropriate for the constructs involved.
 */
export function reifyConstructDependencies(root: IConstruct) {
  for (const dependency of registeredConstructDependencies(root)) {
    dispatchDependencyOperation(dependency);
  }
}

/**
 * Concretely apply a single dependency
 *
 * This finds the closest related dependency containers and applies the right operation.
 */
export function dispatchDependencyOperation(op: DependencyOperation) {
  if (op.source === op.target) {
    // We can summarily skip self-dependencies
    return;
  }

  if (process.env.CDK_DEBUG_DEPS) {
    if (op.kind === 'add') {
      // eslint-disable-next-line no-console
      console.error(`[CDK_DEBUG_DEPS] applying dependency ${op.reason}`);
    } else {
      // eslint-disable-next-line no-console
      console.error(`[CDK_DEBUG_DEPS] removing construct dependency ${op.source.node.path} -> ${op.target.node.path}`);
    }
  }

  const sourcePath = dependencyContainerPath(op.source);
  const targetPath = dependencyContainerPath(op.target);
  const parentI = sharedDependencyContainerIndex(sourcePath, targetPath);

  validateNoStagesCrossed(sourcePath, targetPath, op);
  validateNoParentChildDependency(sourcePath, targetPath, parentI, op);

  if (parentI < sourcePath.length - 1 && parentI < targetPath.length - 1) {
    doDispatch(sourcePath[parentI + 1], targetPath[parentI + 1], op);
  }
}

type DependencyContainer =
  | { type: 'stage'; construct: Stage } // May also be App
  | { type: 'stack'; construct: Stack }
  | { type: 'nested-stack'; construct: NestedStack }
  | { type: 'construct'; construct: IConstruct }
  ;

function validateNoStagesCrossed(sourcePath: DependencyContainer[], targetPath: DependencyContainer[], op: DependencyOperation) {
  const sourceStage = last(sourcePath.filter((c) => c.type === 'stage'))?.construct;
  const targetStage = last(targetPath.filter((c) => c.type === 'stage'))?.construct;
  if (sourceStage !== targetStage) {
    throw new UnscopedValidationError(lit`NoCrossStageDependency`, `You cannot have a dependency from '${op.source.node.path}' (in ${describeStage(sourceStage)}) to '${op.target.node.path}' (in ${describeStage(targetStage)}): dependency cannot cross stage boundaries`);
  }
}

/**
 * Validate parent-child relationships.
 *
 * If the shared parent is the last element in either path, then one of the
 * constructs is a parent of the other. This can only happen for dependency containers,
 * so [nested] stacks and stages.
 */
function validateNoParentChildDependency(
  sourcePath: DependencyContainer[],
  targetPath: DependencyContainer[],
  parentI: number,
  op: DependencyOperation,
) {
  if (parentI === targetPath.length - 1) {
    throw new UnscopedValidationError(lit`CannotDependOnParent`, `'${op.source.node.path}' cannot depend on parent ${targetPath[parentI].type} '${op.target.node.path}'`);
  }

  // The below should be equally illegal because it is as nonsensical as the
  // reverse case. However, we have unit tests in place to verify that we
  // traditionally treated this as a no-op, so we will continue to do so.
  void sourcePath;

  // if (parentI === sourcePath.length - 1) {
  //   throw new UnscopedValidationError(lit`CannotDependOnChild`, `Construct '${op.source.node.path}' cannot depend on a child construct '${op.target.node.path}'`);
  // }
}

function doDispatch(source: DependencyContainer, target: DependencyContainer, op: DependencyOperation) {
  switch (source.type) {
    case 'stage':
      // Stages don't have any dependencies
      break;
    case 'stack':
      if (target.type === 'stack') {
        if (op.kind === 'add') {
          source.construct._addStackDependency(target.construct, op);
        } else {
          source.construct._removeStackDependency(target.construct, op);
        }
      }
      break;
    case 'nested-stack':
    case 'construct':
      // L1 resource dependendencies
      for (const sourceResource of dependingResourcesFor(source)) {
        for (const targetResource of dependingResourcesFor(target)) {
          // Some constructs add a dependency on their containing parent, which would expand to themselves here.
          // Skip self-dependencies.
          if (sourceResource === targetResource) {
            continue;
          }

          if (op.kind === 'add') {
            sourceResource._addResourceDependency(targetResource);
          } else {
            sourceResource._removeResourceDependency(targetResource);
          }
        }
      }
      break;
  }
}

function* dependingResourcesFor(construct: DependencyContainer) {
  if (construct.type === 'nested-stack' && construct.construct.nestedStackResource) {
    yield construct.construct.nestedStackResource;
  }
  if (construct.type === 'construct') {
    yield* iterateStackCfnResources(construct.construct);
  }
}

/**
 * Find the dependency container path for a given construct.
 *
 * The semantics are slightly non-obvious, which follow from the containment
 * relationships that the CloudFormation-based construct tree encodes.
 *
 * - A Stage contains Stacks
 * - A Stack contains resources or NestedStacks (but not other Stacks!)
 * - NestedStacks contain resources or NestedStacks (but not other Stacks!)
 *
 * The reason is that a non-Nested Stack at any level just encodes a top-level
 * Stack, only contained by its enclosing Stage.
 *
 * Dependency paths that cross Stage boundaries are not allowed, so we could have stopped
 * collecting after the first Stage, but for simplicity we collect all Stages up to the root.
 *
 * These are all the dependency containers on the path to the given construct,
 * ending in the construct itself.
 */
function dependencyContainerPath(start: IConstruct): DependencyContainer[] {
  const ret: DependencyContainer[] = [];

  let construct: IConstruct | undefined = start;
  while (construct) {
    let candidate: DependencyContainer | undefined;
    if (STAGE_TYPE.isMarked(construct)) {
      candidate = { type: 'stage', construct };
    } else if (NESTED_STACK_TYPE.isMarked(construct)) {
      candidate = { type: 'nested-stack', construct };
    } else if (STACK_TYPE.isMarked(construct)) {
      candidate = { type: 'stack', construct };
    } else {
      // Only the first time do we do the construct itself; all other constructs on
      // the path do not contribute to the dependency story, so we skip them.
      if (ret.length === 0) {
        candidate = { type: 'construct', construct };
      }
    }
    if (candidate && (ret.length === 0 || contains(candidate, ret[ret.length - 1]))) {
      ret.push(candidate);
    }

    construct = construct.node.scope;
  }

  return ret.reverse();

  function contains(container: DependencyContainer, child: DependencyContainer): boolean {
    switch (child.type) {
      case 'stage':
        // Stages are only contained by stages
        return container.type == 'stage';
      case 'nested-stack':
        // Nested Stacks are only contained by StackLikes
        return container.type == 'nested-stack' || container.type == 'stack';
      case 'stack':
        // Stacks are only contained by stages
        return container.type == 'stage';
      case 'construct':
        // Constructs are contained by anything (though for our purposes only StackLikes matter)
        return true;
    }
  }
}

/**
 * Find the shared dependency container between these paths, which is the rightmost one that is the same between them.
 *
 * Returns -1 if there are no shared parents, which can only be true if the constructs are in different Apps.
 */
function sharedDependencyContainerIndex(as: DependencyContainer[], bs: DependencyContainer[]): number {
  const minLength = Math.min(as.length, bs.length);
  for (let i = minLength - 1; i >= 0; i--) {
    if (as[i].construct === bs[i].construct) {
      return i;
    }
  }
  return -1;
}

/**
 * Return all dependencies registered on this node or any of its children
 */
function* registeredConstructDependencies(root: IConstruct): IterableIterator<DependencyOperation> {
  const found = new Map<IConstruct, Set<IConstruct>>(); // Deduplication map

  for (const source of iterateDfsPreorder(root)) {
    for (const dependable of source.node.dependencies) {
      for (const target of Dependable.of(dependable).dependencyRoots) {
        if (source === target) {
          // Short-circuit a useless operation
          continue;
        }

        let foundTargets = found.get(source);
        if (!foundTargets) { found.set(source, foundTargets = new Set()); }

        if (!foundTargets.has(target)) {
          yield { kind: 'add', source, target, reason: `<${source.node.path}>.node.addDependency(<${target.node.path}>)` };
          foundTargets.add(target);
        }
      }
    }
  }
}

/**
 * Return a string representation of the given assembler, for use in error messages
 */
function describeStage(assembly: Stage | undefined): string {
  if (!assembly) { return 'an unrooted construct tree'; }
  if (!assembly.parentStage) { return 'the App'; }
  return `Stage '${assembly.node.path}'`;
}

export type DependencyOperation =
  | { readonly kind: 'add'; readonly source: IConstruct; readonly target: IConstruct; readonly reason: string }
  | { readonly kind: 'remove'; readonly source: IConstruct; readonly target: IConstruct }
  ;

function last<A>(xs: A[]): A | undefined {
  return xs[xs.length - 1];
}
