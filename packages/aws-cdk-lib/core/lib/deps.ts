import { CfnResource } from './cfn-resource';
import { AssumptionError, UnscopedValidationError } from './errors';
import { lit } from './private/literal-string';
import { Stack } from './stack';
import { Stage } from './stage';
import { findLastCommonElement, pathToTopLevelStack as pathToRoot } from './util';

export type Element = CfnResource | Stack;

/**
 * Adds a dependency between two resources or stacks, across stack and nested
 * stack boundaries.
 *
 * The algorithm consists of:
 * - Try to find the deepest common stack between the two elements
 * - If there isn't a common stack, it means the elements belong to two
 *   disjoined stack-trees and therefore we apply the dependency at the
 *   assembly/app level between the two top-level stacks.
 * - If we did find a common stack, we apply the dependency as a CloudFormation
 *   "DependsOn" between the resources that "represent" our source and target
 *   either directly or through the AWS::CloudFormation::Stack resources that
 *   "lead" to them.
 *
 * @param source The source resource/stack (the dependent)
 * @param target The target resource/stack (the dependency)
 */
export function addDependency(source: Element, target: Element, reason?: string) {
  operateOnDependency(DependencyOperation.ADD, source, target, reason);
}

/**
 * Removes a dependency between two resources or stacks, across stack and nested
 * stack boundaries.
 *
 * The algorithm consists of:
 * - Try to find the deepest common stack between the two elements
 * - If there isn't a common stack, it means the elements belong to two
 *   disjoined stack-trees and therefore we applied the dependency at the
 *   assembly/app level between the two top-level stacks; remove it there.
 * - If we did find a common stack, we applied the dependency as a CloudFormation
 *   "DependsOn" between the resources that "represent" our source and target
 *   either directly or through the AWS::CloudFormation::Stack resources that
 *   "lead" to them and must remove it there.
 *
 * @param source The source resource/stack (the dependent)
 * @param target The target resource/stack (the dependency)
 * @param reason Optional description to associate with the dependency for
 * diagnostics
 */
export function removeDependency(source: Element, target: Element) {
  operateOnDependency(DependencyOperation.REMOVE, source, target);
}

enum DependencyOperation {
  ADD,
  REMOVE,
}

/**
 * Find the appropriate location for a dependency and add or remove it
 *
 * @internal
 */
function operateOnDependency(operation: DependencyOperation, source: Element, target: Element, description?: string) {
  if (source === target) {
    return;
  }

  const sourceStack = Stack.of(source);
  const targetStack = Stack.of(target);

  const sourceStage = Stage.of(sourceStack);
  const targetStage = Stage.of(targetStack);
  if (sourceStage !== targetStage) {
    throw new UnscopedValidationError(lit`CannotDependency`, `You cannot have a dependency from '${source.node.path}' (in ${describeStage(sourceStage)}) to '${target.node.path}' (in ${describeStage(targetStage)}): dependency cannot cross stage boundaries`);
  }

  // find the deepest common stack between the two elements
  const sourcePath = pathToRoot(sourceStack);
  const targetPath = pathToRoot(targetStack);
  const commonStack = findLastCommonElement(sourcePath, targetPath);

  // if there is no common stack, then look for an assembly-level dependency
  // between the two top-level stacks
  if (!commonStack) {
    const topLevelSource = sourcePath[0]; // first path element is the top-level stack
    const topLevelTarget = targetPath[0];
    const reason = { source, target, description };
    switch (operation) {
      case DependencyOperation.ADD: {
        topLevelSource._addAssemblyDependency(topLevelTarget, reason);
        break;
      }
      case DependencyOperation.REMOVE: {
        topLevelSource._removeAssemblyDependency(topLevelTarget, reason);
        break;
      }
      default: {
        throw new AssumptionError(lit`UnsupportedDependencyOperation`, `Unsupported dependency operation: ${operation}`);
      }
    }
    return;
  }

  // assertion: at this point if source and target are stacks, both are nested stacks.
  // since we have a common stack, it is impossible that both are top-level
  // stacks, so let's examine the two cases where one of them is top-level and
  // the other is nested.

  // case 1 - source is top-level and target is nested: this implies that
  // `target` is a direct or indirect nested stack of `source`, and an explicit
  // dependency is not required because nested stacks will always be deployed
  // before their parents.
  if (commonStack === source) {
    return;
  }

  // case 2 - source is nested and target is top-level: this implies that
  // `source` is a direct or indirect nested stack of `target`, and this is not
  // possible (nested stacks cannot depend on their parents).
  if (commonStack === target) {
    throw new UnscopedValidationError(lit`NestedStackCannotDepend`, `Nested stack '${sourceStack.node.path}' cannot depend on a parent stack '${targetStack.node.path}'`);
  }

  // we have a common stack from which we can reach both `source` and `target`
  // now we need to find two resources which are defined directly in this stack
  // and which can "lead us" to the source/target.
  const sourceResource = resourceInCommonStackFor(source, commonStack);
  const targetResource = resourceInCommonStackFor(target, commonStack);
  switch (operation) {
    case DependencyOperation.ADD: {
      sourceResource._addResourceDependency(targetResource);
      break;
    }
    case DependencyOperation.REMOVE: {
      sourceResource._removeResourceDependency(targetResource);
      break;
    }
    default: {
      throw new AssumptionError(lit`UnsupportedDependencyOperation`, `Unsupported dependency operation: ${operation}`);
    }
  }
}

/**
 * Get a list of all resource-to-resource dependencies assembled from this Element, Stack or assembly-dependencies
 * @param source The source resource/stack (the dependent)
 */
export function obtainDependencies(source: Element) {
  let dependencies: Element[] = [];
  if (source instanceof CfnResource) {
    dependencies = source._obtainResourceDependencies();
  }

  let stacks = pathToRoot(Stack.of(source));
  stacks.forEach((stack) => {
    dependencies = [...dependencies, ...stack._obtainAssemblyDependencies({ source: source })];
  });

  return dependencies;
}

/**
 * Find the resource in a common stack that 'points' to the given element
 *
 * @internal
 */
function resourceInCommonStackFor(element: Element, commonStack: Stack): CfnResource {
  const resource: CfnResource = (Stack.isStack(element) ? element.nestedStackResource : element) as CfnResource;
  if (!resource) {
    // see "assertion" in operateOnDependency above
    throw new AssumptionError(lit`UnexpectedValueResourceLooking`, `Unexpected value for resource when looking at ${element}!`);
  }

  const resourceStack = Stack.of(resource);

  // we reached a resource defined in the common stack
  if (commonStack === resourceStack) {
    return resource;
  }

  return resourceInCommonStackFor(resourceStack, commonStack);
}

/**
 * Return a string representation of the given assembler, for use in error messages
 */
function describeStage(assembly: Stage | undefined): string {
  if (!assembly) { return 'an unrooted construct tree'; }
  if (!assembly.parentStage) { return 'the App'; }
  return `Stage '${assembly.node.path}'`;
}
