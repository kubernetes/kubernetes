import type { IConstruct } from 'constructs';
import { isCfnResource, NESTED_STACK_TYPE, STACK_TYPE, STAGE_TYPE } from './core-construct-finders';
import { LinkedQueue } from './linked-queue';
import type { CfnResource } from '../cfn-resource';

/**
 * Depth-first iterator over the construct tree
 *
 * Replaces `node.findAll()` which both uses recursive function
 * calls and accumulates into an array, both of which are much slower
 * than this solution.
 */
export function* iterateDfsPreorder(root: IConstruct) {
  const stack: IConstruct[] = [root];

  let next = stack.pop();
  while (next) {
    // Reverse the children so that they get popped in original array order,
    // consistent with the behavior of `node.findAll()`.
    stack.push(...next.node.children.reverse());
    yield next;

    next = stack.pop();
  }
}

/**
 * Depth-first iterator over CFN resources under the given construct, including the construct itself
 *
 * Will break at stack boundaries that logically form a different deployment unit: nested Stacks,
 * NestedStacks and Stages.
 *
 * Replaces `node.findAll()` which both uses recursive function
 * calls and accumulates into an array, both of which are much slower
 * than this solution.
 */
export function* iterateStackCfnResources(root: IConstruct): IterableIterator<CfnResource> {
  const stack: IConstruct[] = [root];

  let next = stack.pop();
  while (next) {
    if (NESTED_STACK_TYPE.isMarked(next)) {
      if (next.nestedStackResource) {
        yield next.nestedStackResource;
      }
    } else if (STACK_TYPE.isMarked(next) || STAGE_TYPE.isMarked(next)) {
      // New deployment unit, do not recurse
    } else {
      if (isCfnResource(next)) {
        yield next;
      }
      // Reverse the children so that they get popped in original array order,
      // consistent with the behavior of `node.findAll()`.
      stack.push(...next.node.children.reverse());
    }

    next = stack.pop();
  }
}

/**
 * Depth-first iterator over the construct tree (post-order version)
 *
 * Replaces `node.findAll()` which both uses recursive function
 * calls and accumulates into an array, both of which are much slower
 * than this solution.
 */
export function* iterateDfsPostorder(root: IConstruct) {
  type Instruction = { yield: IConstruct } | { visit: IConstruct };
  const stack: Instruction[] = [{ visit: root }];

  let next = stack.pop();
  while (next) {
    if ('yield' in next) {
      yield next.yield;
    } else {
      stack.push({ yield: next.visit });
      stack.push(...next.visit.node.children.reverse().map((visit) => ({ visit })));
    }
    next = stack.pop();
  }
}

/**
 * Breadth-first iterator over the construct tree
 */
export function* iterateBfs(root: IConstruct) {
  // Use a specialized queue data structure. Using `Array.shift()`
  // has a huge performance penalty (difference on the order of
  // ~50ms vs ~1s to iterate a large construct tree)
  const queue = new LinkedQueue<{ construct: IConstruct; parent: IConstruct | undefined }>([{ construct: root, parent: undefined }]);

  let next = queue.shift();
  while (next) {
    for (const child of next.construct.node.children) {
      queue.push({ construct: child, parent: next.construct });
    }
    yield next;

    next = queue.shift();
  }
}
