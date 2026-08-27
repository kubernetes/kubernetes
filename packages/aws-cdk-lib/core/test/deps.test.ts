import { CloudFormationStackArtifact } from '@aws-cdk/cloud-assembly-api';
import { Construct } from 'constructs';
import * as core from '../lib';
import { Names } from '../lib';
import { stackOf } from '../lib/private/core-construct-finders';
import { dispatchDependencyOperation } from '../lib/private/deps';

let app: core.App;

beforeEach(() => {
  app = new core.App();
});

describe('deps', () => {
  describe('dependency methods', () => {
    test('can explicitly add a dependency between resources', () => {
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack, 'Resource2', { type: 'Test::Resource::Fake2' });

      resource1.addResourceDependency(resource2);

      expect(app.synth().getStackByName(stack.stackName).template.Resources).toEqual({
        Resource1: {
          Type: 'Test::Resource::Fake1',
          DependsOn: [
            'Resource2',
          ],
        },
        Resource2: {
          Type: 'Test::Resource::Fake2',
        },
      });
    });

    test('can explicitly remove a dependency between resources', () => {
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack, 'Resource2', { type: 'Test::Resource::Fake2' });
      resource1.addResourceDependency(resource2);
      resource1.removeResourceDependency(resource2);

      expect(app.synth().getStackByName(stack.stackName).template.Resources).toEqual({
        Resource1: {
          Type: 'Test::Resource::Fake1',
        },
        Resource2: {
          Type: 'Test::Resource::Fake2',
        },
      });
    });

    test('can explicitly add, obtain, and remove dependencies across stacks', () => {
      const stack1 = new core.Stack(app, 'TestStack1');
      // Use a really long construct id to identify issues between Names.uniqueId and Names.uniqueResourceName
      const reallyLongConstructId = 'A'.repeat(247);
      const stack2 = new core.Stack(app, reallyLongConstructId, { stackName: 'TestStack2' });
      // Sanity check since this test depends on the discrepancy
      expect(Names.uniqueId(stack2)).not.toBe(Names.uniqueResourceName(stack2, {}));
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack2, 'Resource2', { type: 'Test::Resource::Fake2' });
      const resource3 = new core.CfnResource(stack1, 'Resource3', { type: 'Test::Resource::Fake3' });

      resource1.addResourceDependency(resource2);
      // Adding the same resource dependency twice should be a no-op
      resource1.addDependency(resource2);
      resource1.addDependency(resource3);
      expect(stack1.dependencies.length).toEqual(1);
      expect(stack1.dependencies[0].node.id).toEqual(stack2.node.id);
      // obtainDependencies should assemble and flatten resource-to-resource dependencies even across stacks
      expect(resource1.obtainDependencies().map(x => x.node.path)).toEqual([resource3.node.path, resource2.node.path]);

      resource1.removeResourceDependency(resource2);
      // For symmetry, removing a dependency that doesn't exist should be a no-op
      resource1.removeResourceDependency(resource2);
      expect(stack1.dependencies.length).toEqual(0);
    });

    test('do nothing if source is target', () => {
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      resource1.addResourceDependency(resource1);

      expect(app.synth().getStackByName(stack.stackName).template.Resources).toEqual({
        Resource1: {
          Type: 'Test::Resource::Fake1',
        },
      });
    });

    test('handle source being common stack', () => {
      const stack1 = new core.Stack(app, 'TestStack1');
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });

      // If source is the common stack, this should be a noop
      dispatchDependencyOperation({
        kind: 'add',
        source: stack1,
        target: resource1,
        reason: 'test',
      });
      expect(stack1.dependencies.length).toEqual(0);
    });

    test('throws error if target is common stack', () => {
      const stack1 = new core.Stack(app, 'TestStack1');
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });

      expect(() => dispatchDependencyOperation({
        kind: 'add',
        source: resource1,
        target: stack1,
        reason: 'test',
      })).toThrow(/cannot depend on /);
    });

    test('can explicitly add, obtain, and remove dependencies across nested stacks', () => {
      const stack1 = new core.Stack(app, 'TestStack1');
      const construct1 = new Construct(stack1, 'CommonConstruct');
      // Use a really long construct id to identify issues between Names.uniqueId and Names.uniqueResourceName
      const nestedStack1 = new core.Stack(construct1, 'TestNestedStack1');
      const nestedStack2 = new core.Stack(construct1, 'TestNestedStack2');
      const resource1 = new core.CfnResource(nestedStack1, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(nestedStack2, 'Resource2', { type: 'Test::Resource::Fake2' });

      resource1.addDependency(resource2);
      // Adding the same resource dependency twice should be a no-op
      resource1.addDependency(resource2);
      expect(nestedStack1.dependencies.length).toEqual(1);
      expect(nestedStack1.dependencies[0].node.id).toEqual(nestedStack2.node.id);

      resource1.removeDependency(resource2);
      // For symmetry, removing a dependency that doesn't exist should be a no-op
      resource1.removeDependency(resource2);
      expect(stack1.dependencies.length).toEqual(0);
    });

    test('node.addDependency stack dependencies should not be superlinear in size of stack', () => {
      const nStacks = 10;
      const baseN = 35;

      const growthFactor = 4; // Should take 4x the time, not 16x (or worse)
      const errorMargin = 2; // Have some measurement margin of error

      const small = runTest(baseN);
      const large = runTest(baseN * growthFactor);

      console.log({ small, large });

      expect(large).toBeLessThan(small * growthFactor * errorMargin);

      function runTest(nResources: number) {
        const start = Date.now();

        const innerApp = new core.App();

        let lastStack: core.Stack | undefined;
        for (let i = 0; i < nStacks; i++) {
          const stack = new core.Stack(innerApp, `TestStack${i}`);
          for (let j = 0; j < nResources; j++) {
            new core.CfnResource(stack, `Resource${j}`, { type: 'Test::Resource::Fake' });
          }
          lastStack?.node.addDependency(stack);
          lastStack = stack;
        }

        innerApp.synth();

        return Date.now() - start;
      }
    });
  });

  describe('dependencies involving NestedStacks', () => {
    let fixture: ReturnType<typeof setUp>;
    function setUp() {
      const rootStack = new core.Stack(app, 'Root');
      const compoundConstruct = new Construct(rootStack, 'Compound');
      const nestedStack = new core.NestedStack(compoundConstruct, 'Nested');
      const nestedStackSibling = new core.CfnResource(compoundConstruct, 'NestedSibling', { type: 'Test::Resource::Fake' });

      const nestedChild = new core.CfnResource(nestedStack, 'NestedChild', { type: 'Test::Resource::Fake' });
      const topChild = new core.CfnResource(rootStack, 'TopChild', { type: 'Test::Resource::Fake' });

      return { rootStack, nestedStack, compoundConstruct, nestedStackSibling, nestedChild, topChild };
    }
    beforeEach(() => {
      fixture = setUp();
    });

    test('Top -> Compound/Nested/Child: leads to Top -> Compound/Nested', () => {
      const { topChild, nestedChild } = fixture;

      topChild.node.addDependency(nestedChild);

      expect(resourceSection(topChild).DependsOn).toEqual(['CompoundNestedNestedStackNestedNestedStackResourceBB6D325A']);
    });

    test('Compound/Nested/Child -> Top: leads to Compound/Nested -> Top', () => {
      const { topChild, nestedChild, nestedStack } = fixture;

      nestedChild.node.addDependency(topChild);

      expect(resourceSection(nestedStack.nestedStackResource!).DependsOn).toEqual(['TopChild']);
    });

    test('Top -> Compound[/Nested/Child]: leads to only Top -> Compound/Nested', () => {
      const { topChild, compoundConstruct } = fixture;

      topChild.node.addDependency(compoundConstruct);

      expect(resourceSection(topChild).DependsOn).toEqual(['CompoundNestedNestedStackNestedNestedStackResourceBB6D325A', 'CompoundNestedSibling55188EFC']);
    });
  });

  describe('dependencies involving support Stacks', () => {
    let fixture: ReturnType<typeof setUp>;
    function setUp() {
      const topStack = new core.Stack(app, 'Root');
      const compoundConstruct = new Construct(topStack, 'Compound');
      const supportStack = new core.Stack(compoundConstruct, 'Support');
      const supportStackSibling = new core.CfnResource(compoundConstruct, 'NestedSibling', { type: 'Test::Resource::Fake' });

      const supportChild = new core.CfnResource(supportStack, 'SupportChild', { type: 'Test::Resource::Fake' });
      const topChild = new core.CfnResource(topStack, 'TopChild', { type: 'Test::Resource::Fake' });

      return { topStack, supportStack, compoundConstruct, supportStackSibling, supportChild, topChild };
    }
    beforeEach(() => {
      fixture = setUp();
    });

    test('Top -> Compound/Support/Child: leads to TopStack -[stack]-> SupportStack', () => {
      const { topChild, supportChild, topStack, supportStack } = fixture;

      topChild.node.addDependency(supportChild);

      expect(resourceSection(topChild).DependsOn).toEqual(undefined);
      expect(stackDependencies(topStack)).toEqual([stackArtifact(supportStack)]);
    });

    test('Compound/Support/Child -> Top: leads to SupportStack -[stack]-> TopStack', () => {
      const { topChild, supportChild, topStack, supportStack } = fixture;

      supportChild.node.addDependency(topChild);

      expect(resourceSection(topChild).DependsOn).toEqual(undefined);
      expect(stackDependencies(supportStack)).toEqual([stackArtifact(topStack)]);
    });

    test('Top -> Compound[/Support/Child]: leads to Top -> Compound/Sibling (and no stack dependencies)', () => {
      const { topChild, compoundConstruct, topStack } = fixture;

      topChild.node.addDependency(compoundConstruct);

      expect(resourceSection(topChild).DependsOn).toEqual(['CompoundNestedSibling55188EFC']);
      expect(stackDependencies(topStack)).toEqual([]);
    });
  });
});

function stackArtifact(stack: core.Stack) {
  const asm = app.synth();
  return asm.getStackByName(stack.stackName);
}

/**
 * Return stack artifact dependencies (ignore other types of dependencies like asset manifests)
 */
function stackDependencies(stack: core.Stack) {
  return stackArtifact(stack).dependencies
    .filter(CloudFormationStackArtifact.isCloudFormationStackArtifact);
}

function resourceSection(resource: core.CfnResource) {
  const stack = stackOf(resource);
  const asm = app.synth();
  const logicalId = stack.resolve(resource.logicalId);

  return asm.getStackByName(stack.stackName).template.Resources?.[logicalId];
}
