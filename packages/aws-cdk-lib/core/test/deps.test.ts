import { Construct } from 'constructs';
import * as core from '../lib';
import { Names } from '../lib';
import { addDependency, obtainDependencies, removeDependency } from '../lib/deps';

describe('deps', () => {
  describe('dependency methods', () => {
    test('can explicitly add a dependency between resources', () => {
      const app = new core.App();
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack, 'Resource2', { type: 'Test::Resource::Fake2' });
      addDependency(resource1, resource2);

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
      const app = new core.App();
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack, 'Resource2', { type: 'Test::Resource::Fake2' });
      addDependency(resource1, resource2);
      removeDependency(resource1, resource2);

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
      const app = new core.App();
      const stack1 = new core.Stack(app, 'TestStack1');
      // Use a really long construct id to identify issues between Names.uniqueId and Names.uniqueResourceName
      const reallyLongConstructId = 'A'.repeat(247);
      const stack2 = new core.Stack(app, reallyLongConstructId, { stackName: 'TestStack2' });
      // Sanity check since this test depends on the discrepancy
      expect(Names.uniqueId(stack2)).not.toBe(Names.uniqueResourceName(stack2, {}));
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(stack2, 'Resource2', { type: 'Test::Resource::Fake2' });
      const resource3 = new core.CfnResource(stack1, 'Resource3', { type: 'Test::Resource::Fake3' });

      addDependency(resource1, resource2);
      // Adding the same resource dependency twice should be a no-op
      addDependency(resource1, resource2);
      addDependency(resource1, resource3);
      expect(stack1.dependencies.length).toEqual(1);
      expect(stack1.dependencies[0].node.id).toEqual(stack2.node.id);
      // obtainDependencies should assemble and flatten resource-to-resource dependencies even across stacks
      expect(obtainDependencies(resource1).map(x => x.node.path)).toEqual([resource3.node.path, resource2.node.path]);

      removeDependency(resource1, resource2);
      // For symmetry, removing a dependency that doesn't exist should be a no-op
      removeDependency(resource1, resource2);
      expect(stack1.dependencies.length).toEqual(0);
    });

    test('do nothing if source is target', () => {
      const app = new core.App();
      const stack = new core.Stack(app, 'TestStack');
      const resource1 = new core.CfnResource(stack, 'Resource1', { type: 'Test::Resource::Fake1' });
      addDependency(resource1, resource1);

      expect(app.synth().getStackByName(stack.stackName).template.Resources).toEqual({
        Resource1: {
          Type: 'Test::Resource::Fake1',
        },
      });
    });

    test('handle source being common stack', () => {
      const app = new core.App();
      const stack1 = new core.Stack(app, 'TestStack1');
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });

      // If source is the common stack, this should be a noop
      addDependency(stack1, resource1);
      expect(stack1.dependencies.length).toEqual(0);
    });

    test('throws error if target is common stack', () => {
      const app = new core.App();
      const stack1 = new core.Stack(app, 'TestStack1');
      const resource1 = new core.CfnResource(stack1, 'Resource1', { type: 'Test::Resource::Fake1' });

      expect(() => {
        addDependency(resource1, stack1);
      }).toThrow(/cannot depend on /);
    });

    test('can explicitly add, obtain, and remove dependencies across nested stacks', () => {
      const app = new core.App();
      const stack1 = new core.Stack(app, 'TestStack1');
      const construct1 = new Construct(stack1, 'CommonConstruct');
      // Use a really long construct id to identify issues between Names.uniqueId and Names.uniqueResourceName
      const nestedStack1 = new core.Stack(construct1, 'TestNestedStack1');
      const nestedStack2 = new core.Stack(construct1, 'TestNestedStack2');
      const resource1 = new core.CfnResource(nestedStack1, 'Resource1', { type: 'Test::Resource::Fake1' });
      const resource2 = new core.CfnResource(nestedStack2, 'Resource2', { type: 'Test::Resource::Fake2' });

      addDependency(resource1, resource2);
      // Adding the same resource dependency twice should be a no-op
      addDependency(resource1, resource2);
      expect(nestedStack1.dependencies.length).toEqual(1);
      expect(nestedStack1.dependencies[0].node.id).toEqual(nestedStack2.node.id);

      removeDependency(resource1, resource2);
      // For symmetry, removing a dependency that doesn't exist should be a no-op
      removeDependency(resource1, resource2);
      expect(stack1.dependencies.length).toEqual(0);
    });
  });
});
