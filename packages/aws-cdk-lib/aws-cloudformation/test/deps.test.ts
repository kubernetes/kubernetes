import * as fs from 'fs';
import * as path from 'path';
import { testDeprecated, describeDeprecated } from '@aws-cdk/cdk-build-tools';
import { Template } from '../../assertions';
import { App, CfnResource, Stack } from '../../core';
import * as cxapi from '../../cx-api';
import { NestedStack } from '../lib';

describe('resource dependencies', () => {
  test('between two resources in a top-level stack', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const r1 = new CfnResource(stack, 'r1', { type: 'r1' });
    const r2 = new CfnResource(stack, 'r2', { type: 'r2' });

    // WHEN
    r1.addResourceDependency(r2);

    // THEN
    expect(app.synth().getStackArtifact(stack.artifactId).template?.Resources).toEqual({
      r1: { Type: 'r1', DependsOn: ['r2'] }, r2: { Type: 'r2' },
    });
  });

  describeDeprecated('resource in nested stack depends on a resource in the parent stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const parent = new Stack(undefined, 'root');
    const nested = new NestedStack(parent, 'Nested');
    const resourceInParent = new CfnResource(parent, 'ResourceInParent', { type: 'PARENT' });
    const resourceInNested = new CfnResource(nested, 'ResourceInNested', { type: 'NESTED' });

    // WHEN
    addDep(resourceInNested, resourceInParent);

    // THEN: the dependency needs to transfer from the resource within the
    // nested stack to the nested stack resource itself so the nested stack
    // will only be deployed the dependent resource
    Template.fromStack(parent).hasResource('AWS::CloudFormation::Stack', { DependsOn: ['ResourceInParent'] });
    Template.fromStack(nested).templateMatches({ Resources: { ResourceInNested: { Type: 'NESTED' } } }); // no DependsOn for the actual resource
  }));

  describeDeprecated('resource in nested stack depends on a resource in a grandparent stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const grantparent = new Stack(undefined, 'Grandparent');
    const parent = new NestedStack(grantparent, 'Parent');
    const nested = new NestedStack(parent, 'Nested');
    const resourceInGrandparent = new CfnResource(grantparent, 'ResourceInGrandparent', { type: 'GRANDPARENT' });
    const resourceInNested = new CfnResource(nested, 'ResourceInNested', { type: 'NESTED' });

    // WHEN
    addDep(resourceInNested, resourceInGrandparent);

    // THEN: the dependency needs to transfer from the resource within the
    // nested stack to the *parent* nested stack
    Template.fromStack(grantparent).hasResource('AWS::CloudFormation::Stack', { DependsOn: ['ResourceInGrandparent'] });
    Template.fromStack(nested).templateMatches({ Resources: { ResourceInNested: { Type: 'NESTED' } } }); // no DependsOn for the actual resource
  }));

  describeDeprecated('resource in parent stack depends on resource in nested stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const parent = new Stack(undefined, 'root');
    const nested = new NestedStack(parent, 'Nested');
    const resourceInParent = new CfnResource(parent, 'ResourceInParent', { type: 'PARENT' });
    const resourceInNested = new CfnResource(nested, 'ResourceInNested', { type: 'NESTED' });

    // WHEN
    addDep(resourceInParent, resourceInNested);

    // THEN: resource in parent needs to depend on the nested stack
    Template.fromStack(parent).hasResource('PARENT', {
      DependsOn: [parent.resolve(nested.nestedStackResource!.logicalId)],
    });
  }));

  describeDeprecated('resource in grantparent stack depends on resource in nested stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const grandparent = new Stack(undefined, 'Grandparent');
    const parent = new NestedStack(grandparent, 'Parent');
    const nested = new NestedStack(parent, 'Nested');
    const resourceInGrandparent = new CfnResource(grandparent, 'ResourceInGrandparent', { type: 'GRANDPARENT' });
    const resourceInNested = new CfnResource(nested, 'ResourceInNested', { type: 'NESTED' });

    // WHEN
    addDep(resourceInGrandparent, resourceInNested);

    // THEN: resource in grantparent needs to depend on the top-level nested stack
    Template.fromStack(grandparent).hasResource('GRANDPARENT', {
      DependsOn: [grandparent.resolve(parent.nestedStackResource!.logicalId)],
    });
  }));

  describeDeprecated('resource in sibling stack depends on a resource in nested stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const app = new App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack1 = new Stack(app, 'Stack1');
    const nested1 = new NestedStack(stack1, 'Nested1');
    const resourceInNested1 = new CfnResource(nested1, 'ResourceInNested', { type: 'NESTED' });
    const stack2 = new Stack(app, 'Stack2');
    const resourceInStack2 = new CfnResource(stack2, 'ResourceInSibling', { type: 'SIBLING' });

    // WHEN
    addDep(resourceInStack2, resourceInNested1);

    // THEN: stack2 should depend on stack1 and no "DependsOn" inside templates
    const assembly = app.synth();
    assertAssemblyDependency(assembly, stack1, []);
    assertAssemblyDependency(assembly, stack2, ['Stack1']);
    assertNoDependsOn(assembly, stack1);
    assertNoDependsOn(assembly, stack2);
    assertNoDependsOn(assembly, nested1);
  }));

  describeDeprecated('resource in nested stack depends on a resource in sibling stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const app = new App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack1 = new Stack(app, 'Stack1');
    const nested1 = new NestedStack(stack1, 'Nested1');
    const resourceInNested1 = new CfnResource(nested1, 'ResourceInNested', { type: 'NESTED' });
    const stack2 = new Stack(app, 'Stack2');
    const resourceInStack2 = new CfnResource(stack2, 'ResourceInSibling', { type: 'SIBLING' });

    // WHEN
    addDep(resourceInNested1, resourceInStack2);

    // THEN: stack1 should depend on stack2 and no "DependsOn" inside templates
    const assembly = app.synth();
    assertAssemblyDependency(assembly, stack1, ['Stack2']);
    assertAssemblyDependency(assembly, stack2, []);
    assertNoDependsOn(assembly, stack1);
    assertNoDependsOn(assembly, stack2);
    assertNoDependsOn(assembly, nested1);
  }));

  describeDeprecated('resource in nested stack depends on a resource in nested sibling stack', matrixForResourceDependencyTest((addDep) => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack1');
    const nested1 = new NestedStack(stack, 'Nested1');
    const nested2 = new NestedStack(stack, 'Nested2');
    const resourceInNested1 = new CfnResource(nested1, 'ResourceInNested1', { type: 'NESTED1' });
    const resourceInNested2 = new CfnResource(nested2, 'ResourceInNested2', { type: 'NESTED2' });

    // WHEN
    addDep(resourceInNested1, resourceInNested2);

    // THEN: dependency transfered to nested stack resources
    Template.fromStack(stack).hasResource('AWS::CloudFormation::Stack', {
      DependsOn: [stack.resolve(nested2.nestedStackResource!.logicalId)],
    });

    expect(Template.fromStack(stack).findResources('AWS::CloudFormation::Stack', {
      DependsOn: [stack.resolve(nested1.nestedStackResource!.logicalId)],
    })).toEqual({});
  }));
});

describe('stack dependencies', () => {
  test('top level stack depends on itself', () => {
    // GIVEN
    const app = new App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new Stack(app, 'Stack');

    // WHEN
    stack.addStackDependency(stack);

    // THEN
    const assembly = app.synth();
    assertAssemblyDependency(assembly, stack, []);
    assertNoDependsOn(assembly, stack);
  });

  testDeprecated('nested stack depends on itself', () => {
    // GIVEN
    const app = new App();
    const parent = new Stack(app, 'Parent');
    const nested = new NestedStack(parent, 'Nested');

    // WHEN
    nested.addStackDependency(nested);

    // THEN
    assertNoDependsOn(app.synth(), parent);
  });

  testDeprecated('nested stack cannot depend on any of its parents', () => {
    // GIVEN
    const root = new Stack();
    const nested1 = new NestedStack(root, 'Nested1');
    const nested2 = new NestedStack(nested1, 'Nested2');

    // THEN
    expect(() => nested1.addStackDependency(root)).toThrow(/'Default\/Nested1' cannot depend on parent stack 'Default'/);
    expect(() => nested2.addStackDependency(nested1)).toThrow(/'Default\/Nested1\/Nested2' cannot depend on parent nested-stack 'Default\/Nested1'/);
    expect(() => nested2.addStackDependency(root)).toThrow(/'Default\/Nested1\/Nested2' cannot depend on parent stack 'Default'/);
  });

  testDeprecated('any parent stack is by definition dependent on the nested stack so dependency is ignored', () => {
    // GIVEN
    const root = new Stack();
    const nested1 = new NestedStack(root, 'Nested1');
    const nested2 = new NestedStack(nested1, 'Nested2');

    // WHEN
    root.addStackDependency(nested1);
    root.addStackDependency(nested2);
    nested1.addStackDependency(nested2);
  });

  testDeprecated('sibling nested stacks transfer to resources', () => {
    // GIVEN
    const stack = new Stack();
    const nested1 = new NestedStack(stack, 'Nested1');
    const nested2 = new NestedStack(stack, 'Nested2');

    // WHEN
    nested1.addStackDependency(nested2);

    // THEN
    Template.fromStack(stack).hasResource('AWS::CloudFormation::Stack', {
      DependsOn: [stack.resolve(nested2.nestedStackResource!.logicalId)],
    });
  });

  testDeprecated('nested stack depends on a deeply nested stack', () => {
    // GIVEN
    const stack = new Stack();
    const nested1 = new NestedStack(stack, 'Nested1');
    const nested2 = new NestedStack(stack, 'Nested2');
    const nested21 = new NestedStack(nested2, 'Nested21');

    // WHEN
    nested1.addStackDependency(nested21);

    // THEN: transfered to a resource dep between the resources in the common stack
    Template.fromStack(stack).hasResource('AWS::CloudFormation::Stack', {
      DependsOn: [stack.resolve(nested2.nestedStackResource!.logicalId)],
    });
  });

  testDeprecated('deeply nested stack depends on a parent nested stack', () => {
    // GIVEN
    const stack = new Stack();
    const nested1 = new NestedStack(stack, 'Nested1');
    const nested2 = new NestedStack(stack, 'Nested2');
    const nested21 = new NestedStack(nested2, 'Nested21');

    // WHEN
    nested21.addStackDependency(nested1);

    // THEN: transfered to a resource dep between the resources in the common stack
    Template.fromStack(stack).hasResource('AWS::CloudFormation::Stack', {
      DependsOn: [stack.resolve(nested1.nestedStackResource!.logicalId)],
    });
  });

  testDeprecated('top-level stack depends on a nested stack within a sibling', () => {
    // GIVEN
    const app = new App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack1 = new Stack(app, 'Stack1');
    const nested1 = new NestedStack(stack1, 'Nested1');
    const stack2 = new Stack(app, 'Stack2');

    // WHEN
    stack2.addStackDependency(nested1);

    // THEN: assembly-level dependency between stack2 and stack1
    const assembly = app.synth();
    assertAssemblyDependency(assembly, stack2, ['Stack1']);
    assertAssemblyDependency(assembly, stack1, []);
    assertNoDependsOn(assembly, stack1);
    assertNoDependsOn(assembly, stack2);
    assertNoDependsOn(assembly, nested1);
  });

  testDeprecated('nested stack within a sibling depends on top-level stack', () => {
    // GIVEN
    const app = new App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack1 = new Stack(app, 'Stack1');
    const nested1 = new NestedStack(stack1, 'Nested1');
    const stack2 = new Stack(app, 'Stack2');

    // WHEN
    nested1.addStackDependency(stack2);

    // THEN: assembly-level dependency between stack2 and stack1
    const assembly = app.synth();
    assertAssemblyDependency(assembly, stack2, []);
    assertAssemblyDependency(assembly, stack1, ['Stack2']);
    assertNoDependsOn(assembly, stack1);
    assertNoDependsOn(assembly, stack2);
    assertNoDependsOn(assembly, nested1);
  });
});

/**
 * Given a test function which sets the stage and verifies a dependency scenario
 * between two CloudFormation resources, returns two tests which exercise both
 * "construct dependency" (i.e. node.addDependency) and "resource dependency"
 * (i.e. resource.addDependency).
 *
 * @param testFunction The test function
 */
function matrixForResourceDependencyTest(testFunction: (addDep: (source: CfnResource, target: CfnResource) => void) => void) {
  return () => {
    test('construct dependency', () => {
      testFunction((source, target) => source.node.addDependency(target));
    });
    test('resource dependency', () => {
      testFunction((source, target) => source.addResourceDependency(target));
    });
  };
}

function assertAssemblyDependency(assembly: cxapi.CloudAssembly, stack: Stack, expectedDeps: string[]) {
  const stack1Art = assembly.getStackArtifact(stack.artifactId);
  const stack1Deps = stack1Art.dependencies.map(x => x.id);
  expect(stack1Deps).toEqual(expectedDeps);
}

function assertNoDependsOn(assembly: cxapi.CloudAssembly, stack: Stack) {
  let templateText;
  if (!(stack instanceof NestedStack)) {
    templateText = JSON.stringify(assembly.getStackArtifact(stack.artifactId).template);
  } else {
    templateText = fs.readFileSync(path.join(assembly.directory, stack.templateFile), 'utf-8');
  }

  // verify templates do not have any "DependsOn"
  expect(templateText.includes('DependsOn')).toBeFalsy();
}
