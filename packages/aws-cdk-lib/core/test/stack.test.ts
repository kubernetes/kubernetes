import * as fs from 'fs';
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import type { IConstruct } from 'constructs';
import { Construct, Node } from 'constructs';
import { flattenMeta, getWarnings, toCloudFormation } from './util';
import * as cxapi from '../../cx-api';
import { Fact, RegionInfo } from '../../region-info';
import type { ITaggableV2 } from '../lib';
import {
  App, CfnCondition, CfnInclude, CfnOutput, CfnParameter,
  CfnResource, CrossStackReferences, ReferenceStrength, Lazy, ScopedAws, Stack, validateString,
  Tags, LegacyStackSynthesizer, DefaultStackSynthesizer,
  NestedStack,
  Aws, Fn, ResolutionTypeHint,
  PermissionsBoundary,
  PERMISSIONS_BOUNDARY_CONTEXT_KEY,
  Aspects,
  Stage,
  TagManager,
  TagType,
  Validations,
} from '../lib';
import { Intrinsic } from '../lib/private/intrinsic';
import { resolveReferences } from '../lib/private/refs';
import { PostResolveToken } from '../lib/util';

describe('stack', () => {
  function makeCrossStackApp(context?: Record<string, unknown>): App {
    const app = new App({ context });
    Validations.of(app).acknowledge(
      { id: 'CloudFormation-Validate::F3002', reason: "For cross-stack tests, we don't care about property names being valid" },
      { id: 'CloudFormation-Validate::F3003', reason: "For cross-stack tests, we don't care about property names being valid" },
      { id: 'CloudFormation-Validate::E9004', reason: 'We are using non-existing property names' },
      { id: 'CloudFormation-Validate::F6101', reason: 'We are doing nonsensical type manipulations in these tests' },
    );
    return app;
  }

  test('a stack can be serialized into a CloudFormation template, initially it\'s empty', () => {
    const stack = new Stack();
    expect(toCloudFormation(stack)).toEqual({ });
  });

  test('stack name cannot exceed 128 characters', () => {
    // GIVEN
    const app = new App({});
    const reallyLongStackName = 'LookAtMyReallyLongStackNameThisStackNameIsLongerThan128CharactersThatIsNutsIDontThinkThereIsEnoughAWSAvailableToLetEveryoneHaveStackNamesThisLong';

    // THEN
    expect(() => {
      new Stack(app, 'MyStack', {
        stackName: reallyLongStackName,
      });
    }).toThrow(`Stack name must be <= 128 characters. Stack name: '${reallyLongStackName}'`);
  });

  test.each([
    ['Has:Colon', 'Has_Colon'],
    ['0startWithNumber', '0startWithNumber'],
    ['Has-Dash', 'Has-Dash'],
    [undefined, 'Default'],
    ['With_Underscore', 'With_Underscore'],
    ['with.dot', 'with.dot'],
    ['with/slash', 'with--slash'],
    ['with space', 'with_space'],
    ['UPPERCASE', 'UPPERCASE'],
    ['mixedCase123', 'mixedCase123'],
    ['!@#$%^', '______'],
    ['123456', '123456'],
    ['a-b-c', 'a-b-c'],
    ['x_y_z', 'x_y_z'],
    ['abc.def.ghi', 'abc.def.ghi'],
  ])('valid stack artifact id for construct id \'%s\'', (id, expected) => {
    // GIVEN
    const app = new App({});

    // WHEN
    const stack = new Stack(app, id, {
      stackName: 'ValidStackName',
    });

    // THEN
    expect(stack.stackName).toBe('ValidStackName');
    expect(stack.artifactId).toBe(expected);
  });

  test('stack objects have some template-level properties, such as Description, Version, Transform', () => {
    const stack = new Stack();
    stack.templateOptions.templateFormatVersion = 'MyTemplateVersion';
    stack.templateOptions.description = 'This is my description';
    stack.templateOptions.transforms = ['SAMy'];
    expect(toCloudFormation(stack)).toEqual({
      Description: 'This is my description',
      AWSTemplateFormatVersion: 'MyTemplateVersion',
      Transform: 'SAMy',
    });
  });

  test('Stack.isStack indicates that a construct is a stack', () => {
    const stack = new Stack();
    const c = new Construct(stack, 'Construct');
    expect(Stack.isStack(stack)).toBeDefined();
    expect(!Stack.isStack(c)).toBeDefined();
  });

  test('stack.id is not included in the logical identities of resources within it', () => {
    const stack = new Stack(undefined, 'MyStack');
    new CfnResource(stack, 'MyResource', { type: 'MyResourceType' });

    expect(toCloudFormation(stack)).toEqual({ Resources: { MyResource: { Type: 'MyResourceType' } } });
  });

  test('when stackResourceLimit is default, should give error', () => {
    // GIVEN
    const app = new App({});
    Validations.of(app).acknowledge({
      id: 'CloudFormation-Validate::F0007',
      reason: 'We have our own validator',
    });

    const stack = new Stack(app, 'MyStack');

    // WHEN
    for (let index = 0; index < 1000; index++) {
      new CfnResource(stack, `MyResource-${index}`, { type: 'MyResourceType' });
    }

    expect(() => {
      app.synth();
    }).toThrow('Number of resources in stack \'MyStack\': 1000 is greater than allowed maximum of 500');
  });

  test('when stackResourceLimit is defined, should give the proper error', () => {
    // GIVEN
    const app = new App({
      context: {
        '@aws-cdk/core:stackResourceLimit': 100,
      },
    });

    const stack = new Stack(app, 'MyStack');

    // WHEN
    for (let index = 0; index < 200; index++) {
      new CfnResource(stack, `MyResource-${index}`, { type: 'MyResourceType' });
    }

    expect(() => {
      app.synth();
    }).toThrow('Number of resources in stack \'MyStack\': 200 is greater than allowed maximum of 100');
  });

  test('when stackResourceLimit is 0, should not give error', () => {
    // GIVEN
    const app = makeCrossStackApp({ '@aws-cdk/core:stackResourceLimit': 0 });
    Validations.of(app).acknowledge({
      id: 'CloudFormation-Validate::F0007',
      reason: 'The point of this test is exercising stacks with more than 500 resources',
    });

    const stack = new Stack(app, 'MyStack');

    // WHEN
    for (let index = 0; index < 1000; index++) {
      new CfnResource(stack, `MyResource-${index}`, { type: 'MyResourceType' });
    }

    expect(() => {
      app.synth();
    }).not.toThrow();
  });

  test('stack.templateOptions can be used to set template-level options', () => {
    const stack = new Stack();

    stack.templateOptions.description = 'StackDescription';
    stack.templateOptions.templateFormatVersion = 'TemplateVersion';
    stack.templateOptions.transform = 'DeprecatedField';
    stack.templateOptions.transforms = ['Transform'];
    stack.templateOptions.metadata = {
      MetadataKey: 'MetadataValue',
    };

    expect(toCloudFormation(stack)).toEqual({
      Description: 'StackDescription',
      Transform: ['Transform', 'DeprecatedField'],
      AWSTemplateFormatVersion: 'TemplateVersion',
      Metadata: { MetadataKey: 'MetadataValue' },
    });
  });

  test('stack.templateOptions.transforms removes duplicate values', () => {
    const stack = new Stack();

    stack.templateOptions.transforms = ['A', 'B', 'C', 'A'];

    expect(toCloudFormation(stack)).toEqual({
      Transform: ['A', 'B', 'C'],
    });
  });

  test('stack.addTransform() adds a transform', () => {
    const stack = new Stack();

    stack.addTransform('A');
    stack.addTransform('B');
    stack.addTransform('C');

    expect(toCloudFormation(stack)).toEqual({
      Transform: ['A', 'B', 'C'],
    });
  });

  // This approach will only apply to TypeScript code, but at least it's a temporary
  // workaround for people running into issues caused by SDK-3003.
  // We should come up with a proper solution that involved jsii callbacks (when they exist)
  // so this can be implemented by jsii languages as well.
  test('Overriding `Stack._toCloudFormation` allows arbitrary post-processing of the generated template during synthesis', () => {
    const stack = new StackWithPostProcessor();

    new CfnResource(stack, 'myResource', {
      type: 'AWS::MyResource',
      properties: {
        MyProp1: 'hello',
        MyProp2: 'howdy',
        Environment: {
          Key: 'value',
        },
      },
    });

    expect(stack._toCloudFormation()).toEqual({
      Resources:
      {
        myResource:
         {
           Type: 'AWS::MyResource',
           Properties:
          {
            MyProp1: 'hello',
            MyProp2: 'howdy',
            Environment: { key: 'value' },
          },
         },
      },
    });
  });

  test('Stack.getByPath can be used to find any CloudFormation element (Parameter, Output, etc)', () => {
    const stack = new Stack();

    const p = new CfnParameter(stack, 'MyParam', { type: 'String' });
    const o = new CfnOutput(stack, 'MyOutput', { value: 'boom' });
    const c = new CfnCondition(stack, 'MyCondition');

    expect(stack.node.findChild(p.node.id)).toEqual(p);
    expect(stack.node.findChild(o.node.id)).toEqual(o);
    expect(stack.node.findChild(c.node.id)).toEqual(c);
  });

  test('Stack names can have hyphens in them', () => {
    const root = new App();

    new Stack(root, 'Hello-World');
    // Did not throw
  });

  test('Stacks can have a description given to them', () => {
    const stack = new Stack(new App(), 'MyStack', { description: 'My stack, hands off!' });
    const output = toCloudFormation(stack);
    expect(output.Description).toEqual('My stack, hands off!');
  });

  test('Stack descriptions have a limited length', () => {
    const desc = `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
     incididunt ut labore et dolore magna aliqua. Consequat interdum varius sit amet mattis vulputate
     enim nulla aliquet. At imperdiet dui accumsan sit amet nulla facilisi morbi. Eget lorem dolor sed
     viverra ipsum. Diam volutpat commodo sed egestas egestas. Sit amet porttitor eget dolor morbi non.
     Lorem dolor sed viverra ipsum. Id porta nibh venenatis cras sed felis. Augue interdum velit euismod
     in pellentesque. Suscipit adipiscing bibendum est ultricies integer quis. Condimentum id venenatis a
     condimentum vitae sapien pellentesque habitant morbi. Congue mauris rhoncus aenean vel elit scelerisque
     mauris pellentesque pulvinar.
     Faucibus purus in massa tempor nec. Risus viverra adipiscing at in. Integer feugiat scelerisque varius
     morbi. Malesuada nunc vel risus commodo viverra maecenas accumsan lacus. Vulputate sapien nec sagittis
     aliquam malesuada bibendum arcu vitae. Augue neque gravida in fermentum et sollicitudin ac orci phasellus.
     Ultrices tincidunt arcu non sodales neque sodales.`;
    expect(() => new Stack(new App(), 'MyStack', { description: desc }));
  });

  testDeprecated('Include should support non-hash top-level template elements like "Description"', () => {
    const stack = new Stack();

    const template = {
      Description: 'hello, world',
    };

    new CfnInclude(stack, 'Include', { template });

    const output = toCloudFormation(stack);

    expect(typeof output.Description).toEqual('string');
  });

  test('Pseudo values attached to one stack can be referenced in another stack', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const stack1 = new Stack(app, 'Stack1');
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another stack
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: account1 });

    // THEN
    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    expect(template1).toEqual({
      Outputs: {
        ExportsOutputRefAWSAccountIdAD568057: {
          Value: { Ref: 'AWS::AccountId' },
          Export: { Name: 'Stack1:ExportsOutputRefAWSAccountIdAD568057' },
        },
      },
    });

    expect(template2).toEqual({
      Parameters: {
        SomeParameter: {
          Type: 'String',
          Default: { 'Fn::ImportValue': 'Stack1:ExportsOutputRefAWSAccountIdAD568057' },
        },
      },
    });
  });

  test('Cross-stack references are detected in resource properties', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const resource1 = new CfnResource(stack1, 'Resource', { type: 'BLA' });
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another resource
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::Some::Resource',
      properties: {
        someProperty: new Intrinsic(resource1.ref),
      },
    });

    // THEN
    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    expect(template2?.Resources).toEqual({
      SomeResource: {
        Type: 'AWS::Some::Resource',
        Properties: {
          someProperty: { 'Fn::ImportValue': 'Stack1:ExportsOutputRefResource1D5D905A' },
        },
      },
    });
  });

  test('Cross-stack export names account for stack name lengths', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', {
      stackName: 'SoThisCouldPotentiallyBeAVeryLongStackName',
    });
    let scope: Construct = stack1;

    // WHEN - deeply nested
    for (let i = 0; i < 50; i++) {
      scope = new Construct(scope, `ChildConstruct${i}`);
    }

    const resource1 = new CfnResource(scope, 'Resource', { type: 'BLA' });
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another resource
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::Some::Resource',
      properties: {
        someProperty: new Intrinsic(resource1.ref),
      },
    });

    // THEN
    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;

    const theOutput = template1.Outputs[Object.keys(template1.Outputs)[0]];
    expect(theOutput.Export.Name.length).toEqual(255);
  });

  test('Cross-stack reference export names are relative to the stack (when the flag is set)', () => {
    // GIVEN
    const app = makeCrossStackApp({
      '@aws-cdk/core:stackRelativeExports': 'true',
    });
    const indifferentScope = new Construct(app, 'ExtraScope');

    const stack1 = new Stack(indifferentScope, 'Stack1', {
      stackName: 'Stack1',
    });
    const resource1 = new CfnResource(stack1, 'Resource', { type: 'BLA' });
    const stack2 = new Stack(indifferentScope, 'Stack2');

    // WHEN - used in another resource
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::Some::Resource',
      properties: {
        someProperty: new Intrinsic(resource1.ref),
      },
    });

    // THEN
    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    expect(template2?.Resources).toEqual({
      SomeResource: {
        Type: 'AWS::Some::Resource',
        Properties: {
          someProperty: { 'Fn::ImportValue': 'Stack1:ExportsOutputRefResource1D5D905A' },
        },
      },
    });
  });

  test('cross-stack references in lazy tokens work', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const stack1 = new Stack(app, 'Stack1');
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another stack
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: Lazy.string({ produce: () => account1 }) });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN
    expect(template1).toEqual({
      Outputs: {
        ExportsOutputRefAWSAccountIdAD568057: {
          Value: { Ref: 'AWS::AccountId' },
          Export: { Name: 'Stack1:ExportsOutputRefAWSAccountIdAD568057' },
        },
      },
    });

    expect(template2).toEqual({
      Parameters: {
        SomeParameter: {
          Type: 'String',
          Default: { 'Fn::ImportValue': 'Stack1:ExportsOutputRefAWSAccountIdAD568057' },
        },
      },
    });
  });

  test('Cross-stack use of Region and account returns nonscoped intrinsic because the two stacks must be in the same region anyway', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another stack
    new CfnOutput(stack2, 'DemOutput', { value: stack1.region });
    new CfnOutput(stack2, 'DemAccount', { value: stack1.account });

    // THEN
    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    expect(template2?.Outputs).toEqual({
      DemOutput: {
        Value: { Ref: 'AWS::Region' },
      },
      DemAccount: {
        Value: { Ref: 'AWS::AccountId' },
      },
    });
  });

  test('cross-stack references in strings work', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const stack1 = new Stack(app, 'Stack1');
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another stack
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: `TheAccountIs${account1}` });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN
    expect(template2).toEqual({
      Parameters: {
        SomeParameter: {
          Type: 'String',
          Default: { 'Fn::Join': ['', ['TheAccountIs', { 'Fn::ImportValue': 'Stack1:ExportsOutputRefAWSAccountIdAD568057' }]] },
        },
      },
    });
  });

  test('cross-stack references of lists returned from Fn::GetAtt work', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const exportResource = new CfnResource(stack1, 'exportedResource', {
      type: 'BLA',
    });
    const stack2 = new Stack(app, 'Stack2');
    // L1s represent attribute names with `attr${attributeName}`
    (exportResource as any).attrList = ['magic-attr-value'];

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'BLA',
      properties: {
        Prop: exportResource.getAtt('List', ResolutionTypeHint.STRING_LIST),
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN
    expect(template1).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttexportedResourceList0EA3E0D9: {
          Value: {
            'Fn::Join': [
              '||', {
                'Fn::GetAtt': [
                  'exportedResource',
                  'List',
                ],
              },
            ],
          },
          Export: { Name: 'Stack1:ExportsOutputFnGetAttexportedResourceList0EA3E0D9' },
        },
      },
    });

    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'BLA',
          Properties: {
            Prop: {
              'Fn::Split': [
                '||',
                {
                  'Fn::ImportValue': 'Stack1:ExportsOutputFnGetAttexportedResourceList0EA3E0D9',
                },
              ],
            },
          },
        },
      },
    });
  });

  test('cross-stack references of nested stack lists returned from Fn::GetAtt work', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const producer = new Stack(app, 'Producer');
    const nested = new NestedStack(producer, 'Nestor');
    const exportResource = new CfnResource(nested, 'exportedResource', {
      type: 'BLA',
    });
    const consumer = new Stack(app, 'Consumer');
    // L1s represent attribute names with `attr${attributeName}`
    (exportResource as any).attrList = ['magic-attr-value'];

    // WHEN - used in another stack
    new CfnResource(consumer, 'SomeResource', {
      type: 'BLA',
      properties: {
        Prop: exportResource.getAtt('List', ResolutionTypeHint.STRING_LIST),
      },
    });

    const assembly = app.synth();
    const producerTemplate = assembly.getStackByName(producer.stackName).template;
    const consumerTemplate = assembly.getStackByName(consumer.stackName).template;

    // THEN
    expect(producerTemplate).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttNestorNestedStackNestorNestedStackResourceAE182597OutputsProducerNestorexportedResource5B6DDAA1ListAF9B4148: {
          Value: {
            'Fn::GetAtt': [
              'NestorNestedStackNestorNestedStackResourceAE182597',
              'Outputs.ProducerNestorexportedResource5B6DDAA1List',
            ],
          },
          Export: { Name: 'Producer:ExportsOutputFnGetAttNestorNestedStackNestorNestedStackResourceAE182597OutputsProducerNestorexportedResource5B6DDAA1ListAF9B4148' },
        },
      },
    });

    expect(consumerTemplate).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'BLA',
          Properties: {
            Prop: {
              'Fn::Split': [
                '||',
                {
                  'Fn::ImportValue': 'Producer:ExportsOutputFnGetAttNestorNestedStackNestorNestedStackResourceAE182597OutputsProducerNestorexportedResource5B6DDAA1ListAF9B4148',
                },
              ],
            },
          },
        },
      },
    });
  });

  test('cross-stack references of lists returned from Fn::GetAtt can be used with CFN intrinsics', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const exportResource = new CfnResource(stack1, 'exportedResource', {
      type: 'BLA',
    });
    const stack2 = new Stack(app, 'Stack2');
    // L1s represent attribute names with `attr${attributeName}`
    (exportResource as any).attrList = ['magic-attr-value'];

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'BLA',
      properties: {
        Prop: Fn.select(3, exportResource.getAtt('List', ResolutionTypeHint.STRING_LIST) as any),
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN
    expect(template1).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttexportedResourceList0EA3E0D9: {
          Value: {
            'Fn::Join': [
              '||', {
                'Fn::GetAtt': [
                  'exportedResource',
                  'List',
                ],
              },
            ],
          },
          Export: { Name: 'Stack1:ExportsOutputFnGetAttexportedResourceList0EA3E0D9' },
        },
      },
    });

    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'BLA',
          Properties: {
            Prop: {
              'Fn::Select': [
                3,
                {
                  'Fn::Split': [
                    '||',
                    {
                      'Fn::ImportValue': 'Stack1:ExportsOutputFnGetAttexportedResourceList0EA3E0D9',
                    },
                  ],
                },
              ],
            },
          },
        },
      },
    });
  });

  test('cross-stack references of lists returned from Fn::Ref work', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const param = new CfnParameter(stack1, 'magicParameter', {
      default: 'BLAT,BLAH',
      type: 'List<String>',
    });
    const stack2 = new Stack(app, 'Stack2');

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'BLA',
      properties: {
        Prop: param.value,
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN
    expect(template1).toMatchObject({
      Outputs: {
        ExportsOutputRefmagicParameter4CC6F7BE: {
          Value: {
            'Fn::Join': [
              '||', {
                Ref: 'magicParameter',
              },
            ],
          },
          Export: { Name: 'Stack1:ExportsOutputRefmagicParameter4CC6F7BE' },
        },
      },
    });

    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'BLA',
          Properties: {
            Prop: {
              'Fn::Split': [
                '||',
                {
                  'Fn::ImportValue': 'Stack1:ExportsOutputRefmagicParameter4CC6F7BE',
                },
              ],
            },
          },
        },
      },
    });
  });

  test('cross-region stack references, crossRegionReferences=true', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
      },
    });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;
    const template1 = assembly.getStackByName(stack1.stackName).template;

    // THEN
    expect(template1).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
      Outputs: {
        PublishOutputFnGetAttSomeResourceExportnameC1AF3C83: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'name',
            ],
          },
        },
      },
    });

    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'AWS::S3::Bucket',
          Properties: {
            Name: {
              'Fn::GetStackOutput': {
                StackName: 'Stack1',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportnameC1AF3C83',
              },
            },
          },
        },
      },
    });
  });

  test('cross-region stack references work without crossRegionReferences prop', () => {
    // GIVEN - no crossRegionReferences on consumer, strong flag (default)
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' } });

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
      },
    });

    // THEN - does not throw, uses ExportWriter/ExportReader
    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const resources1 = template1.Resources ?? {};
    const customResources1 = Object.values(resources1).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportWriter',
    );
    expect(customResources1.length).toBeGreaterThan(0);
  });

  test('cross region stack references with multiple stacks, crossRegionReferences=true', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack3 = new Stack(app, 'Stack3', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource3 = new CfnResource(stack3, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
        Other: exportResource.getAtt('other'),
        Other2: exportResource3.getAtt('other2'),
      },
    });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;
    const template3 = assembly.getStackByName(stack3.stackName).template;
    const template1 = assembly.getStackByName(stack1.stackName).template;

    // THEN
    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'AWS::S3::Bucket',
          Properties: {
            Name: {
              'Fn::GetStackOutput': {
                StackName: 'Stack1',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportnameC1AF3C83',
              },
            },
            Other: {
              'Fn::GetStackOutput': {
                StackName: 'Stack1',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportother3693C96F',
              },
            },
            Other2: {
              'Fn::GetStackOutput': {
                StackName: 'Stack3',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportother215E978B3',
              },
            },
          },
        },
      },
    });
    expect(template3).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
      Outputs: {
        PublishOutputFnGetAttSomeResourceExportother215E978B3: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'other2',
            ],
          },
        },
      },
    });
    expect(template1).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
      Outputs: {
        PublishOutputFnGetAttSomeResourceExportnameC1AF3C83: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'name',
            ],
          },
        },
        PublishOutputFnGetAttSomeResourceExportother3693C96F: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'other',
            ],
          },
        },
      },
    });
  });

  test('cross region stack references with multiple stacks and multiple regions, crossRegionReferences=true', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack3 = new Stack(app, 'Stack3', { env: { region: 'us-west-1' }, crossRegionReferences: true });
    const exportResource3 = new CfnResource(stack3, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN - used in another stack
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
        Other: exportResource.getAtt('other'),
        Other2: exportResource3.getAtt('other2'),
      },
    });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;
    const template3 = assembly.getStackByName(stack3.stackName).template;
    const template1 = assembly.getStackByName(stack1.stackName).template;

    // THEN
    expect(template3).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
    });
    expect(template2).toMatchObject({
      Resources: {
        SomeResource: {
          Type: 'AWS::S3::Bucket',
          Properties: {
            Name: {
              'Fn::GetStackOutput': {
                StackName: 'Stack1',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportnameC1AF3C83',
              },
            },
            Other: {
              'Fn::GetStackOutput': {
                StackName: 'Stack1',
                Region: 'us-east-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportother3693C96F',
              },
            },
            Other2: {
              'Fn::GetStackOutput': {
                StackName: 'Stack3',
                Region: 'us-west-1',
                OutputName: 'PublishOutputFnGetAttSomeResourceExportother215E978B3',
              },
            },
          },
        },
      },
    });
    expect(template3).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
      Outputs: {
        PublishOutputFnGetAttSomeResourceExportother215E978B3: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'other2',
            ],
          },
        },
      },
    });
    expect(template1).toMatchObject({
      Resources: {
        SomeResourceExport: {
          Type: 'AWS::S3::Bucket',
        },
      },
      Outputs: {
        PublishOutputFnGetAttSomeResourceExportnameC1AF3C83: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'name',
            ],
          },
        },
        PublishOutputFnGetAttSomeResourceExportother3693C96F: {
          Value: {
            'Fn::GetAtt': [
              'SomeResourceExport',
              'other',
            ],
          },
        },
      },
    });
  });

  test('cross-region references default to strong (ExportWriter/ExportReader) when flag is unset', () => {
    // GIVEN - no crossStackReferenceStrength context set
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has ExportWriter custom resource
    const resources1 = template1.Resources ?? {};
    const customResources1 = Object.values(resources1).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportWriter',
    );
    expect(customResources1.length).toBeGreaterThan(0);

    // THEN - consumer has ExportReader custom resource
    const resources2 = template2.Resources ?? {};
    const customResources2 = Object.values(resources2).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportReader',
    );
    expect(customResources2.length).toBeGreaterThan(0);
  });

  test('emits a single warning per app when cross-stack reference flag is unset', () => {
    // GIVEN - no context flag set, multiple consumer stacks
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const resource1 = new CfnResource(stack1, 'Resource1', { type: 'AWS::S3::Bucket' });
    const resource2 = new CfnResource(stack1, 'Resource2', { type: 'AWS::SNS::Topic' });
    const stack2 = new Stack(app, 'Stack2');
    const stack3 = new Stack(app, 'Stack3');

    // WHEN - cross-stack references from multiple consumers
    new CfnResource(stack2, 'Consumer1', {
      type: 'AWS::S3::Bucket',
      properties: { Prop1: resource1.getAtt('Arn') },
    });
    new CfnResource(stack3, 'Consumer2', {
      type: 'AWS::S3::Bucket',
      properties: { Prop2: resource2.getAtt('Arn') },
    });

    const assembly = app.synth();
    const warnings = getWarnings(assembly);

    // THEN - only one warning in the entire app
    const relevantWarnings = warnings.filter(w =>
      w.message.includes('@aws-cdk/core:crossStackReferencesDefaultStrong'),
    );
    expect(relevantWarnings).toHaveLength(1);
  });

  test('no warning when cross-stack reference flag is explicitly set', () => {
    // GIVEN - context flag explicitly set to 'strong'
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'strong' });
    const stack1 = new Stack(app, 'Stack1');
    const resource1 = new CfnResource(stack1, 'Resource1', { type: 'AWS::S3::Bucket' });
    const stack2 = new Stack(app, 'Stack2');

    // WHEN
    new CfnResource(stack2, 'Consumer1', {
      type: 'AWS::S3::Bucket',
      properties: { Prop1: resource1.getAtt('Arn') },
    });

    const assembly = app.synth();
    const warnings = getWarnings(assembly);

    // THEN - no warning because the flag is explicitly set
    const relevantWarnings = warnings.filter(w =>
      w.message.includes('@aws-cdk/core:crossStackReferencesDefaultStrong'),
    );
    expect(relevantWarnings).toHaveLength(0);
  });

  test('emits transitional warning when cross-stack reference flag is set to both', () => {
    // GIVEN - context flag set to 'both'
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'both' });
    const stack1 = new Stack(app, 'Stack1');
    const resource1 = new CfnResource(stack1, 'Resource1', { type: 'AWS::S3::Bucket' });
    const stack2 = new Stack(app, 'Stack2');

    // WHEN
    new CfnResource(stack2, 'Consumer1', {
      type: 'AWS::S3::Bucket',
      properties: { Prop1: resource1.getAtt('Arn') },
    });

    const assembly = app.synth();
    const warnings = getWarnings(assembly);

    // THEN - transitional warning telling user to move to 'weak'
    const relevantWarnings = warnings.filter(w =>
      w.message.includes('@aws-cdk/core:crossStackReferencesBothTransitional'),
    );
    expect(relevantWarnings).toHaveLength(1);
    expect(relevantWarnings[0].path).toContain('Stack2');
  });

  test.each(['up', 'down'])('no cross-stack reference flag warning when referencing %s across nested stacks', (direction) => {
    // GIVEN - context flag explicitly set to 'strong'
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const stack2 = new NestedStack(stack1, 'Stack2');

    // WHEN
    if (direction === 'up') {
      const resource1 = new CfnResource(stack1, 'Resource1', { type: 'AWS::S3::Bucket' });
      new CfnResource(stack2, 'Resource2', {
        type: 'AWS::S3::Bucket',
        properties: { Prop1: resource1.getAtt('Arn') },
      });
    } else {
      const resource2 = new CfnResource(stack2, 'Resource2', { type: 'AWS::S3::Bucket' });
      new CfnResource(stack1, 'Resource1', {
        type: 'AWS::S3::Bucket',
        properties: { Prop1: resource2.getAtt('Arn') },
      });
    }

    const assembly = app.synth();
    const warnings = getWarnings(assembly);

    // THEN - no warning because nested stacks are not cross-stack references
    const relevantWarnings = warnings.filter(w =>
      w.message.includes('@aws-cdk/core:crossStackReferencesDefaultStrong'),
    );
    expect(relevantWarnings).toHaveLength(0);
  });

  test('cross-region strong references use ExportWriter/ExportReader', () => {
    // GIVEN - strength is explicitly 'strong'
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'strong' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has ExportWriter custom resource
    const resources1 = template1.Resources ?? {};
    const customResources1 = Object.values(resources1).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportWriter',
    );
    expect(customResources1.length).toBeGreaterThan(0);

    // THEN - consumer has ExportReader custom resource
    const resources2 = template2.Resources ?? {};
    const customResources2 = Object.values(resources2).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportReader',
    );
    expect(customResources2.length).toBeGreaterThan(0);
  });

  test('cross-region both references generate ExportWriter AND Fn::GetStackOutput', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'both' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
      },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has ExportWriter (strong side)
    const resources1 = template1.Resources ?? {};
    const customResources1 = Object.values(resources1).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportWriter',
    );
    expect(customResources1.length).toBeGreaterThan(0);

    // THEN - producer has Output (for Fn::GetStackOutput)
    expect(template1.Outputs).toBeDefined();

    // THEN - consumer uses Fn::GetStackOutput (weak side), NOT ExportReader
    const resources2 = template2.Resources ?? {};
    const consumerResource = resources2.SomeResource;
    expect(consumerResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');

    const customResources2 = Object.values(resources2).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportReader',
    );
    expect(customResources2).toHaveLength(0);
  });

  test('cross-region weak references serialize STRING_LIST with Fn::Join and deserialize with Fn::Split', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' }, crossRegionReferences: true });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    (exportResource as any).attrDnsEntries = ['entry1', 'entry2'];
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' }, crossRegionReferences: true });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('DnsEntries', ResolutionTypeHint.STRING_LIST) },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer output wraps list with Fn::Join
    const outputKeys = Object.keys(template1.Outputs ?? {});
    expect(outputKeys.length).toBeGreaterThan(0);
    const output = template1.Outputs[outputKeys[0]];
    expect(output.Value).toEqual({
      'Fn::Join': ['||', { 'Fn::GetAtt': ['SomeResourceExport', 'DnsEntries'] }],
    });

    // THEN - consumer uses Fn::Split around Fn::GetStackOutput
    expect(template2.Resources.SomeResource.Properties.Name).toEqual({
      'Fn::Split': ['||', { 'Fn::GetStackOutput': expect.objectContaining({ OutputName: outputKeys[0] }) }],
    });
  });

  test('cross-account strong references emit warning and use Fn::GetStackOutput', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'strong' });
    const producer = new Stack(app, 'Stack1', {
      env: { region: 'us-east-1', account: '111111111111' },
    });
    const exportResource = new CfnResource(producer, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const consumer = new Stack(app, 'Stack2', {
      env: { region: 'us-east-2', account: '222222222222' },
    });

    // WHEN
    new CfnResource(consumer, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const consumerTemplate = assembly.getStackByName(consumer.stackName).template;

    // THEN - still uses Fn::GetStackOutput (falls through)
    expect(consumerTemplate.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');

    // THEN - warning emitted
    const warnings = consumer.node.metadata.filter(m => m.type === 'aws:cdk:warning');
    expect(warnings.some(w => w.data.includes('cross-account references can only be weak'))).toBe(true);
  });

  test('cross-account references from multiple consumers share a single role with multiple principals', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const producer = new Stack(app, 'Producer', {
      env: { region: 'us-east-1', account: '111111111111' },
    });
    const exportResource = new CfnResource(producer, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const consumer1 = new Stack(app, 'Consumer1', {
      env: { region: 'us-east-1', account: '222222222222' },
    });
    const consumer2 = new Stack(app, 'Consumer2', {
      env: { region: 'us-east-1', account: '333333333333' },
    });

    // WHEN - two different consumers reference the same producer
    new CfnResource(consumer1, 'Resource1', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });
    new CfnResource(consumer2, 'Resource2', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('other') },
    });

    const assembly = app.synth();
    const producerTemplate = assembly.getStackByName(producer.stackName).template;

    // THEN - producer has exactly one IAM::Role
    const roles = Object.values(producerTemplate.Resources).filter(
      (r: any) => r.Type === 'AWS::IAM::Role',
    );
    expect(roles).toHaveLength(1);

    // THEN - the single role has both consumer principals
    const role = roles[0] as any;
    const principals = role.Properties.AssumeRolePolicyDocument.Statement[0].Principal.AWS;
    expect(principals).toHaveLength(2);
    expect(principals).toContainEqual({ 'Fn::Sub': consumer1.synthesizer.cloudFormationExecutionRole });
    expect(principals).toContainEqual({ 'Fn::Sub': consumer2.synthesizer.cloudFormationExecutionRole });

    // THEN - producer has exactly one IAM::Policy
    const policies = Object.values(producerTemplate.Resources).filter(
      (r: any) => r.Type === 'AWS::IAM::Policy',
    );
    expect(policies).toHaveLength(1);
  });

  test('cross-account references from a single consumer produce a single principal string', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const producer = new Stack(app, 'Producer', {
      env: { region: 'us-east-1', account: '111111111111' },
    });
    const exportResource = new CfnResource(producer, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const consumer = new Stack(app, 'Consumer', {
      env: { region: 'us-east-1', account: '222222222222' },
    });

    // WHEN - single consumer references multiple attributes
    new CfnResource(consumer, 'Resource1', {
      type: 'AWS::S3::Bucket',
      properties: {
        Name: exportResource.getAtt('name'),
        Other: exportResource.getAtt('other'),
      },
    });

    const assembly = app.synth();
    const producerTemplate = assembly.getStackByName(producer.stackName).template;

    // THEN - producer has exactly one IAM::Role
    const roles = Object.values(producerTemplate.Resources).filter(
      (r: any) => r.Type === 'AWS::IAM::Role',
    );
    expect(roles).toHaveLength(1);

    // THEN - single principal is a string, not an array
    const role = roles[0] as any;
    const principal = role.Properties.AssumeRolePolicyDocument.Statement[0].Principal.AWS;
    expect(principal).toEqual({ 'Fn::Sub': consumer.synthesizer.cloudFormationExecutionRole });
  });

  test('same-region weak references use Fn::GetStackOutput', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has an Output without Export
    expect(template1.Outputs).toBeDefined();
    const outputKeys = Object.keys(template1.Outputs);
    expect(outputKeys.length).toBeGreaterThan(0);
    const output = template1.Outputs[outputKeys[0]];
    expect(output.Export).toBeUndefined();

    // THEN - consumer uses Fn::GetStackOutput
    expect(template2.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');
  });

  test('same-region both references generate Export AND Fn::GetStackOutput', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'both' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has an Output WITH Export (strong side)
    const outputsWithExport = Object.values(template1.Outputs ?? {}).filter(
      (o: any) => o.Export !== undefined,
    );
    expect(outputsWithExport.length).toBeGreaterThan(0);

    // THEN - producer also has an Output WITHOUT Export (for Fn::GetStackOutput)
    const outputsWithoutExport = Object.values(template1.Outputs ?? {}).filter(
      (o: any) => o.Export === undefined,
    );
    expect(outputsWithoutExport.length).toBeGreaterThan(0);

    // THEN - consumer uses Fn::GetStackOutput (weak side), NOT Fn::ImportValue
    expect(template2.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');
    expect(template2.Resources.SomeResource.Properties.Name).not.toHaveProperty('Fn::ImportValue');
  });

  test('same-region weak references serialize STRING_LIST with Fn::Join and deserialize with Fn::Split', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    (exportResource as any).attrDnsEntries = ['entry1', 'entry2'];
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('DnsEntries', ResolutionTypeHint.STRING_LIST) },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer output wraps list with Fn::Join
    const outputKeys = Object.keys(template1.Outputs);
    const output = template1.Outputs[outputKeys[0]];
    expect(output.Value).toEqual({
      'Fn::Join': ['||', { 'Fn::GetAtt': ['SomeResourceExport', 'DnsEntries'] }],
    });
    expect(output.Export).toBeUndefined();

    // THEN - consumer uses Fn::Split around Fn::GetStackOutput
    expect(template2.Resources.SomeResource.Properties.Name).toEqual({
      'Fn::Split': ['||', { 'Fn::GetStackOutput': expect.objectContaining({ OutputName: outputKeys[0] }) }],
    });
  });

  test('same-region both references serialize STRING_LIST with Fn::Join for publish output', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'both' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    (exportResource as any).attrDnsEntries = ['entry1', 'entry2'];
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('DnsEntries', ResolutionTypeHint.STRING_LIST) },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has a publish output (no Export) with Fn::Join
    const publishOutputs = Object.entries(template1.Outputs ?? {}).filter(
      ([_, o]: [string, any]) => o.Export === undefined,
    );
    expect(publishOutputs.length).toBeGreaterThan(0);
    const [publishKey, publishOutput] = publishOutputs[0] as [string, any];
    expect(publishOutput.Value).toEqual({
      'Fn::Join': ['||', { 'Fn::GetAtt': ['SomeResourceExport', 'DnsEntries'] }],
    });

    // THEN - producer also has a strong export output with Fn::Join
    const exportOutputs = Object.entries(template1.Outputs ?? {}).filter(
      ([_, o]: [string, any]) => o.Export !== undefined,
    );
    expect(exportOutputs.length).toBeGreaterThan(0);
    const exportOutput = exportOutputs[0][1] as any;
    expect(exportOutput.Value).toEqual({
      'Fn::Join': ['||', { 'Fn::GetAtt': ['SomeResourceExport', 'DnsEntries'] }],
    });

    // THEN - consumer uses Fn::Split around Fn::GetStackOutput (weak path)
    expect(template2.Resources.SomeResource.Properties.Name).toEqual({
      'Fn::Split': ['||', { 'Fn::GetStackOutput': expect.objectContaining({ OutputName: publishKey }) }],
    });
  });

  test('invalid cross stack reference strength throws', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'invalid' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    // THEN
    expect(() => app.synth()).toThrow(/Must be "strong", "weak", or "both"/);
  });

  test('per-resource weak override uses Fn::GetStackOutput when global is strong (same region)', () => {
    // GIVEN - global default is strong (no context set)
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    CrossStackReferences.of(exportResource).produce(ReferenceStrength.WEAK);

    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has an Output without Export
    expect(template1.Outputs).toBeDefined();
    const outputKeys = Object.keys(template1.Outputs);
    expect(outputKeys.length).toBeGreaterThan(0);
    const output = template1.Outputs[outputKeys[0]];
    expect(output.Export).toBeUndefined();

    // THEN - consumer uses Fn::GetStackOutput
    expect(template2.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');
  });

  test('per-resource strong override uses Fn::ImportValue when global is weak (same region)', () => {
    // GIVEN - global default is weak
    const app = makeCrossStackApp({ [cxapi.DEFAULT_CROSS_STACK_REFERENCES]: 'weak' });
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    CrossStackReferences.of(exportResource).produce(ReferenceStrength.STRONG);

    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - consumer uses Fn::ImportValue (strong)
    expect(template2.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::ImportValue');
  });

  test('per-resource override only affects the overridden resource', () => {
    // GIVEN - global default is strong
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const weakResource = new CfnResource(stack1, 'WeakResource', {
      type: 'AWS::S3::Bucket',
    });
    CrossStackReferences.of(weakResource).produce(ReferenceStrength.WEAK);

    const strongResource = new CfnResource(stack1, 'StrongResource', {
      type: 'AWS::SNS::Topic',
    });

    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::Lambda::Function',
      properties: {
        WeakRef: weakResource.getAtt('name'),
        StrongRef: strongResource.getAtt('arn'),
      },
    });

    const assembly = app.synth();
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - weak-overridden resource uses Fn::GetStackOutput
    expect(template2.Resources.SomeResource.Properties.WeakRef).toHaveProperty('Fn::GetStackOutput');
    // THEN - non-overridden resource uses Fn::ImportValue (global strong)
    expect(template2.Resources.SomeResource.Properties.StrongRef).toHaveProperty('Fn::ImportValue');
  });

  test('per-resource weak override works for cross-region references', () => {
    // GIVEN - global default is strong
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1' } });
    const exportResource = new CfnResource(stack1, 'SomeResourceExport', {
      type: 'AWS::S3::Bucket',
    });
    CrossStackReferences.of(exportResource).produce(ReferenceStrength.WEAK);

    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-2' } });

    // WHEN
    new CfnResource(stack2, 'SomeResource', {
      type: 'AWS::S3::Bucket',
      properties: { Name: exportResource.getAtt('name') },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer does NOT have ExportWriter
    const resources1 = template1.Resources ?? {};
    const exportWriters = Object.values(resources1).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportWriter',
    );
    expect(exportWriters).toHaveLength(0);

    // THEN - consumer uses Fn::GetStackOutput, no ExportReader
    expect(template2.Resources.SomeResource.Properties.Name).toHaveProperty('Fn::GetStackOutput');
    const resources2 = template2.Resources ?? {};
    const exportReaders = Object.values(resources2).filter(
      (r: any) => r.Type === 'Custom::CrossRegionExportReader',
    );
    expect(exportReaders).toHaveLength(0);
  });

  test('applyCrossStackReferenceStrength on CfnResource stores and exposes value', () => {
    const stack = new Stack();
    const resource = new CfnResource(stack, 'MyResource', { type: 'AWS::S3::Bucket' });

    // WHEN
    resource.applyCrossStackReferenceStrength(ReferenceStrength.WEAK);

    // THEN
    expect(resource._crossStackReferenceStrengthOverride).toBe(ReferenceStrength.WEAK);
  });

  test('CrossStackReferences.of().produce() delegates to applyCrossStackReferenceStrength', () => {
    const stack = new Stack();
    const resource = new CfnResource(stack, 'MyResource', { type: 'AWS::S3::Bucket' });

    // WHEN
    CrossStackReferences.of(resource).produce(ReferenceStrength.WEAK);

    // THEN
    expect(resource._crossStackReferenceStrengthOverride).toBe(ReferenceStrength.WEAK);
  });

  test('CrossStackReferences.of().consume() sets context on the scope', () => {
    const app = makeCrossStackApp();
    const stack = new Stack(app, 'Stack1');

    // WHEN
    CrossStackReferences.of(stack).consume(ReferenceStrength.WEAK);

    // THEN
    expect(stack.node.tryGetContext(cxapi.DEFAULT_CROSS_STACK_REFERENCES)).toBe('weak');
  });

  test('consumeReference weakens a single reference usage (same region)', () => {
    // GIVEN - global default is strong
    const app = makeCrossStackApp();
    const producerStack = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const resource = new CfnResource(producerStack, 'MyResource', { type: 'AWS::S3::Bucket' });

    const consumerStack = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN - one reference uses consumeReference, another uses the raw token
    new CfnResource(consumerStack, 'WeakConsumer', {
      type: 'AWS::Lambda::Function',
      properties: { Description: Stack.consumeReference(resource.getAtt('Arn').toString()) },
    });
    new CfnResource(consumerStack, 'StrongConsumer', {
      type: 'AWS::Lambda::Function',
      properties: { Description: resource.getAtt('Arn') },
    });

    const assembly = app.synth();
    const producerTemplate = assembly.getStackByName(producerStack.stackName).template;
    const consumerTemplate = assembly.getStackByName(consumerStack.stackName).template;

    // THEN - producer has an Output with Export (for the strong reference)
    expect(producerTemplate.Outputs.ExportsOutputFnGetAttMyResourceArnE157F485).toEqual({
      Value: { 'Fn::GetAtt': ['MyResource', 'Arn'] },
      Export: { Name: 'Stack1:ExportsOutputFnGetAttMyResourceArnE157F485' },
    });
    // and an Output without Export (for the BOTH/weak side)
    expect(producerTemplate.Outputs.PublishOutputFnGetAttMyResourceArn8F315E6B).toEqual({
      Value: { 'Fn::GetAtt': ['MyResource', 'Arn'] },
    });

    // THEN - weak consumer uses Fn::GetStackOutput
    expect(consumerTemplate.Resources.WeakConsumer.Properties.Description).toHaveProperty('Fn::GetStackOutput');
    // THEN - strong consumer uses Fn::ImportValue
    expect(consumerTemplate.Resources.StrongConsumer.Properties.Description).toHaveProperty('Fn::ImportValue');
  });

  test('consumeReference with explicit WEAK strength (same region)', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { region: 'us-east-1', account: '111111111111' } });
    const resource = new CfnResource(stack1, 'MyResource', { type: 'AWS::S3::Bucket' });

    const stack2 = new Stack(app, 'Stack2', { env: { region: 'us-east-1', account: '111111111111' } });

    // WHEN
    new CfnResource(stack2, 'WeakConsumer', {
      type: 'AWS::Lambda::Function',
      properties: { Description: Stack.consumeReference(resource.getAtt('Arn').toString(), ReferenceStrength.WEAK) },
    });

    const assembly = app.synth();
    const template1 = assembly.getStackByName(stack1.stackName).template;
    const template2 = assembly.getStackByName(stack2.stackName).template;

    // THEN - producer has Output WITHOUT Export
    const outputs = template1.Outputs ?? {};
    const outputWithExport = Object.values(outputs).find((o: any) => o.Export);
    expect(outputWithExport).toBeUndefined();

    // THEN - consumer uses Fn::GetStackOutput
    expect(template2.Resources.WeakConsumer.Properties.Description).toHaveProperty('Fn::GetStackOutput');
  });

  test('consumeReference throws for non-reference values', () => {
    expect(() => Stack.consumeReference('just-a-string')).toThrow(/consumeReference: the value must be a resource attribute reference/);
  });

  test('cross stack references and dependencies work within child stacks (non-nested)', () => {
    // GIVEN
    const app = makeCrossStackApp({
      '@aws-cdk/core:stackRelativeExports': true,
      [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false,
    });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const child2 = new Stack(parent, 'Child2');
    const resourceA = new CfnResource(child1, 'ResourceA', { type: 'RA' });
    const resourceB = new CfnResource(child1, 'ResourceB', { type: 'RB' });

    // WHEN
    const resource2 = new CfnResource(child2, 'Resource1', {
      type: 'R2',
      properties: {
        RefToResource1: resourceA.ref,
      },
    });
    resource2.addResourceDependency(resourceB);

    // THEN
    const assembly = app.synth();
    const parentTemplate = assembly.getStackArtifact(parent.artifactId).template;
    const child1Template = assembly.getStackArtifact(child1.artifactId).template;
    const child2Template = assembly.getStackArtifact(child2.artifactId).template;

    expect(parentTemplate).toEqual({});
    expect(child1Template).toEqual({
      Resources: {
        ResourceA: { Type: 'RA' },
        ResourceB: { Type: 'RB' },
      },
      Outputs: {
        ExportsOutputRefResourceA461B4EF9: {
          Value: { Ref: 'ResourceA' },
          Export: { Name: 'ParentChild18FAEF419:ExportsOutputRefResourceA461B4EF9' },
        },
      },
    });
    expect(child2Template).toEqual({
      Resources: {
        Resource1: {
          Type: 'R2',
          Properties: {
            RefToResource1: { 'Fn::ImportValue': 'ParentChild18FAEF419:ExportsOutputRefResourceA461B4EF9' },
          },
        },
      },
    });

    expect(assembly.getStackArtifact(child1.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
    expect(assembly.getStackArtifact(child2.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual(['ParentChild18FAEF419']);
  });

  test('cross-stack dependencies can be queried with _stackDependenciesCausedBy', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const childA = new Stack(parent, 'ChildA');
    const resource1 = new CfnResource(child1, 'Resource1', { type: 'R1' });
    const resource2 = new CfnResource(child1, 'Resource2', { type: 'R2' });
    const resourceA = new CfnResource(childA, 'ResourceA', { type: 'RA' });

    resourceA.addResourceDependency(resource1);
    resourceA.addResourceDependency(resource2);

    expect(paths(childA._stackDependenciesCausedBy(resourceA)))
      .toEqual(paths([resource1, resource2]));

    const assembly = app.synth();

    expect(assembly.getStackArtifact(child1.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
    expect(assembly.getStackArtifact(childA.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual(['ParentChild18FAEF419']);
  });

  test('addStackDependency adds one StackDependencyReason with defaults', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const childA = new Stack(parent, 'ChildA');

    childA.addStackDependency(child1, 'reason');

    expect(paths(childA._stackDependenciesCausedBy(childA)))
      .toEqual(paths([child1]));

    const assembly = app.synth();

    expect(assembly.getStackArtifact(child1.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
    expect(assembly.getStackArtifact(childA.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual(['ParentChild18FAEF419']);
  });

  test('addStackDependency raises error on cycle', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const child2 = new Stack(parent, 'Child2');

    child2.addStackDependency(child1);
    expect(() => child1.addStackDependency(child2)).toThrow("'Parent/Child2' depends on");
  });

  test('addStackDependency handles duplicate dependency reasons', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const child2 = new Stack(parent, 'Child2');

    child2.addStackDependency(child1);
    const depsBefore = child2._stackDependenciesCausedBy(child2);
    child2.addStackDependency(child1);
    expect(depsBefore).toEqual(child2._stackDependenciesCausedBy(child2));
  });

  test('_removeAssemblyDependency removes one StackDependencyReason of two from _stackDependencies', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const childA = new Stack(parent, 'ChildA');
    const resource1 = new CfnResource(child1, 'Resource1', { type: 'R1' });
    const resource2 = new CfnResource(child1, 'Resource2', { type: 'R2' });
    const resourceA = new CfnResource(childA, 'ResourceA', { type: 'RA' });

    resourceA.addResourceDependency(resource1);
    resourceA.addResourceDependency(resource2);
    resourceA.removeResourceDependency(resource1);

    expect(paths(childA._stackDependenciesCausedBy(resourceA))).toEqual(paths([resource2]));

    const assembly = app.synth();

    expect(assembly.getStackArtifact(child1.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
    expect(assembly.getStackArtifact(childA.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual(['ParentChild18FAEF419']);
  });

  test('_removeAssemblyDependency removes a StackDependency from _stackDependencies with the last reason', () => {
    const app = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true, [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parent = new Stack(app, 'Parent');
    const child1 = new Stack(parent, 'Child1');
    const childA = new Stack(parent, 'Child2');
    const resource1 = new CfnResource(child1, 'Resource1', { type: 'R1' });
    const resource2 = new CfnResource(child1, 'Resource2', { type: 'R2' });
    const resourceA = new CfnResource(childA, 'ResourceA', { type: 'RA' });

    resourceA.addResourceDependency(resource1);
    resourceA.addResourceDependency(resource2);
    resourceA.removeResourceDependency(resource1);
    resourceA.removeResourceDependency(resource2);

    expect(childA._stackDependenciesCausedBy(childA)).toEqual([]);

    const assembly = app.synth();

    expect(assembly.getStackArtifact(child1.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
    expect(assembly.getStackArtifact(childA.artifactId).dependencies.map((x: { id: any }) => x.id)).toEqual([]);
  });

  test('automatic cross-stack references and manual exports look the same', () => {
    // GIVEN: automatic
    const appA = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true });
    const producerA = new Stack(appA, 'Producer');
    const consumerA = new Stack(appA, 'Consumer');
    const resourceA = new CfnResource(producerA, 'Resource', { type: 'AWS::Resource' });
    new CfnOutput(consumerA, 'SomeOutput', { value: `${resourceA.getAtt('Att')}` });

    // GIVEN: manual
    const appM = makeCrossStackApp();
    const producerM = new Stack(appM, 'Producer');
    const resourceM = new CfnResource(producerM, 'Resource', { type: 'AWS::Resource' });
    producerM.exportValue(resourceM.getAtt('Att'));

    // THEN - producers are the same
    const templateA = appA.synth().getStackByName(producerA.stackName).template;
    const templateM = appM.synth().getStackByName(producerM.stackName).template;

    expect(templateA).toEqual(templateM);
  });

  test('automatic cross-stack references and manual list exports look the same', () => {
    // GIVEN: automatic
    const appA = makeCrossStackApp({ '@aws-cdk/core:stackRelativeExports': true });
    const producerA = new Stack(appA, 'Producer');
    const consumerA = new Stack(appA, 'Consumer');
    const resourceA = new CfnResource(producerA, 'Resource', { type: 'AWS::Resource' });
    (resourceA as any).attrAtt = ['Foo', 'Bar'];
    new CfnOutput(consumerA, 'SomeOutput', { value: `${resourceA.getAtt('Att', ResolutionTypeHint.STRING_LIST)}` });

    // GIVEN: manual
    const appM = makeCrossStackApp();
    const producerM = new Stack(appM, 'Producer');
    const resourceM = new CfnResource(producerM, 'Resource', { type: 'AWS::Resource' });
    (resourceM as any).attrAtt = ['Foo', 'Bar'];
    producerM.exportStringListValue(resourceM.getAtt('Att', ResolutionTypeHint.STRING_LIST));

    // THEN - producers are the same
    const templateA = appA.synth().getStackByName(producerA.stackName).template;
    const templateM = appM.synth().getStackByName(producerM.stackName).template;

    expect(templateA).toEqual(templateM);
  });

  test('throw error if overrideLogicalId is used and logicalId is locked', () => {
    // GIVEN: manual
    const appM = new App();
    Validations.of(appM).acknowledge({
      id: 'CloudFormation-Validate::F0006',
      reason: 'Yes this is an invalid logical ID',
    });
    const producerM = new Stack(appM, 'Producer');
    const resourceM = new CfnResource(producerM, 'ResourceXXX', { type: 'AWS::Resource' });
    producerM.exportValue(resourceM.getAtt('Att'));

    // THEN - producers are the same
    expect(() => {
      resourceM.overrideLogicalId('OVERRIDE_LOGICAL_ID');
    }).toThrow(/The logicalId for resource at path Producer\/ResourceXXX has been locked and cannot be overridden/);
  });

  test('do not throw error if overrideLogicalId is used and logicalId is not locked', () => {
    // GIVEN: manual
    const appM = makeCrossStackApp();
    const producerM = new Stack(appM, 'Producer');
    const resourceM = new CfnResource(producerM, 'ResourceXXX', { type: 'AWS::Resource' });

    // THEN - producers are the same
    resourceM.overrideLogicalId('banana');
    producerM.exportValue(resourceM.getAtt('Att'));

    const template = appM.synth().getStackByName(producerM.stackName).template;
    expect(template).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttbananaAttF2F2ECCE: {
          Export: {
            Name: 'Producer:ExportsOutputFnGetAttbananaAttF2F2ECCE',
          },
          Value: {
            'Fn::GetAtt': [
              'banana',
              'Att',
            ],
          },
        },
      },
      Resources: {
        banana: {
          Type: 'AWS::Resource',
        },
      },
    });
  });

  test('automatic cross-stack references and manual exports look the same: nested stack edition', () => {
    // GIVEN: automatic
    const appA = makeCrossStackApp();
    const producerA = new Stack(appA, 'Producer');
    const nestedA = new NestedStack(producerA, 'Nestor');
    const resourceA = new CfnResource(nestedA, 'Resource', { type: 'AWS::Resource' });

    const consumerA = new Stack(appA, 'Consumer');
    new CfnOutput(consumerA, 'SomeOutput', { value: `${resourceA.getAtt('Att')}` });

    // GIVEN: manual
    const appM = makeCrossStackApp();
    const producerM = new Stack(appM, 'Producer');
    const nestedM = new NestedStack(producerM, 'Nestor');
    const resourceM = new CfnResource(nestedM, 'Resource', { type: 'AWS::Resource' });
    producerM.exportValue(resourceM.getAtt('Att'));

    // THEN - producers are the same
    const templateA = appA.synth().getStackByName(producerA.stackName).template;
    const templateM = appM.synth().getStackByName(producerM.stackName).template;

    expect(templateA).toEqual(templateM);
  });

  test('manual exports require a name if not supplying a resource attribute', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    expect(() => {
      stack.exportValue('someValue');
    }).toThrow(/or make sure to export a resource attribute/);
  });

  test('manual list exports require a name if not supplying a resource attribute', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    expect(() => {
      stack.exportStringListValue(['someValue']);
    }).toThrow(/or make sure to export a resource attribute/);
  });

  test('manual exports can also just be used to create an export of anything', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    const importV = stack.exportValue('someValue', { name: 'MyExport' });

    expect(stack.resolve(importV)).toEqual({ 'Fn::ImportValue': 'MyExport' });
  });

  test('manual list exports can also just be used to create an export of anything', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    const importV = stack.exportStringListValue(['someValue', 'anotherValue'], { name: 'MyExport' });

    expect(stack.resolve(importV)).toEqual(
      {
        'Fn::Split': [
          '||',
          {
            'Fn::ImportValue': 'MyExport',
          },
        ],
      },
    );

    const template = app.synth().getStackByName(stack.stackName).template;

    expect(template).toMatchObject({
      Outputs: {
        ExportMyExport: {
          Value: 'someValue||anotherValue',
          Export: {
            Name: 'MyExport',
          },
        },
      },
    });
  });

  test('exports with name can include description', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    stack.exportValue('someValue', {
      name: 'MyExport',
      description: 'This is a description',
    });

    const template = app.synth().getStackByName(stack.stackName).template;
    expect(template).toMatchObject({
      Outputs: {
        ExportMyExport: {
          Description: 'This is a description',
        },
      },
    });
  });

  test('list exports with name can include description', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    stack.exportStringListValue(['someValue', 'anotherValue'], {
      name: 'MyExport',
      description: 'This is a description',
    });

    const template = app.synth().getStackByName(stack.stackName).template;
    expect(template).toMatchObject({
      Outputs: {
        ExportMyExport: {
          Description: 'This is a description',
        },
      },
    });
  });

  test('exports without name can include description', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    const resource = new CfnResource(stack, 'Resource', { type: 'AWS::Resource' });
    stack.exportValue(resource.getAtt('Att'), {
      description: 'This is a description',
    });

    const template = app.synth().getStackByName(stack.stackName).template;
    expect(template).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttResourceAttB5968E71: {
          Description: 'This is a description',
        },
      },
    });
  });

  test('list exports without name can include description', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');

    const resource = new CfnResource(stack, 'Resource', { type: 'AWS::Resource' });
    (resource as any).attrAtt = ['Foo', 'Bar'];
    stack.exportStringListValue(resource.getAtt('Att', ResolutionTypeHint.STRING_LIST), {
      description: 'This is a description',
    });

    const template = app.synth().getStackByName(stack.stackName).template;
    expect(template).toMatchObject({
      Outputs: {
        ExportsOutputFnGetAttResourceAttB5968E71: {
          Description: 'This is a description',
        },
      },
    });
  });

  test('CfnSynthesisError is ignored when preparing cross references', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'my-stack');

    // WHEN
    class CfnTest extends CfnResource {
      public _toCloudFormation() {
        return new PostResolveToken({
          xoo: 1234,
        }, (props, _context) => {
          validateString(props).assertSuccess();
        });
      }
    }

    new CfnTest(stack, 'MyThing', { type: 'AWS::Type' });

    // THEN
    resolveReferences(app);
  });

  test('Stacks can be children of other stacks (substack) and they will be synthesized separately', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const parentStack = new Stack(app, 'parent');
    const childStack = new Stack(parentStack, 'child');
    new CfnResource(parentStack, 'MyParentResource', { type: 'Resource::Parent' });
    new CfnResource(childStack, 'MyChildResource', { type: 'Resource::Child' });

    // THEN
    const assembly = app.synth();
    expect(assembly.getStackByName(parentStack.stackName).template?.Resources).toEqual({ MyParentResource: { Type: 'Resource::Parent' } });
    expect(assembly.getStackByName(childStack.stackName).template?.Resources).toEqual({ MyChildResource: { Type: 'Resource::Child' } });
  });

  test('Nested Stacks are synthesized with DESTROY policy', () => {
    const app = new App();

    // WHEN
    const parentStack = new Stack(app, 'parent');
    const childStack = new NestedStack(parentStack, 'child');
    new CfnResource(childStack, 'ChildResource', { type: 'Resource::Child' });

    const assembly = app.synth();
    expect(assembly.getStackByName(parentStack.stackName).template).toEqual(expect.objectContaining({
      Resources: {
        childNestedStackchildNestedStackResource7408D03F: expect.objectContaining({
          Type: 'AWS::CloudFormation::Stack',
          DeletionPolicy: 'Delete',
        }),
      },
    }));
  });

  test('asset metadata added to NestedStack resource that contains asset path and property', () => {
    const app = new App();

    // WHEN
    const parentStack = new Stack(app, 'parent');
    parentStack.node.setContext(cxapi.ASSET_RESOURCE_METADATA_ENABLED_CONTEXT, true);
    const childStack = new NestedStack(parentStack, 'child');
    new CfnResource(childStack, 'ChildResource', { type: 'Resource::Child' });

    const assembly = app.synth();
    expect(assembly.getStackByName(parentStack.stackName).template).toEqual(expect.objectContaining({
      Resources: {
        childNestedStackchildNestedStackResource7408D03F: expect.objectContaining({
          Metadata: {
            'aws:asset:path': 'parentchild13F9359B.nested.template.json',
            'aws:asset:property': 'TemplateURL',
          },
        }),
      },
    }));
  });

  test('cross-stack reference (substack references parent stack)', () => {
    // GIVEN
    const app = makeCrossStackApp({ [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false });
    const parentStack = new Stack(app, 'parent');
    const childStack = new Stack(parentStack, 'child');

    // WHEN (a resource from the child stack references a resource from the parent stack)
    const parentResource = new CfnResource(parentStack, 'MyParentResource', { type: 'Resource::Parent' });
    new CfnResource(childStack, 'MyChildResource', {
      type: 'Resource::Child',
      properties: {
        ChildProp: parentResource.getAtt('AttOfParentResource'),
      },
    });

    // THEN
    const assembly = app.synth();
    expect(assembly.getStackByName(parentStack.stackName).template).toEqual({
      Resources: { MyParentResource: { Type: 'Resource::Parent' } },
      Outputs: {
        ExportsOutputFnGetAttMyParentResourceAttOfParentResourceC2D0BB9E: {
          Value: { 'Fn::GetAtt': ['MyParentResource', 'AttOfParentResource'] },
          Export: { Name: 'parent:ExportsOutputFnGetAttMyParentResourceAttOfParentResourceC2D0BB9E' },
        },
      },
    });
    expect(assembly.getStackByName(childStack.stackName).template).toEqual({
      Resources: {
        MyChildResource: {
          Type: 'Resource::Child',
          Properties: {
            ChildProp: {
              'Fn::ImportValue': 'parent:ExportsOutputFnGetAttMyParentResourceAttOfParentResourceC2D0BB9E',
            },
          },
        },
      },
    });
  });

  test('cross-stack reference (parent stack references substack)', () => {
    // GIVEN
    const app = makeCrossStackApp({
      '@aws-cdk/core:stackRelativeExports': true,
      [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false,
    });

    const parentStack = new Stack(app, 'parent');
    const childStack = new Stack(parentStack, 'child');

    // WHEN (a resource from the child stack references a resource from the parent stack)
    const childResource = new CfnResource(childStack, 'MyChildResource', { type: 'Resource::Child' });
    new CfnResource(parentStack, 'MyParentResource', {
      type: 'Resource::Parent',
      properties: {
        ParentProp: childResource.getAtt('AttributeOfChildResource'),
      },
    });

    // THEN
    const assembly = app.synth();
    expect(assembly.getStackByName(parentStack.stackName).template).toEqual({
      Resources: {
        MyParentResource: {
          Type: 'Resource::Parent',
          Properties: {
            ParentProp: { 'Fn::ImportValue': 'parentchild13F9359B:ExportsOutputFnGetAttMyChildResourceAttributeOfChildResource52813264' },
          },
        },
      },
    });

    expect(assembly.getStackByName(childStack.stackName).template).toEqual({
      Resources: { MyChildResource: { Type: 'Resource::Child' } },
      Outputs: {
        ExportsOutputFnGetAttMyChildResourceAttributeOfChildResource52813264: {
          Value: { 'Fn::GetAtt': ['MyChildResource', 'AttributeOfChildResource'] },
          Export: { Name: 'parentchild13F9359B:ExportsOutputFnGetAttMyChildResourceAttributeOfChildResource52813264' },
        },
      },
    });
  });

  test('cannot create cyclic reference between stacks', () => {
    // GIVEN
    const app = new App();
    const stack1 = new Stack(app, 'Stack1');
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2');
    const account2 = new ScopedAws(stack2).accountId;

    // WHEN
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: account1 });
    new CfnParameter(stack1, 'SomeParameter', { type: 'String', default: account2 });

    expect(() => {
      app.synth();
    }).toThrow("'Stack1' depends on 'Stack2' (Stack1 -> Stack2.AWS::AccountId). Adding this dependency (Stack2 -> Stack1.AWS::AccountId) would create a cyclic reference.");
  });

  test('stacks know about their dependencies', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1');
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2');

    // WHEN
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: account1 });

    app.synth();

    // THEN
    expect(stack2.dependencies.map(s => s.node.id)).toEqual(['Stack1']);
  });

  test('cross-account references with strong flag emit warning but still synth', () => {
    // GIVEN
    const app = makeCrossStackApp();
    const stack1 = new Stack(app, 'Stack1', { env: { account: '123456789012', region: 'es-norst-1' } });
    const account1 = new ScopedAws(stack1).accountId;
    const stack2 = new Stack(app, 'Stack2', { env: { account: '11111111111', region: 'es-norst-2' } });

    // WHEN
    new CfnParameter(stack2, 'SomeParameter', { type: 'String', default: account1 });

    // THEN - does not throw, emits warning
    expect(() => {
      app.synth();
    }).not.toThrow();

    const warnings = stack2.node.metadata.filter(m => m.type === 'aws:cdk:warning');
    expect(warnings.some(w => w.data.includes('cross-account references can only be weak'))).toBe(true);
  });

  test('urlSuffix does not imply a stack dependency', () => {
    // GIVEN
    const app = new App();
    const first = new Stack(app, 'First');
    const second = new Stack(app, 'Second');

    // WHEN
    new CfnOutput(second, 'Output', {
      value: first.urlSuffix,
    });

    // THEN
    app.synth();

    expect(second.dependencies.length).toEqual(0);
  });

  test('stack with region supplied via props returns literal value', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack1', { env: { account: '123456789012', region: 'es-norst-1' } });

    // THEN
    expect(stack.resolve(stack.region)).toEqual('es-norst-1');
  });

  describe('stack partition literal feature flag', () => {
    // GIVEN
    const featureFlag = { [cxapi.ENABLE_PARTITION_LITERALS]: true };
    const envForRegion = (region: string) => { return { env: { account: '123456789012', region: region } }; };

    // THEN
    describe('does not change missing or unknown region behaviors', () => {
      test('stacks with no region defined', () => {
        const noRegionStack = new Stack(new App(), 'MissingRegion');
        expect(noRegionStack.partition).toEqual(Aws.PARTITION);
      });

      test('stacks with an unknown region', () => {
        const imaginaryRegionStack = new Stack(new App(), 'ImaginaryRegion', envForRegion('us-area51'));
        expect(imaginaryRegionStack.partition).toEqual(Aws.PARTITION);
      });
    });

    describe('changes known region behaviors only when enabled', () => {
      test('(disabled)', () => {
        const app = new App();
        RegionInfo.regions.forEach(function(region) {
          const regionStack = new Stack(app, `Region-${region.name}`, envForRegion(region.name));
          expect(regionStack.partition).toEqual(Aws.PARTITION);
        });
      });

      test('(enabled)', () => {
        const app = new App({ context: featureFlag });
        RegionInfo.regions.forEach(function(region) {
          const regionStack = new Stack(app, `Region-${region.name}`, envForRegion(region.name));
          expect(regionStack.partition).toEqual(RegionInfo.get(region.name).partition);
        });
      });
    });
  });

  test('overrideLogicalId(id) can be used to override the logical ID of a resource', () => {
    // GIVEN
    const stack = new Stack();
    const bonjour = new CfnResource(stack, 'BonjourResource', { type: 'Resource::Type' });

    // { Ref } and { GetAtt }
    new CfnResource(stack, 'RefToBonjour', {
      type: 'Other::Resource',
      properties: {
        RefToBonjour: bonjour.ref,
        GetAttBonjour: bonjour.getAtt('TheAtt').toString(),
      },
    });

    bonjour.overrideLogicalId('BOOM');

    // THEN
    expect(toCloudFormation(stack)).toEqual({
      Resources:
      {
        BOOM: { Type: 'Resource::Type' },
        RefToBonjour:
         {
           Type: 'Other::Resource',
           Properties:
            {
              RefToBonjour: { Ref: 'BOOM' },
              GetAttBonjour: { 'Fn::GetAtt': ['BOOM', 'TheAtt'] },
            },
         },
      },
    });
  });

  test('Stack name can be overridden via properties', () => {
    // WHEN
    const stack = new Stack(undefined, 'Stack', { stackName: 'otherName' });

    // THEN
    expect(stack.stackName).toEqual('otherName');
  });

  test('Stack name is inherited from App name if available', () => {
    // WHEN
    const root = new App();
    const app = new Construct(root, 'Prod');
    const stack = new Stack(app, 'Stack');

    // THEN
    expect(stack.stackName).toEqual('ProdStackD5279B22');
  });

  test('generated stack names will not exceed 128 characters', () => {
    // WHEN
    const root = new App();
    const app = new Construct(root, 'ProdAppThisNameButItWillOnlyBeTooLongWhenCombinedWithTheStackName' + 'z'.repeat(60));
    const stack = new Stack(app, 'ThisNameIsVeryLongButItWillOnlyBeTooLongWhenCombinedWithTheAppNameStack');

    // THEN
    expect(stack.stackName.length).toEqual(128);
    expect(stack.stackName).toEqual('ProdAppThisNameButItWillOnlyBeTooLongWhenCombinedWithTheStaceryLongButItWillOnlyBeTooLongWhenCombinedWithTheAppNameStack864CC1D3');
  });

  test('stack construct id does not go through stack name validation if there is an explicit stack name', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const stack = new Stack(app, 'invalid as : stack name, but thats fine', {
      stackName: 'valid-stack-name',
    });

    // THEN
    const session = app.synth();
    expect(stack.stackName).toEqual('valid-stack-name');
    expect(session.tryGetArtifact(stack.artifactId)).toBeDefined();
  });

  test('stack validation is performed on explicit stack name', () => {
    // GIVEN
    const app = new App();

    // THEN
    expect(() => new Stack(app, 'boom', { stackName: 'invalid:stack:name' }))
      .toThrow(/Stack name must match the regular expression/);
  });

  test('Stack.of(stack) returns the correct stack', () => {
    const stack = new Stack();
    expect(Stack.of(stack)).toBe(stack);
    const parent = new Construct(stack, 'Parent');
    const construct = new Construct(parent, 'Construct');
    expect(Stack.of(construct)).toBe(stack);
  });

  test('Stack.of() throws when there is no parent Stack', () => {
    const root = new Construct(undefined as any, 'Root');
    const construct = new Construct(root, 'Construct');
    expect(() => Stack.of(construct)).toThrow(/should be created in the scope of a Stack, but no Stack found/);
  });

  test('Stack.of() works for substacks', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const parentStack = new Stack(app, 'ParentStack');
    const parentResource = new CfnResource(parentStack, 'ParentResource', { type: 'parent::resource' });

    // we will define a substack under the /resource/... just for giggles.
    const childStack = new Stack(parentResource, 'ChildStack');
    const childResource = new CfnResource(childStack, 'ChildResource', { type: 'child::resource' });

    // THEN
    expect(Stack.of(parentStack)).toBe(parentStack);
    expect(Stack.of(parentResource)).toBe(parentStack);
    expect(Stack.of(childStack)).toBe(childStack);
    expect(Stack.of(childResource)).toBe(childStack);
  });

  test('stack.availabilityZones falls back to Fn::GetAZ[0],[2] if region is not specified', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');

    // WHEN
    const azs = stack.availabilityZones;

    // THEN
    expect(stack.resolve(azs)).toEqual([
      { 'Fn::Select': [0, { 'Fn::GetAZs': '' }] },
      { 'Fn::Select': [1, { 'Fn::GetAZs': '' }] },
    ]);
  });

  test('allows using the same stack name for two stacks (i.e. in different regions)', () => {
    // WHEN
    const app = new App();
    const stack1 = new Stack(app, 'MyStack1', { stackName: 'thestack' });
    const stack2 = new Stack(app, 'MyStack2', { stackName: 'thestack' });
    const assembly = app.synth();

    // THEN
    expect(assembly.getStackArtifact(stack1.artifactId).templateFile).toEqual('MyStack1.template.json');
    expect(assembly.getStackArtifact(stack2.artifactId).templateFile).toEqual('MyStack2.template.json');
    expect(stack1.templateFile).toEqual('MyStack1.template.json');
    expect(stack2.templateFile).toEqual('MyStack2.template.json');
  });

  test('artifactId and templateFile use the unique id and not the stack name', () => {
    // WHEN
    const app = new App();
    const stack1 = new Stack(app, 'MyStack1', { stackName: 'thestack' });
    const assembly = app.synth();

    // THEN
    expect(stack1.artifactId).toEqual('MyStack1');
    expect(stack1.templateFile).toEqual('MyStack1.template.json');
    expect(assembly.getStackArtifact(stack1.artifactId).templateFile).toEqual('MyStack1.template.json');
  });

  test('use the artifact id as the template name', () => {
    // WHEN
    const app = new App();
    const stack1 = new Stack(app, 'MyStack1');
    const stack2 = new Stack(app, 'MyStack2', { stackName: 'MyRealStack2' });

    // THEN
    expect(stack1.templateFile).toEqual('MyStack1.template.json');
    expect(stack2.templateFile).toEqual('MyStack2.template.json');
  });

  test('metadata is collected at the stack boundary', () => {
    // GIVEN
    const app = new App({
      context: {
        [cxapi.DISABLE_METADATA_STACK_TRACE]: 'true',
      },
    });
    const parent = new Stack(app, 'parent');
    const child = new Stack(parent, 'child');

    // WHEN
    child.node.addMetadata('foo', 'bar');

    // THEN
    const asm = app.synth();
    expect(asm.getStackByName(parent.stackName).findMetadataByType('foo')).toEqual([]);
    expect(asm.getStackByName(child.stackName).findMetadataByType('foo')).toEqual([
      { path: '/parent/child', type: 'foo', data: 'bar' },
    ]);
  });

  test('stack tags are reflected in the stack cloud assembly artifact metadata', () => {
    // GIVEN
    const app = new App({ stackTraces: false, context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });

    const stack1 = new Stack(app, 'stack1');
    const stack2 = new Stack(stack1, 'stack2');

    // WHEN
    Tags.of(app).add('foo', 'bar');

    // THEN
    const asm = app.synth();
    expect(flattenMeta(asm.getStackArtifact(stack1.artifactId).metadata)).toMatchObject({
      '/stack1': {
        'aws:cdk:stack-tags': [
          [{ key: 'foo', value: 'bar' }],
        ],
      },
    });

    expect(flattenMeta(asm.getStackArtifact(stack2.artifactId).metadata)).toMatchObject({
      '/stack1/stack2': {
        'aws:cdk:stack-tags': [
          [{ key: 'foo', value: 'bar' }],
        ],
      },
    });
  });

  test('stack tags are reflected in the stack artifact properties', () => {
    // GIVEN
    const app = new App({ stackTraces: false });
    const stack1 = new Stack(app, 'stack1');
    const stack2 = new Stack(stack1, 'stack2');

    // WHEN
    Tags.of(app).add('foo', 'bar');

    // THEN
    const asm = app.synth();
    const expected = { foo: 'bar' };

    expect(asm.getStackArtifact(stack1.artifactId).tags).toEqual(expected);
    expect(asm.getStackArtifact(stack2.artifactId).tags).toEqual(expected);
  });

  describe.each([false, true])(`with ${cxapi.EXPLICIT_STACK_TAGS} set to %p`, (explicitStackTags) => {
    let app: App;

    beforeEach(() => {
      app = new App({
        stackTraces: false,
        context: {
          [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false,
          [cxapi.EXPLICIT_STACK_TAGS]: explicitStackTags,
        },
      });
    });

    test('stack tags are both in stack metadata and stack artifact properties', () => {
      // GIVEN

      const stack = new Stack(app, 'stack1', {
        tags: {
          foo: 'bar',
        },
      });

      // THEN
      const asm = app.synth();

      const stackArtifact = asm.getStackArtifact(stack.artifactId);
      expect(flattenMeta(stackArtifact.metadata)).toMatchObject({
        '/stack1': {
          'aws:cdk:stack-tags': [
            [{ key: 'foo', value: 'bar' }],
          ],
        },
      });
      expect(stackArtifact.tags).toEqual({ foo: 'bar' });
    });

    test('stack tags do not appear in template', () => {
      const stack = new Stack(app, 'stack1', {
        tags: {
          foo: 'bar',
        },
      });
      new TaggableResource(stack, 'res');

      // THEN
      const asm = app.synth();
      const stackArtifact = asm.getStackArtifact(stack.artifactId);
      expect(stackArtifact.template.Resources.res).toEqual({
        Type: 'AWS::Taggable::Resource',
        Properties: {
          R: 1,
        },
      });
    });

    test('tags added using Tags.of() are or are not applied to Stacks', () => {
      const stack = new Stack(app, 'stack1');
      Tags.of(stack).add('resourceTag', 'resourceValue');

      // THEN
      const asm = app.synth();
      const stackArtifact = asm.getStackArtifact(stack.artifactId);
      if (explicitStackTags) {
        expect(stackArtifact.tags).toEqual({});
      } else {
        expect(stackArtifact.tags).toEqual({ resourceTag: 'resourceValue' });
      }
    });

    test('tags added using Tags.of() are applied to Stacks if explicitly included', () => {
      const stack = new Stack(app, 'stack1');
      Tags.of(stack).add('resourceTag', 'resourceValue', {
        includeResourceTypes: ['aws:cdk:stack'],
      });

      // THEN
      const asm = app.synth();
      const stackArtifact = asm.getStackArtifact(stack.artifactId);
      expect(stackArtifact.tags).toEqual({ resourceTag: 'resourceValue' });
    });
  });

  test('warning when stack tags contain tokens', () => {
    // GIVEN
    const app = new App({
      stackTraces: false,
    });

    new Stack(app, 'stack1', {
      tags: {
        foo: Lazy.string({ produce: () => 'lazy' }),
      },
    });

    const asm = app.synth();
    const stackArtifact = asm.stacks[0];
    expect(flattenMeta(stackArtifact.metadata)).toMatchObject({
      '/stack1': {
        'aws:cdk:warning': [
          expect.stringContaining('Ignoring stack tags that contain deploy-time values'),
        ],
      },
    });
  });

  test('stack notification arns defaults to undefined', () => {
    const app = new App({ stackTraces: false });
    const stack1 = new Stack(app, 'stack1', {});

    const asm = app.synth();

    // It must be undefined and not [] because:
    //
    //  - undefined  =>  cdk ignores it entirely, as if it wasn't supported (allows external management).
    //  - []:        =>  cdk manages it, and the user wants to wipe it out.
    //  - ['arn-1']  =>  cdk manages it, and the user wants to set it to ['arn-1'].
    expect(asm.getStackArtifact(stack1.artifactId).notificationArns).toBeUndefined();
  });

  test('stack notification arns are reflected in the stack artifact properties', () => {
    // GIVEN
    const NOTIFICATION_ARNS = ['arn:aws:sns:bermuda-triangle-1337:123456789012:MyTopic'];
    const app = new App({ stackTraces: false });
    const stack1 = new Stack(app, 'stack1', {
      notificationArns: NOTIFICATION_ARNS,
    });

    // WHEN
    const asm = app.synth();

    // THEN
    expect(asm.getStackArtifact(stack1.artifactId).notificationArns).toEqual(NOTIFICATION_ARNS);
  });

  test('throws if stack notification arns contain tokens', () => {
    // GIVEN
    const NOTIFICATION_ARNS = ['arn:aws:sns:bermuda-triangle-1337:123456789012:MyTopic'];
    const app = new App({ stackTraces: false });

    // THEN
    expect(() => new Stack(app, 'stack1', {
      notificationArns: [...NOTIFICATION_ARNS, Aws.URL_SUFFIX],
    })).toThrow('includes one or more tokens in its notification ARNs');
  });

  test('Termination Protection is reflected in Cloud Assembly artifact', () => {
    // if the root is an app, invoke "synth" to avoid double synthesis
    const app = new App();
    const stack = new Stack(app, 'Stack', { terminationProtection: true });

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);

    expect(artifact.terminationProtection).toEqual(true);
  });

  test('Set termination protection to true with setter', () => {
    // if the root is an app, invoke "synth" to avoid double synthesis
    const app = new App();
    const stack = new Stack(app, 'Stack', {});

    stack.terminationProtection = true;

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);

    expect(artifact.terminationProtection).toEqual(true);
  });

  test('Set termination protection to false with setter', () => {
    // if the root is an app, invoke "synth" to avoid double synthesis
    const app = new App();
    const stack = new Stack(app, 'Stack', { terminationProtection: true });

    stack.terminationProtection = false;

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);

    expect(artifact.terminationProtection).toEqual(false);
  });

  test('context can be set on a stack using a LegacySynthesizer', () => {
    // WHEN
    const stack = new Stack(undefined, undefined, {
      synthesizer: new LegacyStackSynthesizer(),
    });
    stack.node.setContext('something', 'value');

    // THEN: no exception
  });

  test('context can be set on a stack using a DefaultSynthesizer', () => {
    // WHEN
    const stack = new Stack(undefined, undefined, {
      synthesizer: new DefaultStackSynthesizer(),
    });
    stack.node.setContext('something', 'value');

    // THEN: no exception
  });

  test('version reporting can be configured on the app', () => {
    const app = new App({ analyticsReporting: true });
    expect(new Stack(app, 'Stack')._versionReportingEnabled).toBeDefined();
  });

  test('version reporting can be configured with context', () => {
    const app = new App({ context: { 'aws:cdk:version-reporting': true } });
    expect(new Stack(app, 'Stack')._versionReportingEnabled).toBeDefined();
  });

  test('version reporting can be configured on the stack', () => {
    const app = new App();
    expect(new Stack(app, 'Stack', { analyticsReporting: true })._versionReportingEnabled).toBeDefined();
  });

  test('requires bundling when wildcard is specified in BUNDLING_STACKS', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    stack.node.setContext(cxapi.BUNDLING_STACKS, ['*']);
    expect(stack.bundlingRequired).toBe(true);
  });

  test('requires bundling when stackName has an exact match in BUNDLING_STACKS', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    stack.node.setContext(cxapi.BUNDLING_STACKS, ['Stack']);
    expect(stack.bundlingRequired).toBe(true);
  });

  test('does not require bundling when no item from BUILDING_STACKS matches stackName', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    stack.node.setContext(cxapi.BUNDLING_STACKS, ['Stac']);
    expect(stack.bundlingRequired).toBe(false);
  });

  test('does not require bundling when BUNDLING_STACKS is empty', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    stack.node.setContext(cxapi.BUNDLING_STACKS, []);
    expect(stack.bundlingRequired).toBe(false);
  });

  test('account id passed in stack environment must be a string', () => {
    // GIVEN
    const envConfig: any = {
      account: 11111111111,
    };

    // WHEN
    const app = new App();

    // THEN
    expect(() => {
      new Stack(app, 'Stack', {
        env: envConfig,
      });
    }).toThrow('Account id of stack environment must be a \'string\' but received \'number\'');
  });

  test('region passed in stack environment must be a string', () => {
    // GIVEN
    const envConfig: any = {
      region: 2,
    };

    // WHEN
    const app = new App();

    // THEN
    expect(() => {
      new Stack(app, 'Stack', {
        env: envConfig,
      });
    }).toThrow('Region of stack environment must be a \'string\' but received \'number\'');
  });

  test('indent templates when suppressTemplateIndentation is not set', () => {
    const app = new App();

    const stack = new Stack(app, 'Stack');
    new CfnResource(stack, 'MyResource', { type: 'MyResourceType' });

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);
    const templateData = fs.readFileSync(artifact.templateFullPath, 'utf-8');

    expect(templateData).toMatch(/^{\n \"Resources\": {\n  \"MyResource\": {\n   \"Type\": \"MyResourceType\"\n  }\n }/);
  });

  test('do not indent templates when suppressTemplateIndentation is true', () => {
    const app = new App();

    const stack = new Stack(app, 'Stack', { suppressTemplateIndentation: true });
    new CfnResource(stack, 'MyResource', { type: 'MyResourceType' });

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);
    const templateData = fs.readFileSync(artifact.templateFullPath, 'utf-8');

    expect(templateData).toMatch(/^{\"Resources\":{\"MyResource\":{\"Type\":\"MyResourceType\"}}/);
  });

  test('do not indent templates when @aws-cdk/core:suppressTemplateIndentation is true', () => {
    const app = new App({
      context: {
        '@aws-cdk/core:suppressTemplateIndentation': true,
      },
    });

    const stack = new Stack(app, 'Stack');
    new CfnResource(stack, 'MyResource', { type: 'MyResourceType' });

    const assembly = app.synth();
    const artifact = assembly.getStackArtifact(stack.artifactId);
    const templateData = fs.readFileSync(artifact.templateFullPath, 'utf-8');

    expect(templateData).toMatch(/^{\"Resources\":{\"MyResource\":{\"Type\":\"MyResourceType\"}}/);
  });

  test('Stacks log their own creation stack trace', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const res = new CfnResource(stack, 'SomeCfnResource', {
      type: 'Some::Resource',
    });

    // THEN
    const metadata = res.node.metadata.find(m => m.type === 'aws:cdk:creationStack');
    expect(metadata).toBeDefined();
  });
});

describe('permissions boundary', () => {
  test('can specify a valid permissions boundary name', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const stack = new Stack(app, 'Stack', {
      permissionsBoundary: PermissionsBoundary.fromName('valid'),
    });

    // THEN
    const pbContext = stack.node.tryGetContext(PERMISSIONS_BOUNDARY_CONTEXT_KEY);
    expect(pbContext).toEqual({
      name: 'valid',
    });
  });

  test('can specify a valid permissions boundary arn', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const stack = new Stack(app, 'Stack', {
      permissionsBoundary: PermissionsBoundary.fromArn('arn:aws:iam::12345678912:policy/valid'),
    });

    // THEN
    const pbContext = stack.node.tryGetContext(PERMISSIONS_BOUNDARY_CONTEXT_KEY);
    expect(pbContext).toEqual({
      name: undefined,
      arn: 'arn:aws:iam::12345678912:policy/valid',
    });
  });

  test('single aspect is added to stack', () => {
    // GIVEN
    const app = new App();

    // WHEN
    const stage = new Stage(app, 'Stage', {
      permissionsBoundary: PermissionsBoundary.fromArn('arn:aws:iam::12345678912:policy/stage'),
    });
    const stack = new Stack(stage, 'Stack', {
      permissionsBoundary: PermissionsBoundary.fromArn('arn:aws:iam::12345678912:policy/valid'),
    });

    // THEN
    const aspects = Aspects.of(stack).all;
    expect(aspects.length).toEqual(1);
    const pbContext = stack.node.tryGetContext(PERMISSIONS_BOUNDARY_CONTEXT_KEY);
    expect(pbContext).toEqual({
      name: undefined,
      arn: 'arn:aws:iam::12345678912:policy/valid',
    });
  });

  test('throws if pseudo parameters are in the name', () => {
    // GIVEN
    const app = new App();

    // THEN
    expect(() => {
      new Stack(app, 'Stack', {
        permissionsBoundary: PermissionsBoundary.fromArn('arn:aws:iam::${AWS::AccountId}:policy/valid'),
      });
    }).toThrow(/The permissions boundary .* includes a pseudo parameter/);
  });
});

describe('regionalFact', () => {
  Fact.register({ name: 'MyFact', region: 'us-east-1', value: 'x.amazonaws.com' });
  Fact.register({ name: 'MyFact', region: 'eu-west-1', value: 'x.amazonaws.com' });
  Fact.register({ name: 'MyFact', region: 'cn-north-1', value: 'x.amazonaws.com.cn' });

  Fact.register({ name: 'WeirdFact', region: 'us-east-1', value: 'oneformat' });
  Fact.register({ name: 'WeirdFact', region: 'eu-west-1', value: 'otherformat' });

  test('regional facts return a literal value if possible', () => {
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });
    expect(stack.regionalFact('MyFact')).toEqual('x.amazonaws.com');
  });

  test('regional facts are simplified to use URL_SUFFIX token if possible', () => {
    const stack = new Stack();
    expect(stack.regionalFact('MyFact')).toEqual(`x.${Aws.URL_SUFFIX}`);
  });

  test('regional facts are simplified to use concrete values if URL_SUFFIX token is not necessary', () => {
    const stack = new Stack();
    Node.of(stack).setContext(cxapi.TARGET_PARTITIONS, ['aws']);
    expect(stack.regionalFact('MyFact')).toEqual('x.amazonaws.com');
  });

  test('regional facts use the global lookup map if partition is the literal string of "undefined"', () => {
    const stack = new Stack();
    Node.of(stack).setContext(cxapi.TARGET_PARTITIONS, 'undefined');
    new CfnOutput(stack, 'TheFact', {
      value: stack.regionalFact('WeirdFact'),
    });

    expect(toCloudFormation(stack)).toEqual({
      Mappings: {
        WeirdFactMap: {
          'eu-west-1': { value: 'otherformat' },
          'us-east-1': { value: 'oneformat' },
        },
      },
      Outputs: {
        TheFact: {
          Value: {
            'Fn::FindInMap': ['WeirdFactMap', { Ref: 'AWS::Region' }, 'value'],
          },
        },
      },
    });
  });

  test('regional facts generate a mapping if necessary', () => {
    const stack = new Stack();
    new CfnOutput(stack, 'TheFact', {
      value: stack.regionalFact('WeirdFact'),
    });

    expect(toCloudFormation(stack)).toEqual({
      Mappings: {
        WeirdFactMap: {
          'eu-west-1': { value: 'otherformat' },
          'us-east-1': { value: 'oneformat' },
        },
      },
      Outputs: {
        TheFact: {
          Value: {
            'Fn::FindInMap': [
              'WeirdFactMap',
              { Ref: 'AWS::Region' },
              'value',
            ],
          },
        },
      },
    });
  });

  test('stack.addMetadata() adds metadata', () => {
    const stack = new Stack();

    stack.addMetadata('Instances', { Description: 'Information about the instances' });
    stack.addMetadata('Databases', { Description: 'Information about the databases' } );

    expect(toCloudFormation(stack)).toEqual({
      Metadata: {
        Instances: { Description: 'Information about the instances' },
        Databases: { Description: 'Information about the databases' },
      },
    });
  });
});

class StackWithPostProcessor extends Stack {
  public _toCloudFormation() {
    const template = super._toCloudFormation();

    // manipulate template (e.g. rename "Key" to "key")
    template.Resources.myResource.Properties.Environment.key =
      template.Resources.myResource.Properties.Environment.Key;
    delete template.Resources.myResource.Properties.Environment.Key;

    return template;
  }
}

class TaggableResource extends CfnResource implements ITaggableV2 {
  public readonly cdkTagManager = new TagManager(TagType.KEY_VALUE, 'TaggableResource', {}, {
    tagPropertyName: 'Tags',
  });

  constructor(scope: Construct, id: string) {
    super(scope, id, {
      type: 'AWS::Taggable::Resource',
      properties: {
        R: 1,
        Tags: Lazy.any({ produce: () => this.cdkTagManager.renderTags() }),
      },
    });
  }
}

function paths(xs: IConstruct[]): string[] {
  return xs.map(x => x.node.path);
}
