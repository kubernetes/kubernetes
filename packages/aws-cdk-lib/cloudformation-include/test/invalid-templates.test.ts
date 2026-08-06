import * as path from 'path';
import type * as constructs from 'constructs';
import { acknowledgeTestWarnings } from './test-warnings';
import { Template } from '../../assertions';
import * as core from '../../core';
import { Validations } from '../../core';
import * as cxapi from '../../cx-api';
import * as inc from '../lib';

describe('CDK Include', () => {
  let app: core.App;
  let stack: core.Stack;

  beforeEach(() => {
    app = new core.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    acknowledgeTestWarnings(app);
    stack = new core.Stack(app);
  });

  test('throws a validation exception for a template with a missing required top-level resource property', () => {
    expect(() => {
      includeTestTemplate(stack, 'bucket-policy-without-bucket.json');
    }).toThrow(/missing required property: bucket/);
  });

  test('throws a validation exception for a template with a resource property expecting an array assigned the wrong type', () => {
    includeTestTemplate(stack, 'bucket-with-cors-rules-not-an-array.json');

    expect(() => {
      app.synth();
    }).toThrow(/corsRules: "CorsRules!" should be a list/);
  });

  test('throws a validation exception for a template with a null array element of a complex type with required fields', () => {
    includeTestTemplate(stack, 'bucket-with-cors-rules-null-element.json');

    expect(() => {
      app.synth();
    }).toThrow(/allowedMethods: required but missing/);
  });

  test('throws a validation exception for a template with a missing nested resource property', () => {
    includeTestTemplate(stack, 'bucket-with-invalid-cors-rule.json');

    expect(() => {
      app.synth();
    }).toThrow(/allowedOrigins: required but missing/);
  });

  test("throws a validation exception for a template with a DependsOn that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-depends-on.json');
    }).toThrow(/Resource 'Bucket2' depends on 'Bucket1' that doesn't exist/);
  });

  test("throws a validation exception for a template referencing a Condition in the Conditions section that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-condition-in-conditions.json');
    }).toThrow(/Referenced Condition with name 'AlwaysFalse' was not found in the template/);
  });

  test('throws a validation exception for a template using Fn::GetAtt in the Conditions section', () => {
    expect(() => {
      includeTestTemplate(stack, 'getatt-in-conditions.json');
    }).toThrow(/Using GetAtt in Condition definitions is not allowed/);
  });

  test("throws a validation exception for a template referencing a Condition resource attribute that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-condition.json');
    }).toThrow(/Resource 'Bucket' uses Condition 'AlwaysFalseCond' that doesn't exist/);
  });

  test("throws a validation exception for a template referencing a Condition in an If expression that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-condition-in-if.json');
    }).toThrow(/Condition 'AlwaysFalse' used in an Fn::If expression does not exist in the template/);
  });

  test("throws an exception when encountering a CFN function it doesn't support", () => {
    expect(() => {
      includeTestTemplate(stack, 'only-codecommit-repo-using-cfn-functions.json');
    }).toThrow(/Unsupported CloudFormation function 'Fn::ValueOfAll'/);
  });

  test('parses even the unrecognized attributes of resources', () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-resource-attribute.json');
    }).toThrow(/Element used in Ref expression with logical ID: 'NonExistentResource' not found/);
  });

  test("throws a validation exception when encountering a Ref-erence to a template element that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'ref-ing-a-non-existent-element.json');
    }).toThrow(/Element used in Ref expression with logical ID: 'DoesNotExist' not found/);
  });

  test("throws a validation exception when encountering a GetAtt reference to a resource that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'getting-attribute-of-a-non-existent-resource.json');
    }).toThrow(/Resource used in GetAtt expression with logical ID: 'DoesNotExist' not found/);
  });

  test("throws a validation exception when an Output references a Condition that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'output-referencing-nonexistent-condition.json');
    }).toThrow(/Output with name 'SomeOutput' refers to a Condition with name 'NonexistentCondition' which was not found in this template/);
  });

  test("throws a validation exception when a Resource property references a Mapping that doesn't exist", () => {
    expect(() => {
      includeTestTemplate(stack, 'non-existent-mapping.json');
    }).toThrow(/Mapping used in FindInMap expression with name 'NonExistentMapping' was not found in the template/);
  });

  test("throws a validation exception when a Rule references a Parameter that isn't in the template", () => {
    expect(() => {
      includeTestTemplate(stack, 'rule-referencing-a-non-existent-parameter.json');
    }).toThrow(/Rule references parameter 'Subnets' which was not found in the template/);
  });

  test("throws a validation exception when Fn::Sub in string form uses a key that isn't in the template", () => {
    expect(() => {
      includeTestTemplate(stack, 'fn-sub-key-not-in-template-string.json');
    }).toThrow(/Element referenced in Fn::Sub expression with logical ID: 'AFakeResource' was not found in the template/);
  });

  test('throws a validation exception when Fn::Sub has an empty ${} reference', () => {
    expect(() => {
      includeTestTemplate(stack, 'fn-sub-${}-only.json');
    }).toThrow(/Element referenced in Fn::Sub expression with logical ID: '' was not found in the template/);
  });

  test("throws an exception for a template with a non-number string passed to a property with type 'number'", () => {
    includeTestTemplate(stack, 'alphabetical-string-passed-to-number.json');

    expect(() => {
      app.synth();
    }).toThrow(/"abc" should be a number/);
  });

  test('throws an exception for a template with a short-form Fn::GetAtt whose string argument does not contain a dot', () => {
    expect(() => {
      includeTestTemplate(stack, 'short-form-get-att-no-dot.yaml');
    }).toThrow(/Short-form Fn::GetAtt must contain a '.' in its string argument, got: 'Bucket1Arn'/);
  });

  /**
   * A->B
   * B->A
   * simplified version of cycle-in-resources.json, an example of cyclical references
   */
  test('by default does not accept a cycle between resources in the template', () => {
    expect(() => {
      includeTestTemplate(stack, 'cycle-in-resources.json');
    }).toThrow(/Found a cycle between resources in the template: Bucket1 depends on Bucket2 depends on Bucket1/);
  });

  /**
   * A->B
   * B->C
   * C->{D,A}
   * D->B
   * simplified version of multi-cycle-in-resources.json, an example of multiple cyclical references
   */
  test('by default does not accept multiple cycles between resources in the template', () => {
    expect(() => {
      includeTestTemplate(stack, 'multi-cycle-in-resources.json');
    }).toThrow(/Found a cycle between resources in the template: Bucket1 depends on Bucket2 depends on Bucket3 depends on Bucket4 depends on Bucket2/);
  });

  /**
   * A->B
   * B->{C,A}
   * C->A
   * simplified version of multi-cycle-multi-dest-in-resources.json, an example of multiple cyclical references that
   * include visiting the same destination more than once
   */
  test('by default does not accept multiple cycles and multiple destinations between resources in the template', () => {
    expect(() => {
      includeTestTemplate(stack, 'multi-cycle-multi-dest-in-resources.json');
    }).toThrow(/Found a cycle between resources in the template: Bucket1 depends on Bucket2 depends on Bucket3 depends on Bucket1/);
  });

  /**
   * A->B
   * B->A
   * simplified version of cycle-in-resources.json, an example of cyclical references
   */
  test('accepts a cycle between resources in the template if allowed', () => {
    includeTestTemplate(stack, 'cycle-in-resources.json', { allowCyclicalReferences: true });
    Template.fromStack(stack, { skipCyclicalDependenciesCheck: true }).templateMatches(
      {
        Resources: {
          Bucket2: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket1'] },
          Bucket1: {
            Type: 'AWS::S3::Bucket',
            Properties: { BucketName: { Ref: 'Bucket2' } },
          },
        },
      },
    );
  });

  /**
   * A->B
   * B->C
   * C->{D,A}
   * D->B
   * simplified version of multi-cycle-in-resources.json, an example of multiple cyclical references
   */
  test('accepts multiple cycles between resources in the template if allowed', () => {
    includeTestTemplate(stack, 'multi-cycle-in-resources.json', { allowCyclicalReferences: true });
    Template.fromStack(stack, { skipCyclicalDependenciesCheck: true }).templateMatches(
      {
        Resources: {
          Bucket2: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket3'] },
          Bucket3: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket4', 'Bucket1'] },
          Bucket4: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket2'] },
          Bucket1: {
            Type: 'AWS::S3::Bucket',
            Properties: { BucketName: { Ref: 'Bucket2' } },
          },
        },
      },
    );
  });

  /**
   * A->B
   * B->{C,A}
   * C->A
   * simplified version of multi-cycle-multi-dest-in-resources.json, an example of multiple cyclical references that
   * include visiting the same destination more than once
   */
  test('accepts multiple cycles and multiple destinations between resources in the template if allowed', () => {
    includeTestTemplate(stack, 'multi-cycle-multi-dest-in-resources.json', { allowCyclicalReferences: true });
    Template.fromStack(stack, { skipCyclicalDependenciesCheck: true }).templateMatches(
      {
        Resources: {
          Bucket2: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket3', 'Bucket1'] },
          Bucket3: { Type: 'AWS::S3::Bucket', DependsOn: ['Bucket1'] },
          Bucket1: {
            Type: 'AWS::S3::Bucket',
            Properties: { BucketName: { Ref: 'Bucket2' } },
          },
        },
      },
    );
  });

  test('throws an exception if Tags contains invalid intrinsics', () => {
    expect(() => {
      includeTestTemplate(stack, 'tags-with-invalid-intrinsics.json');
    }).toThrow(/expression does not exist in the template/);
  });

  test('non-leaf Intrinsics cannot be used in the top-level creation policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-create-policy.json');
    }).toThrow(/Cannot convert resource 'CreationPolicyIntrinsic' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'CreationPolicyIntrinsic' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the autoscaling creation policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-create-policy-autoscaling.json');
    }).toThrow(/Cannot convert resource 'AutoScalingCreationPolicyIntrinsic' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'AutoScalingCreationPolicyIntrinsic' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the create policy resource signal', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-create-policy-resource-signal.json');
    }).toThrow(/Cannot convert resource 'ResourceSignalIntrinsic' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'ResourceSignalIntrinsic' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the top-level update policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy.json');
    }).toThrow(/Cannot convert resource 'ASG' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'ASG' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the auto scaling rolling update update policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-autoscaling-rolling-update.json');
    }).toThrow(/Cannot convert resource 'ASG' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'ASG' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the auto scaling replacing update update policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-autoscaling-replacing-update.json');
    }).toThrow(/Cannot convert resource 'ASG' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'ASG' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the auto scaling scheduled action update policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-autoscaling-scheduled-action.json');
    }).toThrow(/Cannot convert resource 'ASG' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'ASG' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('Intrinsics cannot be used in the code deploy lambda alias update policy', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-code-deploy-lambda-alias-update.json');
    }).toThrow(/Cannot convert resource 'Alias' to CDK objects: it uses an intrinsic in a resource update or deletion policy to represent a non-primitive value. Specify 'Alias' in the 'dehydratedResources' prop to skip parsing this resource, while still including it in the output./);
  });

  test('FF toggles error checking', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, false);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-code-deploy-lambda-alias-update.json');
    }).not.toThrow();
  });

  test('FF disabled with dehydratedResources does not throw', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, false);
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-update-policy-code-deploy-lambda-alias-update.json', {
        dehydratedResources: ['Alias'],
      });
    }).not.toThrow();
  });

  test('dehydrated resources retain attributes with complex Intrinsics', () => {
    stack.node.setContext(cxapi.CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS, true);
    includeTestTemplate(stack, 'intrinsics-update-policy-code-deploy-lambda-alias-update.json', {
      dehydratedResources: ['Alias'],
    });

    expect(Template.fromStack(stack).hasResource('AWS::Lambda::Alias', {
      UpdatePolicy: {
        CodeDeployLambdaAliasUpdate: {
          'Fn::If': [
            'SomeCondition',
            {
              AfterAllowTrafficHook: 'SomeOtherHook',
              ApplicationName: 'SomeApp',
              BeforeAllowTrafficHook: 'SomeHook',
              DeploymentGroupName: 'SomeDeploymentGroup',
            },
            {
              AfterAllowTrafficHook: 'SomeOtherOtherHook',
              ApplicationName: 'SomeOtherApp',
              BeforeAllowTrafficHook: 'SomeOtherHook',
              DeploymentGroupName: 'SomeOtherDeploymentGroup',

            },
          ],
        },
      },
    }));
  });

  test('dehydrated resources retain all attributes', () => {
    includeTestTemplate(stack, 'resource-all-attributes.json', {
      dehydratedResources: ['Foo'],
    });

    expect(Template.fromStack(stack).hasResource('AWS::Foo::Bar', {
      Properties: { Blinky: 'Pinky' },
      Type: 'AWS::Foo::Bar',
      CreationPolicy: { Inky: 'Clyde' },
      DeletionPolicy: { DeletionPolicyKey: 'DeletionPolicyValue' },
      Metadata: { SomeKey: 'SomeValue' },
      Version: '1.2.3.4.5.6',
      UpdateReplacePolicy: { Oh: 'No' },
      Description: 'This resource does not match the spec, but it does have every possible attribute',
      UpdatePolicy: {
        Foo: 'Bar',
      },
    }));
  });

  test('synth-time validation does not run on dehydrated resources 1', () => {
    // synth-time validation fails if resource is hydrated
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-tags-resource-validation.json');
      Template.fromStack(stack);
    }).toThrow(`Resolution error: Supplied properties not correct for \"CfnLoadBalancerProps\"
  tags: element 1: {} should have a 'key' and a 'value' property.`);

    Validations.of(app).acknowledge(
      { id: 'CloudFormation-Validate::F3002', reason: 'This test is purposely creating invalid templates' },
      { id: 'CloudFormation-Validate::F3003', reason: 'This test is purposely creating invalid templates' },
    );
  });

  test('synth-time validation does not run on dehydrated resources 2', () => {
    // synth-time validation not run if resource is dehydrated
    includeTestTemplate(stack, 'intrinsics-tags-resource-validation.json', {
      dehydratedResources: ['MyLoadBalancer'],
    });

    expect(Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Properties: {
        Tags: [
          {
            Key: 'Name',
            Value: 'MyLoadBalancer',
          },
          {
            data: [
              'IsExtraTag',
              {
                Key: 'Name2',
                Value: 'MyLoadBalancer2',
              },
              {
                data: 'AWS::NoValue',
                type: 'Ref',
                isCfnFunction: true,
              },
            ],
            type: 'Fn::If',
            isCfnFunction: true,
          },
        ],
      },
    }));
  });

  test('throws on dehydrated resources not present in the template', () => {
    expect(() => {
      includeTestTemplate(stack, 'intrinsics-tags-resource-validation.json', {
        dehydratedResources: ['ResourceNotExistingHere'],
      });
    }).toThrow(/Logical ID 'ResourceNotExistingHere' was specified in 'dehydratedResources', but does not belong to a resource in the template./);
  });
});

interface IncludeTestTemplateProps {
  /** @default false */
  readonly allowCyclicalReferences?: boolean;

  /** @default none */
  readonly dehydratedResources?: string[];
}

function includeTestTemplate(scope: constructs.Construct, testTemplate: string, props: IncludeTestTemplateProps = {}): inc.CfnInclude {
  return new inc.CfnInclude(scope, 'MyScope', {
    templateFile: _testTemplateFilePath(testTemplate),
    allowCyclicalReferences: props.allowCyclicalReferences,
    dehydratedResources: props.dehydratedResources,
  });
}

function _testTemplateFilePath(testTemplate: string) {
  return path.join(__dirname, 'test-templates', 'invalid', testTemplate);
}
