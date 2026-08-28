import { Template } from '../../assertions';
import * as cdk from '../../core';
import { DatabaseClusterEngine, ParameterGroup } from '../lib';

describe('parameter group', () => {
  test("does not create a parameter group if it wasn't bound to a cluster or instance", () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key: 'value',
      },
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBParameterGroup', 0);
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBClusterParameterGroup', 0);
  });

  test('create instance parameter group explicitly with forInstance() static method', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    ParameterGroup.forInstance(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      name: 'name',
      parameters: {
        key: 'value',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBParameterGroup', {
      DBParameterGroupName: 'name',
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key: 'value',
      },
    });
  });

  test('create cluster parameter group explicitly with forCluster() static method', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    ParameterGroup.forCluster(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      name: 'name',
      parameters: {
        key: 'value',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBClusterParameterGroup', {
      DBClusterParameterGroupName: 'name',
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key: 'value',
      },
    });
  });

  test('can create both instance and cluster parameter groups', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    ParameterGroup.forInstance(stack, 'InstanceParams', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'instance desc',
      parameters: {
        key: 'value',
      },
    });
    ParameterGroup.forCluster(stack, 'ClusterParams', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'cluster desc',
      parameters: {
        key: 'value',
      },
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBParameterGroup', 1);
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBClusterParameterGroup', 1);
  });

  test('create a parameter group when bound to an instance', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      name: 'name',
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToInstance({});

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBParameterGroup', {
      DBParameterGroupName: 'name',
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key: 'value',
      },
    });
  });

  test('create a parameter group when bound to a cluster', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      name: 'name',
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToCluster({});

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBClusterParameterGroup', {
      DBClusterParameterGroupName: 'name',
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key: 'value',
      },
    });
  });

  test('creates 2 parameter groups when bound to a cluster and an instance', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToCluster({});
    parameterGroup.bindToInstance({});

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBParameterGroup', 1);
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBClusterParameterGroup', 1);
  });

  test('creates 2 parameter groups when bound to a cluster and an instance and they have the correct removal policy', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToCluster({});
    parameterGroup.bindToInstance({});

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBParameterGroup', 1);
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBClusterParameterGroup', 1);

    const instanceParameterGroup = Template.fromStack(stack).findResources('AWS::RDS::DBParameterGroup');
    const clusterParameterGroup = Template.fromStack(stack).findResources('AWS::RDS::DBClusterParameterGroup');

    expect(Object.values(instanceParameterGroup)[0].DeletionPolicy).toEqual('Retain');
    expect(Object.values(clusterParameterGroup)[0].DeletionPolicy).toEqual('Retain');
  });

  test('removal policy is applied when using forInstance() and forCluster() methods', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    ParameterGroup.forInstance(stack, 'InstanceParams', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      parameters: {
        key: 'value',
      },
    });
    ParameterGroup.forCluster(stack, 'ClusterParams', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      parameters: {
        key: 'value',
      },
    });

    // THEN
    const instanceParameterGroup = Template.fromStack(stack).findResources('AWS::RDS::DBParameterGroup');
    const clusterParameterGroup = Template.fromStack(stack).findResources('AWS::RDS::DBClusterParameterGroup');

    expect(Object.values(instanceParameterGroup)[0].DeletionPolicy).toEqual('Retain');
    expect(Object.values(clusterParameterGroup)[0].DeletionPolicy).toEqual('Retain');
  });

  test('addParameter() works with forInstance() static method', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = ParameterGroup.forInstance(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key1: 'value1',
      },
    });
    parameterGroup.addParameter('key2', 'value2');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBParameterGroup', {
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key1: 'value1',
        key2: 'value2',
      },
    });
  });

  test('Add an additional parameter to an existing parameter group with bindToCluster()', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const clusterParameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key1: 'value1',
      },
    });
    clusterParameterGroup.bindToCluster({});
    clusterParameterGroup.addParameter('key2', 'value2');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::RDS::DBClusterParameterGroup', {
      Description: 'desc',
      Family: 'aurora-mysql5.7',
      Parameters: {
        key1: 'value1',
        key2: 'value2',
      },
    });
  });

  test('backward compatibility: bindToInstance after forInstance() returns existing resource', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = ParameterGroup.forInstance(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToInstance({});

    // THEN - should only have 1 resource, not 2
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBParameterGroup', 1);
  });

  test('backward compatibility: bindToCluster after forCluster() returns existing resource', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const parameterGroup = ParameterGroup.forCluster(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
      description: 'desc',
      parameters: {
        key: 'value',
      },
    });
    parameterGroup.bindToCluster({});

    // THEN - should only have 1 resource, not 2
    Template.fromStack(stack).resourceCountIs('AWS::RDS::DBClusterParameterGroup', 1);
  });

  test('dbParameterGroupRef of an imported group formats a colon-separated parameter group ARN', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { account: '123456789012', region: 'us-east-1' } });

    // WHEN
    const parameterGroup = ParameterGroup.fromParameterGroupName(stack, 'Params', 'my-group');

    // THEN
    expect(stack.resolve(parameterGroup.dbParameterGroupRef.dbParameterGroupArn)).toEqual({
      'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':rds:us-east-1:123456789012:pg:my-group']],
    });
  });

  test('fails to read dbParameterGroupArn of a group not bound to a DB instance', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
    });

    // THEN
    expect(() => parameterGroup.dbParameterGroupRef.dbParameterGroupArn).toThrow(
      'this ParameterGroup is not bound to a DB instance, so it has no DB parameter group ARN - bind it with bindToInstance() or create it with ParameterGroup.forInstance()',
    );
  });

  test('dbParameterGroupRef read before binding still resolves once the group is bound', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const parameterGroup = new ParameterGroup(stack, 'Params', {
      engine: DatabaseClusterEngine.AURORA_MYSQL,
    });
    const ref = parameterGroup.dbParameterGroupRef;

    // WHEN
    parameterGroup.bindToInstance({});

    // THEN
    expect(stack.resolve(ref.dbParameterGroupArn)).toEqual({
      'Fn::GetAtt': ['ParamsA8366201', 'DBParameterGroupArn'],
    });
    expect(stack.resolve(ref.dbParameterGroupName)).toEqual({ Ref: 'ParamsA8366201' });
  });
});
