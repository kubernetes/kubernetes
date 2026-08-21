import * as cdk from 'aws-cdk-lib';
import { Annotations, Match, Template } from 'aws-cdk-lib/assertions';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as glue from '../lib';

test('a connection with connection properties', () => {
  const stack = new cdk.Stack();
  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.JDBC,
    properties: {
      JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
      USERNAME: 'username',
      PASSWORD: 'password',
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionProperties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
        USERNAME: 'username',
        PASSWORD: 'password',
      },
      ConnectionType: 'JDBC',
    },
  });
});

test('a connection with a subnet and security group', () => {
  const stack = new cdk.Stack();
  const subnet = ec2.Subnet.fromSubnetAttributes(stack, 'subnet', {
    subnetId: 'subnetId',
    availabilityZone: 'azId',
  });
  const securityGroup = ec2.SecurityGroup.fromSecurityGroupId(stack, 'securityGroup', 'sgId');
  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    securityGroups: [securityGroup],
    subnet,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      PhysicalConnectionRequirements: {
        AvailabilityZone: 'azId',
        SubnetId: 'subnetId',
        SecurityGroupIdList: ['sgId'],
      },
    },
  });
});

test('a connection with a vpc selects a subnet automatically', () => {
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'Vpc');
  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    vpc,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      PhysicalConnectionRequirements: {
        AvailabilityZone: { 'Fn::Select': [0, { 'Fn::GetAZs': '' }] },
        SubnetId: { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
      },
    },
  });
});

test('a connection with a vpc and explicit subnet selection', () => {
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'Vpc');
  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    vpc,
    vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      PhysicalConnectionRequirements: {
        SubnetId: { Ref: 'VpcPublicSubnet1Subnet5C2D37C4' },
      },
    },
  });
});

test('fails when both subnet and vpc are specified', () => {
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const subnet = ec2.Subnet.fromSubnetAttributes(stack, 'subnet', {
    subnetId: 'subnetId',
    availabilityZone: 'azId',
  });

  expect(() => new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    subnet,
    vpc,
  })).toThrow(/cannot specify both `subnet` and `vpc`/);
});

test('fails when vpcSubnets is specified without vpc', () => {
  const stack = new cdk.Stack();

  expect(() => new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
  })).toThrow(/`vpcSubnets` can only be specified together with `vpc`/);
});

test('fails with a clear message when vpcSubnets selects no subnets', () => {
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'Vpc');

  expect(() => new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    vpc,
    vpcSubnets: { subnets: [] },
  })).toThrow(/`vpcSubnets` selected no subnets from the provided `vpc`/);
});

test('does not fail on an empty selection while the vpc lookup is pending', () => {
  const stack = new cdk.Stack(undefined, 'Stack', { env: { account: '1234', region: 'us-east-1' } });
  // Before the context lookup resolves, `selectSubnets` returns an empty set
  // with `isPendingLookup: true`; this is expected and must not throw.
  const vpc = ec2.Vpc.fromLookup(stack, 'Vpc', { vpcId: 'vpc-1234' });

  expect(() => new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    vpc,
    vpcSubnets: { subnetGroupName: 'DoesNotExist' },
  })).not.toThrow();
});

test('a connection with a name and description', () => {
  const stack = new cdk.Stack();
  new glue.Connection(stack, 'Connection', {
    connectionName: 'name',
    description: 'description',
    type: glue.ConnectionType.NETWORK,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      Name: 'name',
      Description: 'description',
    },
  });
});

test('a connection with a custom type', () => {
  const stack = new cdk.Stack();
  new glue.Connection(stack, 'Connection', {
    connectionName: 'name',
    description: 'description',
    type: new glue.ConnectionType('CUSTOM_TYPE'),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionType: 'CUSTOM_TYPE',
      Name: 'name',
      Description: 'description',
    },
  });
});

test('a connection with match criteria', () => {
  const stack = new cdk.Stack();
  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
    matchCriteria: ['c1', 'c2'],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      MatchCriteria: ['c1', 'c2'],
    },
  });
});

test('addProperty', () => {
  const stack = new cdk.Stack();
  const connection = new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.NETWORK,
  });
  connection.addProperty('SomeKey', 'SomeValue');

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    CatalogId: {
      Ref: 'AWS::AccountId',
    },
    ConnectionInput: {
      ConnectionType: 'NETWORK',
      ConnectionProperties: {
        SomeKey: 'SomeValue',
      },
    },
  });
});

test('a secret is wired to the SECRET_ID connection property', () => {
  const stack = new cdk.Stack();
  const secret = new secretsmanager.Secret(stack, 'Secret');

  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.JDBC,
    properties: { JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection' },
    secret,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::Connection', {
    ConnectionInput: {
      ConnectionProperties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
        SECRET_ID: stack.resolve(secret.secretRef.secretId),
      },
    },
  });
});

test('setting both `secret` and a SECRET_ID property throws', () => {
  const stack = new cdk.Stack();
  const secret = new secretsmanager.Secret(stack, 'Secret');

  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.JDBC,
    secret,
    properties: { SECRET_ID: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:other' },
  });

  expect(() => Template.fromStack(stack)).toThrow('cannot set both `secret` and a `SECRET_ID` connection property');
});

test('setting `secret` and adding a SECRET_ID via addProperty throws', () => {
  const stack = new cdk.Stack();
  const secret = new secretsmanager.Secret(stack, 'Secret');

  const connection = new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.JDBC,
    secret,
  });
  connection.addProperty('SECRET_ID', 'arn:aws:secretsmanager:us-east-1:123456789012:secret:other');

  expect(() => Template.fromStack(stack)).toThrow('cannot set both `secret` and a `SECRET_ID` connection property');
});

test('referencing a secret does not trigger the plaintext-secret warning', () => {
  const stack = new cdk.Stack();
  // An imported secret has a literal ARN; the SECRET_ID it populates is a
  // reference, not a credential, so it must not be flagged.
  const secret = secretsmanager.Secret.fromSecretCompleteArn(
    stack, 'Secret', 'arn:aws:secretsmanager:us-east-1:123456789012:secret:creds-AbCdEf');

  new glue.Connection(stack, 'Connection', {
    type: glue.ConnectionType.JDBC,
    secret,
  });

  Annotations.fromStack(stack).hasNoWarning(
    '/Default/Connection',
    Match.stringLikeRegexp('.*plaintext secret.*'));
});

test('fromConnectionName', () => {
  const connectionName = 'name';
  const stack = new cdk.Stack();
  const connection = glue.Connection.fromConnectionName(stack, 'ImportedConnection', connectionName);

  expect(connection.connectionName).toEqual(connectionName);
  expect(connection.connectionArn).toEqual(stack.formatArn({
    service: 'glue',
    resource: 'connection',
    resourceName: connectionName,
  }));
});

test('fromConnectionArn', () => {
  const connectionArn = 'arn:aws:glue:region:account-id:connection/name';
  const stack = new cdk.Stack();
  const connection = glue.Connection.fromConnectionArn(stack, 'ImportedConnection', connectionArn);

  expect(connection.connectionName).toEqual('name');
  expect(connection.connectionArn).toEqual(connectionArn);
});
