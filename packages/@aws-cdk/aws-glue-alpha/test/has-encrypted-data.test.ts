import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import * as glue from '../lib';

const columns = [{ name: 'col', type: glue.Schema.STRING }];

function newDatabase(stack: cdk.Stack): glue.Database {
  return new glue.Database(stack, 'Database');
}

function newConnection(stack: cdk.Stack): glue.Connection {
  return new glue.Connection(stack, 'Connection', {
    connectionName: 'my_connection',
    type: glue.ConnectionType.JDBC,
    properties: {
      JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
      USERNAME: 'username',
      PASSWORD: 'password',
    },
  });
}

describe('S3Table has_encrypted_data', () => {
  test('defaults to true', () => {
    const stack = new cdk.Stack();
    new glue.S3Table(stack, 'Table', {
      database: newDatabase(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ has_encrypted_data: true }),
      },
    });
  });

  test('can be set to false', () => {
    const stack = new cdk.Stack();
    new glue.S3Table(stack, 'Table', {
      database: newDatabase(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      hasEncryptedData: false,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ has_encrypted_data: false }),
      },
    });
  });

  test('typed value cannot be silently overridden by parameters', () => {
    const stack = new cdk.Stack();

    expect(() => new glue.S3Table(stack, 'Table', {
      database: newDatabase(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      hasEncryptedData: true,
      parameters: { has_encrypted_data: 'false' },
    })).toThrow(/`has_encrypted_data` table parameter is managed by the `hasEncryptedData` property/);
  });

  test('a matching value in parameters is tolerated', () => {
    const stack = new cdk.Stack();
    new glue.S3Table(stack, 'Table', {
      database: newDatabase(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      hasEncryptedData: false,
      parameters: { has_encrypted_data: 'false' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ has_encrypted_data: false }),
      },
    });
  });

  test('a tokenized value in parameters is rejected', () => {
    const stack = new cdk.Stack();
    const param = new cdk.CfnParameter(stack, 'Flag', { type: 'String' });

    expect(() => new glue.S3Table(stack, 'Table', {
      database: newDatabase(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      parameters: { has_encrypted_data: param.valueAsString },
    })).toThrow(/`has_encrypted_data` table parameter is managed by the `hasEncryptedData` property/);
  });
});

describe('ExternalTable has_encrypted_data', () => {
  const externalDataLocation = 'default_db.public.test';

  test('defaults to true', () => {
    const stack = new cdk.Stack();
    new glue.ExternalTable(stack, 'Table', {
      database: newDatabase(stack),
      connection: newConnection(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      externalDataLocation,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ has_encrypted_data: true }),
      },
    });
  });

  test('can be set to false', () => {
    const stack = new cdk.Stack();
    new glue.ExternalTable(stack, 'Table', {
      database: newDatabase(stack),
      connection: newConnection(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      externalDataLocation,
      hasEncryptedData: false,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ has_encrypted_data: false }),
      },
    });
  });

  test('typed value cannot be silently overridden by parameters', () => {
    const stack = new cdk.Stack();

    expect(() => new glue.ExternalTable(stack, 'Table', {
      database: newDatabase(stack),
      connection: newConnection(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      externalDataLocation,
      parameters: { has_encrypted_data: 'false' },
    })).toThrow(/`has_encrypted_data` table parameter is managed by the `hasEncryptedData` property/);
  });

  test('user-supplied parameters still flow through', () => {
    const stack = new cdk.Stack();
    new glue.ExternalTable(stack, 'Table', {
      database: newDatabase(stack),
      connection: newConnection(stack),
      columns,
      dataFormat: glue.DataFormat.JSON,
      externalDataLocation,
      parameters: { custom: 'value' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Glue::Table', {
      TableInput: {
        Parameters: Match.objectLike({ custom: 'value', has_encrypted_data: true }),
      },
    });
  });
});
