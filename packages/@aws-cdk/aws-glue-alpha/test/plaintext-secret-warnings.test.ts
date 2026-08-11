import * as cdk from 'aws-cdk-lib';
import { Annotations, Match } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as glue from '../lib';

const WARNING = /appears to hold a plaintext secret|appear to hold a plaintext secret/;

function newRole(stack: cdk.Stack): iam.Role {
  return new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
  });
}

describe('Connection plaintext secret warnings', () => {
  test('warns when a property key looks like a secret with a literal value', () => {
    const stack = new cdk.Stack();
    new glue.Connection(stack, 'Connection', {
      type: glue.ConnectionType.JDBC,
      properties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
        USERNAME: 'username',
        PASSWORD: 'plaintext-password',
      },
    });

    Annotations.fromStack(stack).hasWarning(
      '/Default/Connection',
      Match.stringLikeRegexp('.*plaintext secret.*PASSWORD.*plaintextConnectionSecret.*'),
    );
  });

  test('does not warn when no key looks like a secret', () => {
    const stack = new cdk.Stack();
    new glue.Connection(stack, 'Connection', {
      type: glue.ConnectionType.JDBC,
      properties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
        HOST: 'server',
      },
    });

    Annotations.fromStack(stack).hasNoWarning('/Default/Connection', Match.stringLikeRegexp(WARNING.source));
  });

  test('does not warn when a secret-like key holds a SecretValue token', () => {
    const stack = new cdk.Stack();
    const secret = cdk.SecretValue.unsafePlainText('x'); // token, resolves to {{resolve:...}} in real use
    new glue.Connection(stack, 'Connection', {
      type: glue.ConnectionType.JDBC,
      properties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
        SECRET_ID: cdk.Token.asString(secret),
      },
    });

    Annotations.fromStack(stack).hasNoWarning('/Default/Connection', Match.stringLikeRegexp(WARNING.source));
  });

  test('warns for a secret added via addProperty', () => {
    const stack = new cdk.Stack();
    const connection = new glue.Connection(stack, 'Connection', {
      type: glue.ConnectionType.JDBC,
      properties: {
        JDBC_CONNECTION_URL: 'jdbc:server://server:443/connection',
      },
    });
    connection.addProperty('SECRET_TOKEN', 'plaintext-token');

    Annotations.fromStack(stack).hasWarning(
      '/Default/Connection',
      Match.stringLikeRegexp('.*plaintext secret.*SECRET_TOKEN.*plaintextConnectionSecret.*'),
    );
  });
});

describe('Job defaultArguments plaintext secret warnings', () => {
  test('warns when a defaultArguments key looks like a secret with a literal value', () => {
    const stack = new cdk.Stack();
    new glue.PySparkEtlJob(stack, 'Job', {
      role: newRole(stack),
      script: glue.Code.fromAsset(__filename),
      defaultArguments: {
        '--api_key': 'plaintext-key',
      },
    });

    Annotations.fromStack(stack).hasWarning(
      '/Default/Job',
      Match.stringLikeRegexp('.*plaintext secret.*api_key.*plaintextJobArgumentSecret.*'),
    );
  });

  test('does not warn for ordinary defaultArguments', () => {
    const stack = new cdk.Stack();
    new glue.PySparkEtlJob(stack, 'Job', {
      role: newRole(stack),
      script: glue.Code.fromAsset(__filename),
      defaultArguments: {
        '--extra-files': 's3://bucket/file',
      },
    });

    Annotations.fromStack(stack).hasNoWarning('/Default/Job', Match.stringLikeRegexp(WARNING.source));
  });
});
