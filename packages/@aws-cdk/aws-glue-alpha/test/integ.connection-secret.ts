import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as glue from '../lib';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-glue-connection-secret');

// The credentials live in Secrets Manager; the connection references the secret
// through its SECRET_ID property, so the secret value never enters the template.
const secret = new secretsmanager.Secret(stack, 'ConnectionSecret');

const connection = new glue.Connection(stack, 'JdbcConnection', {
  type: glue.ConnectionType.JDBC,
  secret,
  properties: {
    JDBC_CONNECTION_URL: 'jdbc:mysql://mydb.example.com:3306/mydatabase',
    JDBC_ENFORCE_SSL: 'false',
  },
});

const test = new integ.IntegTest(app, 'ConnectionSecretIntegTest', {
  testCases: [stack],
});

// Verify the deployed connection actually has the secret associated: its
// SECRET_ID property must resolve to the secret's reference.
test.assertions.awsApiCall('Glue', 'getConnection', {
  Name: connection.connectionName,
}).expect(integ.ExpectedResult.objectLike({
  Connection: {
    ConnectionProperties: {
      SECRET_ID: secret.secretRef.secretId,
    },
  },
}));

app.synth();
