import { Template } from '../../assertions';
import * as iam from '../../aws-iam';
import { App, Stack } from '../../core';
import { OpenSearchAccessPolicy } from '../lib/opensearch-access-policy';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'Stack', {
    env: { account: '1234', region: 'testregion' },
  });
});

test('minimal example renders correctly', () => {
  const domainArn = 'arn:aws:es:us-east-1:123456789012:domain/test-domain';

  new OpenSearchAccessPolicy(stack, 'OpenSearchAccessPolicy', {
    domainName: 'TestDomain',
    domainArn: domainArn,
    accessPolicies: [new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['es:ESHttp*'],
      principals: [new iam.AnyPrincipal()],
      resources: [domainArn],

    })],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::OpenSearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'es:UpdateDomainConfig',
        Effect: 'Allow',
        Resource: domainArn,
      }],
    },
  });
});

test('support access policy added inline and later', () => {
  const opensearchAccessPolicy = new OpenSearchAccessPolicy(stack, 'OpenSearchAccessPolicy', {
    domainName: 'TestDomain',
    domainArn: 'arn:aws:es:us-east-1:123456789012:domain/test-domain',
    accessPolicies: [
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['es:ESHttp*'],
        principals: [new iam.AnyPrincipal()],
        resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],
      }),
    ],
  });
  opensearchAccessPolicy.addAccessPolicies(
    new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['*'],
      principals: [new iam.AnyPrincipal()],
      resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],
    }),
  );

  Template.fromStack(stack).hasResourceProperties('Custom::OpenSearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"},{"Action":"*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"},{"Action":"*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
});

test('handling of verbose output via flag explicitly set', () => {
  const domainArn = 'arn:aws:es:us-east-1:123456789012:domain/test-domain';

  new OpenSearchAccessPolicy(stack, 'OpenSearchAccessPolicy', {
    domainName: 'TestDomain',
    domainArn: 'arn:aws:es:us-east-1:123456789012:domain/test-domain',
    accessPolicies: [
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['es:ESHttp*'],
        principals: [new iam.AnyPrincipal()],
        resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],
      }),
    ],
    verboseOutput: true,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::OpenSearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'es:UpdateDomainConfig',
        Effect: 'Allow',
        Resource: domainArn,
      }],
    },
  });
});

test('handling of less verbose output', () => {
  const domainArn = 'arn:aws:es:us-east-1:123456789012:domain/test-domain';

  new OpenSearchAccessPolicy(stack, 'OpenSearchAccessPolicy', {
    domainName: 'TestDomain',
    domainArn: 'arn:aws:es:us-east-1:123456789012:domain/test-domain',
    accessPolicies: [
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['es:ESHttp*'],
        principals: [new iam.AnyPrincipal()],
        resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],
      }),
    ],
    verboseOutput: false,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::OpenSearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies.Status.State', 'DomainConfig.AccessPolicies.Status.UpdateVersion'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateDomainConfig',
      service: 'OpenSearch',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.AccessPolicies.Status.State', 'DomainConfig.AccessPolicies.Status.UpdateVersion'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'es:UpdateDomainConfig',
        Effect: 'Allow',
        Resource: domainArn,
      }],
    },
  });
});

