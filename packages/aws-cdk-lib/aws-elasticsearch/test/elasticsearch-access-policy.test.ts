import { Template } from '../../assertions';
import * as iam from '../../aws-iam';
import { App, Stack } from '../../core';
import { ElasticsearchAccessPolicy } from '../lib/elasticsearch-access-policy';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'Stack', {
    env: { account: '1234', region: 'testregion' },
  });
});

test('minimal example renders correctly', () => {
  new ElasticsearchAccessPolicy(stack, 'ElasticsearchAccessPolicy', {
    domainName: 'TestDomain',
    domainArn: 'arn:aws:es:us-east-1:123456789012:domain/test-domain',
    accessPolicies: [new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['es:ESHttp*'],
      principals: [new iam.AnyPrincipal()],
      resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],

    })],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::ElasticsearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateElasticsearchDomainConfig',
      service: 'ES',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.ElasticsearchClusterConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateElasticsearchDomainConfig',
      service: 'ES',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.ElasticsearchClusterConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
});

test('support access policy added inline and later', () => {
  const elasticsearchAccessPolicy = new ElasticsearchAccessPolicy(stack, 'ElasticsearchAccessPolicy', {
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
  elasticsearchAccessPolicy.addAccessPolicies(
    new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['*'],
      principals: [new iam.AnyPrincipal()],
      resources: ['arn:aws:es:us-east-1:123456789012:domain/test-domain'],
    }),
  );

  Template.fromStack(stack).hasResourceProperties('Custom::ElasticsearchAccessPolicy', {
    ServiceToken: {
      'Fn::GetAtt': [
        'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
        'Arn',
      ],
    },
    Create: JSON.stringify({
      action: 'updateElasticsearchDomainConfig',
      service: 'ES',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"},{"Action":"*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.ElasticsearchClusterConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
    Update: JSON.stringify({
      action: 'updateElasticsearchDomainConfig',
      service: 'ES',
      parameters: {
        DomainName: 'TestDomain',
        AccessPolicies: '{"Statement":[{"Action":"es:ESHttp*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"},{"Action":"*","Effect":"Allow","Principal":{"AWS":"*"},"Resource":"arn:aws:es:us-east-1:123456789012:domain/test-domain"}],"Version":"2012-10-17"}',
      },
      outputPaths: ['DomainConfig.ElasticsearchClusterConfig.AccessPolicies'],
      physicalResourceId: { id: 'TestDomainAccessPolicy' },
    }),
  });
});
