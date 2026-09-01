import { Match, Template } from '../../assertions';
import * as acm from '../../aws-certificatemanager';
import { Bucket } from '../../aws-s3';
import { Fn, Stack, Validations } from '../../core';
import * as apigw from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('domains', () => {
  test('can define either an EDGE or REGIONAL domain name', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    const regionalDomain = new apigw.DomainName(stack, 'my-domain', {
      domainName: 'example.com/region',
      certificate: cert,
      endpointType: apigw.EndpointType.REGIONAL,
    });

    const edgeDomain = new apigw.DomainName(stack, 'your-domain', {
      domainName: 'example.com/edge',
      certificate: cert,
      endpointType: apigw.EndpointType.EDGE,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'example.com/region',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': { 'Ref': 'Cert5C9FAEC1' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'example.com/edge',
      'EndpointConfiguration': { 'Types': ['EDGE'] },
      'CertificateArn': { 'Ref': 'Cert5C9FAEC1' },
    });

    expect(stack.resolve(regionalDomain.domainNameAliasDomainName)).toEqual({ 'Fn::GetAtt': ['mydomain592C948B', 'RegionalDomainName'] });
    expect(stack.resolve(regionalDomain.domainNameAliasHostedZoneId)).toEqual({ 'Fn::GetAtt': ['mydomain592C948B', 'RegionalHostedZoneId'] });
    expect(stack.resolve(edgeDomain.domainNameAliasDomainName)).toEqual({ 'Fn::GetAtt': ['yourdomain5FE30C81', 'DistributionDomainName'] });
    expect(stack.resolve(edgeDomain.domainNameAliasHostedZoneId)).toEqual({ 'Fn::GetAtt': ['yourdomain5FE30C81', 'DistributionHostedZoneId'] });
  });

  test('default endpoint type is REGIONAL', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'my-domain', {
      domainName: 'example.com',
      certificate: cert,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': { 'Ref': 'Cert5C9FAEC1' },
    });
  });

  test('accepts different security policies', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'my-domain', {
      domainName: 'old.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS_1_0,
    });

    new apigw.DomainName(stack, 'your-domain', {
      domainName: 'new.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS_1_2,
    });

    new apigw.DomainName(stack, 'default-domain', {
      domainName: 'default.example.com',
      certificate: cert,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'old.example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': { 'Ref': 'Cert5C9FAEC1' },
      'SecurityPolicy': 'TLS_1_0',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'new.example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': { 'Ref': 'Cert5C9FAEC1' },
      'SecurityPolicy': 'TLS_1_2',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'default.example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': { 'Ref': 'Cert5C9FAEC1' },
      'SecurityPolicy': Match.absent(),
    });
  });

  test('accepts TLS 1.3 security policies', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'tls13-domain', {
      domainName: 'tls13.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
      endpointAccessMode: apigw.EndpointAccessMode.STRICT,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'tls13.example.com',
      'SecurityPolicy': 'SecurityPolicy_TLS13_1_3_2025_09',
      'EndpointAccessMode': 'STRICT',
    });
  });

  test('allows TLS 1.3 for multi-level base paths', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });
    const api = new apigw.RestApi(stack, 'api');
    api.root.addMethod('GET');

    // WHEN - Should not throw error
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'api.example.com',
        certificate: cert,
        securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
        endpointAccessMode: apigw.EndpointAccessMode.STRICT,
        mapping: api,
        basePath: 'v1/users',
      });
    }).not.toThrow();
  });

  test('allows multi-level base paths without security policy (defaults to TLS 1.2)', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });
    const api = new apigw.RestApi(stack, 'api');
    api.root.addMethod('GET');

    // WHEN - Should not throw error, default is TLS 1.2
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'api.example.com',
        certificate: cert,
        mapping: api,
        basePath: 'v1/users',
      });
    }).not.toThrow();
  });

  test('accepts TLS 1.3 with post-quantum cryptography security policy', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'pq-domain', {
      domainName: 'pq.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS13_1_2_PFS_PQ_2025_09,
      endpointAccessMode: apigw.EndpointAccessMode.STRICT,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'pq.example.com',
      'SecurityPolicy': 'SecurityPolicy_TLS13_1_2_PFS_PQ_2025_09',
      'EndpointAccessMode': 'STRICT',
    });
  });

  test('accepts endpointAccessMode property', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'tls13-strict-domain', {
      domainName: 'strict.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
      endpointAccessMode: apigw.EndpointAccessMode.STRICT,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'strict.example.com',
      'SecurityPolicy': 'SecurityPolicy_TLS13_1_3_2025_09',
      'EndpointAccessMode': 'STRICT',
    });
  });

  test('throws if enhanced security policy is used without endpointAccessMode', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // THEN
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'example.com',
        certificate: cert,
        securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
        // Missing endpointAccessMode
      });
    }).toThrow(/Enhanced security policies require endpointAccessMode to be specified/);
  });

  test('throws if endpointAccessMode is set with a legacy security policy', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // THEN
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'example.com',
        certificate: cert,
        securityPolicy: apigw.SecurityPolicy.TLS_1_2,
        endpointAccessMode: apigw.EndpointAccessMode.STRICT,
      });
    }).toThrow(/endpointAccessMode is not supported for legacy security policies/);
  });

  test('throws if mTLS is used with enhanced security policy', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });
    const bucket = Bucket.fromBucketName(stack, 'testBucket', 'example-bucket');

    // THEN
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'mtls.example.com',
        certificate: cert,
        securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
        mtls: {
          bucket,
          key: 'someca.pem',
        },
      });
    }).toThrow(/Mutual TLS \(mTLS\) cannot be enabled on a domain name that uses an enhanced security policy/);
  });

  test('allows mTLS with legacy security policy TLS_1_2', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });
    const bucket = Bucket.fromBucketName(stack, 'testBucket', 'example-bucket');

    // WHEN - Should not throw error
    new apigw.DomainName(stack, 'domain', {
      domainName: 'mtls.example.com',
      certificate: cert,
      securityPolicy: apigw.SecurityPolicy.TLS_1_2,
      mtls: {
        bucket,
        key: 'someca.pem',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'mtls.example.com',
      'SecurityPolicy': 'TLS_1_2',
      'MutualTlsAuthentication': { 'TruststoreUri': 's3://example-bucket/someca.pem' },
    });
  });

  test('throws if regional-only security policy is used with EDGE endpoint', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // THEN
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'example.com',
        certificate: cert,
        endpointType: apigw.EndpointType.EDGE,
        securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
        endpointAccessMode: apigw.EndpointAccessMode.STRICT,
      });
    }).toThrow(/Security policy SecurityPolicy_TLS13_1_3_2025_09 is not supported for edge-optimized endpoints/);
  });

  test('throws if edge-only security policy is used with REGIONAL endpoint', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // THEN
    expect(() => {
      new apigw.DomainName(stack, 'domain', {
        domainName: 'example.com',
        certificate: cert,
        endpointType: apigw.EndpointType.REGIONAL,
        securityPolicy: apigw.SecurityPolicy.TLS13_2025_EDGE,
        endpointAccessMode: apigw.EndpointAccessMode.STRICT,
      });
    }).toThrow(/Security policy SecurityPolicy_TLS13_2025_EDGE is only supported for edge-optimized endpoints/);
  });

  test('allows TLS 1.3 edge policy with EDGE endpoint', () => {
    // GIVEN
    const stack = new Stack();
    const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

    // WHEN
    new apigw.DomainName(stack, 'domain', {
      domainName: 'edge.example.com',
      certificate: cert,
      endpointType: apigw.EndpointType.EDGE,
      securityPolicy: apigw.SecurityPolicy.TLS13_2025_EDGE,
      endpointAccessMode: apigw.EndpointAccessMode.STRICT,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'edge.example.com',
      'SecurityPolicy': 'SecurityPolicy_TLS13_2025_EDGE',
      'EndpointAccessMode': 'STRICT',
      'EndpointConfiguration': { 'Types': ['EDGE'] },
    });
  });

  test('"mapping" can be used to automatically map this domain to the deployment stage of an API', () => {
    // GIVEN
    const stack = new Stack();
    const api = new apigw.RestApi(stack, 'api');
    api.root.addMethod('GET');

    // WHEN
    new apigw.DomainName(stack, 'Domain', {
      domainName: 'foo.com',
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      endpointType: apigw.EndpointType.EDGE,
      mapping: api,
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'Domain66AC69E0',
      },
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
      'Stage': {
        'Ref': 'apiDeploymentStageprod896C8101',
      },
    });
  });

  describe('multi-level mapping', () => {
    test('can add a multi-level path', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      // WHEN
      new apigw.DomainName(stack, 'Domain', {
        domainName: 'foo.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
        endpointType: apigw.EndpointType.REGIONAL,
        mapping: api,
        basePath: 'v1/api',
      });
      Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'Domain66AC69E0',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
        'ApiMappingKey': 'v1/api',
      });
    });

    test('throws if endpointType is not REGIONAL', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      // THEN
      expect(() => {
        new apigw.DomainName(stack, 'Domain', {
          domainName: 'foo.com',
          certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
          endpointType: apigw.EndpointType.EDGE,
          mapping: api,
          basePath: 'v1/api',
        });
      }).toThrow(/multi-level basePath is only supported when endpointType is EndpointType.REGIONAL/);
    });

    test('throws if securityPolicy is TLS_1_0', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      // THEN
      expect(() => {
        new apigw.DomainName(stack, 'Domain', {
          domainName: 'foo.com',
          certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
          mapping: api,
          basePath: 'v1/api',
          securityPolicy: apigw.SecurityPolicy.TLS_1_0,
        });
      }).toThrow(/securityPolicy must be TLS 1.2 or higher for multi-level basePath/);
    });

    test('can use addApiMapping', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      // WHEN
      const domain = new apigw.DomainName(stack, 'Domain', {
        domainName: 'foo.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      });
      Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

      domain.addApiMapping(api.deploymentStage);
      domain.addApiMapping(api.deploymentStage, { basePath: '//' });
      domain.addApiMapping(api.deploymentStage, {
        basePath: 'v1/my-api',
      });
      domain.addApiMapping(api.deploymentStage, {
        basePath: 'v1//my-api',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'Domain66AC69E0',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'Domain66AC69E0',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
        'ApiMappingKey': '//',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'Domain66AC69E0',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
        'ApiMappingKey': 'v1/my-api',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'Domain66AC69E0',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
        'ApiMappingKey': 'v1//my-api',
      });
    });

    test('can use addDomainName', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      const domain = api.addDomainName('Domain', {
        domainName: 'foo.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      });
      Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

      // WHEN
      domain.addApiMapping(api.deploymentStage, {
        basePath: 'v1/my-api',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
        'DomainName': {
          'Ref': 'apiDomain6D60CEFD',
        },
        'RestApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
        'DomainName': {
          'Ref': 'apiDomain6D60CEFD',
        },
        'ApiId': {
          'Ref': 'apiC8550315',
        },
        'Stage': {
          'Ref': 'apiDeploymentStageprod896C8101',
        },
        'ApiMappingKey': 'v1/my-api',
      });
    });

    test('throws if addBasePathMapping tries to add a mapping for a path that is already mapped', () => {
      // GIVEN
      const stack = new Stack();
      const api = new apigw.RestApi(stack, 'api');
      api.root.addMethod('GET');

      // WHEN
      const domain = new apigw.DomainName(stack, 'Domain', {
        domainName: 'foo.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
        mapping: api,
        basePath: 'v1/path',
      });

      // THEN
      expect(() => {
        domain.addApiMapping(api.deploymentStage, {
          basePath: 'v1/path',
        });
      }).toThrow(/DomainName Domain already has a mapping for path v1\/path/);
    });
  });

  test('"addBasePathMapping" can be used to add base path mapping to the domain', () => {
    // GIVEN
    const stack = new Stack();
    const api1 = new apigw.RestApi(stack, 'api1');
    const api2 = new apigw.RestApi(stack, 'api2');
    const domain = new apigw.DomainName(stack, 'my-domain', {
      domainName: 'example.com',
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      endpointType: apigw.EndpointType.REGIONAL,
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });
    api1.root.addMethod('GET');
    api2.root.addMethod('GET');

    // WHEN
    domain.addBasePathMapping(api1, { basePath: 'api1' });
    domain.addBasePathMapping(api2, { basePath: 'api2' });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'mydomain592C948B',
      },
      'BasePath': 'api1',
      'RestApiId': {
        'Ref': 'api1A91238E2',
      },
      'Stage': {
        'Ref': 'api1DeploymentStageprod362746F6',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'mydomain592C948B',
      },
      'BasePath': 'api2',
      'RestApiId': {
        'Ref': 'api2C4850CEA',
      },
      'Stage': {
        'Ref': 'api2DeploymentStageprod4120D74E',
      },
    });
  });

  test('a domain name can be defined with the API', () => {
    // GIVEN
    const domainName = 'my.domain.com';
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'my.domain.com' });

    // WHEN
    const api = new apigw.RestApi(stack, 'api', {
      domainName: { domainName, certificate },
    });

    api.root.addMethod('GET');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'my.domain.com',
      'EndpointConfiguration': {
        'Types': [
          'REGIONAL',
        ],
      },
      'RegionalCertificateArn': {
        'Ref': 'cert56CA94EB',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'apiCustomDomain64773C4F',
      },
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
      'Stage': {
        'Ref': 'apiDeploymentStageprod896C8101',
      },
    });
  });

  test('a domain name can be added later', () => {
    // GIVEN
    const domainName = 'my.domain.com';
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'my.domain.com' });

    // WHEN
    const api = new apigw.RestApi(stack, 'api', {});

    api.root.addMethod('GET');

    api.addDomainName('domainId', { domainName, certificate });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': domainName,
      'EndpointConfiguration': {
        'Types': [
          'REGIONAL',
        ],
      },
      'RegionalCertificateArn': {
        'Ref': 'cert56CA94EB',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'apidomainId102F8DAA',
      },
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
      'Stage': {
        'Ref': 'apiDeploymentStageprod896C8101',
      },
    });
  });

  test('a base path can be defined when adding a domain name', () => {
    // GIVEN
    const domainName = 'my.domain.com';
    const basePath = 'users';
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'my.domain.com' });

    // WHEN
    const api = new apigw.RestApi(stack, 'api', {});

    api.root.addMethod('GET');

    api.addDomainName('domainId', { domainName, certificate, basePath });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'BasePath': 'users',
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
    });
  });

  test('additional base paths can added if addDomainName was called with a non-empty base path', () => {
    // GIVEN
    const domainName = 'my.domain.com';
    const basePath = 'users';
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'my.domain.com' });

    // WHEN
    const api = new apigw.RestApi(stack, 'api', {});

    api.root.addMethod('GET');

    const dn = api.addDomainName('domainId', { domainName, certificate, basePath });
    dn.addBasePathMapping(api, {
      basePath: 'books',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'BasePath': 'users',
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'BasePath': 'books',
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
    });
  });

  test('domain name cannot contain uppercase letters', () => {
    // GIVEN
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'someDomainWithUpercase.domain.com' });

    // WHEN & THEN
    expect(() => {
      new apigw.DomainName(stack, 'someDomain', { domainName: 'someDomainWithUpercase.domain.com', certificate });
    }).toThrow(/uppercase/);
  });

  test('multiple domain names can be added', () => {
    // GIVEN
    const domainName = 'my.domain.com';
    const stack = new Stack();
    const certificate = new acm.Certificate(stack, 'cert', { domainName: 'my.domain.com' });

    // WHEN
    const api = new apigw.RestApi(stack, 'api', {});

    api.root.addMethod('GET');

    const domainName1 = api.addDomainName('domainId', { domainName, certificate });
    api.addDomainName('domainId1', { domainName: 'your.domain.com', certificate });
    api.addDomainName('domainId2', { domainName: 'our.domain.com', certificate });

    expect(api.domainName).toEqual(domainName1);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'my.domain.com',
      'EndpointConfiguration': {
        'Types': [
          'REGIONAL',
        ],
      },
      'RegionalCertificateArn': {
        'Ref': 'cert56CA94EB',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'your.domain.com',
      'EndpointConfiguration': {
        'Types': [
          'REGIONAL',
        ],
      },
      'RegionalCertificateArn': {
        'Ref': 'cert56CA94EB',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'our.domain.com',
      'EndpointConfiguration': {
        'Types': [
          'REGIONAL',
        ],
      },
      'RegionalCertificateArn': {
        'Ref': 'cert56CA94EB',
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'apidomainId102F8DAA',
      },
      'RestApiId': {
        'Ref': 'apiC8550315',
      },
      'Stage': {
        'Ref': 'apiDeploymentStageprod896C8101',
      },
    });
  });

  test('"addBasePathMapping" can be used to add base path mapping to the domain with specific stage', () => {
    // GIVEN
    const stack = new Stack();
    const api1 = new apigw.RestApi(stack, 'api1', {
      deploy: false,
    });
    const api2 = new apigw.RestApi(stack, 'api2');
    const domain = new apigw.DomainName(stack, 'my-domain', {
      domainName: 'example.com',
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      endpointType: apigw.EndpointType.REGIONAL,
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    api1.root.addMethod('GET');
    api2.root.addMethod('GET');

    const testDeploy = new apigw.Deployment(stack, 'test-deployment', {
      api: api1,
    });

    const testStage = new apigw.Stage(stack, 'test-stage', {
      deployment: testDeploy,
    });

    // WHEN
    domain.addBasePathMapping(api1, { basePath: 'api1', stage: testStage });
    domain.addBasePathMapping(api2, { basePath: 'api2' });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'mydomain592C948B',
      },
      'BasePath': 'api1',
      'RestApiId': {
        'Ref': 'api1A91238E2',
      },
      'Stage': stack.resolve(testStage.stageName),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'mydomain592C948B',
      },
      'BasePath': 'api2',
      'RestApiId': {
        'Ref': 'api2C4850CEA',
      },
      'Stage': {
        'Ref': 'api2DeploymentStageprod4120D74E',
      },
    });
  });

  test('accepts a mutual TLS configuration', () => {
    const stack = new Stack();
    const bucket = Bucket.fromBucketName(stack, 'testBucket', 'example-bucket');
    new apigw.DomainName(stack, 'another-domain', {
      domainName: 'example.com',
      mtls: {
        bucket,
        key: 'someca.pem',
      },
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
      'MutualTlsAuthentication': { 'TruststoreUri': 's3://example-bucket/someca.pem' },
    });
  });

  test('mTLS should allow versions to be set on the s3 bucket', () => {
    const stack = new Stack();
    const bucket = Bucket.fromBucketName(stack, 'testBucket', 'example-bucket');
    new apigw.DomainName(stack, 'another-domain', {
      domainName: 'example.com',
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert2', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      mtls: {
        bucket,
        key: 'someca.pem',
        version: 'version',
      },
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::DomainName', {
      'DomainName': 'example.com',
      'EndpointConfiguration': { 'Types': ['REGIONAL'] },
      'RegionalCertificateArn': 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
      'MutualTlsAuthentication': { 'TruststoreUri': 's3://example-bucket/someca.pem', 'TruststoreVersion': 'version' },
    });
  });

  test('base path mapping configures stage for RestApi creation', () => {
    // GIVEN
    const stack = new Stack();
    new apigw.RestApi(stack, 'restApiWithStage', {
      domainName: {
        domainName: 'example.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
        endpointType: apigw.EndpointType.REGIONAL,
      },
    }).root.addMethod('GET');
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'restApiWithStageCustomDomainC4749625',
      },
      'RestApiId': {
        'Ref': 'restApiWithStageD4F931D0',
      },
      'Stage': {
        'Ref': 'restApiWithStageDeploymentStageprodC82A6648',
      },
    });
  });

  test('base path mapping configures stage for SpecRestApi creation', () => {
    // GIVEN
    const stack = new Stack();
    Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::W3660',
      reason: 'We mix resources and Flutter definitions on purpose',
    });

    const definition = {
      key1: 'val1',
    };

    new apigw.SpecRestApi(stack, 'specRestApiWithStage', {
      apiDefinition: apigw.ApiDefinition.fromInline(definition),
      domainName: {
        domainName: 'example.com',
        certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
        endpointType: apigw.EndpointType.REGIONAL,
      },
    }).root.addMethod('GET');
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::BasePathMapping', {
      'DomainName': {
        'Ref': 'specRestApiWithStageCustomDomain8A36A5C9',
      },
      'RestApiId': {
        'Ref': 'specRestApiWithStageC1492575',
      },
      'Stage': {
        'Ref': 'specRestApiWithStageDeploymentStageprod2D3037ED',
      },
    });
  });

  test('allows REST API to be mapped with enhanced security policy and multi-level base path', () => {
    // GIVEN
    const stack = new Stack();
    const api = new apigw.RestApi(stack, 'api');
    api.root.addMethod('GET');

    const domain = new apigw.DomainName(stack, 'Domain', {
      domainName: 'foo.com',
      certificate: acm.Certificate.fromCertificateArn(stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d'),
      securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
      endpointAccessMode: apigw.EndpointAccessMode.STRICT,
    });
    Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'hardcoded ARN intentional for tests' });

    // WHEN - should not throw
    expect(() => {
      domain.addApiMapping(api.deploymentStage, {
        basePath: 'v1/my-api',
      });
    }).not.toThrow();

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGatewayV2::ApiMapping', {
      'ApiMappingKey': 'v1/my-api',
    });
  });

  describe('token handling', () => {
    test('allows token-based endpointAccessMode with enhanced security policy', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN - using a token for endpointAccessMode (e.g., from CfnParameter)
      const tokenValue = Fn.ref('AccessModeParameter');

      // THEN - should not throw during synthesis
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'token.example.com',
          certificate: cert,
          securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
          endpointAccessMode: tokenValue as any,
        });
      }).not.toThrow();
    });

    test('allows token-based securityPolicy with endpointAccessMode', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN - using a token for securityPolicy (e.g., from CfnParameter)
      const tokenValue = Fn.ref('SecurityPolicyParameter');

      // THEN - should not throw during synthesis
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'token.example.com',
          certificate: cert,
          securityPolicy: tokenValue as any,
          endpointAccessMode: apigw.EndpointAccessMode.STRICT,
        });
      }).not.toThrow();
    });

    test('allows token-based endpointType with security policy validation', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN - using a token for endpointType (e.g., from cross-stack reference)
      const tokenValue = Fn.importValue('EndpointTypeExport');

      // THEN - should not throw during synthesis
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'token.example.com',
          certificate: cert,
          endpointType: tokenValue as any,
          securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
          endpointAccessMode: apigw.EndpointAccessMode.STRICT,
        });
      }).not.toThrow();
    });

    test('allows token-based endpointType with edge-only security policy', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN - using a token for endpointType with edge-only policy
      const tokenValue = Fn.ref('EndpointTypeParameter');

      // THEN - should not throw during synthesis (validation deferred to CloudFormation)
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'token.example.com',
          certificate: cert,
          endpointType: tokenValue as any,
          securityPolicy: apigw.SecurityPolicy.TLS13_2025_EDGE,
          endpointAccessMode: apigw.EndpointAccessMode.STRICT,
        });
      }).not.toThrow();
    });

    test('accepts BASIC endpointAccessMode with enhanced security policy', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN/THEN - BASIC is a valid value for enhanced security policies
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'example.com',
          certificate: cert,
          securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09,
          endpointAccessMode: apigw.EndpointAccessMode.BASIC,
        });
      }).not.toThrow();
    });

    test('still validates non-token endpointType with incompatible security policy', () => {
      // GIVEN
      const stack = new Stack();
      const cert = new acm.Certificate(stack, 'Cert', { domainName: 'example.com' });

      // WHEN/THEN - non-token values should still be validated
      expect(() => {
        new apigw.DomainName(stack, 'domain', {
          domainName: 'example.com',
          certificate: cert,
          endpointType: apigw.EndpointType.EDGE,
          securityPolicy: apigw.SecurityPolicy.TLS13_1_3_2025_09, // Regional-only policy
          endpointAccessMode: apigw.EndpointAccessMode.STRICT,
        });
      }).toThrow(/Security policy SecurityPolicy_TLS13_1_3_2025_09 is not supported for edge-optimized endpoints/);
    });
  });
});
