import { Template } from '../../assertions';
import * as firehose from '../../aws-kinesisfirehose';
import * as logs from '../../aws-logs';
import * as cdk from '../../core';
import * as apigateway from '../lib';
import { ApiDefinition } from '../lib';

describe('stage', () => {
  test('minimal setup', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', { deployment });

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        testapiD6451F70: {
          Type: 'AWS::ApiGateway::RestApi',
          Properties: {
            Name: 'test-api',
          },
        },
        testapiGETD8DE4ED1: {
          Type: 'AWS::ApiGateway::Method',
          Properties: {
            HttpMethod: 'GET',
            ResourceId: {
              'Fn::GetAtt': [
                'testapiD6451F70',
                'RootResourceId',
              ],
            },
            RestApiId: {
              Ref: 'testapiD6451F70',
            },
            AuthorizationType: 'NONE',
            Integration: {
              Type: 'MOCK',
            },
          },
        },
        mydeployment71ED3B4B5ce82e617e0729f75657ddcca51e3b91: {
          Type: 'AWS::ApiGateway::Deployment',
          Properties: {
            RestApiId: {
              Ref: 'testapiD6451F70',
            },
          },
          DependsOn: [
            'testapiGETD8DE4ED1',
          ],
        },
        mystage7483BE9A: {
          Type: 'AWS::ApiGateway::Stage',
          Properties: {
            RestApiId: {
              Ref: 'testapiD6451F70',
            },
            DeploymentId: {
              Ref: 'mydeployment71ED3B4B5ce82e617e0729f75657ddcca51e3b91',
            },
            StageName: 'prod',
          },
        },
      },
    });
  });

  test('RestApi - stage depends on the CloudWatch role when it exists', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: true, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', { deployment });

    // THEN
    Template.fromStack(stack).hasResource('AWS::ApiGateway::Stage', {
      DependsOn: ['testapiAccount9B907665'],
    });
  });

  test('SpecRestApi - stage depends on the CloudWatch role when it exists', () => {
    // GIVEN
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::W3660',
      reason: 'We mix resources and Flutter definitions on purpose',
    });
    const api = new apigateway.SpecRestApi(stack, 'test-api', { deploy: false, apiDefinition: apigateway.ApiDefinition.fromInline( { foo: 'bar' }) });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', { deployment });

    // THEN
    Template.fromStack(stack).hasResource('AWS::ApiGateway::Stage', {
      DependsOn: ['testapiAccount9B907665'],
    });
  });

  test('common method settings can be set at the stage level', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      loggingLevel: apigateway.MethodLoggingLevel.INFO,
      throttlingRateLimit: 12,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      MethodSettings: [
        {
          DataTraceEnabled: false,
          HttpMethod: '*',
          LoggingLevel: 'INFO',
          ResourcePath: '/*',
          ThrottlingRateLimit: 12,
        },
      ],
    });
  });

  test('"stageResourceArn" returns the ARN for the stage', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api');
    const deployment = new apigateway.Deployment(stack, 'test-deploymnet', {
      api,
    });
    api.root.addMethod('GET');

    // WHEN
    const stage = new apigateway.Stage(stack, 'test-stage', {
      deployment,
    });

    // THEN
    expect(stack.resolve(stage.stageArn)).toEqual({
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':apigateway:',
          { Ref: 'AWS::Region' },
          '::/restapis/',
          { Ref: 'testapiD6451F70' },
          '/stages/',
          { Ref: 'teststage8788861E' },
        ],
      ],
    });
  });

  test('custom method settings can be set by their path', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      loggingLevel: apigateway.MethodLoggingLevel.INFO,
      throttlingRateLimit: 12,
      methodOptions: {
        '/goo/bar/GET': {
          loggingLevel: apigateway.MethodLoggingLevel.ERROR,
        },
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      MethodSettings: [
        {
          DataTraceEnabled: false,
          HttpMethod: '*',
          LoggingLevel: 'INFO',
          ResourcePath: '/*',
          ThrottlingRateLimit: 12,
        },
        {
          DataTraceEnabled: false,
          HttpMethod: 'GET',
          LoggingLevel: 'ERROR',
          ResourcePath: '/~1goo~1bar',
        },
      ],
    });
  });

  test('default "cacheClusterSize" is 0.5 (if cache cluster is enabled)', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      cacheClusterEnabled: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      CacheClusterEnabled: true,
      CacheClusterSize: '0.5',
    });
  });

  test('setting "cacheClusterSize" implies "cacheClusterEnabled"', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      cacheClusterSize: '0.5',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      CacheClusterEnabled: true,
      CacheClusterSize: '0.5',
    });
  });

  test('fails when "cacheClusterEnabled" is "false" and "cacheClusterSize" is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // THEN
    expect(() => new apigateway.Stage(stack, 'my-stage', {
      deployment,
      cacheClusterSize: '0.5',
      cacheClusterEnabled: false,
    })).toThrow(/Cannot set "cacheClusterSize" to 0.5 and "cacheClusterEnabled" to "false"/);
  });

  test('if "cachingEnabled" in method settings, implicitly enable cache cluster', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      cachingEnabled: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      CacheClusterEnabled: true,
      CacheClusterSize: '0.5',
      MethodSettings: [
        {
          DataTraceEnabled: false,
          CachingEnabled: true,
          HttpMethod: '*',
          ResourcePath: '/*',
        },
      ],
      StageName: 'prod',
    });
  });

  test('if caching cluster is explicitly disabled, do not auto-enable cache cluster when "cachingEnabled" is set in method options', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // THEN
    expect(() => new apigateway.Stage(stack, 'my-stage', {
      cacheClusterEnabled: false,
      deployment,
      cachingEnabled: true,
    })).toThrow(/Cannot enable caching for method \/\*\/\* since cache cluster is disabled on stage/);
  });

  test('if only the custom log destination log group is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      AccessLogSetting: {
        DestinationArn: {
          'Fn::GetAtt': [
            'LogGroupF5B46931',
            'Arn',
          ],
        },
        Format: '$context.identity.sourceIp $context.identity.caller $context.identity.user [$context.requestTime] "$context.httpMethod $context.resourcePath $context.protocol" $context.status $context.responseLength $context.requestId',
      },
      StageName: 'prod',
    });
  });

  test('if the custom log destination log group and format is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
    const testFormat = apigateway.AccessLogFormat.jsonWithStandardFields();
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
      accessLogFormat: testFormat,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      AccessLogSetting: {
        DestinationArn: {
          'Fn::GetAtt': [
            'LogGroupF5B46931',
            'Arn',
          ],
        },
        Format: '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","user":"$context.identity.user","caller":"$context.identity.caller","requestTime":"$context.requestTime","httpMethod":"$context.httpMethod","resourcePath":"$context.resourcePath","status":"$context.status","protocol":"$context.protocol","responseLength":"$context.responseLength"}',
      },
      StageName: 'prod',
    });
  });

  test('if only the custom log destination firehose delivery stream is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    const testDeliveryStream = new firehose.CfnDeliveryStream(stack, 'MyStream', {
      deliveryStreamName: 'amazon-apigateway-delivery-stream',
    });
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      accessLogDestination: new apigateway.FirehoseLogDestination(testDeliveryStream),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      AccessLogSetting: {
        DestinationArn: {
          'Fn::GetAtt': [
            'MyStream',
            'Arn',
          ],
        },
        Format: '$context.identity.sourceIp $context.identity.caller $context.identity.user [$context.requestTime] "$context.httpMethod $context.resourcePath $context.protocol" $context.status $context.responseLength $context.requestId',
      },
      StageName: 'prod',
    });
  });

  test('if the custom log destination firehose delivery stream and format is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
    const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
    api.root.addMethod('GET');

    // WHEN
    const testDeliveryStream = new firehose.CfnDeliveryStream(stack, 'MyStream', {
      deliveryStreamName: 'amazon-apigateway-delivery-stream',
    });
    const testFormat = apigateway.AccessLogFormat.jsonWithStandardFields();
    new apigateway.Stage(stack, 'my-stage', {
      deployment,
      accessLogDestination: new apigateway.FirehoseLogDestination(testDeliveryStream),
      accessLogFormat: testFormat,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      AccessLogSetting: {
        DestinationArn: {
          'Fn::GetAtt': [
            'MyStream',
            'Arn',
          ],
        },
        Format: '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","user":"$context.identity.user","caller":"$context.identity.caller","requestTime":"$context.requestTime","httpMethod":"$context.httpMethod","resourcePath":"$context.resourcePath","status":"$context.status","protocol":"$context.protocol","responseLength":"$context.responseLength"}',
      },
      StageName: 'prod',
    });
  });

  describe('access log check', () => {
    test('fails when access log format does not contain `contextRequestId()` or `contextExtendedRequestId()', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom('');

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).toThrow('Access log must include either `AccessLogFormat.contextRequestId()` or `AccessLogFormat.contextExtendedRequestId()`');
    });

    test('succeeds when access log format contains `contextRequestId()`', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom(JSON.stringify({
        requestId: apigateway.AccessLogField.contextRequestId(),
      }));

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).not.toThrow();
    });

    test('succeeds when access log format contains `contextExtendedRequestId()`', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom(JSON.stringify({
        extendedRequestId: apigateway.AccessLogField.contextExtendedRequestId(),
      }));

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).not.toThrow();
    });

    test('succeeds when access log format contains both `contextRequestId()` and `contextExtendedRequestId`', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom(JSON.stringify({
        requestId: apigateway.AccessLogField.contextRequestId(),
        extendedRequestId: apigateway.AccessLogField.contextExtendedRequestId(),
      }));

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).not.toThrow();
    });

    test('fails when access log format contains `contextRequestIdXxx`', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom(JSON.stringify({
        requestIdXxx: '$context.requestIdXxx',
      }));

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).toThrow('Access log must include either `AccessLogFormat.contextRequestId()` or `AccessLogFormat.contextExtendedRequestId()`');
    });

    test('does not fail when access log format is a token', () => {
    // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testLogGroup = new logs.LogGroup(stack, 'LogGroup');
      const testFormat = apigateway.AccessLogFormat.custom(cdk.Lazy.string({ produce: () => 'test' }));

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogDestination: new apigateway.LogGroupLogDestination(testLogGroup),
        accessLogFormat: testFormat,
      })).not.toThrow();
    });

    test('fails when access log destination is empty', () => {
    // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testFormat = apigateway.AccessLogFormat.jsonWithStandardFields();

      // THEN
      expect(() => new apigateway.Stage(stack, 'my-stage', {
        deployment,
        accessLogFormat: testFormat,
      })).toThrow(/Access log format is specified without a destination/);
    });

    test('fails if firehose delivery stream name does not start with amazon-apigateway-', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false, deploy: false });
      const deployment = new apigateway.Deployment(stack, 'my-deployment', { api });
      api.root.addMethod('GET');

      // WHEN
      const testDeliveryStream = new firehose.CfnDeliveryStream(stack, 'MyStream', {
        deliveryStreamName: 'invalid',
      });
      expect(() => {
        new apigateway.Stage(stack, 'my-stage', {
          deployment,
          accessLogDestination: new apigateway.FirehoseLogDestination(testDeliveryStream),
        });
      }).toThrow(/Firehose delivery stream name for access log destination must begin with 'amazon-apigateway-', got 'invalid'/);
    });
  });

  test('default throttling settings', () => {
    // GIVEN
    const stack = new cdk.Stack();
    new apigateway.SpecRestApi(stack, 'testapi', {
      apiDefinition: ApiDefinition.fromInline({
        openapi: '3.0.2',
      }),
      deployOptions: {
        throttlingBurstLimit: 0,
        throttlingRateLimit: 0,
        metricsEnabled: false,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Stage', {
      MethodSettings: [{
        DataTraceEnabled: false,
        HttpMethod: '*',
        ResourcePath: '/*',
        ThrottlingBurstLimit: 0,
        ThrottlingRateLimit: 0,
      }],
    });
  });

  test('addApiKey is supported', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false });
    api.root.addMethod('GET');

    // WHEN
    api.deploymentStage.addApiKey('MyKey');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::ApiKey', {
      StageKeys: [
        {
          RestApiId: {
            Ref: 'testapiD6451F70',
          },
          StageName: {
            Ref: 'testapiDeploymentStageprod5C9E92A4',
          },
        },
      ],
    });
  });

  test('addApiKey is supported on an imported stage', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'test-api', { cloudWatchRole: false });
    api.root.addMethod('GET');
    const stage = apigateway.Stage.fromStageAttributes(stack, 'Stage', {
      restApi: api,
      stageName: 'MyStage',
    });

    // WHEN
    stage.addApiKey('MyKey');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::ApiKey', {
      StageKeys: [
        {
          RestApiId: {
            Ref: 'testapiD6451F70',
          },
          StageName: 'MyStage',
        },
      ],
    });
  });

  describe('Metrics', () => {
    test('metric', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const metricName = '4XXError';
      const statistic = 'Sum';
      const metric = api.deploymentStage.metric(metricName, { statistic });

      // THEN
      expect(metric.namespace).toEqual('AWS/ApiGateway');
      expect(metric.metricName).toEqual(metricName);
      expect(metric.statistic).toEqual(statistic);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricClientError', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricClientError({ color });

      // THEN
      expect(metric.metricName).toEqual('4XXError');
      expect(metric.statistic).toEqual('Sum');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricServerError', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricServerError({ color });

      // THEN
      expect(metric.metricName).toEqual('5XXError');
      expect(metric.statistic).toEqual('Sum');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricCacheHitCount', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricCacheHitCount({ color });

      // THEN
      expect(metric.metricName).toEqual('CacheHitCount');
      expect(metric.statistic).toEqual('Sum');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricCacheMissCount', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricCacheMissCount({ color });

      // THEN
      expect(metric.metricName).toEqual('CacheMissCount');
      expect(metric.statistic).toEqual('Sum');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricCount', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricCount({ color });

      // THEN
      expect(metric.metricName).toEqual('Count');
      expect(metric.statistic).toEqual('SampleCount');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricIntegrationLatency', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricIntegrationLatency({ color });

      // THEN
      expect(metric.metricName).toEqual('IntegrationLatency');
      expect(metric.statistic).toEqual('Average');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });

    test('metricLatency', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const api = new apigateway.RestApi(stack, 'test-api');
      const color = '#00ff00';
      const metric = api.deploymentStage.metricLatency({ color });

      // THEN
      expect(metric.metricName).toEqual('Latency');
      expect(metric.statistic).toEqual('Average');
      expect(metric.color).toEqual(color);
      expect(metric.dimensions).toEqual({ ApiName: 'test-api', Stage: api.deploymentStage.stageName });
    });
  });
});
