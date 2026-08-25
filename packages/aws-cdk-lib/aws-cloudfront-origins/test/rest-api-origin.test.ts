import * as apigateway from '../../aws-apigateway';
import { Duration, Stack } from '../../core';
import { RestApiOrigin } from '../lib';

let stack: Stack;

beforeEach(() => {
  stack = new Stack();
});

test('Correctly renders the origin', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  const origin = new RestApiOrigin(api);
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(stack.resolve(originBindConfig.originProperty)).toEqual({
    id: 'StackOrigin029E19582',
    domainName: {
      'Fn::Select': [2, {
        'Fn::Split': ['/', {
          'Fn::Join': ['', [
            'https://', { Ref: 'RestApi0C43BF4B' },
            '.execute-api.',
            { Ref: 'AWS::Region' },
            '.',
            { Ref: 'AWS::URLSuffix' },
            '/',
            { Ref: 'RestApiDeploymentStageprod3855DE66' },
            '/',
          ]],
        }],
      }],
    },
    originPath: {
      'Fn::Join': ['', ['/', {
        'Fn::Select': [3, {
          'Fn::Split': ['/', {
            'Fn::Join': ['', [
              'https://',
              { Ref: 'RestApi0C43BF4B' },
              '.execute-api.',
              { Ref: 'AWS::Region' },
              '.',
              { Ref: 'AWS::URLSuffix' },
              '/',
              { Ref: 'RestApiDeploymentStageprod3855DE66' },
              '/',
            ]],
          }],
        }],
      }]],
    },
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
    },
  });
});

test('Correctly renders the origin, with custom originId', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  const origin = new RestApiOrigin(api, { originId: 'MyCustomOrigin' });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(stack.resolve(originBindConfig.originProperty)).toEqual({
    id: 'MyCustomOrigin',
    domainName: {
      'Fn::Select': [2, {
        'Fn::Split': ['/', {
          'Fn::Join': ['', [
            'https://', { Ref: 'RestApi0C43BF4B' },
            '.execute-api.',
            { Ref: 'AWS::Region' },
            '.',
            { Ref: 'AWS::URLSuffix' },
            '/',
            { Ref: 'RestApiDeploymentStageprod3855DE66' },
            '/',
          ]],
        }],
      }],
    },
    originPath: {
      'Fn::Join': ['', ['/', {
        'Fn::Select': [3, {
          'Fn::Split': ['/', {
            'Fn::Join': ['', [
              'https://',
              { Ref: 'RestApi0C43BF4B' },
              '.execute-api.',
              { Ref: 'AWS::Region' },
              '.',
              { Ref: 'AWS::URLSuffix' },
              '/',
              { Ref: 'RestApiDeploymentStageprod3855DE66' },
              '/',
            ]],
          }],
        }],
      }]],
    },
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
    },
  });
});

test('Correctly renders the origin, with custom originPath', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  const origin = new RestApiOrigin(api, { originPath: '/my/custom/path' });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(stack.resolve(originBindConfig.originProperty)).toEqual({
    id: 'StackOrigin029E19582',
    domainName: {
      'Fn::Select': [2, {
        'Fn::Split': ['/', {
          'Fn::Join': ['', [
            'https://', { Ref: 'RestApi0C43BF4B' },
            '.execute-api.',
            { Ref: 'AWS::Region' },
            '.',
            { Ref: 'AWS::URLSuffix' },
            '/',
            { Ref: 'RestApiDeploymentStageprod3855DE66' },
            '/',
          ]],
        }],
      }],
    },
    originPath: '/my/custom/path',
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
    },
  });
});

test('renders responseCompletionTimeout in origin property', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  const origin = new RestApiOrigin(api, {
    responseCompletionTimeout: Duration.seconds(120),
  });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(originBindConfig.originProperty?.responseCompletionTimeout).toEqual(120);
});

test('configure both responseCompletionTimeout and readTimeout', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  const origin = new RestApiOrigin(api, {
    responseCompletionTimeout: Duration.seconds(60),
    readTimeout: Duration.seconds(60),
  });

  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });
  expect(originBindConfig.originProperty?.responseCompletionTimeout).toEqual(60);
});

test('throw error for configuring readTimeout less than responseCompletionTimeout value', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  expect(() => {
    new RestApiOrigin(api, {
      responseCompletionTimeout: Duration.seconds(30),
      readTimeout: Duration.seconds(60),
    });
  }).toThrow('responseCompletionTimeout must be equal to or greater than readTimeout (60s), got: 30s.');
});

test('validates readTimeout is at least 1 second', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  expect(() => {
    new RestApiOrigin(api, { readTimeout: Duration.seconds(0) });
  }).toThrow('readTimeout: Must be an int 1 seconds or greater; received 0.');
});

test.each([
  Duration.seconds(121),
  Duration.minutes(5),
])('accepts readTimeout above the default quota, which the service validates at deploy time', (readTimeout) => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  expect(() => {
    new RestApiOrigin(api, { readTimeout });
  }).not.toThrow();
});

test('validates keepaliveTimeout is at least 1 second', () => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  expect(() => {
    new RestApiOrigin(api, { keepaliveTimeout: Duration.seconds(0) });
  }).toThrow('keepaliveTimeout: Must be an int 1 seconds or greater; received 0.');
});

test.each([
  Duration.seconds(301),
  Duration.minutes(10),
])('accepts keepaliveTimeout above the default quota, which the service validates at deploy time', (keepaliveTimeout) => {
  const api = new apigateway.RestApi(stack, 'RestApi');
  api.root.addMethod('GET');

  expect(() => {
    new RestApiOrigin(api, { keepaliveTimeout });
  }).not.toThrow();
});
