import * as cloudfront from '../../aws-cloudfront';
import { App, Duration, Stack } from '../../core';
import { HttpOrigin } from '../lib';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'Stack', {
    env: { account: '1234', region: 'testregion' },
  });
});

test('Renders minimal example with just a domain name', () => {
  const origin = new HttpOrigin('www.example.com');
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(originBindConfig.originProperty).toEqual({
    id: 'StackOrigin029E19582',
    domainName: 'www.example.com',
    originCustomHeaders: undefined,
    originPath: undefined,
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
    },
  });
});

test('renders an example with all available props', () => {
  const origin = new HttpOrigin('www.example.com', {
    originId: 'MyCustomOrigin',
    originPath: '/app',
    connectionTimeout: Duration.seconds(5),
    connectionAttempts: 2,
    customHeaders: { AUTH: 'NONE' },
    protocolPolicy: cloudfront.OriginProtocolPolicy.MATCH_VIEWER,
    httpPort: 8080,
    httpsPort: 8443,
    readTimeout: Duration.seconds(45),
    keepaliveTimeout: Duration.seconds(3),
    originSslProtocols: [cloudfront.OriginSslPolicy.TLS_V1_2],
  });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(originBindConfig.originProperty).toEqual({
    id: 'MyCustomOrigin',
    domainName: 'www.example.com',
    originPath: '/app',
    connectionTimeout: 5,
    connectionAttempts: 2,
    originCustomHeaders: [{
      headerName: 'AUTH',
      headerValue: 'NONE',
    }],
    customOriginConfig: {
      originProtocolPolicy: 'match-viewer',
      originSslProtocols: [
        'TLSv1.2',
      ],
      httpPort: 8080,
      httpsPort: 8443,
      originReadTimeout: 45,
      originKeepaliveTimeout: 3,
    },
  });
});

test.each([
  Duration.seconds(0),
])('validates readTimeout is at least 1 second', (readTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      readTimeout,
    });
  }).toThrow(`readTimeout: Must be an int 1 seconds or greater; received ${readTimeout.toSeconds()}.`);
});

test.each([
  Duration.seconds(121),
  Duration.minutes(5),
])('accepts readTimeout above the default quota, which the service validates at deploy time', (readTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      readTimeout,
    });
  }).not.toThrow();
});

test.each([
  Duration.seconds(0.5),
  Duration.seconds(60.5),
])('validates readTimeout is a whole number of seconds', (readTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      readTimeout,
    });
  }).toThrow(/must be a whole number of/);
});

test.each([
  Duration.seconds(0),
])('validates keepaliveTimeout is at least 1 second', (keepaliveTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      keepaliveTimeout,
    });
  }).toThrow(`keepaliveTimeout: Must be an int 1 seconds or greater; received ${keepaliveTimeout.toSeconds()}.`);
});

test.each([
  Duration.seconds(301),
  Duration.minutes(10),
])('accepts keepaliveTimeout above the default quota, which the service validates at deploy time', (keepaliveTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      keepaliveTimeout,
    });
  }).not.toThrow();
});

test.each([
  Duration.seconds(0.5),
  Duration.seconds(60.5),
])('validates keepaliveTimeout is a whole number of seconds', (keepaliveTimeout) => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      keepaliveTimeout,
    });
  }).toThrow(/must be a whole number of/);
});

test.each([
  cloudfront.OriginIpAddressType.IPV4,
  cloudfront.OriginIpAddressType.IPV6,
  cloudfront.OriginIpAddressType.DUALSTACK,
])('renders with %s address type', (ipAddressType) => {
  const origin = new HttpOrigin('www.example.com', {
    ipAddressType,
  });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(originBindConfig.originProperty).toEqual({
    id: 'StackOrigin029E19582',
    domainName: 'www.example.com',
    originCustomHeaders: undefined,
    originPath: undefined,
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
      ipAddressType,
    },
  });
});

test('renders responseCompletionTimeout in origin property', () => {
  const origin = new HttpOrigin('www.example.com', {
    responseCompletionTimeout: Duration.seconds(120),
  });
  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });

  expect(originBindConfig.originProperty).toEqual({
    id: 'StackOrigin029E19582',
    domainName: 'www.example.com',
    originCustomHeaders: undefined,
    originPath: undefined,
    responseCompletionTimeout: 120,
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
    },
  });
});

test('configure both responseCompletionTimeout and readTimeout', () => {
  const origin = new HttpOrigin('www.example.com', {
    responseCompletionTimeout: Duration.seconds(60),
    readTimeout: Duration.seconds(60),
  });

  const originBindConfig = origin.bind(stack, { originId: 'StackOrigin029E19582' });
  expect(originBindConfig.originProperty).toEqual({
    id: 'StackOrigin029E19582',
    domainName: 'www.example.com',
    originCustomHeaders: undefined,
    originPath: undefined,
    responseCompletionTimeout: 60,
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: [
        'TLSv1.2',
      ],
      originReadTimeout: 60,
    },
  });
});

test.each([
  [0],
  [-1],
  [79],
  [81],
  [442],
  [444],
  [1023],
  [65536],
])('fails for invalid httpPort %d', (httpPort) => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpPort });
  }).toThrow(`'httpPort' must be 80, 443, or an integer between 1024 and 65535; received ${httpPort}.`);
});

test('fails for non-integer httpPort', () => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpPort: 1024.5 });
  }).toThrow('\'httpPort\' must be 80, 443, or an integer between 1024 and 65535; received 1024.5.');
});

test.each([
  [80],
  [443],
  [1024],
  [65535],
])('accepts valid httpPort %d', (httpPort) => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpPort });
  }).not.toThrow();
});

test.each([
  [0],
  [-1],
  [79],
  [81],
  [442],
  [444],
  [1023],
  [65536],
])('fails for invalid httpsPort %d', (httpsPort) => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpsPort });
  }).toThrow(`'httpsPort' must be 80, 443, or an integer between 1024 and 65535; received ${httpsPort}.`);
});

test('fails for non-integer httpsPort', () => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpsPort: 1024.5 });
  }).toThrow('\'httpsPort\' must be 80, 443, or an integer between 1024 and 65535; received 1024.5.');
});

test.each([
  [80],
  [443],
  [1024],
  [65535],
])('accepts valid httpsPort %d', (httpsPort) => {
  expect(() => {
    new HttpOrigin('www.example.com', { httpsPort });
  }).not.toThrow();
});

test('throw error for configuring readTimeout less than responseCompletionTimeout value', () => {
  expect(() => {
    new HttpOrigin('www.example.com', {
      responseCompletionTimeout: Duration.seconds(30),
      readTimeout: Duration.seconds(60),
    });
  }).toThrow('responseCompletionTimeout must be equal to or greater than readTimeout (60s), got: 30s.');
});
