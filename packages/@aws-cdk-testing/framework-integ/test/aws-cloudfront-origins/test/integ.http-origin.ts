import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as cdk from 'aws-cdk-lib';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'cloudfront-http-origin');

// 200 seconds is deliberately above the 180 second ceiling the CDK used to
// hardcode for keepaliveTimeout, so this stack only synthesizes because that
// ceiling is gone. Both values sit within the default CloudFront quotas
// (`Response timeout per origin` 1-120s, `Keep-alive timeout per origin` 1-300s),
// so deploying this test needs no quota increase.
new cloudfront.Distribution(stack, 'Distribution', {
  defaultBehavior: {
    origin: new origins.HttpOrigin('www.example.com', {
      ipAddressType: cloudfront.OriginIpAddressType.DUALSTACK,
      readTimeout: cdk.Duration.seconds(100),
      keepaliveTimeout: cdk.Duration.seconds(200),
    }),
  },
});

new IntegTest(app, 'http-origin-test-integ', {
  testCases: [stack],
});
