#!/usr/bin/env node
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cdk from 'aws-cdk-lib';
import { AwsApiCall, ExpectedResult, IntegTest } from '@aws-cdk/integ-tests-alpha';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-firehose-delivery-stream-s3-custom-time-zone');

const bucket = new s3.Bucket(stack, 'Bucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

// 'UTC' is special-cased by the customTimeZone validation (it is not a
// standard Region/City IANA identifier but is accepted by the service).
const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
  destination: new firehose.S3Bucket(bucket, {
    timeZone: cdk.TimeZone.of('UTC'),
  }),
});

const testCase = new IntegTest(app, 'integ-tests', {
  testCases: [stack],
  regions: ['us-east-1'],
});

testCase.assertions.awsApiCall('Firehose', 'putRecord', {
  DeliveryStreamName: deliveryStream.deliveryStreamName,
  Record: {
    Data: 'testData123',
  },
});

const s3ApiCall = testCase.assertions.awsApiCall('S3', 'listObjectsV2', {
  Bucket: bucket.bucketName,
  MaxKeys: 1,
}).expect(ExpectedResult.objectLike({
  KeyCount: 1,
})).waitForAssertions({
  interval: cdk.Duration.seconds(30),
  totalTimeout: cdk.Duration.minutes(10),
});

if (s3ApiCall instanceof AwsApiCall && s3ApiCall.waiterProvider) {
  s3ApiCall.waiterProvider.addToRolePolicy({
    Effect: 'Allow',
    Action: ['s3:GetObject', 's3:ListBucket'],
    Resource: ['*'],
  });
}
