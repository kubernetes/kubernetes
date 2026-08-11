#!/usr/bin/env node
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose';
import * as cdk from 'aws-cdk-lib';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-firehose-delivery-stream-s3-all-properties');

const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
  destination: new firehose.HttpEndpoint({
    endpointConfig: {
      url: 'https://postman-echo.com/post',
    },
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
