import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-url-pull');

new medialive.Input(stack, 'UrlPullInput', {
  inputName: 'integ-url-pull',
  input: medialive.InputConfiguration.urlPull([medialive.InputSource.url('https://example.com/stream.m3u8')]),
});

new medialive.Input(stack, 'UrlPullRedundant', {
  inputName: 'integ-url-pull-redundant',
  input: medialive.InputConfiguration.urlPull([
    medialive.InputSource.url('https://primary.example.com/stream.m3u8'),
    medialive.InputSource.url('https://backup.example.com/stream.m3u8'),
  ]),
});

new IntegTest(app, 'cdk-integ-medialive-input-url-pull', {
  testCases: [stack],
});

app.synth();
