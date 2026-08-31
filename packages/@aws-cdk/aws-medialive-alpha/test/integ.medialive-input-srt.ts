import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Secret } from 'aws-cdk-lib/aws-secretsmanager';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-srt');

// Security group required for push-type inputs
const sg = new medialive.InputSecurityGroup(stack, 'InputSG', {
  allowlistRules: ['203.0.113.0/24'],
});

// Secret for SRT encryption
const srtSecret = new Secret(stack, 'SrtSecret', {
  secretName: 'integ-srt-passphrase',
});

new medialive.Input(stack, 'SrtCallerInput', {
  inputName: 'integ-srt-caller',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 9000,
    minimumLatency: cdk.Duration.millis(1000),
  }]),
});

new medialive.Input(stack, 'SrtListenerInput', {
  inputName: 'integ-srt-listener',
  input: medialive.InputConfiguration.srtListener({
    minimumLatency: cdk.Duration.millis(500),
    streamId: 'integ-stream-id',
    inputSecurityGroups: [sg],
    decryption: {
      algorithm: medialive.SrtDecryptionAlgorithm.AES256,
      passphraseSecret: srtSecret,
    },
  }),
});

new IntegTest(app, 'cdk-integ-medialive-input-srt', {
  testCases: [stack],
});

app.synth();
