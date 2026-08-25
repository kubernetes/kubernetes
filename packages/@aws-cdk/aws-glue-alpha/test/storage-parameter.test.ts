import * as cdk from 'aws-cdk-lib';
import * as kms from 'aws-cdk-lib/aws-kms';
import { StorageParameter } from '../lib';

test('writeKmsKeyId takes a KMS key and renders its ARN', () => {
  const stack = new cdk.Stack();
  const key = kms.Key.fromKeyArn(stack, 'Key', 'arn:aws:kms:us-east-1:123456789012:key/abcd-1234');

  const param = StorageParameter.writeKmsKeyId(key);

  expect(param.key).toEqual('write.kms.key.id');
  expect(param.value).toEqual(stack.resolve(key.keyArn));
});

test('custom takes a string key and value', () => {
  const param = StorageParameter.custom('separatorChar', ',');

  expect(param.key).toEqual('separatorChar');
  expect(param.value).toEqual(',');
});
