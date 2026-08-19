import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as glue from '../lib';

test('throws when a security configuration has no encryption config', () => {
  const stack = new cdk.Stack();

  expect(() => new glue.SecurityConfiguration(stack, 'SecurityConfiguration'))
    .toThrow(/One of cloudWatchEncryption, jobBookmarksEncryption or s3Encryption must be defined/);
});

test('a security configuration with encryption configuration requiring kms key and providing an explicit one', () => {
  const stack = new cdk.Stack();
  const keyArn = 'arn:aws:kms:us-west-2:111122223333:key/test-key';
  const key = kms.Key.fromKeyArn(stack, 'ImportedKey', keyArn);

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    cloudWatchEncryption: glue.CloudWatchEncryption.kms(key),
  });

  expect(securityConfiguration.cloudWatchEncryptionKey?.keyRef.keyArn).toEqual(keyArn);
  expect(securityConfiguration.jobBookmarksEncryptionKey).toBeUndefined();
  expect(securityConfiguration.s3EncryptionKey).toBeUndefined();

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    Name: 'SecurityConfiguration',
    EncryptionConfiguration: {
      CloudWatchEncryption: {
        CloudWatchEncryptionMode: 'SSE-KMS',
        KmsKeyArn: keyArn,
      },
    },
  });
});

test('a security configuration with an encryption configuration requiring kms key but not providing an explicit one', () => {
  const stack = new cdk.Stack();

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
  });

  expect(securityConfiguration.cloudWatchEncryptionKey).toBeDefined();
  expect(securityConfiguration.jobBookmarksEncryptionKey).toBeUndefined();
  expect(securityConfiguration.s3EncryptionKey).toBeUndefined();

  Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 1);

  // Auto-created keys have rotation enabled.
  Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
    EnableKeyRotation: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    Name: 'SecurityConfiguration',
    EncryptionConfiguration: {
      CloudWatchEncryption: {
        CloudWatchEncryptionMode: 'SSE-KMS',
        KmsKeyArn: stack.resolve(securityConfiguration.cloudWatchEncryptionKey?.keyRef.keyArn),
      },
    },
  });
});

test('a security configuration with all encryption configs and mixed kms key inputs', () => {
  const stack = new cdk.Stack();
  const keyArn = 'arn:aws:kms:us-west-2:111122223333:key/test-key';
  const key = kms.Key.fromKeyArn(stack, 'ImportedKey', keyArn);

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
    jobBookmarksEncryption: glue.JobBookmarksEncryption.clientSideKms(key),
    s3Encryption: glue.S3Encryption.s3Managed(),
  });

  expect(securityConfiguration.cloudWatchEncryptionKey).toBeDefined();
  expect(securityConfiguration.jobBookmarksEncryptionKey?.keyRef.keyArn).toEqual(keyArn);
  expect(securityConfiguration.s3EncryptionKey).toBeUndefined();

  Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 1);

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    Name: 'SecurityConfiguration',
    EncryptionConfiguration: {
      CloudWatchEncryption: {
        CloudWatchEncryptionMode: 'SSE-KMS',
        // auto-created kms key
        KmsKeyArn: stack.resolve(securityConfiguration.cloudWatchEncryptionKey?.keyRef.keyArn),
      },
      JobBookmarksEncryption: {
        JobBookmarksEncryptionMode: 'CSE-KMS',
        // explicitly provided kms key
        KmsKeyArn: keyArn,
      },
      S3Encryptions: [{
        S3EncryptionMode: 'SSE-S3',
      }],
    },
  });
});

test('S3Encryption.kms with an explicit key emits SSE-KMS with that key', () => {
  const stack = new cdk.Stack();
  const keyArn = 'arn:aws:kms:us-west-2:111122223333:key/test-key';
  const key = kms.Key.fromKeyArn(stack, 'ImportedKey', keyArn);

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    s3Encryption: glue.S3Encryption.kms(key),
  });

  expect(securityConfiguration.s3EncryptionKey?.keyRef.keyArn).toEqual(keyArn);

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    EncryptionConfiguration: {
      S3Encryptions: [{
        S3EncryptionMode: 'SSE-KMS',
        KmsKeyArn: keyArn,
      }],
    },
  });
});

test('S3Encryption.kms without a key auto-creates one', () => {
  const stack = new cdk.Stack();

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    s3Encryption: glue.S3Encryption.kms(),
  });

  expect(securityConfiguration.s3EncryptionKey).toBeDefined();
  Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 1);
  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    EncryptionConfiguration: {
      S3Encryptions: [{
        S3EncryptionMode: 'SSE-KMS',
        KmsKeyArn: stack.resolve(securityConfiguration.s3EncryptionKey?.keyRef.keyArn),
      }],
    },
  });
});

test('S3Encryption.s3Managed emits SSE-S3 with no key', () => {
  const stack = new cdk.Stack();

  const securityConfiguration = new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    s3Encryption: glue.S3Encryption.s3Managed(),
  });

  expect(securityConfiguration.s3EncryptionKey).toBeUndefined();
  Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 0);
  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    EncryptionConfiguration: {
      S3Encryptions: [{
        S3EncryptionMode: 'SSE-S3',
      }],
    },
  });
});

test('fromSecurityConfigurationName', () => {
  const stack = new cdk.Stack();
  const name = 'name';

  const securityConfiguration = glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'ImportedSecurityConfiguration', name);

  expect(securityConfiguration.securityConfigurationName).toEqual(name);
});

test('can specify a physical name', () => {
  const stack = new cdk.Stack();
  new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    securityConfigurationName: 'MySecurityConfiguration',
    cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
  });
  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    Name: 'MySecurityConfiguration',
  });
});

test('removalPolicy can be overridden to DESTROY', () => {
  const stack = new cdk.Stack();
  new glue.SecurityConfiguration(stack, 'SecurityConfiguration', {
    cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
    removalPolicy: cdk.RemovalPolicy.DESTROY,
  });

  Template.fromStack(stack).hasResource('AWS::Glue::SecurityConfiguration', {
    DeletionPolicy: 'Delete',
    UpdateReplacePolicy: 'Delete',
  });
});
