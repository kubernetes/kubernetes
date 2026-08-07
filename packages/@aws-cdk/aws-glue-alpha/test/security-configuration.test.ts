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
    cloudWatchEncryption: {
      mode: glue.CloudWatchEncryptionMode.KMS,
      kmsKey: key,
    },
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
    cloudWatchEncryption: {
      mode: glue.CloudWatchEncryptionMode.KMS,
    },
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
    cloudWatchEncryption: {
      mode: glue.CloudWatchEncryptionMode.KMS,
    },
    jobBookmarksEncryption: {
      mode: glue.JobBookmarksEncryptionMode.CLIENT_SIDE_KMS,
      kmsKey: key,
    },
    s3Encryption: {
      mode: glue.S3EncryptionMode.S3_MANAGED,
    },
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
    cloudWatchEncryption: {
      mode: glue.CloudWatchEncryptionMode.KMS,
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::Glue::SecurityConfiguration', {
    Name: 'MySecurityConfiguration',
  });
});
