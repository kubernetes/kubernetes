import * as cdk from 'aws-cdk-lib';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as glue from '../lib';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-glue-security-configuration');

const key = new kms.Key(stack, 'Key');

// SecurityConfiguration for all 3 (s3, cloudwatch and job bookmarks) in modes requiring kms keys
new glue.SecurityConfiguration(stack, 'KeyedSC', {
  securityConfigurationName: 'KeyedSC',
  jobBookmarksEncryption: glue.JobBookmarksEncryption.clientSideKms(key),
  cloudWatchEncryption: glue.CloudWatchEncryption.kms(key),
  s3Encryption: glue.S3Encryption.kms(key),
});

// SecurityConfiguration for all 3 (s3, cloudwatch and job bookmarks) in modes requiring kms keys without one provided
new glue.SecurityConfiguration(stack, 'KeylessSC', {
  securityConfigurationName: 'KeylessSC',
  jobBookmarksEncryption: glue.JobBookmarksEncryption.clientSideKms(),
  cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
  s3Encryption: glue.S3Encryption.kms(),
});

// SecurityConfiguration for s3 not requiring kms key
new glue.SecurityConfiguration(stack, 'S3SC', {
  securityConfigurationName: 'S3SC',
  s3Encryption: glue.S3Encryption.s3Managed(),
});

app.synth();
