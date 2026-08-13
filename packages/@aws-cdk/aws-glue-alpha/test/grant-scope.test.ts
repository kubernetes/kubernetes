import * as cdk from 'aws-cdk-lib';
import { Annotations, Match } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from '../lib';

const WARNING = /granting access to the entire data bucket/;

function newTable(stack: cdk.Stack, props: Partial<glue.S3TableProps> = {}): glue.S3Table {
  const database = new glue.Database(stack, 'Database');
  return new glue.S3Table(stack, 'Table', {
    database,
    columns: [{ name: 'col', type: glue.Schema.STRING }],
    dataFormat: glue.DataFormat.JSON,
    ...props,
  });
}

test('warns when granting on a user-provided bucket with an empty s3Prefix', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'DataBucket');
  const table = newTable(stack, { bucket });

  table.grantRead(new iam.Role(stack, 'Role', { assumedBy: new iam.ServicePrincipal('glue.amazonaws.com') }));

  Annotations.fromStack(stack).hasWarning('/Default/Table', Match.stringLikeRegexp('.*entire data bucket.*grantScopedToWholeBucket.*'));
});

test('does not warn when a user-provided bucket is scoped with an s3Prefix', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'DataBucket');
  const table = newTable(stack, { bucket, s3Prefix: 'data/' });

  table.grantRead(new iam.Role(stack, 'Role', { assumedBy: new iam.ServicePrincipal('glue.amazonaws.com') }));

  Annotations.fromStack(stack).hasNoWarning('/Default/Table', Match.stringLikeRegexp(WARNING.source));
});

test('does not warn when the bucket was created by the table (empty prefix is safe)', () => {
  const stack = new cdk.Stack();
  const table = newTable(stack); // no bucket -> auto-created, table owns the whole bucket

  table.grantReadWrite(new iam.Role(stack, 'Role', { assumedBy: new iam.ServicePrincipal('glue.amazonaws.com') }));

  Annotations.fromStack(stack).hasNoWarning('/Default/Table', Match.stringLikeRegexp(WARNING.source));
});
