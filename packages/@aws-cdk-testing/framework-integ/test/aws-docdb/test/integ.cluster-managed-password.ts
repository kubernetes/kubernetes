import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as cdk from 'aws-cdk-lib';
import type * as constructs from 'constructs';
import { DatabaseCluster } from 'aws-cdk-lib/aws-docdb';
import { ExpectedResult, IntegTest, Match } from '@aws-cdk/integ-tests-alpha';

class TestStack extends cdk.Stack {
  public readonly cluster: DatabaseCluster;

  constructor(scope: constructs.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'VPC', { maxAzs: 2, restrictDefaultSecurityGroup: false });

    const kmsKey = new kms.Key(this, 'SecretKey', {
      description: 'KMS key for DocumentDB managed master password',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.cluster = new DatabaseCluster(this, 'Database', {
      manageMasterUserPassword: true,
      masterUser: {
        username: 'docdbuser',
      },
      masterUserSecretKmsKey: kmsKey,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.LARGE),
      vpc,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
  }
}

const app = new cdk.App();

const stack = new TestStack(app, 'aws-cdk-docdb-cluster-managed-password');

const test = new IntegTest(app, 'aws-cdk-docdb-cluster-managed-password-integ', {
  testCases: [stack],
});

// Verify that the managed master user secret was created for the cluster
const describeClusters = test.assertions.awsApiCall('DocDB', 'describeDBClusters', {
  DBClusterIdentifier: stack.cluster.clusterIdentifier,
});
describeClusters.expect(ExpectedResult.objectLike({
  DBClusters: Match.arrayWith([
    Match.objectLike({
      MasterUserSecret: Match.objectLike({
        SecretArn: Match.stringLikeRegexp('^arn:aws:secretsmanager:'),
      }),
    }),
  ]),
}));
