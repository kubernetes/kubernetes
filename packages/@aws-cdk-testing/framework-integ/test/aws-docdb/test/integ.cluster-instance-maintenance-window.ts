import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as cdk from 'aws-cdk-lib';
import { ExpectedResult, IntegTest } from '@aws-cdk/integ-tests-alpha';
import type * as constructs from 'constructs';
import { DatabaseCluster } from 'aws-cdk-lib/aws-docdb';
import { DOCDB_ENGINE_VERSION } from './docdb-integ-test-constraints';

const CLUSTER_MAINTENANCE_WINDOW = 'tue:04:17-tue:04:47';
const INSTANCE_MAINTENANCE_WINDOW = 'sat:09:00-sat:09:30';

class TestStack extends cdk.Stack {
  public readonly cluster: DatabaseCluster;

  constructor(scope: constructs.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'VPC', { maxAzs: 2, restrictDefaultSecurityGroup: false });

    this.cluster = new DatabaseCluster(this, 'Database', {
      engineVersion: DOCDB_ENGINE_VERSION,
      masterUser: {
        username: 'docdb',
        password: cdk.SecretValue.unsafePlainText('7959866cacc02c2d243ecfe177464fe6'),
      },
      instances: 2,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.LARGE),
      vpc,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      preferredMaintenanceWindow: CLUSTER_MAINTENANCE_WINDOW,
      instanceMaintenanceWindow: INSTANCE_MAINTENANCE_WINDOW,
    });
  }
}

const app = new cdk.App();
const stack = new TestStack(app, 'aws-cdk-docdb-instance-maintenance-window');

const integ = new IntegTest(app, 'DocdbClusterInstanceMaintenanceWindowInteg', {
  testCases: [stack],
});

// Verify the cluster-level maintenance window is applied to the DB cluster.
integ.assertions.awsApiCall('DocDB', 'describeDBClusters', {
  DBClusterIdentifier: stack.cluster.clusterIdentifier,
}).expect(ExpectedResult.objectLike({
  DBClusters: [
    {
      PreferredMaintenanceWindow: CLUSTER_MAINTENANCE_WINDOW,
    },
  ],
}));

// Verify the instance-level maintenance window is applied to every auto-created instance.
integ.assertions.awsApiCall('DocDB', 'describeDBInstances', {
  DBInstanceIdentifier: stack.cluster.instanceIdentifiers[0],
}).expect(ExpectedResult.objectLike({
  DBInstances: [
    {
      PreferredMaintenanceWindow: INSTANCE_MAINTENANCE_WINDOW,
    },
  ],
}));

integ.assertions.awsApiCall('DocDB', 'describeDBInstances', {
  DBInstanceIdentifier: stack.cluster.instanceIdentifiers[1],
}).expect(ExpectedResult.objectLike({
  DBInstances: [
    {
      PreferredMaintenanceWindow: INSTANCE_MAINTENANCE_WINDOW,
    },
  ],
}));
