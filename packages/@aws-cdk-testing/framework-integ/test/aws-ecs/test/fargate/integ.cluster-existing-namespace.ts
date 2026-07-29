import * as cdk from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import type { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as cloudmap from 'aws-cdk-lib/aws-servicediscovery';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import { EC2_RESTRICT_DEFAULT_SECURITY_GROUP } from 'aws-cdk-lib/cx-api';

/**
 * This integration test verifies that an existing Cloud Map namespace
 * can be used as the default namespace for an ECS cluster.
 */
class ExistingNamespaceStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, false);

    const vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2 });

    // Create namespace separately (simulating an existing namespace)
    const existingNamespace = new cloudmap.PrivateDnsNamespace(this, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    // Create cluster and use the existing namespace
    const cluster = new ecs.Cluster(this, 'EcsCluster', { vpc });

    cluster.addExistingDefaultCloudMapNamespace({
      namespace: existingNamespace,
      useForServiceConnect: true,
    });

    const td = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      cpu: 256,
      memoryLimitMiB: 512,
    });

    td.addContainer('container', {
      containerName: 'web',
      image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      portMappings: [
        {
          name: 'api',
          containerPort: 80,
          appProtocol: ecs.AppProtocol.http,
        },
      ],
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'web',
      }),
    });

    new ecs.FargateService(this, 'Service', {
      taskDefinition: td,
      cluster,
      serviceConnectConfiguration: {
        services: [
          {
            portMappingName: 'api',
            dnsName: 'api',
            port: 80,
          },
        ],
        logDriver: ecs.LogDrivers.awsLogs({
          streamPrefix: 'sc',
        }),
      },
    });
  }
}

const app = new cdk.App({
  analyticsReporting: false,
  postCliContext: {
    '@aws-cdk/aws-ecs:removeDefaultDeploymentAlarm': false,
  },
});
const stack = new ExistingNamespaceStack(app, 'aws-ecs-cluster-existing-namespace');

const test = new integ.IntegTest(app, 'ClusterExistingNamespace', {
  testCases: [stack],
});

// Verify the namespace exists
const listNamespaceCall = test.assertions.awsApiCall('ServiceDiscovery', 'listNamespaces');
listNamespaceCall.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['servicediscovery:ListNamespaces'],
  Resource: ['*'],
});
listNamespaceCall.expect(integ.ExpectedResult.objectLike({
  Namespaces: integ.Match.arrayWith([
    integ.Match.objectLike({
      Name: 'existing.local',
      Type: 'DNS_PRIVATE',
    }),
  ]),
}));
