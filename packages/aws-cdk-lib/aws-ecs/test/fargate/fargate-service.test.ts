import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Annotations, Match, Template } from '../../../assertions';
import * as appscaling from '../../../aws-applicationautoscaling';
import * as batch from '../../../aws-batch';
import * as cloudwatch from '../../../aws-cloudwatch';
import * as ec2 from '../../../aws-ec2';
import * as elb from '../../../aws-elasticloadbalancing';
import * as elbv2 from '../../../aws-elasticloadbalancingv2';
import * as iam from '../../../aws-iam';
import * as kms from '../../../aws-kms';
import * as logs from '../../../aws-logs';
import * as s3 from '../../../aws-s3';
import * as secretsmanager from '../../../aws-secretsmanager';
import * as cloudmap from '../../../aws-servicediscovery';
import * as cdk from '../../../core';
import { App } from '../../../core';
import * as cxapi from '../../../cx-api';
import { ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME } from '../../../cx-api';
import * as ecs from '../../lib';
import type { ServiceConnectProps } from '../../lib/base/base-service';
import { DeploymentControllerType, LaunchType, PropagatedTagSource } from '../../lib/base/base-service';
import { ServiceManagedVolume } from '../../lib/base/service-managed-volume';
import { acknowledgeTestValidationRules, addDefaultCapacityProvider } from '../util';

describe('fargate service', () => {
  describe('When creating a Fargate Service', () => {
    test('with only required properties set, it correctly sets default properties', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      const service = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        LaunchType: LaunchType.FARGATE,
        EnableECSManagedTags: false,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'FargateServiceSecurityGroup0A0E79CB',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
        AvailabilityZoneRebalancing: Match.absent(),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Default/FargateService/SecurityGroup',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      expect(service.node.defaultChild).toBeDefined();
    });

    test('can create service with default settings if VPC only has public subnets', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {
        subnetConfiguration: [
          {
            cidrMask: 28,
            name: 'public-only',
            subnetType: ec2.SubnetType.PUBLIC,
          },
        ],
      });
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      // WHEN
      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // THEN -- did not throw
    });

    testDeprecated('does not set launchType when capacity provider strategies specified (deprecated)', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        capacityProviders: ['FARGATE', 'FARGATE_SPOT'],
      });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        capacityProviderStrategies: [
          {
            capacityProvider: 'FARGATE_SPOT',
            weight: 2,
          },
          {
            capacityProvider: 'FARGATE',
            weight: 1,
          },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
        CapacityProviders: Match.absent(),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
        CapacityProviders: ['FARGATE', 'FARGATE_SPOT'],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        // no launch type
        CapacityProviderStrategy: [
          {
            CapacityProvider: 'FARGATE_SPOT',
            Weight: 2,
          },
          {
            CapacityProvider: 'FARGATE',
            Weight: 1,
          },
        ],
        EnableECSManagedTags: false,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'FargateServiceSecurityGroup0A0E79CB',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
      });
    });

    [false, undefined].forEach((value) => {
      test('set cloudwatch permissions based on falsy feature flag when no cloudwatch log configured', () => {
        // GIVEN
        const app = new App(
          {
            context: {
              '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': value,
            },
          },
        );
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'Ec2TaskDef');

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        new ecs.FargateService(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          enableExecuteCommand: true,
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
          PolicyDocument: {
            Statement: [
              {
                Action: [
                  'ssmmessages:CreateControlChannel',
                  'ssmmessages:CreateDataChannel',
                  'ssmmessages:OpenControlChannel',
                  'ssmmessages:OpenDataChannel',
                ],
                Effect: 'Allow',
                Resource: '*',
              },
              {
                Action: 'logs:DescribeLogGroups',
                Effect: 'Allow',
                Resource: '*',
              },
              {
                Action: [
                  'logs:CreateLogStream',
                  'logs:DescribeLogStreams',
                  'logs:PutLogEvents',
                ],
                Effect: 'Allow',
                Resource: '*',
              },
            ],
            Version: '2012-10-17',
          },
          PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
          Roles: [
            {
              Ref: 'Ec2TaskDefTaskRole400FA349',
            },
          ],
        });
      });
    });

    test('set cloudwatch permissions based on true feature flag when no cloudwatch log configured', () => {
      // GIVEN
      const app = new App(
        {
          context: {
            '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': true,
          },
        },
      );
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('set cloudwatch permissions based on true feature flag when cloudwatch log is configured', () => {
      // GIVEN
      const app = new App(
        {
          context: {
            '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': true,
          },
        },
      );
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchLogGroup: new logs.LogGroup(stack, 'LogGroup'),
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('does not set launchType when capacity provider strategies specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
      });
      cluster.enableFargateCapacityProviders();

      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        capacityProviderStrategies: [
          {
            capacityProvider: 'FARGATE_SPOT',
            weight: 2,
          },
          {
            capacityProvider: 'FARGATE',
            weight: 1,
          },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
        CapacityProviders: Match.absent(),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
        CapacityProviders: ['FARGATE', 'FARGATE_SPOT'],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        // no launch type
        LaunchType: Match.absent(),
        CapacityProviderStrategy: [
          {
            CapacityProvider: 'FARGATE_SPOT',
            Weight: 2,
          },
          {
            CapacityProvider: 'FARGATE',
            Weight: 1,
          },
        ],
        EnableECSManagedTags: false,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'FargateServiceSecurityGroup0A0E79CB',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
      });
    });

    test('with custom cloudmap namespace', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      const cloudMapNamespace = new cloudmap.PrivateDnsNamespace(stack, 'TestCloudMapNamespace', {
        name: 'scorekeep.com',
        vpc,
      });

      cdk.Validations.of(stack).acknowledge(
        { id: 'CloudFormation-Validate::F3034', reason: 'failureThreshold intentionally exceeds the CloudFormation maximum of 10; the test asserts it is passed through to HealthCheckCustomConfig unmodified' },
      );

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
          failureThreshold: 20,
          cloudMapNamespace,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'A',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'TestCloudMapNamespace1FB9B446',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 20,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'TestCloudMapNamespace1FB9B446',
            'Id',
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::PrivateDnsNamespace', {
        Name: 'scorekeep.com',
        Vpc: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });
    });

    test('with user-provided cloudmap service', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      const cloudMapNamespace = new cloudmap.PrivateDnsNamespace(stack, 'TestCloudMapNamespace', {
        name: 'scorekeep.com',
        vpc,
      });

      const cloudMapService = new cloudmap.Service(stack, 'Service', {
        name: 'service-name',
        namespace: cloudMapNamespace,
        dnsRecordType: cloudmap.DnsRecordType.SRV,
      });

      const ecsService = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // WHEN
      ecsService.associateCloudMapService({
        service: cloudMapService,
        container: container,
        containerPort: 8000,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            ContainerName: 'web',
            ContainerPort: 8000,
            RegistryArn: { 'Fn::GetAtt': ['ServiceDBC79909', 'Arn'] },
          },
        ],
      });
    });

    test('errors when more than one service registry used', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      const cloudMapNamespace = new cloudmap.PrivateDnsNamespace(stack, 'TestCloudMapNamespace', {
        name: 'scorekeep.com',
        vpc,
      });

      const ecsService = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      ecsService.enableCloudMap({
        cloudMapNamespace,
      });

      const cloudMapService = new cloudmap.Service(stack, 'Service', {
        name: 'service-name',
        namespace: cloudMapNamespace,
        dnsRecordType: cloudmap.DnsRecordType.SRV,
      });

      // WHEN / THEN
      expect(() => {
        ecsService.associateCloudMapService({
          service: cloudMapService,
          container: container,
          containerPort: 8000,
        });
      }).toThrow(/at most one service registry/i);
    });

    test('with all properties set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      cdk.Validations.of(stack).acknowledge(
        { id: 'CloudFormation-Validate::F3034', reason: 'failureThreshold intentionally exceeds the CloudFormation maximum of 10; this test exercises every property, not valid value ranges' },
      );

      const svc = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        desiredCount: 2,
        assignPublicIp: true,
        cloudMapOptions: {
          name: 'myapp',
          dnsRecordType: cloudmap.DnsRecordType.A,
          dnsTtl: cdk.Duration.seconds(50),
          failureThreshold: 20,
        },
        healthCheckGracePeriod: cdk.Duration.seconds(60),
        maxHealthyPercent: 150,
        minHealthyPercent: 55,
        deploymentController: {
          type: ecs.DeploymentControllerType.ECS,
        },
        circuitBreaker: { rollback: true },
        securityGroups: [new ec2.SecurityGroup(stack, 'SecurityGroup1', {
          allowAllOutbound: true,
          description: 'Example',
          securityGroupName: 'Bob',
          vpc,
        })],
        serviceName: 'bonjour',
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      // THEN
      expect(svc.cloudMapService).toBeDefined();

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 150,
          MinimumHealthyPercent: 55,
          DeploymentCircuitBreaker: {
            Enable: true,
            Rollback: true,
          },
        },
        DeploymentController: {
          Type: ecs.DeploymentControllerType.ECS,
        },
        DesiredCount: 2,
        HealthCheckGracePeriodSeconds: 60,
        LaunchType: LaunchType.FARGATE,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'ENABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'SecurityGroup1F554B36F',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPublicSubnet1SubnetF6608456',
              },
              {
                Ref: 'MyVpcPublicSubnet2Subnet492B6BFB',
              },
            ],
          },
        },
        ServiceName: 'bonjour',
        ServiceRegistries: [
          {
            RegistryArn: {
              'Fn::GetAtt': [
                'FargateServiceCloudmapService9544B753',
                'Arn',
              ],
            },
          },
        ],
        AvailabilityZoneRebalancing: 'ENABLED',
      });
    });

    test('throws when task definition is not Fargate compatible', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.TaskDefinition(stack, 'Ec2TaskDef', {
        compatibility: ecs.Compatibility.EC2,
      });
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryReservationMiB: 10,
      });

      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
        });
      }).toThrow(/Supplied TaskDefinition is not configured for compatibility with Fargate/);
    });

    test('throws whith secret json field on unsupported platform version', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaksDef');
      const secret = new secretsmanager.Secret(stack, 'Secret');
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        secrets: {
          SECRET_KEY: ecs.Secret.fromSecretsManager(secret, 'specificKey'),
        },
      });

      // Errors on validation, not on construction.
      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        platformVersion: ecs.FargatePlatformVersion.VERSION1_3,
      });

      // THEN
      expect(() => {
        Template.fromStack(stack);
      }).toThrow(new RegExp(`uses at least one container that references a secret JSON field.+platform version ${ecs.FargatePlatformVersion.VERSION1_4} or later`));
    });

    test('ignore task definition and launch type if deployment controller is set to be EXTERNAL', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: DeploymentControllerType.EXTERNAL,
        },
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/FargateService', 'taskDefinition and launchType are blanked out when using external deployment controller. [ack: @aws-cdk/aws-ecs:externalDeploymentController]');
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        DeploymentController: {
          Type: 'EXTERNAL',
        },
        EnableECSManagedTags: false,
      });
    });

    test('add warning to annotations if circuitBreaker is specified with a non-ECS DeploymentControllerType', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      const service = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: DeploymentControllerType.EXTERNAL,
        },
        circuitBreaker: { rollback: true },
      });
      app.synth();

      // THEN
      expect(service.node.metadata[1].data).toEqual('Deployment circuit breaker requires the ECS deployment controller.');
      expect(service.node.metadata[0].data).toEqual('taskDefinition and launchType are blanked out when using external deployment controller. [ack: @aws-cdk/aws-ecs:externalDeploymentController]');
    });

    test('errors when no container specified on task definition', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      // Errors on validation, not on construction.
      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // THEN
      expect(() => {
        Template.fromStack(stack);
      }).toThrow(/one essential container/);
    });

    test('errors when platform version does not support containers which references secret JSON field', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef', {
        runtimePlatform: {
          operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
          cpuArchitecture: ecs.CpuArchitecture.ARM64,
        },
        memoryLimitMiB: 512,
        cpu: 256,
      });

      // Errors on validation, not on construction.
      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        platformVersion: ecs.FargatePlatformVersion.VERSION1_2,
      });

      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('somecontainer'),
        secrets: {
          envName: batch.Secret.fromSecretsManager(new secretsmanager.Secret(stack, 'testSecret'), 'secretField'),
        },
      });

      // THEN
      expect(() => {
        Template.fromStack(stack);
      }).toThrow(/This feature requires platform version/);
    });

    test('errors when platform version does not support ephemeralStorageGiB', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef', {
        runtimePlatform: {
          operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
          cpuArchitecture: ecs.CpuArchitecture.ARM64,
        },
        memoryLimitMiB: 512,
        cpu: 256,
        ephemeralStorageGiB: 100,
      });

      // WHEN
      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
          platformVersion: ecs.FargatePlatformVersion.VERSION1_2,
        });
      }).toThrow(/The ephemeralStorageGiB feature requires platform version/);
    });

    test('errors when platform version does not support pidMode', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef', {
        runtimePlatform: {
          operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
          cpuArchitecture: ecs.CpuArchitecture.ARM64,
        },
        memoryLimitMiB: 512,
        cpu: 256,
        pidMode: ecs.PidMode.TASK,
      });

      // WHEN
      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
          platformVersion: ecs.FargatePlatformVersion.VERSION1_2,
        });
      }).toThrow(/The pidMode feature requires platform version/);
    });

    test('allows adding the default container after creating the service', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // Add the container *after* creating the service
      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('somecontainer'),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::TaskDefinition', {
        ContainerDefinitions: [
          Match.objectLike({
            Name: 'main',
          }),
        ],
      });
    });

    test('allows specifying assignPublicIP as enabled', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        assignPublicIp: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'ENABLED',
          },
        },
      });
    });

    test('allows specifying 0 for minimumHealthyPercent', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        assignPublicIp: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'ENABLED',
          },
        },
      });
    });

    test('throws when availability zone rebalancing is enabled and maxHealthyPercent is 100', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
          availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
          maxHealthyPercent: 100,
        });
      }).toThrow(/requires maxHealthyPercent > 100/);
    });

    test('sets task definition to family when CODE_DEPLOY deployment controller is specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: ecs.DeploymentControllerType.CODE_DEPLOY,
        },
      });

      // THEN
      Template.fromStack(stack).hasResource('AWS::ECS::Service', {
        Properties: {
          TaskDefinition: 'FargateTaskDef',
          DeploymentController: {
            Type: 'CODE_DEPLOY',
          },
        },
        DependsOn: [
          'FargateTaskDefC6FB60B4',
          'FargateTaskDefTaskRole0B257552',
        ],
      });
    });

    testDeprecated('throws when securityGroup and securityGroups are supplied', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const securityGroup1 = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
        allowAllOutbound: true,
        description: 'Example',
        securityGroupName: 'Bingo',
        vpc,
      });
      const securityGroup2 = new ec2.SecurityGroup(stack, 'SecurityGroup2', {
        allowAllOutbound: false,
        description: 'Example',
        securityGroupName: 'Rolly',
        vpc,
      });

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
          securityGroup: securityGroup1,
          securityGroups: [securityGroup2],
        });
      }).toThrow(/Only one of SecurityGroup or SecurityGroups can be populated./);
    });

    test('with multiple securty groups, it correctly updates cloudformation template', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const securityGroup1 = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
        allowAllOutbound: true,
        description: 'Example',
        securityGroupName: 'Bingo',
        vpc,
      });
      const securityGroup2 = new ec2.SecurityGroup(stack, 'SecurityGroup2', {
        allowAllOutbound: false,
        description: 'Example',
        securityGroupName: 'Rolly',
        vpc,
      });

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        securityGroups: [securityGroup1, securityGroup2],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        LaunchType: LaunchType.FARGATE,
        EnableECSManagedTags: false,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'SecurityGroup1F554B36F',
                  'GroupId',
                ],
              },
              {
                'Fn::GetAtt': [
                  'SecurityGroup23BE86BB7',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Example',
        GroupName: 'Bingo',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Example',
        GroupName: 'Rolly',
        SecurityGroupEgress: [
          {
            CidrIp: '255.255.255.255/32',
            Description: 'Disallow all traffic',
            FromPort: 252,
            IpProtocol: 'icmp',
            ToPort: 86,
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });
    });

    test('with deployment alarms', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');

      new ecs.FargateService(stack, 'ExternalService', {
        cluster,
        taskDefinition,
        deploymentAlarms: {
          alarmNames: [myAlarm.alarmName],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        DeploymentConfiguration: {
          Alarms: {
            Enable: true,
            Rollback: true,
            AlarmNames: [myAlarm.alarmName],
          },
        },
      });
    });

    test('no network configuration with external deployment controller', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'ExternalService', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: ecs.DeploymentControllerType.EXTERNAL,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        NetworkConfiguration: Match.absent(),
      });
    });

    test('network configuration exists when explicitly specifying a deployment controller type other than EXTERNAL', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'ExternalService', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: ecs.DeploymentControllerType.ECS,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'ExternalServiceSecurityGroup45E0A4FC',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
      });
    });

    test('warning if minHealthyPercent not set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/FargateService', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });

    test('no warning if minHealthyPercent set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        minHealthyPercent: 50,
      });

      // THEN
      Annotations.fromStack(stack).hasNoWarning('/Default/FargateService', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });
  });

  describe('when enabling service connect', () => {
    describe('when validating service connect configurations', () => {
      let service: ecs.FargateService;

      beforeEach(() => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        });

        service = new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
        });
      });

      test('throws an exception if serviceconnectservice.port is a string and it does not exists on the task definition', () => {
        // GIVEN
        const config: ServiceConnectProps = {
          services: [
            {
              portMappingName: '100',
              dnsName: 'backend.prod',
            },
          ],
          namespace: 'test namespace',
        };
        expect(() => {
          service.enableServiceConnect(config);
        }).toThrow(/Port Mapping '100' does not exist on the task definition./);
      });

      test('throws an exception when adding multiple services without different discovery names', () => {
        // GIVEN
        service.taskDefinition.addContainer('mobile', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          portMappings: [
            {
              containerPort: 100,
              name: 'abc',
            },
          ],
        });
        const config: ServiceConnectProps = {
          services: [
            {
              portMappingName: 'abc',
              dnsName: 'backend.prod',
              port: 5005,
            },
            {
              portMappingName: 'abc',
              dnsName: 'backend.prod.local',
            },
          ],
          namespace: 'test namespace',
        };
        expect(() => {
          service.enableServiceConnect(config);
        }).toThrow(/Cannot create multiple services with the discoveryName 'abc'./);
      });

      test('throws an exception if ingressPortOverride is not valid.', () => {
        // GIVEN
        service.taskDefinition.addContainer('mobile', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          portMappings: [
            {
              containerPort: 100,
              name: '100',
            },
          ],
        });
        const config: ServiceConnectProps = {
          services: [
            {
              portMappingName: '100',
              dnsName: 'backend.prod',
              port: 5005,
              ingressPortOverride: 100000,
            },
          ],
          namespace: 'test namespace',
        };
        expect(() => {
          service.enableServiceConnect(config);
        }).toThrow(/ingressPortOverride 100000 is not valid./);
      });

      test('throws an exception if Client Alias port is not valid', () => {
        // GIVEN
        service.taskDefinition.addContainer('mobile', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          portMappings: [
            {
              containerPort: 100,
              name: '100',
            },
          ],
        });
        const config: ServiceConnectProps = {
          services: [
            {
              portMappingName: '100',
              dnsName: 'backend.prod',
              port: 100000,
              ingressPortOverride: 3000,
            },
          ],
          namespace: 'test namespace',
        };
        expect(() => {
          service.enableServiceConnect(config);
        }).toThrow(/Client Alias port 100000 is not valid./);
      });
    });

    describe('when creating a FargateService with service connect', () => {
      let service: ecs.FargateService;
      let stack: cdk.Stack;
      let cluster: ecs.Cluster;

      beforeEach(() => {
        // GIVEN
        stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          portMappings: [
            {
              containerPort: 80,
              name: 'api',
            },
          ],
        });

        service = new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
        });
      });

      test('service connect cannot be enabled twice', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect();

        // THEN

        expect(() => {
          service.enableServiceConnect({});
        }).toThrow('Service connect configuration cannot be specified more than once.');
      });

      test('client alias port is defaulted to containerport', () => {
        service.enableServiceConnect({
          namespace: 'cool',
          services: [
            {
              portMappingName: 'api',
            },
          ],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                ClientAliases: [
                  {
                    Port: 80,
                  },
                ],
              },
            ],
          },
        });
      });

      test('with explicit enable', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect({});

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
          },
        });
      });

      test('with explicit enable and no props', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect();

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
          },
        });
      });

      test('explicit enable and non default namespace', () => {
        // WHEN
        const ns = new cloudmap.HttpNamespace(stack, 'ns', {
          name: 'cool',
        });
        service.enableServiceConnect({
          namespace: ns.namespaceName,
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
          },
        });
      });

      test('namespace inferred from cluster', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect({});

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
          },
        });
      });

      test('namespace inferred from cluster; empty props', () => {
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect();

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
          },
        });
      });

      test('no namespace errors out', () => {
        // THEN
        expect(() => {
          service.enableServiceConnect({});
        }).toThrow();
      });

      test('error when enabling service connect with no container', () => {
        // GIVEN
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'td2');
        const svc = new ecs.FargateService(stack, 'svc2', {
          cluster,
          taskDefinition,
        });
        expect(() => {
          svc.enableServiceConnect({
            logDriver: ecs.LogDrivers.awsLogs({
              streamPrefix: 'sc',
            }),
          });
        }).toThrow('Task definition must have at least one container to enable service connect.');
      });

      test('with all options exercised', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
              discoveryName: 'svc',
              ingressPortOverride: 1000,
              port: 80,
              dnsName: 'api',
              idleTimeout: cdk.Duration.seconds(10),
              perRequestTimeout: cdk.Duration.seconds(10),
            },
          ],
          namespace: 'cool',
          logDriver: ecs.LogDrivers.awsLogs({
            streamPrefix: 'sc',
          }),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                IngressPortOverride: 1000,
                DiscoveryName: 'svc',
                ClientAliases: [
                  {
                    Port: 80,
                    DnsName: 'api',
                  },
                ],
                Timeout: {
                  IdleTimeoutSeconds: 10,
                  PerRequestTimeoutSeconds: 10,
                },
              },
            ],
            LogConfiguration: {
              LogDriver: 'awslogs',
              Options: {
                'awslogs-stream-prefix': 'sc',
              },
            },
          },
        });
      });

      test('can set idleTimeout without perRequestTimeout', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
              idleTimeout: cdk.Duration.seconds(10),
            },
          ],
          namespace: 'cool',
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                Timeout: {
                  IdleTimeoutSeconds: 10,
                  PerRequestTimeoutSeconds: Match.absent(),
                },
              },
            ],
          },
        });
      });

      test('can set perRequestTimeout without idleTimeout', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
              perRequestTimeout: cdk.Duration.seconds(10),
            },
          ],
          namespace: 'cool',
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                Timeout: {
                  IdleTimeoutSeconds: Match.absent(),
                  PerRequestTimeoutSeconds: 10,
                },
              },
            ],
          },
        });
      });

      test('can set idleTimeout and perRequestTimeout to 0', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
              idleTimeout: cdk.Duration.seconds(0),
              perRequestTimeout: cdk.Duration.seconds(0),
            },
          ],
          namespace: 'cool',
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                Timeout: {
                  IdleTimeoutSeconds: 0,
                  PerRequestTimeoutSeconds: 0,
                },
              },
            ],
          },
        });
      });

      test('throws if idleTimeout is less than 1 second and not 0', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        expect(() => {
          service.enableServiceConnect({
            services: [
              {
                portMappingName: 'api',
                idleTimeout: cdk.Duration.millis(10),
              },
            ],
            namespace: 'cool',
          });
        }).toThrow(/idleTimeout must be at least 1 second or 0 to disable it, got 10ms./);
      });

      test('throws if perRequestTimeout is less than 1 second and not 0', () => {
        // WHEN
        new cloudmap.HttpNamespace(stack, 'httpnamespace', {
          name: 'cool',
        });
        expect(() => {
          service.enableServiceConnect({
            services: [
              {
                portMappingName: 'api',
                perRequestTimeout: cdk.Duration.millis(10),
              },
            ],
            namespace: 'cool',
          });
        }).toThrow(/perRequestTimeout must be at least 1 second or 0 to disable it, got 10ms./);
      });

      test('with no alias name', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
              port: 80,
            },
          ],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                ClientAliases: [
                  {
                    Port: 80,
                  },
                ],
              },
            ],
          },
        });
      });

      test('with no alias specified', () => {
        // WHEN
        cluster.addDefaultCloudMapNamespace({
          name: 'cool',
        });
        service.enableServiceConnect({
          services: [
            {
              portMappingName: 'api',
            },
          ],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          ServiceConnectConfiguration: {
            Enabled: true,
            Namespace: 'cool',
            Services: [
              {
                PortName: 'api',
                ClientAliases: [
                  {
                    Port: 80,
                  },
                ],
              },
            ],
          },
        });
      });
    });
  });

  describe('When setting up a service volume configurations', ()=>{
    let service: ecs.FargateService;
    let stack: cdk.Stack;
    let cluster: ecs.Cluster;
    let taskDefinition: ecs.TaskDefinition;
    let container: ecs.ContainerDefinition;
    let role: iam.IRole;
    let app: cdk.App;

    beforeEach(() => {
      // GIVEN
      app = new cdk.App();
      stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
      });
      container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });
      service = new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });
    });
    test('success when adding a service volume', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          role: role,
          size: cdk.Size.gibibytes(20),
          fileSystemType: ecs.FileSystemType.XFS,
          tagSpecifications: [{
            tags: {
              purpose: 'production',
            },
            propagateTags: ecs.EbsPropagatedTagSource.SERVICE,
          }],
        },
      }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
              SizeInGiB: 20,
              FilesystemType: 'xfs',
              TagSpecifications: [
                {
                  PropagateTags: 'SERVICE',
                  ResourceType: 'volume',
                  Tags: [
                    {
                      Key: 'purpose',
                      Value: 'production',
                    },
                  ],
                },
              ],
            },
            Name: 'nginx-vol',
          },
        ],
      });
    });

    test('success when mounting via ServiceManagedVolume', () => {
      // WHEN
      const volume = new ServiceManagedVolume(stack, 'EBS Volume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          role: role,
          size: cdk.Size.gibibytes(20),
          tagSpecifications: [{
            tags: {
              purpose: 'production',
            },
            propagateTags: ecs.EbsPropagatedTagSource.SERVICE,
          }],
        },
      });
      taskDefinition.addVolume(volume);
      service.addVolume(volume);
      volume.mountIn(container, {
        containerPath: '/var/lib',
        readOnly: false,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
              SizeInGiB: 20,
              TagSpecifications: [
                {
                  PropagateTags: 'SERVICE',
                  ResourceType: 'volume',
                  Tags: [
                    {
                      Key: 'purpose',
                      Value: 'production',
                    },
                  ],
                },
              ],
            },
            Name: 'nginx-vol',
          },
        ],
      });
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::TaskDefinition', {
        ContainerDefinitions: [
          {
            MountPoints: [
              {
                ContainerPath: '/var/lib',
                ReadOnly: false,
                SourceVolume: 'nginx-vol',
              },
            ],
          },
        ],
        Volumes: [
          {
            Name: 'nginx-vol',
            ConfiguredAtLaunch: true,
          },
        ],
      });
    });

    test('throw an error when multiple volume configurations are added to ECS service', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });
      const vol1 = new ServiceManagedVolume(stack, 'EBSVolume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          fileSystemType: ecs.FileSystemType.XFS,
          size: cdk.Size.gibibytes(15),
        },
      });
      const vol2 = new ServiceManagedVolume(stack, 'ebs1', {
        name: 'ebs1',
        managedEBSVolume: {
          fileSystemType: ecs.FileSystemType.XFS,
          size: cdk.Size.gibibytes(15),
        },
      });
      service.addVolume(vol1);
      service.addVolume(vol2);
      expect(() => {
        app.synth();
      }).toThrow(/Only one EBS volume can be specified for 'volumeConfigurations', got: 2/);
    });

    test('create a default ebsrole when not provided', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          size: cdk.Size.gibibytes(20),
          fileSystemType: ecs.FileSystemType.XFS,
          tagSpecifications: [{
            tags: {
              purpose: 'production',
            },
            propagateTags: ecs.EbsPropagatedTagSource.SERVICE,
          }],
        },
      }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['EBSVolumeEBSRoleD38B9F31', 'Arn'] },
              SizeInGiB: 20,
              FilesystemType: 'xfs',
              TagSpecifications: [
                {
                  PropagateTags: 'SERVICE',
                  ResourceType: 'volume',
                  Tags: [
                    {
                      Key: 'purpose',
                      Value: 'production',
                    },
                  ],
                },
              ],
            },
            Name: 'nginx-vol',
          },
        ],
      });
    });

    test('throw an error when both size and snapshotId are not provided', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
          },
        }));
      }).toThrow("'size' or 'snapShotId' must be specified");
    });

    test('throw an error snapshot does not match pattern', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            snapShotId: 'snap-0d48decab5c493eee_',
          },
        }));
      }).toThrow("'snapshotId' does match expected pattern. Expected 'snap-<hexadecmial value>' (ex: 'snap-05abe246af') or Token, got: snap-0d48decab5c493eee_");
    });

    test('success when snapshotId matches the pattern', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });
      const vol = new ServiceManagedVolume(stack, 'EBS Volume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          fileSystemType: ecs.FileSystemType.XFS,
          snapShotId: 'snap-0d48decab5c493eee',
        },
      });
      service.addVolume(vol);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['EBSVolumeEBSRoleD38B9F31', 'Arn'] },
              SnapshotId: 'snap-0d48decab5c493eee',
              FilesystemType: 'xfs',
            },
            Name: 'nginx-vol',
          },
        ],
      });
    });

    test('throw an error when size is greater than 16384 for gp2', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            size: cdk.Size.gibibytes(16390),
          },
        }));
      }).toThrow(/'gp2' volumes must have a size between 1 and 16384 GiB, got 16390 GiB/);
    });

    test('throw an error when size is less than 4 for volume type io1', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.IO1,
            size: cdk.Size.gibibytes(0),
          },
        }));
      }).toThrow(/'io1' volumes must have a size between 4 and 16384 GiB, got 0 GiB/);
    });
    test('throw an error when size is greater than 1024 for volume type standard', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.STANDARD,
            size: cdk.Size.gibibytes(1500),
          },
        }));
      }).toThrow(/'standard' volumes must have a size between 1 and 1024 GiB, got 1500 GiB/);
    });

    test('throw an error if throughput is configured for volumetype gp2', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            size: cdk.Size.gibibytes(10),
            throughput: 0,
          },
        }));
      }).toThrow(/'throughput' can only be configured with gp3 volume type, got gp2/);
    });

    test('throw an error if iops is not supported for volume type sc1', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.SC1,
            size: cdk.Size.gibibytes(125),
            iops: 0,
          },
        }));
      }).toThrow(/'iops' cannot be specified with sc1, st1, gp2 and standard volume types, got sc1/);
    });

    test('throw an error if iops is not supported for volume type sc1', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            size: cdk.Size.gibibytes(125),
            iops: 0,
          },
        }));
      }).toThrow(/'iops' cannot be specified with sc1, st1, gp2 and standard volume types, got gp2/);
    });

    test('throw an error if if iops is required but not provided for volume type io2', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.IO2,
            size: cdk.Size.gibibytes(125),
          },
        }));
      }).toThrow(/'iops' must be specified with io1 or io2 volume types, got io2/);
    });

    test('throw an error if if iops is less than 100 for volume type io2', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.IO2,
            size: cdk.Size.gibibytes(125),
            iops: 0,
          },
        }));
      }).toThrow("io2' volumes must have 'iops' between 100 and 256000, got 0");
    });

    test('throw an error if if iops is greater than 256000 for volume type io2', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.IO2,
            size: cdk.Size.gibibytes(125),
            iops: 256001,
          },
        }));
      }).toThrow("io2' volumes must have 'iops' between 100 and 256000, got 256001");
    });

    test('configure volume initialization role', () => {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      const vol = new ServiceManagedVolume(stack, 'EBS Volume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          fileSystemType: ecs.FileSystemType.XFS,
          snapShotId: 'snap-0d48decab5c493eee',
          volumeInitializationRate: cdk.Size.mebibytes(100),
        },
      });
      service.addVolume(vol);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['EBSVolumeEBSRoleD38B9F31', 'Arn'] },
              SnapshotId: 'snap-0d48decab5c493eee',
              FilesystemType: 'xfs',
              VolumeInitializationRate: 100,
            },
            Name: 'nginx-vol',
          },
        ],
      });
    });

    test.each([
      cdk.Size.mebibytes(99),
      cdk.Size.mebibytes(301),
    ])('throw an error if if volume initialization rate is out of range for 100-300 MiB/s', (volumeInitializationRate) => {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            snapShotId: 'snap-0d48decab5c493eee',
            volumeInitializationRate,
          },
        }));
      }).toThrow(`'volumeInitializationRate' must be between 100 and 300 MiB/s, got ${volumeInitializationRate.toMebibytes()} MiB/s.`);
    });

    test('throw an error if if volume initialization rate is specified without specifying snapshot ID', () => {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            volumeInitializationRate: cdk.Size.mebibytes(101),
          },
        }));
      }).toThrow('\'volumeInitializationRate\' can only be specified when \'snapShotId\' is provided.');
    });

    test('success adding gp3 volume with throughput 0', ()=> {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      service.addVolume(new ServiceManagedVolume(stack, 'EBSVolume', {
        name: 'nginx-vol',
        managedEBSVolume: {
          fileSystemType: ecs.FileSystemType.XFS,
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          size: cdk.Size.gibibytes(15),
          throughput: 0,
        },
      }));
      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        VolumeConfigurations: [
          {
            ManagedEBSVolume: {
              RoleArn: { 'Fn::GetAtt': ['EBSVolumeEBSRoleC27DD941', 'Arn'] },
              SizeInGiB: 15,
              FilesystemType: 'xfs',
              VolumeType: 'gp3',
              Throughput: 0,
            },
            Name: 'nginx-vol',
          },
        ],
      });
    });

    test('throw an error if throughput is greater than 2000 for volume type gp3', () => {
      // WHEN
      container.addMountPoints({
        containerPath: '/var/lib',
        readOnly: false,
        sourceVolume: 'nginx-vol',
      });

      expect(() => {
        service.addVolume(new ServiceManagedVolume(stack, 'EBS Volume', {
          name: 'nginx-vol',
          managedEBSVolume: {
            fileSystemType: ecs.FileSystemType.XFS,
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            size: cdk.Size.gibibytes(10),
            throughput: 2001,
          },
        }));
      }).toThrow("'throughput' must be less than or equal to 2000 MiB/s, got 2001 MiB/s");
    });
  });

  describe('When setting up a health check', () => {
    test('grace period is respected', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });

      // WHEN
      new ecs.FargateService(stack, 'Svc', {
        cluster,
        taskDefinition,
        healthCheckGracePeriod: cdk.Duration.seconds(10),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        HealthCheckGracePeriodSeconds: 10,
      });
    });
  });

  describe('When adding an app load balancer', () => {
    test('allows auto scaling by ALB request per target', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });
      const service = new ecs.FargateService(stack, 'Service', { cluster, taskDefinition });

      const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
        targets: [service],
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnRequestCount('ScaleOnRequests', {
        requestsPerTarget: 1000,
        targetGroup,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
        MaxCapacity: 10,
        MinCapacity: 1,
        ResourceId: {
          'Fn::Join': [
            '',
            [
              'service/',
              {
                Ref: 'EcsCluster97242B84',
              },
              '/',
              {
                'Fn::GetAtt': [
                  'ServiceD69D759B',
                  'Name',
                ],
              },
            ],
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
        TargetTrackingScalingPolicyConfiguration: {
          PredefinedMetricSpecification: {
            PredefinedMetricType: 'ALBRequestCountPerTarget',
            ResourceLabel: {
              'Fn::Join': ['', [
                { 'Fn::Select': [1, { 'Fn::Split': ['/', { Ref: 'lblistener657ADDEC' }] }] }, '/',
                { 'Fn::Select': [2, { 'Fn::Split': ['/', { Ref: 'lblistener657ADDEC' }] }] }, '/',
                { 'Fn::Select': [3, { 'Fn::Split': ['/', { Ref: 'lblistener657ADDEC' }] }] }, '/',
                { 'Fn::GetAtt': ['lblistenertargetGroupC7489D1E', 'TargetGroupFullName'] },
              ]],
            },
          },
          TargetValue: 1000,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        // if any load balancer is configured and healthCheckGracePeriodSeconds is not
        // set, then it should default to 60 seconds.
        HealthCheckGracePeriodSeconds: 60,
      });
    });

    test('allows auto scaling by ALB with new service arn format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
        targets: [service],
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnRequestCount('ScaleOnRequests', {
        requestsPerTarget: 1000,
        targetGroup,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
        MaxCapacity: 10,
        MinCapacity: 1,
        ResourceId: {
          'Fn::Join': [
            '',
            [
              'service/',
              {
                Ref: 'EcsCluster97242B84',
              },
              '/',
              {
                'Fn::GetAtt': [
                  'ServiceD69D759B',
                  'Name',
                ],
              },
            ],
          ],
        },
      });
    });

    describe('allows specify any existing container name and port in a service', () => {
      test('with default setting', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001 });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });
        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          LoadBalancers: [
            {
              ContainerName: 'MainContainer',
              ContainerPort: 8000,
              TargetGroupArn: {
                Ref: 'lblistenertargetGroupC7489D1E',
              },
            },
          ],
        });

        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
          Description: 'Load balancer to target',
          FromPort: 8000,
          ToPort: 8000,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
          Description: 'Load balancer to target',
          FromPort: 8000,
          ToPort: 8000,
        });
      });

      test('with TCP protocol and container hostPort unset', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001, protocol: ecs.Protocol.TCP });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8001,
            protocol: ecs.Protocol.TCP,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
          Port: 80,
          Protocol: 'HTTP',
        });
      });

      test('with TCP protocol and container hostPort set', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
          portMappings: [{
            containerPort: 8000,
            hostPort: 8000,
          }],
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8000,
            protocol: ecs.Protocol.TCP,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
          Port: 80,
          Protocol: 'HTTP',
        });
      });

      test('with UDP protocol and container hostPort unset', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001, protocol: ecs.Protocol.UDP });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8001,
            protocol: ecs.Protocol.UDP,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
          Port: 80,
          Protocol: 'HTTP',
        });
      });

      test('with UDP protocol and container hostPort set', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
          portMappings: [{
            containerPort: 8000,
            hostPort: 8000,
            protocol: ecs.Protocol.UDP,
          }],
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8000,
            protocol: ecs.Protocol.UDP,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
          Port: 80,
          Protocol: 'HTTP',
        });
      });

      test('throws when protocol does not match', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001, protocol: ecs.Protocol.UDP });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        // THEN
        expect(() => {
          listener.addTargets('target', {
            port: 80,
            targets: [service.loadBalancerTarget({
              containerName: 'MainContainer',
              containerPort: 8001,
              protocol: ecs.Protocol.TCP,
            })],
          });
        }).toThrow(/Container 'Default\/FargateTaskDef\/MainContainer' has no mapping for port 8001 and protocol tcp. Did you call "container.addPortMappings\(\)"\?/);
      });

      test('throws when port does not match', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001 });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        // THEN
        expect(() => {
          listener.addTargets('target', {
            port: 80,
            targets: [service.loadBalancerTarget({
              containerName: 'MainContainer',
              containerPort: 8002,
            })],
          });
        }).toThrow(/Container 'Default\/FargateTaskDef\/MainContainer' has no mapping for port 8002 and protocol tcp. Did you call "container.addPortMappings\(\)"\?/);
      });

      test('throws when container does not exist', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001 });

        const service = new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });

        // THEN
        expect(() => {
          listener.addTargets('target', {
            port: 80,
            targets: [service.loadBalancerTarget({
              containerName: 'SideContainer',
              containerPort: 8001,
            })],
          });
        }).toThrow(/No container named 'SideContainer'. Did you call "addContainer\(\)"?/);
      });
    });

    describe('allows load balancing to any container and port of service', () => {
      describe('with application load balancers', () => {
        test('with default target group port and protocol', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.applicationListener(listener),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 80,
            Protocol: 'HTTP',
          });
        });

        test('with default target group port and HTTP protocol', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.applicationListener(listener, {
                protocol: elbv2.ApplicationProtocol.HTTP,
              }),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 80,
            Protocol: 'HTTP',
          });
        });

        test('with default target group port and HTTPS protocol', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.applicationListener(listener, {
                protocol: elbv2.ApplicationProtocol.HTTPS,
              }),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 443,
            Protocol: 'HTTPS',
          });
        });

        test('with any target group port and protocol', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.applicationListener(listener, {
                port: 83,
                protocol: elbv2.ApplicationProtocol.HTTP,
              }),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 83,
            Protocol: 'HTTP',
          });
        });

        test('throws when containerPortRange is used instead of containerPort', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
            portMappings: [{
              containerPort: ecs.ContainerDefinition.CONTAINER_PORT_USE_RANGE,
              containerPortRange: '8000-8001',
            }],
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          // THEN
          expect(() => service.registerLoadBalancerTargets({
            containerName: 'MainContainer',
            containerPort: 8000,
            listener: ecs.ListenerConfig.applicationListener(listener),
            newTargetGroupId: 'target1',
          })).toThrow(/Container 'Default\/FargateTaskDef\/MainContainer' has no mapping for port 8000 and protocol tcp. Did you call "container.addPortMappings\(\)"\?/);
        });
      });

      describe('with network load balancers', () => {
        test('with default target group port', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.networkListener(listener),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 80,
            Protocol: 'TCP',
          });
        });

        test('with any target group port', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
          });
          container.addPortMappings({ containerPort: 8000 });

          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          service.registerLoadBalancerTargets(
            {
              containerName: 'MainContainer',
              containerPort: 8000,
              listener: ecs.ListenerConfig.networkListener(listener, {
                port: 81,
              }),
              newTargetGroupId: 'target1',
            },
          );

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
            LoadBalancers: [
              {
                ContainerName: 'MainContainer',
                ContainerPort: 8000,
                TargetGroupArn: {
                  Ref: 'lblistenertarget1Group1A1A5C9E',
                },
              },
            ],
          });

          Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
            Port: 81,
            Protocol: 'TCP',
          });
        });

        test('throws when containerPortRange is used instead of containerPort', () => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
          const service = new ecs.FargateService(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
            portMappings: [{
              containerPort: ecs.ContainerDefinition.CONTAINER_PORT_USE_RANGE,
              containerPortRange: '8000-8001',
            }],
          });

          // WHEN
          const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });

          // THEN
          expect(() => service.registerLoadBalancerTargets({
            containerName: 'MainContainer',
            containerPort: 8000,
            listener: ecs.ListenerConfig.networkListener(listener),
            newTargetGroupId: 'target1',
          })).toThrow(/Container 'Default\/FargateTaskDef\/MainContainer' has no mapping for port 8000 and protocol tcp. Did you call "container.addPortMappings\(\)"\?/);
        });
      });
    });
  });

  describe('When adding a classic load balancer', () => {
    test('throws when AvailabilityZoneRebalancing.ENABLED', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });

      // THEN
      expect(() => {
        lb.addTarget(service);
      }).toThrow('AvailabilityZoneRebalancing.ENABLED disallows using the service as a target of a Classic Load Balancer');
    });
  });

  describe('autoscaling tests', () => {
    test('allows scaling on a specified scheduled time', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnSchedule('ScaleOnSchedule', {
        schedule: appscaling.Schedule.cron({ hour: '8', minute: '0' }),
        minCapacity: 10,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
        ScheduledActions: [
          {
            ScalableTargetAction: {
              MinCapacity: 10,
            },
            Schedule: 'cron(0 8 * * ? *)',
            ScheduledActionName: 'ScaleOnSchedule',
          },
        ],
      });
    });

    test('allows scaling on a specified metric value', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnMetric('ScaleOnMetric', {
        metric: new cloudwatch.Metric({ namespace: 'Test', metricName: 'Metric' }),
        scalingSteps: [
          { upper: 0, change: -1 },
          { lower: 100, change: +1 },
          { lower: 500, change: +5 },
        ],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
        PolicyType: 'StepScaling',
        ScalingTargetId: {
          Ref: 'ServiceTaskCountTarget23E25614',
        },
        StepScalingPolicyConfiguration: {
          AdjustmentType: 'ChangeInCapacity',
          MetricAggregationType: 'Average',
          StepAdjustments: [
            {
              MetricIntervalUpperBound: 0,
              ScalingAdjustment: -1,
            },
          ],
        },
      });
    });

    test('allows scaling on a target CPU utilization', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnCpuUtilization('ScaleOnCpu', {
        targetUtilizationPercent: 30,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
        PolicyType: 'TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration: {
          PredefinedMetricSpecification: { PredefinedMetricType: 'ECSServiceAverageCPUUtilization' },
          TargetValue: 30,
        },
      });
    });

    test('allows scaling on memory utilization', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnMemoryUtilization('ScaleOnMemory', {
        targetUtilizationPercent: 30,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
        PolicyType: 'TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration: {
          PredefinedMetricSpecification: { PredefinedMetricType: 'ECSServiceAverageMemoryUtilization' },
          TargetValue: 30,
        },
      });
    });

    test('allows scaling on custom CloudWatch metric', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleToTrackCustomMetric('ScaleOnCustomMetric', {
        metric: new cloudwatch.Metric({ namespace: 'Test', metricName: 'Metric' }),
        targetValue: 5,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
        PolicyType: 'TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration: {
          CustomizedMetricSpecification: {
            MetricName: 'Metric',
            Namespace: 'Test',
            Statistic: 'Average',
          },
          TargetValue: 5,
        },
      });
    });

    test('scheduled scaling shows warning when minute is not defined in cron', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnSchedule('ScaleOnSchedule', {
        schedule: appscaling.Schedule.cron({ hour: '8' }),
        minCapacity: 10,
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Service/TaskCount/Target', "cron: If you don't pass 'minute', by default the event runs every minute. Pass 'minute: '*'' if that's what you intend, or 'minute: 0' to run once per hour instead. [ack: @aws-cdk/aws-applicationautoscaling:defaultRunEveryMinute]");
    });

    test('scheduled scaling shows no warning when minute is * in cron', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        minHealthyPercent: 50, // must be set to avoid warning causing test failure
        circuitBreaker: {}, // Set to silence an unrelated warning
      });

      // WHEN
      const capacity = service.autoScaleTaskCount({ maxCapacity: 10, minCapacity: 1 });
      capacity.scaleOnSchedule('ScaleOnSchedule', {
        schedule: appscaling.Schedule.cron({ hour: '8', minute: '*' }),
        minCapacity: 10,
      });

      // THEN
      const annotations = Annotations.fromStack(stack).findWarning('*', Match.anyValue());
      expect(annotations.length).toBe(0);
    });
  });

  describe('When enabling service discovery', () => {
    test('throws if namespace has not been added to cluster', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            name: 'myApp',
          },
        });
      }).toThrow(/Cannot enable service discovery if a Cloudmap Namespace has not been created in the cluster./);
    });

    test('creates cloud map service for Private DNS namespace', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'A',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('creates AWS Cloud Map service for Private DNS namespace with SRV records with proper defaults', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
          dnsRecordType: cloudmap.DnsRecordType.SRV,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('creates AWS Cloud Map service for Private DNS namespace with SRV records with overriden defaults', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
          dnsRecordType: cloudmap.DnsRecordType.SRV,
          dnsTtl: cdk.Duration.seconds(10),
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 10,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('user can select any container and port', () => {
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
      const mainContainer = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      mainContainer.addPortMappings({ containerPort: 8000 });

      const otherContainer = taskDefinition.addContainer('OtherContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      otherContainer.addPortMappings({ containerPort: 8001 });

      new ecs.FargateService(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          dnsRecordType: cloudmap.DnsRecordType.SRV,
          container: otherContainer,
          containerPort: 8001,
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            RegistryArn: { 'Fn::GetAtt': ['ServiceCloudmapService046058A4', 'Arn'] },
            ContainerName: 'OtherContainer',
            ContainerPort: 8001,
          },
        ],
      });
    });
  });

  test('Metric', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});
    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');
    taskDefinition.addContainer('Container', {
      image: ecs.ContainerImage.fromRegistry('hello'),
    });

    // WHEN
    const service = new ecs.FargateService(stack, 'Service', {
      cluster,
      taskDefinition,
    });

    // THEN
    expect(stack.resolve(service.metricCpuUtilization())).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
        ServiceName: { 'Fn::GetAtt': ['ServiceD69D759B', 'Name'] },
      },
      namespace: 'AWS/ECS',
      metricName: 'CPUUtilization',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });
  });

  describe('When import a Fargate Service', () => {
    test('fromFargateServiceArn old format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);

      // WHEN
      const service = ecs.FargateService.fromFargateServiceArn(stack, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service');

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');
    });

    test('fromFargateServiceArn new format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);

      // WHEN
      const service = ecs.FargateService.fromFargateServiceArn(stack, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');
    });

    describe('fromFargateServiceArn tokenized ARN', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);

        // WHEN
        const service = ecs.FargateService.fromFargateServiceArn(stack, 'EcsService', new cdk.CfnParameter(stack, 'ARN').valueAsString);

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            1,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });
        const stack = new cdk.Stack(app);

        // WHEN
        const service = ecs.FargateService.fromFargateServiceArn(stack, 'EcsService', new cdk.CfnParameter(stack, 'ARN').valueAsString);

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            2,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });
      });
    });

    test('with serviceArn old format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN
      const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
        serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
        cluster,
      });

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');

      expect(service.env.account).toEqual('123456789012');
      expect(service.env.region).toEqual('us-west-2');
    });

    test('with serviceArn new format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN
      const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
        serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service',
        cluster,
      });

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');

      expect(service.env.account).toEqual('123456789012');
      expect(service.env.region).toEqual('us-west-2');
    });

    describe('with serviceArn tokenized ARN', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          serviceArn: new cdk.CfnParameter(stack, 'ARN').valueAsString,
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            1,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });

        expect(stack.resolve(service.env.account)).toEqual({
          'Fn::Select': [
            4,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
        expect(stack.resolve(service.env.region)).toEqual({
          'Fn::Select': [
            3,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });
        const stack = new cdk.Stack(app);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          serviceArn: new cdk.CfnParameter(stack, 'ARN').valueAsString,
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            2,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });

        expect(stack.resolve(service.env.account)).toEqual({
          'Fn::Select': [
            4,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
        expect(stack.resolve(service.env.region)).toEqual({
          'Fn::Select': [
            3,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
      });
    });

    describe('with serviceName', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();
        acknowledgeTestValidationRules(stack);
        const pseudo = new cdk.ScopedAws(stack);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          serviceName: 'my-http-service',
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual(stack.resolve(`arn:${pseudo.partition}:ecs:${pseudo.region}:${pseudo.accountId}:service/my-http-service`));
        expect(service.serviceName).toEqual('my-http-service');
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });

        const stack = new cdk.Stack(app);
        const pseudo = new cdk.ScopedAws(stack);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          serviceName: 'my-http-service',
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual(stack.resolve(`arn:${pseudo.partition}:ecs:${pseudo.region}:${pseudo.accountId}:service/${cluster.clusterName}/my-http-service`));
        expect(service.serviceName).toEqual('my-http-service');
      });
    });

    test('with circuit breaker', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('Container', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });

      // WHEN
      new ecs.FargateService(stack, 'EcsService', {
        cluster,
        taskDefinition,
        circuitBreaker: { rollback: true },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
          DeploymentCircuitBreaker: {
            Enable: true,
            Rollback: true,
          },
        },
        DeploymentController: {
          Type: ecs.DeploymentControllerType.ECS,
        },
      });
    });

    test('with circuit breaker and deployment controller feature flag enabled', () => {
      // GIVEN
      const disableCircuitBreakerEcsDeploymentControllerFeatureFlag =
        { [cxapi.ECS_DISABLE_EXPLICIT_DEPLOYMENT_CONTROLLER_FOR_CIRCUIT_BREAKER]: true };
      const app = new App({ context: disableCircuitBreakerEcsDeploymentControllerFeatureFlag });
      const stack = new cdk.Stack(app);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('Container', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });

      // WHEN
      new ecs.FargateService(stack, 'EcsService', {
        cluster,
        taskDefinition,
        circuitBreaker: { rollback: true },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
          DeploymentCircuitBreaker: {
            Enable: true,
            Rollback: true,
          },
        },
      });
    });

    test('throws an exception if both serviceArn and serviceName were provided for fromEc2ServiceAttributes', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      expect(() => {
        ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
          serviceName: 'my-http-service',
          cluster,
        });
      }).toThrow(/only specify either serviceArn or serviceName/);
    });

    test('throws an exception if neither serviceArn nor serviceName were provided for fromEc2ServiceAttributes', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      expect(() => {
        ecs.FargateService.fromFargateServiceAttributes(stack, 'EcsService', {
          cluster,
        });
      }).toThrow(/only specify either serviceArn or serviceName/);
    });

    test('allows setting enable execute command', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'FargateTaskDefC6FB60B4',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        LaunchType: LaunchType.FARGATE,
        EnableECSManagedTags: false,
        EnableExecuteCommand: true,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'DISABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'FargateServiceSecurityGroup0A0E79CB',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
              },
              {
                Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
              },
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'FargateTaskDefTaskRoleDefaultPolicy8EB25BBD',
        Roles: [
          {
            Ref: 'FargateTaskDefTaskRole0B257552',
          },
        ],
      });
    });

    test('no logging enabled when logging field is set to NONE', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logging: ecs.ExecuteCommandLogging.NONE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        logging: ecs.LogDrivers.awsLogs({
          logGroup,
          streamPrefix: 'log-group',
        }),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'FargateTaskDefTaskRoleDefaultPolicy8EB25BBD',
        Roles: [
          {
            Ref: 'FargateTaskDefTaskRole0B257552',
          },
        ],
      });
    });

    test('enables execute command logging with logging field set to OVERRIDE', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      const execBucket = new s3.Bucket(stack, 'ExecBucket');

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            s3Bucket: execBucket,
            s3KeyPrefix: 'exec-output',
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'ExecBucket29559356',
                    },
                    '/*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'FargateTaskDefTaskRoleDefaultPolicy8EB25BBD',
        Roles: [
          {
            Ref: 'FargateTaskDefTaskRole0B257552',
          },
        ],
      });
    });

    test('enables only execute command session encryption', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const kmsKey = new kms.Key(stack, 'KmsKey');

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      const execBucket = new s3.Bucket(stack, 'EcsExecBucket');

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          kmsKey,
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            s3Bucket: execBucket,
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });

      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'kms:Decrypt',
                'kms:GenerateDataKey',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'KmsKey46693ADD',
                  'Arn',
                ],
              },
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                    '/*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'FargateTaskDefTaskRoleDefaultPolicy8EB25BBD',
        Roles: [
          {
            Ref: 'FargateTaskDefTaskRole0B257552',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
        KeyPolicy: {
          Statement: [
            {
              Action: 'kms:*',
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':root',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    test('enables encryption for execute command logging', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const kmsKey = new kms.Key(stack, 'KmsKey');

      const logGroup = new logs.LogGroup(stack, 'LogGroup', {
        encryptionKey: kmsKey,
      });

      const execBucket = new s3.Bucket(stack, 'EcsExecBucket', {
        encryptionKey: kmsKey,
      });

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          kmsKey,
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            cloudWatchEncryptionEnabled: true,
            s3Bucket: execBucket,
            s3EncryptionEnabled: true,
            s3KeyPrefix: 'exec-output',
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });

      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.FargateService(stack, 'FargateService', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'kms:Decrypt',
                'kms:GenerateDataKey',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'KmsKey46693ADD',
                  'Arn',
                ],
              },
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetEncryptionConfiguration',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'FargateTaskDefTaskRoleDefaultPolicy8EB25BBD',
        Roles: [
          {
            Ref: 'FargateTaskDefTaskRole0B257552',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
        KeyPolicy: {
          Statement: [
            {
              Action: 'kms:*',
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':root',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
            {
              Action: [
                'kms:Encrypt*',
                'kms:Decrypt*',
                'kms:ReEncrypt*',
                'kms:GenerateDataKey*',
                'kms:Describe*',
              ],
              Condition: {
                ArnLike: {
                  'kms:EncryptionContext:aws:logs:arn': {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        {
                          Ref: 'AWS::Partition',
                        },
                        ':logs:',
                        {
                          Ref: 'AWS::Region',
                        },
                        ':',
                        {
                          Ref: 'AWS::AccountId',
                        },
                        ':*',
                      ],
                    ],
                  },
                },
              },
              Effect: 'Allow',
              Principal: {
                Service: {
                  'Fn::Join': [
                    '',
                    [
                      'logs.',
                      {
                        Ref: 'AWS::Region',
                      },
                      '.amazonaws.com',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    testDeprecated('with both propagateTags and propagateTaskTagsFrom defined', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.FargateTaskDefinition(stack, 'FargateTaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      // THEN
      expect(() => {
        new ecs.FargateService(stack, 'FargateService', {
          cluster,
          taskDefinition,
          propagateTags: PropagatedTagSource.SERVICE,
          propagateTaskTagsFrom: PropagatedTagSource.SERVICE,
        });
      }).toThrow(/You can only specify either propagateTags or propagateTaskTagsFrom. Alternatively, you can leave both blank/);
    });
  });
});
