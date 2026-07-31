import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as cdk from 'aws-cdk-lib/core';
import * as integ from '@aws-cdk/integ-tests-alpha';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'CapacityProviderAllFieldsStack');

const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 2 });
const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc });

const operatorRole = new iam.Role(stack, 'OperatorRole', {
  assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
  managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLambdaManagedEC2ResourceOperator')],
});

const logGroup = new logs.LogGroup(stack, 'CpLogGroup', {
  logGroupName: '/aws/lambda/capacity-provider/integ-test',
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

new lambda.CapacityProvider(stack, 'CapacityProvider', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  operatorRole: operatorRole,
  architectures: [lambda.Architecture.X86_64],
  instanceTypeFilter: lambda.InstanceTypeFilter.allow([ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE)]),
  maxVCpuCount: 12,
  scalingOptions: lambda.ScalingOptions.manual([
    lambda.TargetTrackingScalingPolicy.cpuUtilization(70),
  ]),
  logGroup: logGroup,
  systemLogLevel: lambda.SystemLogLevel.DEBUG,
});

new integ.IntegTest(app, 'CapacityProviderAllFieldsTest', {
  testCases: [stack],
  regions: ['us-east-1', 'us-east-2', 'us-west-2', 'eu-west-1', 'eu-central-1', 'eu-north-1', 'ap-south-1', 'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1'],
});

