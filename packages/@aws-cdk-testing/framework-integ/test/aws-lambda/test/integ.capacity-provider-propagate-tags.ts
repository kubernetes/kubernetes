import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as cdk from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'CapacityProviderPropagateTagsStack');

const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 2 });
const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc });

new lambda.CapacityProvider(stack, 'CpWithExplicitTags', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  propagateTags: lambda.PropagateTags.explicit({
    Environment: 'Test',
    Project: 'CDK-Integ',
  }),
});

new lambda.CapacityProvider(stack, 'CpWithNone', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  propagateTags: lambda.PropagateTags.none(),
});

new integ.IntegTest(app, 'CapacityProviderPropagateTagsTest', {
  testCases: [stack],
});
