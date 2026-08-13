/// !cdk-integ cdk-route53-ebs-hostedzoneid-integ
import { App, Aws, CfnMapping, Stack, Token } from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as targets from 'aws-cdk-lib/aws-route53-targets';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as elasticbeanstalk from 'aws-cdk-lib/aws-elasticbeanstalk';
import * as custom from 'aws-cdk-lib/custom-resources';
import { RegionInfo } from 'aws-cdk-lib/region-info';
import { SOLUTION_STACK_NAME } from '../../utils/aws-elasticbeanstalk';

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
  },
});
const stack = new Stack(app, 'cdk-route53-ebs-hostedzoneid-integ');

// RegionInfo.get() requires a concrete region string at synth time, but stack.region
// is a Token (AWS::Region) when no explicit env is set. A CfnMapping lets us resolve
// the EB hosted zone ID at deploy time using Fn::FindInMap with AWS::Region.
const ebsHostedZoneMapping = new CfnMapping(stack, 'EbsHostedZoneMapping');
for (const region of RegionInfo.regions) {
  if (region.ebsEnvEndpointHostedZoneId) {
    ebsHostedZoneMapping.setValue(region.name, 'hostedZoneId', region.ebsEnvEndpointHostedZoneId);
  }
}

const zone = new route53.PublicHostedZone(stack, 'HostedZone', {
  zoneName: 'test.public',
});

const ebsApp = new elasticbeanstalk.CfnApplication(stack, 'Application', {});

const instanceRole = new iam.Role(stack, 'EbsEc2Role', {
  assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
});
instanceRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AWSElasticBeanstalkWebTier'));
instanceRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AWSElasticBeanstalkManagedUpdatesCustomerRolePolicy'));

const instanceProfile = new iam.CfnInstanceProfile(stack, 'EbsInstanceProfile', {
  roles: [instanceRole.roleName],
});

const ebsEnv = new elasticbeanstalk.CfnEnvironment(stack, 'Environment', {
  applicationName: ebsApp.ref,
  solutionStackName: SOLUTION_STACK_NAME.NODEJS_22,
  optionSettings: [
    {
      namespace: 'aws:autoscaling:launchconfiguration',
      optionName: 'IamInstanceProfile',
      value: instanceProfile.ref,
    },
    {
      namespace: 'aws:autoscaling:launchconfiguration',
      optionName: 'RootVolumeType',
      value: 'gp3',
    },
    {
      namespace: 'aws:ec2:instances',
      optionName: 'InstanceTypes',
      value: 't3.micro',
    },
    {
      namespace: 'aws:elasticbeanstalk:environment',
      optionName: 'EnvironmentType',
      value: 'SingleInstance',
    },
  ],
});
ebsEnv.addResourceDependency(ebsApp);

const getEnvironmentUrl = new custom.AwsCustomResource(stack, 'GetEnvironmentUrl', {
  onCreate: {
    service: 'ElasticBeanstalk',
    action: 'describeEnvironments',
    parameters: {
      EnvironmentNames: [Token.asString(ebsEnv.environmentName)],
    },
    physicalResourceId: custom.PhysicalResourceId.of('EnvironmentUrl'),
    logging: custom.Logging.withDataHidden(),
  },
  policy: custom.AwsCustomResourcePolicy.fromSdkCalls({ resources: custom.AwsCustomResourcePolicy.ANY_RESOURCE }),
});
getEnvironmentUrl.node.addDependency(ebsEnv);

new route53.ARecord(stack, 'AliasRecord', {
  zone,
  recordName: 'test',
  target: route53.RecordTarget.fromAlias(new targets.ElasticBeanstalkEnvironmentEndpointTarget(getEnvironmentUrl.getResponseField('Environments.0.CNAME'), {
    hostedZoneId: ebsHostedZoneMapping.findInMap(Aws.REGION, 'hostedZoneId'),
  })),
});

new IntegTest(app, 'cdk-route53-ebs-hostedzoneid-integ-test', {
  testCases: [stack],
});
