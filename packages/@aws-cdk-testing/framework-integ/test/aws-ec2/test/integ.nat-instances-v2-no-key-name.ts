import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { EC2_RESTRICT_DEFAULT_SECURITY_GROUP } from 'aws-cdk-lib/cx-api';

/**
 * NatInstanceProviderV2 must not pass the deprecated `keyName` down to its
 * instances when it was never provided (https://github.com/aws/aws-cdk/issues/30806).
 * The integ runner synthesizes with JSII_DEPRECATED=fail, so this app fails to
 * synth if that regresses.
 */
class NatInstanceNoKeyNameStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, false);

    const natGatewayProvider = ec2.NatProvider.instanceV2({
      instanceType: new ec2.InstanceType('t3.small'),
      defaultAllowedTraffic: ec2.NatTrafficDirection.OUTBOUND_ONLY,
    });

    new ec2.Vpc(this, 'MyVpc', {
      natGatewayProvider,
      natGateways: 1,
      maxAzs: 1,
    });
  }
}

const app = new cdk.App();
const stack = new NatInstanceNoKeyNameStack(app, 'aws-cdk-vpc-nat-instances-v2-no-key-name');

new IntegTest(app, 'nat-instance-v2-no-key-name-integ-test', {
  testCases: [stack],
});
