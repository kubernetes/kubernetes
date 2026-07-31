/// !cdk-integ pragma:disable-update-workflow
import type { StackProps } from 'aws-cdk-lib';
import { App, Stack } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import { getClusterVersionConfig } from './integ-tests-kubernetes-version';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { EC2_RESTRICT_DEFAULT_SECURITY_GROUP } from 'aws-cdk-lib/cx-api';
interface EksFargateClusterStackProps extends StackProps {
  authMode?: eks.AuthenticationMode;
  vpc?: ec2.IVpc;
}
class EksFargateClusterStack extends Stack {
  public readonly vpc: ec2.IVpc;
  constructor(scope: App, id: string, props?: EksFargateClusterStackProps) {
    super(scope, id, props);

    this.node.setContext(EC2_RESTRICT_DEFAULT_SECURITY_GROUP, false);
    this.vpc = props?.vpc ?? this.createDummyVpc();
    new eks.FargateCluster(this, 'FargateCluster', {
      ...getClusterVersionConfig(this, eks.KubernetesVersion.V1_36),
      prune: false,
      authenticationMode: props?.authMode,
      vpc: this.vpc,
    });
  }
  private createDummyVpc(): ec2.IVpc {
    return new ec2.Vpc(this, 'DummyVpc', { maxAzs: 2, natGateways: 1, restrictDefaultSecurityGroup: false });
  }
}

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
    '@aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy': false,
  },
});

// create fargate cluster with undefined auth mode
const stack1 = new EksFargateClusterStack(app, 'aws-cdk-eks-fargate-cluster-test-stack1', {
  authMode: eks.AuthenticationMode.API,
});
// create the 2nd fargate cluster in the same vpc, but with api auth mode
const stack2 = new EksFargateClusterStack(app, 'aws-cdk-eks-fargate-cluster-test-stack2', {
  authMode: eks.AuthenticationMode.API,
  vpc: stack1.vpc,
});

new integ.IntegTest(app, 'aws-cdk-eks-fargate-cluster', {
  testCases: [stack1, stack2],
  // Test includes assets that are updated weekly. If not disabled, the upgrade PR will fail.
  diffAssets: false,
});
