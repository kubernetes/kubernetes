/// !cdk-integ pragma:disable-update-workflow
import * as integ from '@aws-cdk/integ-tests-alpha';
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
import type { StackProps } from 'aws-cdk-lib';
import { App, Stack } from 'aws-cdk-lib';
import type * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks-v2';

interface EksFargateClusterStackProps extends StackProps {
  vpc?: ec2.IVpc;
}
class EksFargateClusterStack extends Stack {
  constructor(scope: App, id: string, props?: EksFargateClusterStackProps) {
    super(scope, id, props);

    new eks.FargateCluster(this, 'FargateTestCluster', {
      vpc: props?.vpc,
      version: eks.KubernetesVersion.V1_36,
      prune: false,
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV36Layer(this, 'kubectlLayer'),
      },
    });
  }
}

const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy': true,
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
  },
});
const stack = new EksFargateClusterStack(app, 'eks-fargate-cluster-test-stack', {});
new integ.IntegTest(app, 'eks-fargate-cluster', {
  testCases: [stack],
  diffAssets: false,
});
