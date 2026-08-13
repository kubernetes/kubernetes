/// !cdk-integ pragma:disable-update-workflow
import type { StackProps } from 'aws-cdk-lib';
import { App, Stack } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as eks from 'aws-cdk-lib/aws-eks-v2';

const clusterName = 'eks-v2-provisioned-control-plane-test';

/**
 * This test verifies that an aws-eks-v2 cluster can be created with a provisioned control
 * plane scaling tier. We use TIER_XL, the smallest provisioned tier, since the default
 * STANDARD tier would not exercise the configuration at all.
 */
class EksV2ProvisionedControlPlaneStack extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    new eks.Cluster(this, 'Cluster', {
      version: eks.KubernetesVersion.V1_32,
      clusterName,
      controlPlaneScalingTier: eks.ControlPlaneScalingTier.TIER_XL,
    });
  }
}

const app = new App();

const stack = new EksV2ProvisionedControlPlaneStack(app, 'EksV2ProvisionedControlPlaneStack');

const test = new integ.IntegTest(app, 'eks-v2-provisioned-control-plane-integ', {
  testCases: [stack],
  diffAssets: false,
});

test.assertions.awsApiCall('eks', 'describeCluster', {
  name: clusterName,
}).expect(integ.ExpectedResult.objectLike({
  cluster: integ.Match.objectLike({
    controlPlaneScalingConfig: {
      tier: 'tier-xl',
    },
  }),
})).waitForAssertions();
