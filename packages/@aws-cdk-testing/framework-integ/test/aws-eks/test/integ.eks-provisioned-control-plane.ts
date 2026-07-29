/// !cdk-integ pragma:disable-update-workflow
import type { StackProps } from 'aws-cdk-lib';
import { App, Stack } from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import { getClusterVersionConfig } from './integ-tests-kubernetes-version';
import * as eks from 'aws-cdk-lib/aws-eks';

const clusterName = 'eks-provisioned-control-plane-test';

/**
 * This test verifies that a cluster can be created with a provisioned control plane
 * scaling tier. We use TIER_XL, the smallest provisioned tier, since the default
 * STANDARD tier would not exercise the configuration at all.
 */
class EksProvisionedControlPlaneStack extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    new eks.Cluster(this, 'Cluster', {
      ...getClusterVersionConfig(this, eks.KubernetesVersion.V1_32),
      clusterName,
      controlPlaneScalingTier: eks.ControlPlaneScalingTier.TIER_XL,
    });
  }
}

const app = new App();

const stack = new EksProvisionedControlPlaneStack(app, 'EksProvisionedControlPlaneStack');

const test = new integ.IntegTest(app, 'eks-provisioned-control-plane-integ', {
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

app.synth();
