import { testFixture } from './util';
import { Match, Template } from '../../assertions';
import * as eks from '../lib';

const CLUSTER_VERSION = eks.KubernetesVersion.V1_33;

describe('controlPlaneScalingTier', () => {
  test.each([
    [eks.ControlPlaneScalingTier.STANDARD, 'standard'],
    [eks.ControlPlaneScalingTier.TIER_XL, 'tier-xl'],
    [eks.ControlPlaneScalingTier.TIER_2XL, 'tier-2xl'],
    [eks.ControlPlaneScalingTier.TIER_4XL, 'tier-4xl'],
    [eks.ControlPlaneScalingTier.TIER_8XL, 'tier-8xl'],
  ])(
    'controlPlaneScalingTier(%s) configures ControlPlaneScalingConfig on the native cluster resource',
    (tier, expected) => {
      // GIVEN
      const { stack } = testFixture();

      // WHEN
      new eks.Cluster(stack, 'Cluster', {
        version: CLUSTER_VERSION,
        controlPlaneScalingTier: tier,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EKS::Cluster', {
        ControlPlaneScalingConfig: {
          Tier: expected,
        },
      });
    },
  );

  test('ControlPlaneScalingTier.of() creates a tier with a custom value', () => {
    // a tier that is not (yet) exposed as a named member is reachable via the escape hatch
    expect(eks.ControlPlaneScalingTier.of('tier-16xl').value).toEqual('tier-16xl');
  });

  test('ControlPlaneScalingConfig is not set when controlPlaneScalingTier is not provided', () => {
    // GIVEN
    const { stack } = testFixture();

    // WHEN
    new eks.Cluster(stack, 'Cluster', {
      version: CLUSTER_VERSION,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Cluster', {
      ControlPlaneScalingConfig: Match.absent(),
    });
  });
});
