import type * as lambda from 'aws-cdk-lib/aws-lambda';
import { KubectlV29Layer } from '@aws-cdk/lambda-layer-kubectl-v29';
import { KubectlV30Layer } from '@aws-cdk/lambda-layer-kubectl-v30';
import { KubectlV31Layer } from '@aws-cdk/lambda-layer-kubectl-v31';
import { KubectlV32Layer } from '@aws-cdk/lambda-layer-kubectl-v32';
import { KubectlV33Layer } from '@aws-cdk/lambda-layer-kubectl-v33';
import { KubectlV34Layer } from '@aws-cdk/lambda-layer-kubectl-v34';
import { KubectlV35Layer } from '@aws-cdk/lambda-layer-kubectl-v35';
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
import type { Construct } from 'constructs';
import * as eks from 'aws-cdk-lib/aws-eks-v2';

// This object maps Kubernetes version strings to their corresponding
// KubectlLayerVersion constructor functions. This allows us to dynamically
// create the appropriate KubectlLayerVersion instance based on the
// Kubernetes version.
const versionMap: { [key: string]: new (scope: Construct, id: string) => lambda.ILayerVersion } = {
  '1.29': KubectlV29Layer,
  '1.30': KubectlV30Layer,
  '1.31': KubectlV31Layer,
  '1.32': KubectlV32Layer,
  '1.33': KubectlV33Layer,
  '1.34': KubectlV34Layer,
  '1.35': KubectlV35Layer,
  '1.36': KubectlV36Layer,
};

const sortedVersions = Object.keys(versionMap).sort((a, b) => parseFloat(a) - parseFloat(b));

/**
 * Returns the N latest version strings from the versionMap (default 2).
 * e.g. ['1.34', '1.35'] for the latest two.
 */
export function getLatestVersions(count = 2): string[] {
  return sortedVersions.slice(-count);
}

/**
 * Returns the version and kubectlProviderOptions for an EKS v2 cluster,
 * dynamically picking the latest available version as fallback.
 */
export function getClusterVersionConfig(scope: Construct, version?: eks.KubernetesVersion) {
  const latestVersion = sortedVersions[sortedVersions.length - 1];
  const _version = version ?? eks.KubernetesVersion.of(latestVersion);
  return {
    version: _version,
    kubectlProviderOptions: {
      kubectlLayer: new versionMap[_version.version](scope, 'KubectlLayer') as unknown as lambda.ILayerVersion,
    },
  };
}
