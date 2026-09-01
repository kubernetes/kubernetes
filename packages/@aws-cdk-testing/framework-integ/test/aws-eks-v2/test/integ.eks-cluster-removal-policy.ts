import * as integ from '@aws-cdk/integ-tests-alpha';
import { KubectlV32Layer } from '@aws-cdk/lambda-layer-kubectl-v32';
import type { StackProps } from 'aws-cdk-lib';
import { App, RemovalPolicy, Stack } from 'aws-cdk-lib';
import { EKS_USE_NATIVE_OIDC_PROVIDER } from 'aws-cdk-lib/cx-api';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks-v2';
import * as iam from 'aws-cdk-lib/aws-iam';

/**
 * This test checks that all EKS resources can be deployed with removal policies.
 * We use the DESTROY policy here to avoid leaving orphaned resources behind, but if it works for DESTROY, it should work for other values as well.
 */
class EksClusterRemovalPolicyStack extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1, restrictDefaultSecurityGroup: false });

    const cluster = new eks.Cluster(this, 'Cluster', {
      removalPolicy: RemovalPolicy.DESTROY,
      prune: false,
      vpc,
      version: eks.KubernetesVersion.V1_32,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV32Layer(this, 'Kubectl'),
      },
    });

    cluster.clusterSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(vpc.vpcCidrBlock),
      ec2.Port.tcp(443),
      'Allow VPC traffic',
    );

    // Access Entry
    const accessPolicy = new eks.AccessPolicy({
      accessScope: { type: eks.AccessScopeType.CLUSTER },
      policy: eks.AccessPolicyArn.AMAZON_EKS_CLUSTER_ADMIN_POLICY,
    });
    const accessEntryRole = new iam.Role(this, 'AccessEntryRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    new eks.AccessEntry(this, 'AccessEntry', {
      cluster,
      principal: accessEntryRole.roleArn,
      accessPolicies: [accessPolicy],
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Addon
    new eks.Addon(this, 'Addon', {
      cluster,
      addonName: 'kube-proxy',
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ALB Controller
    new eks.AlbController(this, 'AlbControllerConstruct', {
      cluster,
      version: eks.AlbControllerVersion.V2_8_2,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Nodegroup
    new eks.Nodegroup(this, 'Nodegroup', {
      cluster,
      instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM)],
      removalPolicy: RemovalPolicy.DESTROY,
    });

    new eks.FargateProfile(this, 'FargateProfileConstruct', {
      cluster,
      fargateProfileName: 'fp-construct',
      selectors: [{ namespace: 'fp-construct-ns' }],
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Service Account
    new eks.ServiceAccount(this, 'PodIdentityServiceAccount', {
      cluster,
      name: 'test-pod-identity',
      removalPolicy: RemovalPolicy.DESTROY,
      identityType: eks.IdentityType.POD_IDENTITY,
    });

    new eks.ServiceAccount(this, 'ServiceAccount', {
      cluster,
      name: 'test-irsa',
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Helm Chart
    new eks.HelmChart(this, 'HelmChart', {
      cluster,
      chart: 'redis',
      repository: 'oci://registry-1.docker.io/bitnamicharts/redis',
      version: '28.0.7',
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Kubernetes Manifest
    new eks.KubernetesManifest(this, 'K8sManifest', {
      cluster,
      manifest: [{
        apiVersion: 'v1',
        kind: 'ConfigMap',
        metadata: { name: 'test-config' },
        data: { key: 'value' },
      }],
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Kubernetes Object Value
    new eks.KubernetesObjectValue(this, 'K8sObjectValue', {
      cluster,
      objectType: 'service',
      objectName: 'kubernetes',
      objectNamespace: 'default',
      jsonPath: '.spec.type',
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Kubernetes Patch
    new eks.KubernetesPatch(this, 'K8sPatch', {
      cluster,
      resourceName: 'deployment/coredns',
      resourceNamespace: 'kube-system',
      applyPatch: { spec: { replicas: 3 } },
      restorePatch: { spec: { replicas: 2 } },
      removalPolicy: RemovalPolicy.DESTROY,
    });
  }
}

const app = new App({
  postCliContext: {
    [EKS_USE_NATIVE_OIDC_PROVIDER]: true,
  },
});

const stack = new EksClusterRemovalPolicyStack(app, 'EksClusterV2RemovalPolicyStack');

new integ.IntegTest(app, 'eks-cluster-removal-policy-integ', {
  testCases: [stack],
  diffAssets: false,
});

app.synth();
