/// !cdk-integ pragma:disable-update-workflow
import * as integ from '@aws-cdk/integ-tests-alpha';
import { KubectlV34Layer } from '@aws-cdk/lambda-layer-kubectl-v34';
import { App, Stack, RemovalPolicy } from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks-v2';

/**
 * Integration test for kubectl security group configurations.
 *
 * This test creates a single stack with one VPC and three EKS clusters,
 * each demonstrating different security group configurations:
 * 1. Multiple security groups (securityGroups)
 * 2. Single security group (securityGroup - backwards compatibility)
 * 3. Default behavior (cluster security group)
 *
 * Benefits of single-stack approach:
 * - Shared VPC reduces costs
 * - Clusters may be created in parallel by CloudFormation
 * - Faster test execution compared to sequential stack deployment
 * - Simpler resource management
 */
class EksKubectlSecurityGroupsStack extends Stack {
  constructor(scope: App, id: string) {
    super(scope, id);

    // Shared VPC for all three clusters
    const vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 2,
      natGateways: 1,
      restrictDefaultSecurityGroup: false,
    });

    const privateSubnets = vpc.selectSubnets({
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    }).subnets;

    // ========================================================================
    // Test Case 1: Multiple security groups (securityGroups)
    // ========================================================================

    const kubectlSg1 = new ec2.SecurityGroup(this, 'KubectlSG1', {
      vpc,
      description: 'First security group for kubectl handler (Cluster1)',
      allowAllOutbound: true,
    });

    const kubectlSg2 = new ec2.SecurityGroup(this, 'KubectlSG2', {
      vpc,
      description: 'Second security group for kubectl handler (Cluster1)',
      allowAllOutbound: true,
    });

    const cluster1 = new eks.Cluster(this, 'Cluster1-MultipleSecurityGroups', {
      vpc,
      version: eks.KubernetesVersion.V1_34,
      endpointAccess: eks.EndpointAccess.PRIVATE,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      defaultCapacity: 0, // No default capacity to speed up test
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV34Layer(this, 'KubectlLayer1'),
        privateSubnets,
        securityGroups: [kubectlSg1, kubectlSg2], // Multiple SGs
      },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    cluster1.clusterSecurityGroup.addIngressRule(
      kubectlSg1,
      ec2.Port.tcp(443),
      'Allow kubectl from custom SG1',
    );
    cluster1.clusterSecurityGroup.addIngressRule(
      kubectlSg2,
      ec2.Port.tcp(443),
      'Allow kubectl from custom SG2',
    );

    // Validation: Add a manifest to verify kubectl works
    cluster1.addManifest('TestConfigMap1', {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: {
        name: 'test-configmap-multiple-sgs',
        namespace: 'default',
      },
      data: {
        test: 'security-groups-multiple',
        cluster: 'cluster1',
      },
    });

    // ========================================================================
    // Test Case 2: Single security group via securityGroups array
    // ========================================================================

    const kubectlSg3 = new ec2.SecurityGroup(this, 'KubectlSG3', {
      vpc,
      description: 'Custom security group for kubectl handler (Cluster2)',
      allowAllOutbound: true,
    });

    const cluster2 = new eks.Cluster(this, 'Cluster2-SingleSecurityGroup', {
      vpc,
      version: eks.KubernetesVersion.V1_34,
      endpointAccess: eks.EndpointAccess.PRIVATE,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      defaultCapacity: 0,
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV34Layer(this, 'KubectlLayer2'),
        privateSubnets,
        securityGroups: [kubectlSg3], // Single SG passed via securityGroups array
      },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    cluster2.clusterSecurityGroup.addIngressRule(
      kubectlSg3,
      ec2.Port.tcp(443),
      'Allow kubectl from custom SG',
    );

    cluster2.addManifest('TestConfigMap2', {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: {
        name: 'test-configmap-single-sg',
        namespace: 'default',
      },
      data: {
        test: 'security-group-single',
        cluster: 'cluster2',
      },
    });

    // ========================================================================
    // Test Case 3: Default behavior (no custom security group)
    // ========================================================================

    const cluster3 = new eks.Cluster(this, 'Cluster3-DefaultSecurityGroup', {
      vpc,
      version: eks.KubernetesVersion.V1_34,
      endpointAccess: eks.EndpointAccess.PRIVATE,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      defaultCapacity: 0,
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV34Layer(this, 'KubectlLayer3'),
        privateSubnets,
        // No securityGroup or securityGroups specified
        // Should use cluster security group by default
      },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    cluster3.addManifest('TestConfigMap3', {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: {
        name: 'test-configmap-default-sg',
        namespace: 'default',
      },
      data: {
        test: 'default-security-group',
        cluster: 'cluster3',
      },
    });
  }
}

// CDK App
const app = new App({
  postCliContext: {
    '@aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy': true,
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
  },
});

// Single stack with all three test cases
const stack = new EksKubectlSecurityGroupsStack(
  app,
  'aws-cdk-eks-kubectl-security-groups-test',
);

// Integration test
new integ.IntegTest(app, 'EksKubectlSecurityGroupsTest', {
  testCases: [stack],
  // Test includes assets that are updated weekly
  diffAssets: false,
  // Longer timeout for cluster creation (3 clusters)
  cdkCommandOptions: {
    deploy: {
      args: {
        rollback: false,
      },
    },
  },
});

app.synth();
