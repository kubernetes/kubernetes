import * as fs from 'fs';
import * as path from 'path';
import type { Construct } from 'constructs';
import { Node } from 'constructs';
import * as YAML from 'yaml';
import type { IAccessPolicy, IAccessEntry, AccessEntryType } from './access-entry';
import { AccessEntry, AccessPolicy, AccessScopeType } from './access-entry';
import type { IAddon } from './addon';
import { Addon } from './addon';
import type { AlbControllerOptions } from './alb-controller';
import { AlbController } from './alb-controller';
import { AwsAuth } from './aws-auth';
import { ClusterResource, clusterArnComponents } from './cluster-resource';
import type { ClusterReference, IClusterRef } from './eks.generated';
import type { FargateProfileOptions } from './fargate-profile';
import { FargateProfile } from './fargate-profile';
import type { HelmChartOptions } from './helm-chart';
import { HelmChart } from './helm-chart';
import { INSTANCE_TYPES } from './instance-types';
import type { KubernetesManifestOptions } from './k8s-manifest';
import { KubernetesManifest } from './k8s-manifest';
import { KubernetesObjectValue } from './k8s-object-value';
import { KubernetesPatch } from './k8s-patch';
import type { IKubectlProvider } from './kubectl-provider';
import { KubectlProvider } from './kubectl-provider';
import type { NodegroupOptions } from './managed-nodegroup';
import { Nodegroup } from './managed-nodegroup';
import { OpenIdConnectProvider, OidcProviderNative } from './oidc-provider';
import { BottleRocketImage } from './private/bottlerocket';
import type { ServiceAccountOptions } from './service-account';
import { ServiceAccount } from './service-account';
import { LifecycleLabel, renderAmazonLinuxUserData, renderBottlerocketUserData } from './user-data';
import * as autoscaling from '../../aws-autoscaling';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import type * as kms from '../../aws-kms';
import type * as lambda from '../../aws-lambda';
import * as ssm from '../../aws-ssm';
import type { IResource, Duration, Size, RemovalPolicy } from '../../core';
import { Annotations, CfnOutput, CfnResource, Resource, Stack, Tags, Token, ValidationError, UnscopedValidationError, RemovalPolicies, FeatureFlags, Validations } from '../../core';
import { memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import { EKS_USE_NATIVE_OIDC_PROVIDER } from '../../cx-api';

// defaults are based on https://eksctl.io
const DEFAULT_CAPACITY_COUNT = 2;
const DEFAULT_CAPACITY_TYPE = ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE);

/**
 * An EKS cluster
 */
export interface ICluster extends IResource, ec2.IConnectable, IClusterRef {
  /**
   * The VPC in which this Cluster was created
   */
  readonly vpc: ec2.IVpc;

  /**
   * The physical name of the Cluster
   * @attribute
   */
  readonly clusterName: string;

  /**
   * The unique ARN assigned to the service by AWS
   * in the form of arn:aws:eks:
   * @attribute
   */
  readonly clusterArn: string;

  /**
   * The API Server endpoint URL
   * @attribute
   */
  readonly clusterEndpoint: string;

  /**
   * The certificate-authority-data for your cluster.
   * @attribute
   */
  readonly clusterCertificateAuthorityData: string;

  /**
   * The id of the cluster security group that was created by Amazon EKS for the cluster.
   * @attribute
   */
  readonly clusterSecurityGroupId: string;

  /**
   * The cluster security group that was created by Amazon EKS for the cluster.
   * @attribute
   */
  readonly clusterSecurityGroup: ec2.ISecurityGroup;

  /**
   * Amazon Resource Name (ARN) or alias of the customer master key (CMK).
   * @attribute
   */
  readonly clusterEncryptionConfigKeyArn: string;

  /**
   * The Open ID Connect Provider of the cluster used to configure Service Accounts.
   */
  readonly openIdConnectProvider: iam.IOpenIdConnectProvider;

  /**
   * The EKS Pod Identity Agent addon for the EKS cluster.
   *
   * The EKS Pod Identity Agent is responsible for managing the temporary credentials
   * used by pods in the cluster to access AWS resources. It runs as a DaemonSet on
   * each node and provides the necessary credentials to the pods based on their
   * associated service account.
   *
   * This property returns the `CfnAddon` resource representing the EKS Pod Identity
   * Agent addon. If the addon has not been created yet, it will be created and
   * returned.
   */
  readonly eksPodIdentityAgent?: IAddon;

  /**
   * An IAM role that can perform kubectl operations against this cluster.
   *
   * The role should be mapped to the `system:masters` Kubernetes RBAC role.
   */
  readonly kubectlRole?: iam.IRole;

  /**
   * Custom environment variables when running `kubectl` against this cluster.
   */
  readonly kubectlEnvironment?: { [key: string]: string };

  /**
   * A security group to use for `kubectl` execution.
   *
   * If this is undefined, the k8s endpoint is expected to be accessible
   * publicly.
   */
  readonly kubectlSecurityGroup?: ec2.ISecurityGroup;

  /**
   * Subnets to host the `kubectl` compute resources.
   *
   * If this is undefined, the k8s endpoint is expected to be accessible
   * publicly.
   */
  readonly kubectlPrivateSubnets?: ec2.ISubnet[];

  /**
   * An IAM role that can perform kubectl operations against this cluster.
   *
   * The role should be mapped to the `system:masters` Kubernetes RBAC role.
   *
   * This role is directly passed to the lambda handler that sends Kube Ctl commands to the cluster.
   */
  readonly kubectlLambdaRole?: iam.IRole;

  /**
   * An AWS Lambda layer that includes `kubectl` and `helm`
   *
   */
  readonly kubectlLayer?: lambda.ILayerVersion;

  /**
   * Specify which IP family is used to assign Kubernetes pod and service IP addresses.
   *
   * @default - IpFamily.IP_V4
   * @see https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-ipFamily
   */
  readonly ipFamily?: IpFamily;

  /**
   * An AWS Lambda layer that contains the `aws` CLI.
   *
   * If not defined, a default layer will be used containing the AWS CLI 1.x.
   */
  readonly awscliLayer?: lambda.ILayerVersion;

  /**
   * Kubectl Provider for issuing kubectl commands against it
   *
   * If not defined, a default provider will be used
   */
  readonly kubectlProvider?: IKubectlProvider;

  /**
   * Amount of memory to allocate to the provider's lambda function.
   */
  readonly kubectlMemory?: Size;

  /**
   * A security group to associate with the Cluster Handler's Lambdas.
   * The Cluster Handler's Lambdas are responsible for calling AWS's EKS API.
   *
   * Requires `placeClusterHandlerInVpc` to be set to true.
   *
   * @default - No security group.
   * @attribute
   */
  readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup;

  /**
   * An AWS Lambda layer that includes the NPM dependency `proxy-agent`.
   *
   * If not defined, a default layer will be used.
   */
  readonly onEventLayer?: lambda.ILayerVersion;

  /**
   * Indicates whether Kubernetes resources can be automatically pruned. When
   * this is enabled (default), prune labels will be allocated and injected to
   * each resource. These labels will then be used when issuing the `kubectl
   * apply` operation with the `--prune` switch.
   */
  readonly prune: boolean;

  /**
   * The authentication mode for the cluster.
   * @default AuthenticationMode.CONFIG_MAP
   */
  readonly authenticationMode?: AuthenticationMode;

  /**
   * Creates a new service account with corresponding IAM Role (IRSA).
   *
   * @param id logical id of service account
   * @param options service account options
   */
  addServiceAccount(id: string, options?: ServiceAccountOptions): ServiceAccount;

  /**
   * Defines a Kubernetes resource in this cluster.
   *
   * The manifest will be applied/deleted using kubectl as needed.
   *
   * @param id logical id of this manifest
   * @param manifest a list of Kubernetes resource specifications
   * @returns a `KubernetesManifest` object.
   */
  addManifest(id: string, ...manifest: Record<string, any>[]): KubernetesManifest;

  /**
   * Defines a Helm chart in this cluster.
   *
   * @param id logical id of this chart.
   * @param options options of this chart.
   * @returns a `HelmChart` construct
   */
  addHelmChart(id: string, options: HelmChartOptions): HelmChart;

  /**
   * Defines a CDK8s chart in this cluster.
   *
   * @param id logical id of this chart.
   * @param chart the cdk8s chart.
   * @returns a `KubernetesManifest` construct representing the chart.
   */
  addCdk8sChart(id: string, chart: Construct, options?: KubernetesManifestOptions): KubernetesManifest;

  /**
   * Connect capacity in the form of an existing AutoScalingGroup to the EKS cluster.
   *
   * The AutoScalingGroup must be running an EKS-optimized AMI containing the
   * /etc/eks/bootstrap.sh script. This method will configure Security Groups,
   * add the right policies to the instance role, apply the right tags, and add
   * the required user data to the instance's launch configuration.
   *
   * Spot instances will be labeled `lifecycle=Ec2Spot` and tainted with `PreferNoSchedule`.
   * If kubectl is enabled, the
   * [spot interrupt handler](https://github.com/awslabs/ec2-spot-labs/tree/master/ec2-spot-eks-solution/spot-termination-handler)
   * daemon will be installed on all spot instances to handle
   * [EC2 Spot Instance Termination Notices](https://aws.amazon.com/blogs/aws/new-ec2-spot-instance-termination-notices/).
   *
   * Prefer to use `addAutoScalingGroupCapacity` if possible.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/launch-workers.html
   * @param autoScalingGroup [disable-awslint:ref-via-interface]
   * @param options options for adding auto scaling groups, like customizing the bootstrap script
   */
  connectAutoScalingGroupCapacity(autoScalingGroup: autoscaling.AutoScalingGroup, options: AutoScalingGroupOptions): void;
}

/**
 * Attributes for EKS clusters.
 */
export interface ClusterAttributes {
  /**
   * The VPC in which this Cluster was created
   * @default - if not specified `cluster.vpc` will throw an error
   */
  readonly vpc?: ec2.IVpc;

  /**
   * The physical name of the Cluster
   */
  readonly clusterName: string;

  /**
   * The API Server endpoint URL
   * @default - if not specified `cluster.clusterEndpoint` will throw an error.
   */
  readonly clusterEndpoint?: string;

  /**
   * The certificate-authority-data for your cluster.
   * @default - if not specified `cluster.clusterCertificateAuthorityData` will
   * throw an error
   */
  readonly clusterCertificateAuthorityData?: string;

  /**
   * The cluster security group that was created by Amazon EKS for the cluster.
   * @default - if not specified `cluster.clusterSecurityGroupId` will throw an
   * error
   */
  readonly clusterSecurityGroupId?: string;

  /**
   * Amazon Resource Name (ARN) or alias of the customer master key (CMK).
   * @default - if not specified `cluster.clusterEncryptionConfigKeyArn` will
   * throw an error
   */
  readonly clusterEncryptionConfigKeyArn?: string;

  /**
   * Specify which IP family is used to assign Kubernetes pod and service IP addresses.
   *
   * @default - IpFamily.IP_V4
   * @see https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-ipFamily
   */
  readonly ipFamily?: IpFamily;

  /**
   * Additional security groups associated with this cluster.
   * @default - if not specified, no additional security groups will be
   * considered in `cluster.connections`.
   */
  readonly securityGroupIds?: string[];

  /**
   * An IAM role with cluster administrator and "system:masters" permissions.
   * @default - if not specified, it not be possible to issue `kubectl` commands
   * against an imported cluster.
   */
  readonly kubectlRoleArn?: string;

  /**
   * An IAM role that can perform kubectl operations against this cluster.
   *
   * The role should be mapped to the `system:masters` Kubernetes RBAC role.
   *
   * This role is directly passed to the lambda handler that sends Kube Ctl commands
   * to the cluster.
   * @default - if not specified, the default role created by a lambda function will
   * be used.
   */
  readonly kubectlLambdaRole?: iam.IRole;

  /**
   * Environment variables to use when running `kubectl` against this cluster.
   * @default - no additional variables
   */
  readonly kubectlEnvironment?: { [name: string]: string };

  /**
   * A security group to use for `kubectl` execution. If not specified, the k8s
   * endpoint is expected to be accessible publicly.
   * @default - k8s endpoint is expected to be accessible publicly
   */
  readonly kubectlSecurityGroupId?: string;

  /**
   * Subnets to host the `kubectl` compute resources. If not specified, the k8s
   * endpoint is expected to be accessible publicly.
   * @default - k8s endpoint is expected to be accessible publicly
   */
  readonly kubectlPrivateSubnetIds?: string[];

  /**
   * An Open ID Connect provider for this cluster that can be used to configure service accounts.
   * You can either import an existing provider using `iam.OpenIdConnectProvider.fromProviderArn`,
   * or create a new provider using `new eks.OpenIdConnectProvider`
   * @default - if not specified `cluster.openIdConnectProvider` and `cluster.addServiceAccount` will throw an error.
   */
  readonly openIdConnectProvider?: iam.IOpenIdConnectProvider;

  /**
   * An AWS Lambda Layer which includes `kubectl` and Helm.
   *
   * This layer is used by the kubectl handler to apply manifests and install
   * helm charts. You must pick an appropriate releases of one of the
   * `@aws-cdk/layer-kubectl-vXX` packages, that works with the version of
   * Kubernetes you have chosen.
   *
   * The handler expects the layer to include the following executables:
   *
   * ```
   * /opt/helm/helm
   * /opt/kubectl/kubectl
   * ```
   *
   * @default - No default layer will be provided
   */
  readonly kubectlLayer?: lambda.ILayerVersion;

  /**
   * An AWS Lambda layer that contains the `aws` CLI.
   *
   * The handler expects the layer to include the following executables:
   *
   * ```
   * /opt/awscli/aws
   * ```
   *
   * @default - a default layer with the AWS CLI 1.x
   */
  readonly awscliLayer?: lambda.ILayerVersion;

  /**
   * KubectlProvider for issuing kubectl commands.
   *
   * @default - Default CDK provider
   */
  readonly kubectlProvider?: IKubectlProvider;

  /**
   * Amount of memory to allocate to the provider's lambda function.
   *
   * @default Size.gibibytes(1)
   */
  readonly kubectlMemory?: Size;

  /**
   * A security group id to associate with the Cluster Handler's Lambdas.
   * The Cluster Handler's Lambdas are responsible for calling AWS's EKS API.
   *
   * @default - No security group.
   */
  readonly clusterHandlerSecurityGroupId?: string;

  /**
   * An AWS Lambda Layer which includes the NPM dependency `proxy-agent`. This layer
   * is used by the onEvent handler to route AWS SDK requests through a proxy.
   *
   * The handler expects the layer to include the following node_modules:
   *
   *    proxy-agent
   *
   * @default - a layer bundled with this module.
   */
  readonly onEventLayer?: lambda.ILayerVersion;

  /**
   * Indicates whether Kubernetes resources added through `addManifest()` can be
   * automatically pruned. When this is enabled (default), prune labels will be
   * allocated and injected to each resource. These labels will then be used
   * when issuing the `kubectl apply` operation with the `--prune` switch.
   *
   * @default true
   */
  readonly prune?: boolean;
}

/**
 * Options for configuring an EKS cluster.
 */
export interface CommonClusterOptions {
  /**
   * The VPC in which to create the Cluster.
   *
   * @default - a VPC with default configuration will be created and can be accessed through `cluster.vpc`.
   */
  readonly vpc?: ec2.IVpc;

  /**
   * Where to place EKS Control Plane ENIs
   *
   * For example, to only select private subnets, supply the following:
   *
   * `vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }]`
   *
   * @default - All public and private subnets
   */
  readonly vpcSubnets?: ec2.SubnetSelection[];

  /**
   * Role that provides permissions for the Kubernetes control plane to make calls to AWS API operations on your behalf.
   *
   * @default - A role is automatically created for you
   */
  readonly role?: iam.IRole;

  /**
   * Name for the cluster.
   *
   * @default - Automatically generated name
   */
  readonly clusterName?: string;

  /**
   * Security Group to use for Control Plane ENIs
   *
   * @default - A security group is automatically created
   */
  readonly securityGroup?: ec2.ISecurityGroup;

  /**
   * The Kubernetes version to run in the cluster
   */
  readonly version: KubernetesVersion;

  /**
   * Determines whether a CloudFormation output with the name of the cluster
   * will be synthesized.
   *
   * @default false
   */
  readonly outputClusterName?: boolean;

  /**
   * Determines whether a CloudFormation output with the `aws eks
   * update-kubeconfig` command will be synthesized. This command will include
   * the cluster name and the ARN of the masters IAM role.
   *
   * Note: If mastersRole is not specified, this property will be ignored and no config command will be emitted.
   * @see https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_eks-readme.html#masters-role
   *
   * @default true
   */
  readonly outputConfigCommand?: boolean;
}

/**
 * Options for EKS clusters.
 */
export interface ClusterOptions extends CommonClusterOptions {
  /**
   * An IAM role that will be added to the `system:masters` Kubernetes RBAC
   * group.
   *
   * @see https://kubernetes.io/docs/reference/access-authn-authz/rbac/#default-roles-and-role-bindings
   *
   * @default - no masters role.
   */
  readonly mastersRole?: iam.IRole;

  /**
   * Controls the "eks.amazonaws.com/compute-type" annotation in the CoreDNS
   * configuration on your cluster to determine which compute type to use
   * for CoreDNS.
   *
   * @default CoreDnsComputeType.EC2 (for `FargateCluster` the default is FARGATE)
   */
  readonly coreDnsComputeType?: CoreDnsComputeType;

  /**
   * Determines whether a CloudFormation output with the ARN of the "masters"
   * IAM role will be synthesized (if `mastersRole` is specified).
   *
   * @default false
   */
  readonly outputMastersRoleArn?: boolean;

  /**
   * Configure access to the Kubernetes API server endpoint..
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/cluster-endpoint.html
   *
   * @default EndpointAccess.PUBLIC_AND_PRIVATE
   */
  readonly endpointAccess?: EndpointAccess;

  /**
   * Environment variables for the kubectl execution. Only relevant for kubectl enabled clusters.
   *
   * @default - No environment variables.
   */
  readonly kubectlEnvironment?: { [key: string]: string };

  /**
   * An AWS Lambda Layer which includes `kubectl` and Helm.
   *
   * This layer is used by the kubectl handler to apply manifests and install
   * helm charts. You must pick an appropriate releases of one of the
   * `@aws-cdk/layer-kubectl-vXX` packages, that works with the version of
   * Kubernetes you have chosen.
   *
   * The handler expects the layer to include the following executables:
   *
   * ```
   * /opt/helm/helm
   * /opt/kubectl/kubectl
   * ```
   */
  readonly kubectlLayer: lambda.ILayerVersion;

  /**
   * An AWS Lambda layer that contains the `aws` CLI.
   *
   * The handler expects the layer to include the following executables:
   *
   * ```
   * /opt/awscli/aws
   * ```
   *
   * @default - a default layer with the AWS CLI 1.x
   */
  readonly awscliLayer?: lambda.ILayerVersion;

  /**
   * Amount of memory to allocate to the provider's lambda function.
   *
   * @default Size.gibibytes(1)
   */
  readonly kubectlMemory?: Size;

  /**
   * Custom environment variables when interacting with the EKS endpoint to manage the cluster lifecycle.
   *
   * @default - No environment variables.
   */
  readonly clusterHandlerEnvironment?: { [key: string]: string };

  /**
   * A security group to associate with the Cluster Handler's Lambdas.
   * The Cluster Handler's Lambdas are responsible for calling AWS's EKS API.
   *
   * Requires `placeClusterHandlerInVpc` to be set to true.
   *
   * @default - No security group.
   */
  readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup;

  /**
   * An AWS Lambda Layer which includes the NPM dependency `proxy-agent`. This layer
   * is used by the onEvent handler to route AWS SDK requests through a proxy.
   *
   * By default, the provider will use the layer included in the
   * "aws-lambda-layer-node-proxy-agent" SAR application which is available in all
   * commercial regions.
   *
   * To deploy the layer locally define it in your app as follows:
   *
   * ```ts
   * const layer = new lambda.LayerVersion(this, 'proxy-agent-layer', {
   *   code: lambda.Code.fromAsset(`${__dirname}/layer.zip`),
   *   compatibleRuntimes: [lambda.Runtime.NODEJS_LATEST],
   * });
   * ```
   *
   * @default - a layer bundled with this module.
   */
  readonly onEventLayer?: lambda.ILayerVersion;

  /**
   * Indicates whether Kubernetes resources added through `addManifest()` can be
   * automatically pruned. When this is enabled (default), prune labels will be
   * allocated and injected to each resource. These labels will then be used
   * when issuing the `kubectl apply` operation with the `--prune` switch.
   *
   * @default true
   */
  readonly prune?: boolean;

  /**
   * If set to true, the cluster handler functions will be placed in the private subnets
   * of the cluster vpc, subject to the `vpcSubnets` selection strategy.
   *
   * @default false
   */
  readonly placeClusterHandlerInVpc?: boolean;

  /**
   * KMS secret for envelope encryption for Kubernetes secrets.
   *
   * @default - By default, Kubernetes stores all secret object data within etcd and
   *            all etcd volumes used by Amazon EKS are encrypted at the disk-level
   *            using AWS-Managed encryption keys.
   */
  readonly secretsEncryptionKey?: kms.IKeyRef;

  /**
   * Specify which IP family is used to assign Kubernetes pod and service IP addresses.
   *
   * @default - IpFamily.IP_V4
   * @see https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-ipFamily
   */
  readonly ipFamily?: IpFamily;

  /**
   * The CIDR block to assign Kubernetes service IP addresses from.
   *
   * @default - Kubernetes assigns addresses from either the
   *            10.100.0.0/16 or 172.20.0.0/16 CIDR blocks
   * @see https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-serviceIpv4Cidr
   */
  readonly serviceIpv4Cidr?: string;

  /**
   * Install the AWS Load Balancer Controller onto the cluster.
   *
   * @see https://kubernetes-sigs.github.io/aws-load-balancer-controller
   *
   * @default - The controller is not installed.
   */
  readonly albController?: AlbControllerOptions;
  /**
   * The cluster log types which you want to enable.
   *
   * @default - none
   */
  readonly clusterLogging?: ClusterLoggingTypes[];

  /**
   * The desired authentication mode for the cluster.
   * @default AuthenticationMode.CONFIG_MAP
   */
  readonly authenticationMode?: AuthenticationMode;

  /**
   * IPv4 CIDR blocks defining the expected address range of hybrid nodes
   * that will join the cluster.
   * @default - none
   */
  readonly remoteNodeNetworks?: RemoteNodeNetwork[];

  /**
   * IPv4 CIDR blocks for Pods running Kubernetes webhooks on hybrid nodes.
   * @default - none
   */
  readonly remotePodNetworks?: RemotePodNetwork[];

  /**
   * The removal policy applied to all CloudFormation resources created by this construct
   * when they are no longer managed by CloudFormation.
   *
   * This can happen in one of three situations:
     - The resource is removed from the template, so CloudFormation stops managing it;
     - A change to the resource is made that requires it to be replaced, so CloudFormation stops managing it;
     - The stack is deleted, so CloudFormation stops managing all resources in it.
   *
   * This affects the EKS cluster itself, associated IAM roles, node groups, security groups, VPC
   * and any other CloudFormation resources managed by this construct.
   *
   * @default - Resources will be deleted.
   */
  readonly removalPolicy?: RemovalPolicy;

  /**
   * The current deletion protection setting for the cluster.
   * When true, deletion protection is enabled and the cluster cannot be deleted until protection is disabled.
   * When false, the cluster can be deleted normally. This setting only applies to clusters in an active state.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/deletion-protection.html
   *
   * @default - deletion protection is disabled
   */
  readonly deletionProtection?: boolean;

  /**
   * The control plane scaling tier for EKS Provisioned Control Plane.
   *
   * Provisioned Control Plane allows you to select a scaling tier to ensure
   * high and predictable performance for demanding workloads such as
   * AI training/inference, high-performance computing, or large-scale data processing.
   *
   * @default - Standard control plane (no provisioned tier)
   * @see https://docs.aws.amazon.com/eks/latest/userguide/eks-provisioned-control-plane.html
   */
  readonly controlPlaneScalingTier?: ControlPlaneScalingTier;
}

/**
 * Group access configuration together.
 */
interface EndpointAccessConfig {

  /**
   * Indicates if private access is enabled.
   */
  readonly privateAccess: boolean;

  /**
   * Indicates if public access is enabled.
   */
  readonly publicAccess: boolean;
  /**
   * Public access is allowed only from these CIDR blocks.
   * An empty array means access is open to any address.
   *
   * @default - No restrictions.
   */
  readonly publicCidrs?: string[];

}

/**
 * Endpoint access characteristics.
 */
export class EndpointAccess {
  /**
   * The cluster endpoint is accessible from outside of your VPC.
   * Worker node traffic will leave your VPC to connect to the endpoint.
   *
   * By default, the endpoint is exposed to all addresses. You can optionally limit the CIDR blocks that can access the public endpoint using the `PUBLIC.onlyFrom` method.
   * If you limit access to specific CIDR blocks, you must ensure that the CIDR blocks that you
   * specify include the addresses that worker nodes and Fargate pods (if you use them)
   * access the public endpoint from.
   *
   * @param cidr The CIDR blocks.
   */
  public static readonly PUBLIC = new EndpointAccess({ privateAccess: false, publicAccess: true });

  /**
   * The cluster endpoint is only accessible through your VPC.
   * Worker node traffic to the endpoint will stay within your VPC.
   */
  public static readonly PRIVATE = new EndpointAccess({ privateAccess: true, publicAccess: false });

  /**
   * The cluster endpoint is accessible from outside of your VPC.
   * Worker node traffic to the endpoint will stay within your VPC.
   *
   * By default, the endpoint is exposed to all addresses. You can optionally limit the CIDR blocks that can access the public endpoint using the `PUBLIC_AND_PRIVATE.onlyFrom` method.
   * If you limit access to specific CIDR blocks, you must ensure that the CIDR blocks that you
   * specify include the addresses that worker nodes and Fargate pods (if you use them)
   * access the public endpoint from.
   *
   * @param cidr The CIDR blocks.
   */
  public static readonly PUBLIC_AND_PRIVATE = new EndpointAccess({ privateAccess: true, publicAccess: true });

  private constructor(
    /**
     * Configuration properties.
     *
     * @internal
     */
    public readonly _config: EndpointAccessConfig) {
    if (!_config.publicAccess && _config.publicCidrs && _config.publicCidrs.length > 0) {
      throw new UnscopedValidationError(lit`CidrBlocksOnlyConfigured`, 'CIDR blocks can only be configured when public access is enabled');
    }
  }

  /**
   * Restrict public access to specific CIDR blocks.
   * If public access is disabled, this method will result in an error.
   *
   * @param cidr CIDR blocks.
   */
  public onlyFrom(...cidr: string[]) {
    if (!this._config.privateAccess) {
      // when private access is disabled, we can't restric public
      // access since it will render the kubectl provider unusable.
      throw new UnscopedValidationError(lit`CannotRestricPublicAccessEndpoint`, 'Cannot restric public access to endpoint when private access is disabled. Use PUBLIC_AND_PRIVATE.onlyFrom() instead.');
    }
    return new EndpointAccess({
      ...this._config,
      // override CIDR
      publicCidrs: cidr,
    });
  }
}

/**
 * Common configuration props for EKS clusters.
 */
export interface ClusterProps extends ClusterOptions {

  /**
   * Number of instances to allocate as an initial capacity for this cluster.
   * Instance type can be configured through `defaultCapacityInstanceType`,
   * which defaults to `m5.large`.
   *
   * Use `cluster.addAutoScalingGroupCapacity` to add additional customized capacity. Set this
   * to `0` is you wish to avoid the initial capacity allocation.
   *
   * @default 2
   */
  readonly defaultCapacity?: number;

  /**
   * The instance type to use for the default capacity. This will only be taken
   * into account if `defaultCapacity` is > 0.
   *
   * @default m5.large
   */
  readonly defaultCapacityInstance?: ec2.InstanceType;

  /**
   * The default capacity type for the cluster.
   *
   * @default NODEGROUP
   */
  readonly defaultCapacityType?: DefaultCapacityType;

  /**
   * The IAM role to pass to the Kubectl Lambda Handler.
   *
   * @default - Default Lambda IAM Execution Role
   */
  readonly kubectlLambdaRole?: iam.IRole;

  /**
   * Whether or not IAM principal of the cluster creator was set as a cluster admin access entry
   * during cluster creation time.
   *
   * Changing this value after the cluster has been created will result in the cluster being replaced.
   *
   * @default true
   */
  readonly bootstrapClusterCreatorAdminPermissions?: boolean;

  /**
   * If you set this value to False when creating a cluster, the default networking add-ons will not be installed.
   * The default networking addons include vpc-cni, coredns, and kube-proxy.
   * Use this option when you plan to install third-party alternative add-ons or self-manage the default networking add-ons.
   *
   * Changing this value after the cluster has been created will result in the cluster being replaced.
   *
   * @default true
   */
  readonly bootstrapSelfManagedAddons?: boolean;

  /**
   * The tags assigned to the EKS cluster
   *
   * @default - none
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Kubernetes cluster version
 * @see https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html#kubernetes-release-calendar
 */
export class KubernetesVersion {
  /**
   * Kubernetes version 1.14
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_14 = KubernetesVersion.of('1.14');

  /**
   * Kubernetes version 1.15
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_15 = KubernetesVersion.of('1.15');

  /**
   * Kubernetes version 1.16
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_16 = KubernetesVersion.of('1.16');

  /**
   * Kubernetes version 1.17
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_17 = KubernetesVersion.of('1.17');

  /**
   * Kubernetes version 1.18
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_18 = KubernetesVersion.of('1.18');

  /**
   * Kubernetes version 1.19
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_19 = KubernetesVersion.of('1.19');

  /**
   * Kubernetes version 1.20
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_20 = KubernetesVersion.of('1.20');

  /**
   * Kubernetes version 1.21
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_21 = KubernetesVersion.of('1.21');

  /**
   * Kubernetes version 1.22
   * @deprecated Use newer version of EKS
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV22Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v22`.
   * @deprecated Use newer version of EKS
   */
  public static readonly V1_22 = KubernetesVersion.of('1.22');

  /**
   * Kubernetes version 1.23
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV23Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v23`.
   */
  public static readonly V1_23 = KubernetesVersion.of('1.23');

  /**
   * Kubernetes version 1.24
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV24Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v24`.
   */
  public static readonly V1_24 = KubernetesVersion.of('1.24');

  /**
   * Kubernetes version 1.25
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV25Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v25`.
   */
  public static readonly V1_25 = KubernetesVersion.of('1.25');

  /**
   * Kubernetes version 1.26
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV26Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v26`.
   */
  public static readonly V1_26 = KubernetesVersion.of('1.26');

  /**
   * Kubernetes version 1.27
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV27Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v27`.
   */
  public static readonly V1_27 = KubernetesVersion.of('1.27');

  /**
   * Kubernetes version 1.28
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV28Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v28`.
   */
  public static readonly V1_28 = KubernetesVersion.of('1.28');

  /**
   * Kubernetes version 1.29
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV29Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v29`.
   */
  public static readonly V1_29 = KubernetesVersion.of('1.29');

  /**
   * Kubernetes version 1.30
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV30Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v30`.
   */
  public static readonly V1_30 = KubernetesVersion.of('1.30');

  /**
   * Kubernetes version 1.31
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV31Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v31`.
   */
  public static readonly V1_31 = KubernetesVersion.of('1.31');

  /**
   * Kubernetes version 1.32
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV32Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v32`.
   */
  public static readonly V1_32 = KubernetesVersion.of('1.32');

  /**
   * Kubernetes version 1.33
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV33Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v33`.
   */
  public static readonly V1_33 = KubernetesVersion.of('1.33');

  /**
   * Kubernetes version 1.34
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV34Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v34`.
   */
  public static readonly V1_34 = KubernetesVersion.of('1.34');

  /**
   * Kubernetes version 1.35
   *
   * When creating a `Cluster` with this version, you need to also specify the
   * `kubectlLayer` property with a `KubectlV35Layer` from
   * `@aws-cdk/lambda-layer-kubectl-v35`.
   */
  public static readonly V1_35 = KubernetesVersion.of('1.35');

  /**
   * Custom cluster version
   * @param version custom version number
   */
  public static of(version: string) { return new KubernetesVersion(version); }
  /**
   *
   * @param version cluster version number
   */
  private constructor(public readonly version: string) { }
}

// Shared definition with packages/@aws-cdk/custom-resource-handlers/test/aws-eks/compare-log.test.ts
/**
 * EKS cluster logging types
 */
export enum ClusterLoggingTypes {
  /**
   * Logs pertaining to API requests to the cluster.
   */
  API = 'api',
  /**
   * Logs pertaining to cluster access via the Kubernetes API.
   */
  AUDIT = 'audit',
  /**
   * Logs pertaining to authentication requests into the cluster.
   */
  AUTHENTICATOR = 'authenticator',
  /**
   * Logs pertaining to state of cluster controllers.
   */
  CONTROLLER_MANAGER = 'controllerManager',
  /**
   * Logs pertaining to scheduling decisions.
   */
  SCHEDULER = 'scheduler',
}

/**
 * EKS cluster IP family.
 */
export enum IpFamily {
  /**
   * Use IPv4 for pods and services in your cluster.
   */
  IP_V4 = 'ipv4',
  /**
   * Use IPv6 for pods and services in your cluster.
   */
  IP_V6 = 'ipv6',
}

/**
 * Represents the authentication mode for an Amazon EKS cluster.
 */
export enum AuthenticationMode {
  /**
   * Authenticates using a Kubernetes ConfigMap.
   */
  CONFIG_MAP = 'CONFIG_MAP',
  /**
   * Authenticates using both the Kubernetes API server and a ConfigMap.
   */
  API_AND_CONFIG_MAP = 'API_AND_CONFIG_MAP',
  /**
   * Authenticates using the Kubernetes API server.
   */
  API = 'API',
}

/**
 * Control plane scaling tier for EKS Provisioned Control Plane.
 *
 * Provisioned Control Plane allows cluster administrators to select from a set
 * of scaling tiers to ensure high and predictable performance for demanding workloads
 * such as AI training/inference, high-performance computing, or large-scale data processing.
 *
 * @see https://docs.aws.amazon.com/eks/latest/userguide/eks-provisioned-control-plane.html
 */
export enum ControlPlaneScalingTier {
  /**
   * Standard control plane (default, no additional cost).
   */
  STANDARD = 'standard',
  /**
   * Extra-large provisioned tier.
   */
  TIER_XL = 'tier-xl',
  /**
   * 2x extra-large provisioned tier.
   */
  TIER_2XL = 'tier-2xl',
  /**
   * 4x extra-large provisioned tier.
   */
  TIER_4XL = 'tier-4xl',
  /**
   * 8x extra-large provisioned tier.
   */
  TIER_8XL = 'tier-8xl',
}

abstract class ClusterBase extends Resource implements ICluster {
  public abstract readonly connections: ec2.Connections;
  public abstract readonly vpc: ec2.IVpc;
  public abstract readonly clusterName: string;
  public abstract readonly clusterArn: string;
  public abstract readonly clusterEndpoint: string;
  public abstract readonly clusterCertificateAuthorityData: string;
  public abstract readonly clusterSecurityGroupId: string;
  public abstract readonly clusterSecurityGroup: ec2.ISecurityGroup;
  public abstract readonly clusterEncryptionConfigKeyArn: string;
  public abstract readonly ipFamily?: IpFamily;
  public abstract readonly kubectlRole?: iam.IRole;
  public abstract readonly kubectlLambdaRole?: iam.IRole;
  public abstract readonly kubectlLayer?: lambda.ILayerVersion;
  public abstract readonly kubectlEnvironment?: { [key: string]: string };
  public abstract readonly kubectlSecurityGroup?: ec2.ISecurityGroup;
  public abstract readonly kubectlPrivateSubnets?: ec2.ISubnet[];
  public abstract readonly kubectlMemory?: Size;
  public abstract readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup;
  public abstract readonly prune: boolean;
  public abstract readonly openIdConnectProvider: iam.IOpenIdConnectProvider;
  public abstract readonly awsAuth: AwsAuth;

  private _spotInterruptHandler?: HelmChart;

  /**
   * Manages the aws-auth config map.
   *
   * @internal
   */
  protected _awsAuth?: AwsAuth;

  /**
   * Defines a Kubernetes resource in this cluster.
   *
   * The manifest will be applied/deleted using kubectl as needed.
   *
   * @param id logical id of this manifest
   * @param manifest a list of Kubernetes resource specifications
   * @returns a `KubernetesResource` object.
   */
  public addManifest(id: string, ...manifest: Record<string, any>[]): KubernetesManifest {
    return new KubernetesManifest(this, `manifest-${id}`, { cluster: this, manifest });
  }

  /**
   * Defines a Helm chart in this cluster.
   *
   * @param id logical id of this chart.
   * @param options options of this chart.
   * @returns a `HelmChart` construct
   */
  public addHelmChart(id: string, options: HelmChartOptions): HelmChart {
    return new HelmChart(this, `chart-${id}`, { cluster: this, ...options });
  }

  /**
   * Defines a CDK8s chart in this cluster.
   *
   * @param id logical id of this chart.
   * @param chart the cdk8s chart.
   * @returns a `KubernetesManifest` construct representing the chart.
   */
  public addCdk8sChart(id: string, chart: Construct, options: KubernetesManifestOptions = {}): KubernetesManifest {
    const cdk8sChart = chart as any;

    // see https://github.com/awslabs/cdk8s/blob/master/packages/cdk8s/src/chart.ts#L84
    if (typeof cdk8sChart.toJson !== 'function') {
      throw new ValidationError(lit`InvalidCdkChartContainJson`, `Invalid cdk8s chart. Must contain a 'toJson' method, but found ${typeof cdk8sChart.toJson}`, this);
    }

    const manifest = new KubernetesManifest(this, id, {
      cluster: this,
      manifest: cdk8sChart.toJson(),
      ...options,
    });

    return manifest;
  }

  public addServiceAccount(id: string, options: ServiceAccountOptions = {}): ServiceAccount {
    return new ServiceAccount(this, id, {
      ...options,
      cluster: this,
    });
  }

  /**
   * Installs the AWS spot instance interrupt handler on the cluster if it's not
   * already added.
   */
  private addSpotInterruptHandler() {
    if (!this._spotInterruptHandler) {
      this._spotInterruptHandler = this.addHelmChart('spot-interrupt-handler', {
        chart: 'aws-node-termination-handler',
        version: '0.27.0',
        repository: 'oci://public.ecr.aws/aws-ec2/helm/aws-node-termination-handler',
        namespace: 'kube-system',
        values: {
          nodeSelector: {
            lifecycle: LifecycleLabel.SPOT,
          },
        },
      });
    }

    return this._spotInterruptHandler;
  }

  /**
   * Connect capacity in the form of an existing AutoScalingGroup to the EKS cluster.
   *
   * The AutoScalingGroup must be running an EKS-optimized AMI containing the
   * /etc/eks/bootstrap.sh script. This method will configure Security Groups,
   * add the right policies to the instance role, apply the right tags, and add
   * the required user data to the instance's launch configuration.
   *
   * Spot instances will be labeled `lifecycle=Ec2Spot` and tainted with `PreferNoSchedule`.
   * If kubectl is enabled, the
   * [spot interrupt handler](https://github.com/awslabs/ec2-spot-labs/tree/master/ec2-spot-eks-solution/spot-termination-handler)
   * daemon will be installed on all spot instances to handle
   * [EC2 Spot Instance Termination Notices](https://aws.amazon.com/blogs/aws/new-ec2-spot-instance-termination-notices/).
   *
   * Prefer to use `addAutoScalingGroupCapacity` if possible.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/launch-workers.html
   * @param autoScalingGroup [disable-awslint:ref-via-interface]
   * @param options options for adding auto scaling groups, like customizing the bootstrap script
   */
  public connectAutoScalingGroupCapacity(autoScalingGroup: autoscaling.AutoScalingGroup, options: AutoScalingGroupOptions) {
    // self rules
    autoScalingGroup.connections.allowInternally(ec2.Port.allTraffic());

    // Cluster to:nodes rules
    autoScalingGroup.connections.allowFrom(this, ec2.Port.tcp(443));
    autoScalingGroup.connections.allowFrom(this, ec2.Port.tcpRange(1025, 65535));

    // Allow HTTPS from Nodes to Cluster
    autoScalingGroup.connections.allowTo(this, ec2.Port.tcp(443));

    // Allow all node outbound traffic
    autoScalingGroup.connections.allowToAnyIpv4(ec2.Port.allTcp());
    autoScalingGroup.connections.allowToAnyIpv4(ec2.Port.allUdp());
    autoScalingGroup.connections.allowToAnyIpv4(ec2.Port.allIcmp());

    // allow traffic to/from managed node groups (eks attaches this security group to the managed nodes)
    autoScalingGroup.addSecurityGroup(this.clusterSecurityGroup);

    const bootstrapEnabled = options.bootstrapEnabled ?? true;
    if (options.bootstrapOptions && !bootstrapEnabled) {
      throw new ValidationError(lit`CannotSpecifyBootstrapOptionsBootstrap`, 'Cannot specify "bootstrapOptions" if "bootstrapEnabled" is false', this);
    }

    if (bootstrapEnabled) {
      const userData = options.machineImageType === MachineImageType.BOTTLEROCKET ?
        renderBottlerocketUserData(this) :
        renderAmazonLinuxUserData(this, autoScalingGroup, options.bootstrapOptions);
      autoScalingGroup.addUserData(...userData);
    }

    autoScalingGroup.role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'));
    autoScalingGroup.role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'));
    autoScalingGroup.role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'));

    // EKS Required Tags
    // https://docs.aws.amazon.com/eks/latest/userguide/worker.html
    Tags.of(autoScalingGroup).add(`kubernetes.io/cluster/${this.clusterName}`, 'owned', {
      applyToLaunchedInstances: true,
      // exclude security groups to avoid multiple "owned" security groups.
      // (the cluster security group already has this tag)
      excludeResourceTypes: ['AWS::EC2::SecurityGroup'],
    });

    // do not attempt to map the role if `kubectl` is not enabled for this
    // cluster or if `mapRole` is set to false. By default this should happen.
    let mapRole = options.mapRole ?? true;
    if (mapRole && !(this instanceof Cluster)) {
      // do the mapping...
      Annotations.of(autoScalingGroup).addWarningV2('@aws-cdk/aws-eks:clusterUnsupportedAutoMappingAwsAutoRole', 'Auto-mapping aws-auth role for imported cluster is not supported, please map role manually');
      mapRole = false;
    }
    if (mapRole) {
      // see https://docs.aws.amazon.com/en_us/eks/latest/userguide/add-user-role.html
      this.awsAuth.addRoleMapping(autoScalingGroup.role, {
        username: 'system:node:{{EC2PrivateDNSName}}',
        groups: [
          'system:bootstrappers',
          'system:nodes',
        ],
      });
    } else {
      // since we are not mapping the instance role to RBAC, synthesize an
      // output so it can be pasted into `aws-auth-cm.yaml`
      new CfnOutput(autoScalingGroup, 'InstanceRoleARN', {
        value: autoScalingGroup.role.roleArn,
      });
    }

    const addSpotInterruptHandler = options.spotInterruptHandler ?? true;
    // if this is an ASG with spot instances, install the spot interrupt handler (only if kubectl is enabled).
    if (autoScalingGroup.spotPrice && addSpotInterruptHandler) {
      this.addSpotInterruptHandler();
    }

    if (this instanceof Cluster && this.albController) {
      // the controller runs on the worker nodes so they cannot
      // be deleted before the controller.
      Node.of(this.albController).addDependency(autoScalingGroup);
    }
  }

  public get clusterRef(): ClusterReference {
    return {
      clusterArn: this.clusterArn,
      clusterName: this.clusterName,
    };
  }
}

/**
 * Options for fetching a ServiceLoadBalancerAddress.
 */
export interface ServiceLoadBalancerAddressOptions {

  /**
   * Timeout for waiting on the load balancer address.
   *
   * @default Duration.minutes(5)
   */
  readonly timeout?: Duration;

  /**
   * The namespace the service belongs to.
   *
   * @default 'default'
   */
  readonly namespace?: string;

}

/**
 * Options for fetching an IngressLoadBalancerAddress.
 */
export interface IngressLoadBalancerAddressOptions extends ServiceLoadBalancerAddressOptions {}

/**
 * Internal helper interface for adding access entries to a cluster.
 */
interface AddAccessEntryOptions {
  readonly id: string;
  readonly principal: string;
  readonly policies: IAccessPolicy[];
  readonly accessEntryType?: AccessEntryType;
}

/**
 * Options for granting access to a cluster.
 */
export interface GrantAccessOptions {
  /**
   * The type of the access entry.
   *
   * Specify `AccessEntryType.EC2` for EKS Auto Mode node roles,
   * `AccessEntryType.HYBRID_LINUX` for EKS Hybrid Nodes, or
   * `AccessEntryType.HYPERPOD_LINUX` for SageMaker HyperPod.
   *
   * Note that EC2, HYBRID_LINUX, and HYPERPOD_LINUX types cannot
   * have access policies attached per AWS EKS API constraints.
   *
   * @default AccessEntryType.STANDARD - Standard access entry type that supports access policies
   */
  readonly accessEntryType?: AccessEntryType;
}

/**
 * A Cluster represents a managed Kubernetes Service (EKS)
 *
 * This is a fully managed cluster of API Servers (control-plane)
 * The user is still required to create the worker nodes.
 */
@propertyInjectable
export class Cluster extends ClusterBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-eks.Cluster';

  /**
   * Import an existing cluster
   *
   * @param scope the construct scope, in most cases 'this'
   * @param id the id or name to import as
   * @param attrs the cluster properties to use for importing information
   */
  public static fromClusterAttributes(scope: Construct, id: string, attrs: ClusterAttributes): ICluster {
    return new ImportedCluster(scope, id, attrs);
  }

  private accessEntries: Map<string, IAccessEntry> = new Map();

  /**
   * The VPC in which this Cluster was created
   */
  public readonly vpc: ec2.IVpc;

  @memoizedGetter
  public get clusterName(): string {
    return this.getResourceNameAttribute(this._clusterResource.ref);
  }

  @memoizedGetter
  public get clusterArn(): string {
    return this.getResourceArnAttribute(this._clusterResource.attrArn, clusterArnComponents(this.physicalName));
  }

  /**
   * The endpoint URL for the Cluster
   *
   * This is the URL inside the kubeconfig file to use with kubectl
   *
   * For example, `https://5E1D0CEXAMPLEA591B746AFC5AB30262.yl4.us-west-2.eks.amazonaws.com`
   */
  public readonly clusterEndpoint: string;

  /**
   * The certificate-authority-data for your cluster.
   */
  public readonly clusterCertificateAuthorityData: string;

  /**
   * The id of the cluster security group that was created by Amazon EKS for the cluster.
   */
  public readonly clusterSecurityGroupId: string;

  /**
   * The cluster security group that was created by Amazon EKS for the cluster.
   */
  public readonly clusterSecurityGroup: ec2.ISecurityGroup;

  /**
   * Amazon Resource Name (ARN) or alias of the customer master key (CMK).
   */
  public readonly clusterEncryptionConfigKeyArn: string;

  /**
   * Manages connection rules (Security Group Rules) for the cluster
   *
   * @type {ec2.Connections}
   * @memberof Cluster
   */
  public readonly connections: ec2.Connections;

  /**
   * IAM role assumed by the EKS Control Plane
   */
  public readonly role: iam.IRole;

  /**
   * The auto scaling group that hosts the default capacity for this cluster.
   * This will be `undefined` if the `defaultCapacityType` is not `EC2` or
   * `defaultCapacityType` is `EC2` but default capacity is set to 0.
   */
  public readonly defaultCapacity?: autoscaling.AutoScalingGroup;

  /**
   * The node group that hosts the default capacity for this cluster.
   * This will be `undefined` if the `defaultCapacityType` is `EC2` or
   * `defaultCapacityType` is `NODEGROUP` but default capacity is set to 0.
   */
  public readonly defaultNodegroup?: Nodegroup;

  /**
   * An IAM role that can perform kubectl operations against this cluster.
   *
   * The role should be mapped to the `system:masters` Kubernetes RBAC role.
   */
  public readonly kubectlRole?: iam.IRole;

  /**
   * An IAM role that can perform kubectl operations against this cluster.
   *
   * The role should be mapped to the `system:masters` Kubernetes RBAC role.
   *
   * This role is directly passed to the lambda handler that sends Kube Ctl commands to the cluster.
   * @default - if not specified, the default role created by a lambda function will
   * be used.
   */

  public readonly kubectlLambdaRole?: iam.IRole;

  /**
   * Custom environment variables when running `kubectl` against this cluster.
   */
  public readonly kubectlEnvironment?: { [key: string]: string };

  /**
   * A security group to use for `kubectl` execution.
   *
   * @default - If not specified, the k8s endpoint is expected to be accessible
   * publicly.
   */
  public readonly kubectlSecurityGroup?: ec2.ISecurityGroup;

  /**
   * Subnets to host the `kubectl` compute resources.
   *
   * @default - If not specified, the k8s endpoint is expected to be accessible
   * publicly.
   */
  public readonly kubectlPrivateSubnets?: ec2.ISubnet[];

  /**
   * Specify which IP family is used to assign Kubernetes pod and service IP addresses.
   *
   * @default - IpFamily.IP_V4
   * @see https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-ipFamily
   */
  public readonly ipFamily?: IpFamily;

  /**
   * An IAM role with administrative permissions to create or update the
   * cluster. This role also has `systems:master` permissions.
   */
  public readonly adminRole: iam.Role;

  /**
   * If the cluster has one (or more) FargateProfiles associated, this array
   * will hold a reference to each.
   */
  private readonly _fargateProfiles: FargateProfile[] = [];

  /**
   * an Open ID Connect Provider instance
   */
  private _openIdConnectProvider?: iam.IOpenIdConnectProvider;

  /**
   * an EKS Pod Identity Agent instance
   */
  private _eksPodIdentityAgent?: IAddon;

  /**
   * An AWS Lambda layer that includes `kubectl` and `helm`
   *
   */
  readonly kubectlLayer?: lambda.ILayerVersion;

  /**
   * An AWS Lambda layer that contains the `aws` CLI.
   *
   * If not defined, a default layer will be used containing the AWS CLI 1.x.
   */
  readonly awscliLayer?: lambda.ILayerVersion;

  /**
   * The amount of memory allocated to the kubectl provider's lambda function.
   */
  public readonly kubectlMemory?: Size;

  /**
   * A security group to associate with the Cluster Handler's Lambdas.
   * The Cluster Handler's Lambdas are responsible for calling AWS's EKS API.
   *
   * Requires `placeClusterHandlerInVpc` to be set to true.
   *
   * @default - No security group.
   */
  public readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup;

  /**
   * The AWS Lambda layer that contains the NPM dependency `proxy-agent`. If
   * undefined, a SAR app that contains this layer will be used.
   */
  readonly onEventLayer?: lambda.ILayerVersion;

  /**
   * Determines if Kubernetes resources can be pruned automatically.
   */
  public readonly prune: boolean;

  /**
   * The ALB Controller construct defined for this cluster.
   * Will be undefined if `albController` wasn't configured.
   */
  public readonly albController?: AlbController;

  /**
   * The authentication mode for the Amazon EKS cluster.
   *
   * The authentication mode determines how users and applications authenticate to the Kubernetes API server.
   *
   * @property {AuthenticationMode} [authenticationMode] - The authentication mode for the Amazon EKS cluster.
   *
   * @default CONFIG_MAP.
   */
  public readonly authenticationMode?: AuthenticationMode;

  /**
   * If this cluster is kubectl-enabled, returns the `ClusterResource` object
   * that manages it. If this cluster is not kubectl-enabled (i.e. uses the
   * stock `CfnCluster`), this is `undefined`.
   */
  private readonly _clusterResource: ClusterResource;

  private _neuronDevicePlugin?: KubernetesManifest;

  private readonly endpointAccess: EndpointAccess;

  private readonly vpcSubnets: ec2.SubnetSelection[];

  private readonly version: KubernetesVersion;

  private readonly logging?: { [key: string]: [ { [key: string]: any } ] };

  /**
   * A dummy CloudFormation resource that is used as a wait barrier which
   * represents that the cluster is ready to receive "kubectl" commands.
   *
   * Specifically, all fargate profiles are automatically added as a dependency
   * of this barrier, which means that it will only be "signaled" when all
   * fargate profiles have been successfully created.
   *
   * When kubectl resources call `_attachKubectlResourceScope()`, this resource
   * is added as their dependency which implies that they can only be deployed
   * after the cluster is ready.
   */
  private readonly _kubectlReadyBarrier: CfnResource;

  private readonly _kubectlResourceProvider: KubectlProvider;

  private readonly _removalPolicy?: RemovalPolicy;

  /**
   * Initiates an EKS Cluster with the supplied arguments
   *
   * @param scope a Construct, most likely a cdk.Stack created
   * @param id the id of the Construct to create
   * @param props properties in the IClusterProps interface
   */
  constructor(scope: Construct, id: string, props: ClusterProps) {
    super(scope, id, {
      physicalName: props.clusterName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const stack = Stack.of(this);

    this.prune = props.prune ?? true;
    this.vpc = props.vpc || new ec2.Vpc(this, 'DefaultVpc');

    this.version = props.version;

    // since this lambda role needs to be added to the trust policy of the creation role,
    // we must create it in this scope (instead of the KubectlProvider nested stack) to avoid
    // a circular dependency.
    this.kubectlLambdaRole = props.kubectlLambdaRole ? props.kubectlLambdaRole : new iam.Role(this, 'KubectlHandlerRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')],
    });

    this.tagSubnets();

    // this is the role used by EKS when interacting with AWS resources
    this.role = props.role || new iam.Role(this, 'Role', {
      assumedBy: new iam.ServicePrincipal('eks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSClusterPolicy'),
      ],
    });

    const securityGroup = props.securityGroup || new ec2.SecurityGroup(this, 'ControlPlaneSecurityGroup', {
      vpc: this.vpc,
      description: 'EKS Control Plane Security Group',
    });

    this.vpcSubnets = props.vpcSubnets ?? [{ subnetType: ec2.SubnetType.PUBLIC }, { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }];

    const selectedSubnetIdsPerGroup = this.vpcSubnets.map(s => this.vpc.selectSubnets(s).subnetIds);
    if (selectedSubnetIdsPerGroup.some(Token.isUnresolved) && selectedSubnetIdsPerGroup.length > 1) {
      throw new ValidationError(lit`EksClusterCannotSelectMultiple`, 'eks.Cluster: cannot select multiple subnet groups from a VPC imported from list tokens with unknown length. Select only one subnet group, pass a length to Fn.split, or switch to Vpc.fromLookup.', this);
    }

    // Get subnetIds for all selected subnets
    const subnetIds = Array.from(new Set(flatten(selectedSubnetIdsPerGroup)));

    this.logging = props.clusterLogging ? {
      clusterLogging: [
        {
          enabled: true,
          types: Object.values(props.clusterLogging),
        },
      ],
    } : undefined;

    this.endpointAccess = props.endpointAccess ?? EndpointAccess.PUBLIC_AND_PRIVATE;
    this.kubectlEnvironment = props.kubectlEnvironment;
    this.kubectlLayer = props.kubectlLayer;
    this.awscliLayer = props.awscliLayer;
    this.kubectlMemory = props.kubectlMemory;
    this.ipFamily = props.ipFamily ?? IpFamily.IP_V4;
    this.onEventLayer = props.onEventLayer;
    this.clusterHandlerSecurityGroup = props.clusterHandlerSecurityGroup;
    this._removalPolicy = props.removalPolicy;

    const privateSubnets = this.selectPrivateSubnets().slice(0, 16);
    const publicAccessDisabled = !this.endpointAccess._config.publicAccess;
    const publicAccessRestricted = !publicAccessDisabled
      && this.endpointAccess._config.publicCidrs
      && this.endpointAccess._config.publicCidrs.length !== 0;

    // Check if any subnet selection is pending lookup
    const hasPendingLookup = this.vpcSubnets.some(placement =>
      this.vpc.selectSubnets(placement).isPendingLookup,
    );

    // validate endpoint access configuration
    if (!hasPendingLookup) {
      if (privateSubnets.length === 0 && publicAccessDisabled) {
        // no private subnets and no public access at all, no good.
        throw new ValidationError(lit`VpcContainPrivateSubnetsPublic`, 'Vpc must contain private subnets when public endpoint access is disabled', this);
      }

      if (privateSubnets.length === 0 && publicAccessRestricted) {
      // no private subnets and public access is restricted, no good.
        throw new ValidationError(lit`VpcContainPrivateSubnetsPublic`, 'Vpc must contain private subnets when public endpoint access is restricted', this);
      }
    }

    const placeClusterHandlerInVpc = props.placeClusterHandlerInVpc ?? false;

    if (!hasPendingLookup) {
      if (placeClusterHandlerInVpc && privateSubnets.length === 0) {
        throw new ValidationError(lit`CannotPlaceClusterHandlerSince`, 'Cannot place cluster handler in the VPC since no private subnets could be selected', this);
      }
    }

    if (props.clusterHandlerSecurityGroup && !placeClusterHandlerInVpc) {
      throw new ValidationError(lit`CannotSpecifyClusterHandlerSecurity`, 'Cannot specify clusterHandlerSecurityGroup without placeClusterHandlerInVpc set to true', this);
    }

    if (props.serviceIpv4Cidr && props.ipFamily == IpFamily.IP_V6) {
      throw new ValidationError(lit`CannotSpecifyServiceIpvCidr`, 'Cannot specify serviceIpv4Cidr with ipFamily equal to IpFamily.IP_V6', this);
    }

    // Check if the cluster name exceeds 100 characters
    if (!Token.isUnresolved(this.physicalName) && this.physicalName.length > 100) {
      throw new ValidationError(lit`ClusterNameCannotCharacters`, 'Cluster name cannot be more than 100 characters', this);
    }

    this.validateRemoteNetworkConfig(props);

    this.authenticationMode = props.authenticationMode;

    const resource = this._clusterResource = new ClusterResource(this, 'Resource', {
      name: this.physicalName,
      environment: props.clusterHandlerEnvironment,
      roleArn: this.role.roleArn,
      version: props.version.version,
      accessconfig: {
        authenticationMode: props.authenticationMode,
        bootstrapClusterCreatorAdminPermissions: props.bootstrapClusterCreatorAdminPermissions,
      },
      ...(props.remoteNodeNetworks ? {
        remoteNetworkConfig: {
          remoteNodeNetworks: props.remoteNodeNetworks,
          ...(props.remotePodNetworks ? {
            remotePodNetworks: props.remotePodNetworks,
          }: {}),
        },
      } : {}),
      resourcesVpcConfig: {
        securityGroupIds: [securityGroup.securityGroupId],
        subnetIds,
      },
      ...(props.secretsEncryptionKey ? {
        encryptionConfig: [{
          provider: {
            keyArn: props.secretsEncryptionKey.keyRef.keyArn,
          },
          resources: ['secrets'],
        }],
      } : {}),
      kubernetesNetworkConfig: {
        ipFamily: this.ipFamily,
        serviceIpv4Cidr: props.serviceIpv4Cidr,
      },
      endpointPrivateAccess: this.endpointAccess._config.privateAccess,
      endpointPublicAccess: this.endpointAccess._config.publicAccess,
      publicAccessCidrs: this.endpointAccess._config.publicCidrs,
      secretsEncryptionKey: props.secretsEncryptionKey,
      vpc: this.vpc,
      subnets: placeClusterHandlerInVpc ? privateSubnets : undefined,
      clusterHandlerSecurityGroup: this.clusterHandlerSecurityGroup,
      onEventLayer: this.onEventLayer,
      tags: props.tags,
      logging: this.logging,
      bootstrapSelfManagedAddons: props.bootstrapSelfManagedAddons,
      deletionProtection: props.deletionProtection,
      controlPlaneScalingConfig: props.controlPlaneScalingTier
        ? { tier: props.controlPlaneScalingTier }
        : undefined,
    });

    this.node.defaultChild = resource;

    if (this.endpointAccess._config.privateAccess && privateSubnets.length !== 0) {
      // when private access is enabled and the vpc has private subnets, lets connect
      // the provider to the vpc so that it will work even when restricting public access.

      // validate VPC properties according to: https://docs.aws.amazon.com/eks/latest/userguide/cluster-endpoint.html
      if (this.vpc instanceof ec2.Vpc && !(this.vpc.dnsHostnamesEnabled && this.vpc.dnsSupportEnabled)) {
        throw new ValidationError(lit`RequiresPrivateEndpointAccess`, 'Private endpoint access requires the VPC to have DNS support and DNS hostnames enabled. Use `enableDnsHostnames: true` and `enableDnsSupport: true` when creating the VPC.', this);
      }

      // Validate that kubectl subnets are not isolated. Isolated subnets have no
      // internet access by definition, so the kubectl Lambda will not be able to
      // reach the EKS API, STS, or other AWS service endpoints required for
      // kubectl operations (including the CoreDNS compute type patch).
      // Only check for CDK-created VPCs where we know isolated subnets have no egress.
      // Imported VPCs may have VPC endpoints configured that we can't detect.
      // See https://github.com/aws/aws-cdk/issues/26613
      if (this.vpc instanceof ec2.Vpc) {
        const isolatedSubnetIds = new Set(this.vpc.isolatedSubnets.map(s => s.subnetId));
        const hasIsolatedSubnets = privateSubnets.some(s => isolatedSubnetIds.has(s.subnetId));
        if (hasIsolatedSubnets) {
          Annotations.of(this).addWarningV2(
            '@aws-cdk/aws-eks:isolatedSubnetsForKubectlPrivateSubnets',
            'Isolated subnets are being used for kubectl private subnets. Isolated subnets have no internet access. '
            + 'Ensure that VPC endpoints are configured for STS, EKS, ECR, S3 and other AWS services detailed here '
            + 'https://docs.aws.amazon.com/eks/latest/userguide/private-clusters.html',
          );
        }
      }

      this.kubectlPrivateSubnets = privateSubnets;

      // the vpc must exist in order to properly delete the cluster (since we run `kubectl delete`).
      // this ensures that.
      this._clusterResource.node.addDependency(this.vpc);
    }

    this.adminRole = resource.adminRole;

    // we use an SSM parameter as a barrier because it's free and fast.
    this._kubectlReadyBarrier = new CfnResource(this, 'KubectlReadyBarrier', {
      type: 'AWS::SSM::Parameter',
      properties: {
        Type: 'String',
        Value: 'aws:cdk:eks:kubectl-ready',
      },
    });

    // add the cluster resource itself as a dependency of the barrier
    this._kubectlReadyBarrier.node.addDependency(this._clusterResource);

    this.clusterEndpoint = resource.attrEndpoint;
    this.clusterCertificateAuthorityData = resource.attrCertificateAuthorityData;
    this.clusterSecurityGroupId = resource.attrClusterSecurityGroupId;
    this.clusterEncryptionConfigKeyArn = resource.attrEncryptionConfigKeyArn;

    this.clusterSecurityGroup = ec2.SecurityGroup.fromSecurityGroupId(this, 'ClusterSecurityGroup', this.clusterSecurityGroupId);

    this.connections = new ec2.Connections({
      securityGroups: [this.clusterSecurityGroup, securityGroup],
      defaultPort: ec2.Port.tcp(443), // Control Plane has an HTTPS API
    });

    // we can use the cluster security group since its already attached to the cluster
    // and configured to allow connections from itself.
    this.kubectlSecurityGroup = this.clusterSecurityGroup;

    this.adminRole.assumeRolePolicy?.addStatements(new iam.PolicyStatement({
      actions: ['sts:AssumeRole'],
      principals: [this.kubectlLambdaRole],
    }));

    // use the cluster creation role to issue kubectl commands against the cluster because when the
    // cluster is first created, that's the only role that has "system:masters" permissions
    this.kubectlRole = this.adminRole;

    this._kubectlResourceProvider = this.defineKubectlProvider(props.removalPolicy);

    const updateConfigCommandPrefix = `aws eks update-kubeconfig --name ${this.clusterName}`;
    const getTokenCommandPrefix = `aws eks get-token --cluster-name ${this.clusterName}`;
    const commonCommandOptions = [`--region ${stack.region}`];

    if (props.outputClusterName) {
      new CfnOutput(this, 'ClusterName', { value: this.clusterName });
    }

    const supportAuthenticationApi = (this.authenticationMode === AuthenticationMode.API ||
      this.authenticationMode === AuthenticationMode.API_AND_CONFIG_MAP) ? true : false;

    // do not create a masters role if one is not provided. Trusting the accountRootPrincipal() is too permissive.
    if (props.mastersRole) {
      const mastersRole = props.mastersRole;

      // if we support authentication API we create an access entry for this mastersRole
      // with cluster scope.
      if (supportAuthenticationApi) {
        this.grantAccess('mastersRoleAccess', props.mastersRole.roleArn, [
          AccessPolicy.fromAccessPolicyName('AmazonEKSClusterAdminPolicy', {
            accessScopeType: AccessScopeType.CLUSTER,
          }),
        ]);
      } else {
        // if we don't support authentication API we should fallback to configmap
        // this would avoid breaking changes as well if authenticationMode is undefined
        this.awsAuth.addMastersRole(mastersRole);
      }

      if (props.outputMastersRoleArn) {
        new CfnOutput(this, 'MastersRoleArn', { value: mastersRole.roleArn });
      }

      commonCommandOptions.push(`--role-arn ${mastersRole.roleArn}`);
    }

    if (props.albController) {
      this.albController = AlbController.create(this, { ...props.albController, cluster: this });
    }

    // allocate default capacity if non-zero (or default).
    const minCapacity = props.defaultCapacity ?? DEFAULT_CAPACITY_COUNT;
    if (minCapacity > 0) {
      const instanceType = props.defaultCapacityInstance || DEFAULT_CAPACITY_TYPE;
      this.defaultCapacity = props.defaultCapacityType === DefaultCapacityType.EC2 ?
        this.addAutoScalingGroupCapacity('DefaultCapacity', { instanceType, minCapacity }) : undefined;

      this.defaultNodegroup = props.defaultCapacityType !== DefaultCapacityType.EC2 ?
        this.addNodegroupCapacity('DefaultCapacity', { instanceTypes: [instanceType], minSize: minCapacity }) : undefined;
    }

    if (props.outputConfigCommand && !props.mastersRole) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-eks:clusterMastersroleNotSpecified', '\'outputConfigCommand\' will be ignored as \'mastersRole\' has not been specified.');
    }

    const outputConfigCommand = (props.outputConfigCommand ?? true) && props.mastersRole;
    if (outputConfigCommand) {
      const postfix = commonCommandOptions.join(' ');
      new CfnOutput(this, 'ConfigCommand', { value: `${updateConfigCommandPrefix} ${postfix}` });
      new CfnOutput(this, 'GetTokenCommand', { value: `${getTokenCommandPrefix} ${postfix}` });
    }

    this.defineCoreDnsComputeType(props.coreDnsComputeType ?? CoreDnsComputeType.EC2);

    // Apply removal policy to all CFN resources created under this construct.
    if (props.removalPolicy) {
      RemovalPolicies.of(this).apply(props.removalPolicy);
    }
  }

  /**
   * Grants the specified IAM principal access to the EKS cluster based on the provided access policies.
   *
   * This method creates an `AccessEntry` construct that grants the specified IAM principal the access permissions
   * defined by the provided `IAccessPolicy` array. This allows the IAM principal to perform the actions permitted
   * by the access policies within the EKS cluster.
   *
   * @param id - The ID of the `AccessEntry` construct to be created.
   * @param principal - The IAM principal (role or user) to be granted access to the EKS cluster.
   * @param accessPolicies - An array of `IAccessPolicy` objects that define the access permissions to be granted to the IAM principal.
   * @param options - Additional options for granting access.
   */
  @MethodMetadata()
  public grantAccess(id: string, principal: string, accessPolicies: IAccessPolicy[], options?: GrantAccessOptions) {
    this.addToAccessEntry({
      id,
      principal,
      policies: accessPolicies,
      accessEntryType: options?.accessEntryType,
    });
  }

  /**
   * Fetch the load balancer address of a service of type 'LoadBalancer'.
   *
   * @param serviceName The name of the service.
   * @param options Additional operation options.
   */
  @MethodMetadata()
  public getServiceLoadBalancerAddress(serviceName: string, options: ServiceLoadBalancerAddressOptions = {}): string {
    const loadBalancerAddress = new KubernetesObjectValue(this, `${serviceName}LoadBalancerAddress`, {
      cluster: this,
      objectType: 'service',
      objectName: serviceName,
      objectNamespace: options.namespace,
      jsonPath: '.status.loadBalancer.ingress[0].hostname',
      timeout: options.timeout,
    });

    return loadBalancerAddress.value;
  }

  /**
   * Fetch the load balancer address of an ingress backed by a load balancer.
   *
   * @param ingressName The name of the ingress.
   * @param options Additional operation options.
   */
  @MethodMetadata()
  public getIngressLoadBalancerAddress(ingressName: string, options: IngressLoadBalancerAddressOptions = {}): string {
    const loadBalancerAddress = new KubernetesObjectValue(this, `${ingressName}LoadBalancerAddress`, {
      cluster: this,
      objectType: 'ingress',
      objectName: ingressName,
      objectNamespace: options.namespace,
      jsonPath: '.status.loadBalancer.ingress[0].hostname',
      timeout: options.timeout,
    });

    return loadBalancerAddress.value;
  }

  /**
   * Add nodes to this EKS cluster
   *
   * The nodes will automatically be configured with the right VPC and AMI
   * for the instance type and Kubernetes version.
   *
   * Note that if you specify `updateType: RollingUpdate` or `updateType: ReplacingUpdate`, your nodes might be replaced at deploy
   * time without notice in case the recommended AMI for your machine image type has been updated by AWS.
   * The default behavior for `updateType` is `None`, which means only new instances will be launched using the new AMI.
   *
   * Spot instances will be labeled `lifecycle=Ec2Spot` and tainted with `PreferNoSchedule`.
   * In addition, the [spot interrupt handler](https://github.com/awslabs/ec2-spot-labs/tree/master/ec2-spot-eks-solution/spot-termination-handler)
   * daemon will be installed on all spot instances to handle
   * [EC2 Spot Instance Termination Notices](https://aws.amazon.com/blogs/aws/new-ec2-spot-instance-termination-notices/).
   */
  @MethodMetadata()
  public addAutoScalingGroupCapacity(id: string, options: AutoScalingGroupCapacityOptions): autoscaling.AutoScalingGroup {
    if (options.machineImageType === MachineImageType.BOTTLEROCKET && options.bootstrapOptions !== undefined) {
      throw new ValidationError(lit`BootstrapOptionsSupportedBottlerocket`, 'bootstrapOptions is not supported for Bottlerocket', this);
    }
    const asg = new autoscaling.AutoScalingGroup(this, id, {
      ...options,
      vpc: this.vpc,
      machineImage: options.machineImageType === MachineImageType.BOTTLEROCKET ?
        new BottleRocketImage({
          kubernetesVersion: this.version.version,
        }) :
        new EksOptimizedImage({
          nodeType: nodeTypeForInstanceType(options.instanceType),
          cpuArch: cpuArchForInstanceType(options.instanceType),
          kubernetesVersion: this.version.version,
        }),
    });

    this.connectAutoScalingGroupCapacity(asg, {
      mapRole: options.mapRole,
      bootstrapOptions: options.bootstrapOptions,
      bootstrapEnabled: options.bootstrapEnabled,
      machineImageType: options.machineImageType,
      spotInterruptHandler: options.spotInterruptHandler,
    });

    if (nodeTypeForInstanceType(options.instanceType) === NodeType.INFERENTIA ||
      nodeTypeForInstanceType(options.instanceType) === NodeType.TRAINIUM) {
      this.addNeuronDevicePlugin();
    }

    return asg;
  }

  /**
   * Add managed nodegroup to this Amazon EKS cluster
   *
   * This method will create a new managed nodegroup and add into the capacity.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html
   * @param id The ID of the nodegroup
   * @param options options for creating a new nodegroup
   */
  @MethodMetadata()
  public addNodegroupCapacity(id: string, options?: NodegroupOptions): Nodegroup {
    const hasInferentiaOrTrainiumInstanceType = [
      options?.instanceType,
      ...options?.instanceTypes ?? [],
    ].some(i => i && (nodeTypeForInstanceType(i) === NodeType.INFERENTIA ||
      nodeTypeForInstanceType(i) === NodeType.TRAINIUM));

    if (hasInferentiaOrTrainiumInstanceType) {
      this.addNeuronDevicePlugin();
    }
    return new Nodegroup(this, `Nodegroup${id}`, {
      cluster: this,
      ...options,
    });
  }

  /**
   * Lazily creates the AwsAuth resource, which manages AWS authentication mapping.
   */
  public get awsAuth() {
    if (!this._awsAuth) {
      this._awsAuth = new AwsAuth(this, 'AwsAuth', { cluster: this });
    }

    return this._awsAuth;
  }

  /**
   * If this cluster is kubectl-enabled, returns the OpenID Connect issuer url.
   * This is because the values is only be retrieved by the API and not exposed
   * by CloudFormation. If this cluster is not kubectl-enabled (i.e. uses the
   * stock `CfnCluster`), this is `undefined`.
   * @attribute
   */
  public get clusterOpenIdConnectIssuerUrl(): string {
    return this._clusterResource.attrOpenIdConnectIssuerUrl;
  }

  /**
   * If this cluster is kubectl-enabled, returns the OpenID Connect issuer.
   * This is because the values is only be retrieved by the API and not exposed
   * by CloudFormation. If this cluster is not kubectl-enabled (i.e. uses the
   * stock `CfnCluster`), this is `undefined`.
   * @attribute
   */
  public get clusterOpenIdConnectIssuer(): string {
    return this._clusterResource.attrOpenIdConnectIssuer;
  }

  /**
   * An `OpenIdConnectProvider` resource associated with this cluster, and which can be used
   * to link this cluster to AWS IAM.
   *
   * A provider will only be defined if this property is accessed (lazy initialization).
   *
   */
  public get openIdConnectProvider(): iam.IOpenIdConnectProvider {
    if (!this._openIdConnectProvider) {
      if (FeatureFlags.of(this).isEnabled(EKS_USE_NATIVE_OIDC_PROVIDER)) {
        this._openIdConnectProvider = new OidcProviderNative(this, 'OidcProviderNative', {
          url: this.clusterOpenIdConnectIssuerUrl,
          removalPolicy: this._removalPolicy,
        });
      } else {
        this._openIdConnectProvider = new OpenIdConnectProvider(this, 'OpenIdConnectProvider', {
          url: this.clusterOpenIdConnectIssuerUrl,
          removalPolicy: this._removalPolicy,
        });
      }
    }

    return this._openIdConnectProvider;
  }

  /**
   * Retrieves the EKS Pod Identity Agent addon for the EKS cluster.
   *
   * The EKS Pod Identity Agent is responsible for managing the temporary credentials
   * used by pods in the cluster to access AWS resources. It runs as a DaemonSet on
   * each node and provides the necessary credentials to the pods based on their
   * associated service account.
   *
   */
  public get eksPodIdentityAgent(): IAddon | undefined {
    if (!this._eksPodIdentityAgent) {
      this._eksPodIdentityAgent = new Addon(this, 'EksPodIdentityAgentAddon', {
        cluster: this,
        addonName: 'eks-pod-identity-agent',
        removalPolicy: this._removalPolicy,
      });
    }

    return this._eksPodIdentityAgent;
  }

  /**
   * Adds a Fargate profile to this cluster.
   * @see https://docs.aws.amazon.com/eks/latest/userguide/fargate-profile.html
   *
   * @param id the id of this profile
   * @param options profile options
   */
  @MethodMetadata()
  public addFargateProfile(id: string, options: FargateProfileOptions) {
    return new FargateProfile(this, `fargate-profile-${id}`, {
      ...options,
      cluster: this,
    });
  }

  /**
   * Internal API used by `FargateProfile` to keep inventory of Fargate profiles associated with
   * this cluster, for the sake of ensuring the profiles are created sequentially.
   *
   * @returns the list of FargateProfiles attached to this cluster, including the one just attached.
   * @internal
   */
  public _attachFargateProfile(fargateProfile: FargateProfile): FargateProfile[] {
    this._fargateProfiles.push(fargateProfile);

    // add all profiles as a dependency of the "kubectl-ready" barrier because all kubectl-
    // resources can only be deployed after all fargate profiles are created.
    this._kubectlReadyBarrier.node.addDependency(fargateProfile);

    return this._fargateProfiles;
  }

  /**
   * Adds a resource scope that requires `kubectl` to this cluster and returns
   * the `KubectlProvider` which is the custom resource provider that should be
   * used as the resource provider.
   *
   * Called from `HelmResource` and `KubernetesResource`
   *
   * @param resourceScope the construct scope in which kubectl resources are defined.
   *
   * @internal
   */
  public _attachKubectlResourceScope(resourceScope: Construct): KubectlProvider {
    Node.of(resourceScope).addDependency(this._kubectlReadyBarrier);
    return this._kubectlResourceProvider;
  }

  /**
   * Adds an access entry to the cluster's access entries map.
   *
   * If an entry already exists for the given principal, it adds the provided access policies to the existing entry.
   * If no entry exists for the given principal, it creates a new access entry with the provided access policies.
   *
   * @param props - Options for adding the access entry.
   *
   * @throws {Error} If the uniqueName generated for the new access entry is not unique.
   *
   * @returns {void}
   */
  private addToAccessEntry(props: AddAccessEntryOptions) {
    const entry = this.accessEntries.get(props.principal);
    if (entry) {
      (entry as AccessEntry).addAccessPolicies(props.policies);
    } else {
      const newEntry = new AccessEntry(this, props.id, {
        principal: props.principal,
        cluster: this,
        accessPolicies: props.policies,
        accessEntryType: props.accessEntryType,
      });
      this.accessEntries.set(props.principal, newEntry);
    }
  }

  private defineKubectlProvider(removalPolicy?: RemovalPolicy) {
    const uid = '@aws-cdk/aws-eks.KubectlProvider';

    // since we can't have the provider connect to multiple networks, and we
    // wanted to avoid resource tear down, we decided for now that we will only
    // support a single EKS cluster per CFN stack.
    if (this.stack.node.tryFindChild(uid)) {
      throw new ValidationError(lit`SingleClusterDefinedWithinCloud`, 'Only a single EKS cluster can be defined within a CloudFormation stack', this);
    }

    return new KubectlProvider(this.stack, uid, { cluster: this, removalPolicy });
  }

  private selectPrivateSubnets(): ec2.ISubnet[] {
    const privateSubnets: ec2.ISubnet[] = [];
    const vpcPrivateSubnetIds = this.vpc.privateSubnets.map(s => s.subnetId);
    const vpcPublicSubnetIds = this.vpc.publicSubnets.map(s => s.subnetId);

    for (const placement of this.vpcSubnets) {
      for (const subnet of this.vpc.selectSubnets(placement).subnets) {
        if (vpcPrivateSubnetIds.includes(subnet.subnetId)) {
          // definitely private, take it.
          privateSubnets.push(subnet);
          continue;
        }

        if (vpcPublicSubnetIds.includes(subnet.subnetId)) {
          // definitely public, skip it.
          continue;
        }

        // neither public and nor private - what is it then? this means its a subnet instance that was explicitly passed
        // in the subnet selection. since ISubnet doesn't contain information on type, we have to assume its private and let it
        // fail at deploy time :\ (its better than filtering it out and preventing a possibly successful deployment)
        privateSubnets.push(subnet);
      }
    }

    return privateSubnets;
  }

  /**
   * Installs the Neuron device plugin on the cluster if it's not
   * already added.
   */
  private addNeuronDevicePlugin() {
    if (!this._neuronDevicePlugin) {
      const fileContents = fs.readFileSync(path.join(__dirname, 'addons', 'neuron-device-plugin.yaml'), 'utf8');
      const sanitized = YAML.parse(fileContents);
      this._neuronDevicePlugin = this.addManifest('NeuronDevicePlugin', sanitized);
    }

    return this._neuronDevicePlugin;
  }

  /**
   * Opportunistically tag subnets with the required tags.
   *
   * If no subnets could be found (because this is an imported VPC), add a warning.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/network_reqs.html
   */
  private tagSubnets() {
    const tagAllSubnets = (type: string, subnets: ec2.ISubnet[], tag: string) => {
      for (const subnet of subnets) {
        // if this is not a concrete subnet, attach a construct warning
        if (!ec2.Subnet.isVpcSubnet(subnet)) {
          // message (if token): "could not auto-tag public/private subnet with tag..."
          // message (if not token): "count not auto-tag public/private subnet xxxxx with tag..."
          const subnetID = Token.isUnresolved(subnet.subnetId) || Token.isUnresolved([subnet.subnetId]) ? '' : ` ${subnet.subnetId}`;
          Annotations.of(this).addWarningV2('@aws-cdk/aws-eks:clusterMustManuallyTagSubnet', `Could not auto-tag ${type} subnet${subnetID} with "${tag}=1", please remember to do this manually`);
          continue;
        }

        Tags.of(subnet).add(tag, '1');
      }
    };

    // https://docs.aws.amazon.com/eks/latest/userguide/network_reqs.html
    tagAllSubnets('private', this.vpc.privateSubnets, 'kubernetes.io/role/internal-elb');
    tagAllSubnets('public', this.vpc.publicSubnets, 'kubernetes.io/role/elb');
  }

  /**
   * Patches the CoreDNS deployment configuration and sets the "eks.amazonaws.com/compute-type"
   * annotation to either "ec2" or "fargate". Note that if "ec2" is selected, the resource is
   * omitted/removed, since the cluster is created with the "ec2" compute type by default.
   */
  private defineCoreDnsComputeType(type: CoreDnsComputeType) {
    // ec2 is the "built in" compute type of the cluster so if this is the
    // requested type we can simply omit the resource. since the resource's
    // `restorePatch` is configured to restore the value to "ec2" this means
    // that deletion of the resource will change to "ec2" as well.
    if (type === CoreDnsComputeType.EC2) {
      return;
    }

    // this is the json patch we merge into the resource based off of:
    // https://docs.aws.amazon.com/eks/latest/userguide/fargate-getting-started.html#fargate-gs-coredns
    const renderPatch = (computeType: CoreDnsComputeType) => ({
      spec: {
        template: {
          metadata: {
            annotations: {
              'eks.amazonaws.com/compute-type': computeType,
            },
          },
        },
      },
    });

    new KubernetesPatch(this, 'CoreDnsComputeTypePatch', {
      cluster: this,
      resourceName: 'deployment/coredns',
      resourceNamespace: 'kube-system',
      applyPatch: renderPatch(CoreDnsComputeType.FARGATE),
      restorePatch: renderPatch(CoreDnsComputeType.EC2),
    });
  }

  private validateRemoteNetworkConfig(props: ClusterProps) {
    if (!props.remoteNodeNetworks) {
      if (props.remotePodNetworks) {
        Annotations.of(this).addWarningV2(
          '@aws-cdk/aws-eks:clusterRemotePodNetworksWithoutNodeNetworks',
          'remotePodNetworks is specified without remoteNodeNetworks. remotePodNetworks will be ignored.',
        );
      }
      return;
    }

    this.validateNetworkCidrs(props.remoteNodeNetworks, 'node');

    if (!props.remotePodNetworks) return;

    this.validateNetworkCidrs(props.remotePodNetworks, 'pod');
    this.validateCrossNetworkOverlap(props.remoteNodeNetworks, props.remotePodNetworks);
  }

  /**
   * Validates all CIDR rules for a single network type (within same network + across networks).
   */
  private validateNetworkCidrs(networks: RemoteNodeNetwork[] | RemotePodNetwork[], networkType: 'node' | 'pod') {
    const resolvedCidrs = networks.map(n => n.cidrs.filter(c => !Token.isUnresolved(c)));

    // Within same network
    resolvedCidrs.forEach((cidrs, index) => {
      for (let i = 0; i < cidrs.length; i++) {
        for (let j = i + 1; j < cidrs.length; j++) {
          if (ec2.NetworkUtils.validateCidrPairOverlap(cidrs[i], cidrs[j])) {
            throw new ValidationError(
              lit`RemoteNetworkCidrOverlap`,
              `CIDR ${cidrs[i]} should not overlap with another CIDR in remote ${networkType} network #${index + 1}`,
              this,
            );
          }
        }
      }
    });

    // Across different networks
    for (let i = 0; i < resolvedCidrs.length; i++) {
      if (resolvedCidrs[i].length === 0) continue;
      for (let j = i + 1; j < resolvedCidrs.length; j++) {
        if (resolvedCidrs[j].length === 0) continue;
        const [overlap, cidr1, cidr2] = ec2.NetworkUtils.validateCidrBlocksOverlap(resolvedCidrs[i], resolvedCidrs[j]);
        if (overlap) {
          throw new ValidationError(
            lit`RemoteNetworkCidrBlockOverlap`,
            `CIDR block ${cidr1} in remote ${networkType} network #${i + 1} should not overlap with CIDR block ${cidr2} in remote ${networkType} network #${j + 1}`,
            this,
          );
        }
      }
    }
  }

  /**
   * Validates that node network CIDRs do not overlap with pod network CIDRs.
   */
  private validateCrossNetworkOverlap(nodeNetworks: RemoteNodeNetwork[], podNetworks: RemotePodNetwork[]) {
    for (const nodeNetwork of nodeNetworks) {
      const nodeCidrs = nodeNetwork.cidrs.filter(c => !Token.isUnresolved(c));
      if (nodeCidrs.length === 0) continue;

      for (const podNetwork of podNetworks) {
        const podCidrs = podNetwork.cidrs.filter(c => !Token.isUnresolved(c));
        if (podCidrs.length === 0) continue;

        const [overlap, nodeCidr, podCidr] = ec2.NetworkUtils.validateCidrBlocksOverlap(nodeCidrs, podCidrs);
        if (overlap) {
          throw new ValidationError(
            lit`RemoteNodePodNetworkOverlap`,
            `Remote node network CIDR block ${nodeCidr} should not overlap with remote pod network CIDR block ${podCidr}`,
            this,
          );
        }
      }
    }
  }
}

/**
 * Options for adding worker nodes
 */
export interface AutoScalingGroupCapacityOptions extends autoscaling.CommonAutoScalingGroupProps {
  /**
   * Instance type of the instances to start
   */
  readonly instanceType: ec2.InstanceType;

  /**
   * Will automatically update the aws-auth ConfigMap to map the IAM instance
   * role to RBAC.
   *
   * This cannot be explicitly set to `true` if the cluster has kubectl disabled.
   *
   * @default - true if the cluster has kubectl enabled (which is the default).
   */
  readonly mapRole?: boolean;

  /**
   * Configures the EC2 user-data script for instances in this autoscaling group
   * to bootstrap the node (invoke `/etc/eks/bootstrap.sh`) and associate it
   * with the EKS cluster.
   *
   * If you wish to provide a custom user data script, set this to `false` and
   * manually invoke `autoscalingGroup.addUserData()`.
   *
   * @default true
   */
  readonly bootstrapEnabled?: boolean;

  /**
   * EKS node bootstrapping options.
   *
   * @default - none
   */
  readonly bootstrapOptions?: BootstrapOptions;

  /**
   * Machine image type
   *
   * @default MachineImageType.AMAZON_LINUX_2
   */
  readonly machineImageType?: MachineImageType;

  /**
   * Installs the AWS spot instance interrupt handler on the cluster if it's not
   * already added. Only relevant if `spotPrice` is used.
   *
   * @default true
   */
  readonly spotInterruptHandler?: boolean;
}

/**
 * EKS node bootstrapping options.
 */
export interface BootstrapOptions {
  /**
   * Sets `--max-pods` for the kubelet based on the capacity of the EC2 instance.
   *
   * @default true
   */
  readonly useMaxPods?: boolean;

  /**
   * Restores the docker default bridge network.
   *
   * @default false
   */
  readonly enableDockerBridge?: boolean;

  /**
   * Number of retry attempts for AWS API call (DescribeCluster).
   *
   * @default 3
   */
  readonly awsApiRetryAttempts?: number;

  /**
   * The contents of the `/etc/docker/daemon.json` file. Useful if you want a
   * custom config differing from the default one in the EKS AMI.
   *
   * @default - none
   */
  readonly dockerConfigJson?: string;

  /**
   * Overrides the IP address to use for DNS queries within the
   * cluster.
   *
   * @default - 10.100.0.10 or 172.20.0.10 based on the IP
   * address of the primary interface.
   */
  readonly dnsClusterIp?: string;

  /**
   * Extra arguments to add to the kubelet. Useful for adding labels or taints.
   *
   * For example, `--node-labels foo=bar,goo=far`.
   *
   * @default - none
   */
  readonly kubeletExtraArgs?: string;

  /**
   * Additional command line arguments to pass to the `/etc/eks/bootstrap.sh`
   * command.
   *
   * @see https://github.com/awslabs/amazon-eks-ami/blob/master/files/bootstrap.sh
   * @default - none
   */
  readonly additionalArgs?: string;
}

/**
 * Options for adding an AutoScalingGroup as capacity
 */
export interface AutoScalingGroupOptions {
  /**
   * Will automatically update the aws-auth ConfigMap to map the IAM instance
   * role to RBAC.
   *
   * This cannot be explicitly set to `true` if the cluster has kubectl disabled.
   *
   * @default - true if the cluster has kubectl enabled (which is the default).
   */
  readonly mapRole?: boolean;

  /**
   * Configures the EC2 user-data script for instances in this autoscaling group
   * to bootstrap the node (invoke `/etc/eks/bootstrap.sh`) and associate it
   * with the EKS cluster.
   *
   * If you wish to provide a custom user data script, set this to `false` and
   * manually invoke `autoscalingGroup.addUserData()`.
   *
   * @default true
   */
  readonly bootstrapEnabled?: boolean;

  /**
   * Allows options for node bootstrapping through EC2 user data.
   * @default - default options
   */
  readonly bootstrapOptions?: BootstrapOptions;

  /**
   * Allow options to specify different machine image type
   *
   * @default MachineImageType.AMAZON_LINUX_2
   */
  readonly machineImageType?: MachineImageType;

  /**
   * Installs the AWS spot instance interrupt handler on the cluster if it's not
   * already added. Only relevant if `spotPrice` is configured on the auto-scaling group.
   *
   * @default true
   */
  readonly spotInterruptHandler?: boolean;
}

/**
 * Network configuration of nodes run on-premises with EKS Hybrid Nodes.
 */
export interface RemoteNodeNetwork {
  /**
   * Specifies the list of remote node CIDRs.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-eks-cluster-remotenodenetwork.html#cfn-eks-cluster-remotenodenetwork-cidrs
   */
  readonly cidrs: string[];
}

/**
 * Network configuration of pods run on-premises with EKS Hybrid Nodes.
 */
export interface RemotePodNetwork {
  /**
   * Specifies the list of remote pod CIDRs.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-eks-cluster-remotepodnetwork.html#cfn-eks-cluster-remotepodnetwork-cidrs
   */
  readonly cidrs: string[];
}

/**
 * Import a cluster to use in another stack
 */
@propertyInjectable
class ImportedCluster extends ClusterBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-eks.ImportedCluster';
  public readonly clusterName: string;
  public readonly clusterArn: string;
  public readonly connections = new ec2.Connections();
  public readonly kubectlRole?: iam.IRole;
  public readonly kubectlLambdaRole?: iam.IRole;
  public readonly kubectlEnvironment?: { [key: string]: string } | undefined;
  public readonly kubectlSecurityGroup?: ec2.ISecurityGroup | undefined;
  public readonly kubectlPrivateSubnets?: ec2.ISubnet[] | undefined;
  public readonly kubectlLayer?: lambda.ILayerVersion;
  public readonly ipFamily?: IpFamily;
  public readonly awscliLayer?: lambda.ILayerVersion;
  public readonly kubectlProvider?: IKubectlProvider;
  public readonly onEventLayer?: lambda.ILayerVersion;
  public readonly kubectlMemory?: Size;
  public readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup | undefined;
  public readonly prune: boolean;

  // so that `clusterSecurityGroup` on `ICluster` can be configured without optionality, avoiding users from having
  // to null check on an instance of `Cluster`, which will always have this configured.
  private readonly _clusterSecurityGroup?: ec2.ISecurityGroup;

  constructor(scope: Construct, id: string, private readonly props: ClusterAttributes) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.clusterName = props.clusterName;
    this.clusterArn = this.stack.formatArn(clusterArnComponents(props.clusterName));
    this.kubectlRole = props.kubectlRoleArn ? iam.Role.fromRoleArn(this, 'KubectlRole', props.kubectlRoleArn) : undefined;
    this.kubectlLambdaRole = props.kubectlLambdaRole;
    this.kubectlSecurityGroup = props.kubectlSecurityGroupId ? ec2.SecurityGroup.fromSecurityGroupId(this, 'KubectlSecurityGroup', props.kubectlSecurityGroupId) : undefined;
    this.kubectlEnvironment = props.kubectlEnvironment;
    this.kubectlPrivateSubnets = props.kubectlPrivateSubnetIds ? props.kubectlPrivateSubnetIds.map((subnetid, index) => ec2.Subnet.fromSubnetId(this, `KubectlSubnet${index}`, subnetid)) : undefined;
    this.kubectlLayer = props.kubectlLayer;
    this.ipFamily = props.ipFamily;
    this.awscliLayer = props.awscliLayer;
    this.kubectlMemory = props.kubectlMemory;
    this.clusterHandlerSecurityGroup = props.clusterHandlerSecurityGroupId ? ec2.SecurityGroup.fromSecurityGroupId(this, 'ClusterHandlerSecurityGroup', props.clusterHandlerSecurityGroupId) : undefined;
    this.kubectlProvider = props.kubectlProvider;
    this.onEventLayer = props.onEventLayer;
    this.prune = props.prune ?? true;

    let i = 1;
    for (const sgid of props.securityGroupIds ?? []) {
      this.connections.addSecurityGroup(ec2.SecurityGroup.fromSecurityGroupId(this, `SecurityGroup${i}`, sgid));
      i++;
    }

    if (props.clusterSecurityGroupId) {
      this._clusterSecurityGroup = ec2.SecurityGroup.fromSecurityGroupId(this, 'ClusterSecurityGroup', this.clusterSecurityGroupId);
      this.connections.addSecurityGroup(this._clusterSecurityGroup);
    }
  }

  public get vpc() {
    if (!this.props.vpc) {
      throw new ValidationError(lit`VpcDefinedImportedCluster`, '"vpc" is not defined for this imported cluster', this);
    }
    return this.props.vpc;
  }

  public get clusterSecurityGroup(): ec2.ISecurityGroup {
    if (!this._clusterSecurityGroup) {
      throw new ValidationError(lit`ClusterSecurityGroupDefinedImported`, '"clusterSecurityGroup" is not defined for this imported cluster', this);
    }
    return this._clusterSecurityGroup;
  }

  public get clusterSecurityGroupId(): string {
    if (!this.props.clusterSecurityGroupId) {
      throw new ValidationError(lit`ClusterSecurityGroupIdDefined`, '"clusterSecurityGroupId" is not defined for this imported cluster', this);
    }
    return this.props.clusterSecurityGroupId;
  }

  public get clusterEndpoint(): string {
    if (!this.props.clusterEndpoint) {
      throw new ValidationError(lit`ClusterendpointDefinedImportedCluster`, '"clusterEndpoint" is not defined for this imported cluster', this);
    }
    return this.props.clusterEndpoint;
  }

  public get clusterCertificateAuthorityData(): string {
    if (!this.props.clusterCertificateAuthorityData) {
      throw new ValidationError(lit`ClusterCertificateAuthorityDataDefined`, '"clusterCertificateAuthorityData" is not defined for this imported cluster', this);
    }
    return this.props.clusterCertificateAuthorityData;
  }

  public get clusterEncryptionConfigKeyArn(): string {
    if (!this.props.clusterEncryptionConfigKeyArn) {
      throw new ValidationError(lit`ClusterEncryptionConfigKeyArn`, '"clusterEncryptionConfigKeyArn" is not defined for this imported cluster', this);
    }
    return this.props.clusterEncryptionConfigKeyArn;
  }

  public get openIdConnectProvider(): iam.IOpenIdConnectProvider {
    if (!this.props.openIdConnectProvider) {
      throw new ValidationError(lit`OpenIdConnectProviderDefined`, '"openIdConnectProvider" is not defined for this imported cluster', this);
    }
    return this.props.openIdConnectProvider;
  }

  public get awsAuth(): AwsAuth {
    throw new ValidationError(lit`AwsauthSupportedImportedClusters`, '"awsAuth" is not supported on imported clusters', this);
  }
}

/**
 * Properties for EksOptimizedImage
 */
export interface EksOptimizedImageProps {
  /**
   * What instance type to retrieve the image for (standard or GPU-optimized)
   *
   * @default NodeType.STANDARD
   */
  readonly nodeType?: NodeType;

  /**
   * What cpu architecture to retrieve the image for (arm64 or x86_64)
   *
   * @default CpuArch.X86_64
   */
  readonly cpuArch?: CpuArch;

  /**
   * The Kubernetes version to use
   *
   * @default - The latest version
   */
  readonly kubernetesVersion?: string;
}

/**
 * Construct an Amazon Linux 2 image from the latest EKS Optimized AMI published in SSM
 */
export class EksOptimizedImage implements ec2.IMachineImage {
  private readonly nodeType?: NodeType;
  private readonly cpuArch?: CpuArch;
  private readonly kubernetesVersion?: string;
  private readonly amiParameterName: string;

  /**
   * Constructs a new instance of the EcsOptimizedAmi class.
   */
  public constructor(props: EksOptimizedImageProps = {}) {
    this.nodeType = props.nodeType ?? NodeType.STANDARD;
    this.cpuArch = props.cpuArch ?? CpuArch.X86_64;
    this.kubernetesVersion = props.kubernetesVersion ?? LATEST_KUBERNETES_VERSION;

    // set the SSM parameter name
    this.amiParameterName = `/aws/service/eks/optimized-ami/${this.kubernetesVersion}/`
      + (this.nodeType === NodeType.STANDARD ? this.cpuArch === CpuArch.X86_64 ?
        'amazon-linux-2/' : 'amazon-linux-2-arm64/' : '')
      + (this.nodeType === NodeType.GPU ? 'amazon-linux-2-gpu/' : '')
      + (this.nodeType === NodeType.INFERENTIA ? 'amazon-linux-2-gpu/' : '')
      + (this.nodeType === NodeType.TRAINIUM ? 'amazon-linux-2-gpu/' : '')
      + 'recommended/image_id';
  }

  /**
   * Return the correct image
   */
  public getImage(scope: Construct): ec2.MachineImageConfig {
    Validations.of(scope).acknowledge({
      id: 'CloudFormation-Validate::W2506',
      reason: 'SSM parameter is typed as String instead of AWS::SSM::Parameter::Value<AWS::EC2::Image::Id> for historical reasons.',
    });
    const ami = ssm.StringParameter.valueForStringParameter(scope, this.amiParameterName);
    return {
      imageId: ami,
      osType: ec2.OperatingSystemType.LINUX,
      userData: ec2.UserData.forLinux(),
    };
  }
}

// MAINTAINERS: use ./scripts/kube_bump.sh to update LATEST_KUBERNETES_VERSION
const LATEST_KUBERNETES_VERSION = '1.24';

/**
 * Whether the worker nodes should support GPU or just standard instances
 */
export enum NodeType {
  /**
   * Standard instances
   */
  STANDARD = 'Standard',

  /**
   * GPU instances
   */
  GPU = 'GPU',

  /**
   * Inferentia instances
   */
  INFERENTIA = 'INFERENTIA',

  /**
   * Trainium instances
   */
  TRAINIUM = 'TRAINIUM',
}

/**
 * CPU architecture
 */
export enum CpuArch {
  /**
   * arm64 CPU type
   */
  ARM_64 = 'arm64',

  /**
   * x86_64 CPU type
   */
  X86_64 = 'x86_64',
}

/**
 * The type of compute resources to use for CoreDNS.
 */
export enum CoreDnsComputeType {
  /**
   * Deploy CoreDNS on EC2 instances.
   */
  EC2 = 'ec2',

  /**
   * Deploy CoreDNS on Fargate-managed instances.
   */
  FARGATE = 'fargate',
}

/**
 * The default capacity type for the cluster
 */
export enum DefaultCapacityType {
  /**
   * managed node group
   */
  NODEGROUP,
  /**
   * EC2 autoscaling group
   */
  EC2,
}

/**
 * The machine image type
 */
export enum MachineImageType {
  /**
   * Amazon EKS-optimized Linux AMI
   */
  AMAZON_LINUX_2,
  /**
   * Bottlerocket AMI
   */
  BOTTLEROCKET,
}

function nodeTypeForInstanceType(instanceType: ec2.InstanceType) {
  if (INSTANCE_TYPES.gpu.includes(instanceType.toString().substring(0, 2))) {
    return NodeType.GPU;
  } else if (INSTANCE_TYPES.inferentia.includes(instanceType.toString().substring(0, 4))) {
    return NodeType.INFERENTIA;
  } else if (INSTANCE_TYPES.trainium.includes(instanceType.toString().substring(0, 4))) {
    return NodeType.TRAINIUM;
  }
  return NodeType.STANDARD;
}

function cpuArchForInstanceType(instanceType: ec2.InstanceType) {
  return INSTANCE_TYPES.graviton2.includes(instanceType.toString().substring(0, 3)) ? CpuArch.ARM_64 :
    INSTANCE_TYPES.graviton3.includes(instanceType.toString().substring(0, 3)) ? CpuArch.ARM_64 :
      INSTANCE_TYPES.graviton.includes(instanceType.toString().substring(0, 2)) ? CpuArch.ARM_64 :
        CpuArch.X86_64;
}

function flatten<A>(xss: A[][]): A[] {
  return Array.prototype.concat.call([], ...xss);
}
