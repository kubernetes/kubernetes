import { Construct } from 'constructs';
import { CLUSTER_RESOURCE_TYPE } from './cluster-resource-handler/consts';
import { ClusterResourceProvider } from './cluster-resource-provider';
import type { CfnCluster } from './eks.generated';
import type * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import type * as kms from '../../aws-kms';
import type * as lambda from '../../aws-lambda';
import type { ArnComponents } from '../../core';
import { CustomResource, Token, Stack, Lazy, ValidationError } from '../../core';
import { lit } from '../../core/lib/private/literal-string';

export interface ClusterResourceProps {
  readonly resourcesVpcConfig: CfnCluster.ResourcesVpcConfigProperty;
  readonly roleArn: string;
  readonly encryptionConfig?: Array<CfnCluster.EncryptionConfigProperty>;
  readonly kubernetesNetworkConfig?: CfnCluster.KubernetesNetworkConfigProperty;
  readonly name: string;
  readonly version?: string;
  readonly endpointPrivateAccess: boolean;
  readonly endpointPublicAccess: boolean;
  readonly publicAccessCidrs?: string[];
  readonly vpc: ec2.IVpc;
  readonly environment?: { [key: string]: string };
  readonly subnets?: ec2.ISubnet[];
  readonly secretsEncryptionKey?: kms.IKeyRef;
  readonly onEventLayer?: lambda.ILayerVersion;
  readonly clusterHandlerSecurityGroup?: ec2.ISecurityGroup;
  readonly tags?: { [key: string]: string };
  readonly logging?: { [key: string]: [ { [key: string]: any } ] };
  readonly accessconfig?: CfnCluster.AccessConfigProperty;
  readonly remoteNetworkConfig?: CfnCluster.RemoteNetworkConfigProperty;
  readonly bootstrapSelfManagedAddons?: boolean;
  readonly deletionProtection?: boolean;
  readonly controlPlaneScalingConfig?: CfnCluster.ControlPlaneScalingConfigProperty;
}

/**
 * A low-level CFN resource Amazon EKS cluster implemented through a custom
 * resource.
 *
 * Implements EKS create/update/delete through a CloudFormation custom resource
 * in order to allow us to control the IAM role which creates the cluster. This
 * is required in order to be able to allow CloudFormation to interact with the
 * cluster via `kubectl` to enable Kubernetes management capabilities like apply
 * manifest and IAM role/user RBAC mapping.
 */
export class ClusterResource extends Construct {
  public readonly ref: string;

  public readonly adminRole: iam.Role;

  private readonly resource: CustomResource;

  constructor(scope: Construct, id: string, props: ClusterResourceProps) {
    super(scope, id);

    if (!props.roleArn) {
      throw new ValidationError(lit`IsRequiredRolearnRequired`, '"roleArn" is required', this);
    }

    const provider = ClusterResourceProvider.getOrCreate(this, {
      subnets: props.subnets,
      vpc: props.vpc,
      environment: props.environment,
      onEventLayer: props.onEventLayer,
      securityGroup: props.clusterHandlerSecurityGroup,
    });

    this.adminRole = this.createAdminRole(provider, props);

    this.resource = new CustomResource(this, 'Resource', {
      resourceType: CLUSTER_RESOURCE_TYPE,
      serviceToken: provider.serviceToken,
      properties: {
        // the structure of config needs to be that of 'aws.EKS.CreateClusterRequest' since its passed as is
        // to the eks.createCluster sdk invocation.
        Config: {
          name: props.name,
          version: props.version,
          roleArn: props.roleArn,
          encryptionConfig: props.encryptionConfig,
          kubernetesNetworkConfig: props.kubernetesNetworkConfig,
          resourcesVpcConfig: {
            subnetIds: (props.resourcesVpcConfig as CfnCluster.ResourcesVpcConfigProperty).subnetIds,
            securityGroupIds: (props.resourcesVpcConfig as CfnCluster.ResourcesVpcConfigProperty).securityGroupIds,
            endpointPublicAccess: props.endpointPublicAccess,
            endpointPrivateAccess: props.endpointPrivateAccess,
            publicAccessCidrs: props.publicAccessCidrs,
          },
          tags: props.tags,
          logging: props.logging,
          accessConfig: props.accessconfig,
          remoteNetworkConfig: props.remoteNetworkConfig,
          bootstrapSelfManagedAddons: props.bootstrapSelfManagedAddons,
          deletionProtection: props.deletionProtection,
          controlPlaneScalingConfig: props.controlPlaneScalingConfig,
        },
        AssumeRoleArn: this.adminRole.roleArn,

        // IMPORTANT: increment this number when you add new attributes to the
        // resource. Otherwise, CloudFormation will error with "Vendor response
        // doesn't contain XXX key in object" (see #8276) by incrementing this
        // number, you will effectively cause a "no-op update" to the cluster
        // which will return the new set of attribute.
        AttributesRevision: 5,
      },
    });

    this.resource.node.addDependency(this.adminRole);

    this.ref = this.resource.ref;
  }

  public get attrEndpoint(): string {
    return Token.asString(this.resource.getAtt('Endpoint'));
  }

  public get attrArn(): string {
    return Token.asString(this.resource.getAtt('Arn'));
  }

  public get attrCertificateAuthorityData(): string {
    return Token.asString(this.resource.getAtt('CertificateAuthorityData'));
  }

  public get attrClusterSecurityGroupId(): string {
    return Token.asString(this.resource.getAtt('ClusterSecurityGroupId'));
  }

  public get attrEncryptionConfigKeyArn(): string {
    return Token.asString(this.resource.getAtt('EncryptionConfigKeyArn'));
  }

  public get attrOpenIdConnectIssuerUrl(): string {
    return Token.asString(this.resource.getAtt('OpenIdConnectIssuerUrl'));
  }

  public get attrOpenIdConnectIssuer(): string {
    return Token.asString(this.resource.getAtt('OpenIdConnectIssuer'));
  }

  private createAdminRole(provider: ClusterResourceProvider, props: ClusterResourceProps) {
    const stack = Stack.of(this);

    // the role used to create the cluster. this becomes the administrator role
    // of the cluster.
    const creationRole = new iam.Role(this, 'CreationRole', {
      // the role would be assumed by the provider handlers, as they are the ones making
      // the requests.
      assumedBy: new iam.CompositePrincipal(provider.provider.onEventHandler.role!, provider.provider.isCompleteHandler!.role!),
    });

    // the CreateCluster API will allow the cluster to assume this role, so we
    // need to allow the lambda execution role to pass it.
    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      resources: [props.roleArn],
    }));

    // if we know the cluster name, restrict the policy to only allow
    // interacting with this specific cluster otherwise, we will have to grant
    // this role to manage all clusters in the account. this must be lazy since
    // `props.name` may contain a lazy value that conditionally resolves to a
    // physical name.
    const resourceArns = Lazy.list({
      produce: () => {
        const arn = stack.formatArn(clusterArnComponents(stack.resolve(props.name)));
        return stack.resolve(props.name)
          ? [arn, `${arn}/*`] // see https://github.com/aws/aws-cdk/issues/6060
          : ['*'];
      },
    });

    const fargateProfileResourceArn = Lazy.string({
      produce: () => stack.resolve(props.name)
        ? stack.formatArn({ service: 'eks', resource: 'fargateprofile', resourceName: stack.resolve(props.name) + '/*' })
        : '*',
    });

    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'eks:CreateCluster',
        'eks:DescribeCluster',
        'eks:DescribeUpdate',
        'eks:DeleteCluster',
        'eks:UpdateClusterVersion',
        'eks:UpdateClusterConfig',
        'eks:CreateFargateProfile',
        'eks:TagResource',
        'eks:UntagResource',
      ],
      resources: resourceArns,
    }));

    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: ['eks:DescribeFargateProfile', 'eks:DeleteFargateProfile'],
      resources: [fargateProfileResourceArn],
    }));

    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: ['iam:GetRole', 'iam:listAttachedRolePolicies'],
      resources: ['*'],
    }));

    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: ['iam:CreateServiceLinkedRole'],
      resources: ['*'],
    }));

    // see https://github.com/aws/aws-cdk/issues/9027
    // these actions are the combined 'ec2:Describe*' actions taken from the EKS SLR policies.
    // (AWSServiceRoleForAmazonEKS, AWSServiceRoleForAmazonEKSForFargate, AWSServiceRoleForAmazonEKSNodegroup)
    creationRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ec2:DescribeInstances',
        'ec2:DescribeNetworkInterfaces',
        'ec2:DescribeSecurityGroups',
        'ec2:DescribeSubnets',
        'ec2:DescribeRouteTables',
        'ec2:DescribeDhcpOptions',
        'ec2:DescribeVpcs',
      ],
      resources: ['*'],
    }));

    // grant cluster creation role sufficient permission to access the specified key
    // see https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html
    if (props.secretsEncryptionKey) {
      creationRole.addToPolicy(new iam.PolicyStatement({
        actions: [
          'kms:Encrypt',
          'kms:Decrypt',
          'kms:DescribeKey',
          'kms:CreateGrant',
        ],
        resources: [props.secretsEncryptionKey.keyRef.keyArn],
      }));
    }

    return creationRole;
  }
}

export function clusterArnComponents(clusterName: string): ArnComponents {
  return {
    service: 'eks',
    resource: 'cluster',
    resourceName: clusterName,
  };
}
