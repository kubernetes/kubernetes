import { Construct } from 'constructs';
import type { ICluster } from './cluster';
import { CfnPodIdentityAssociation } from './eks.generated';
import { KubernetesManifest } from './k8s-manifest';
import type { AddToPrincipalPolicyResult, IPrincipal, IRole, PrincipalPolicyFragment } from '../../aws-iam';
import {
  OpenIdConnectPrincipal, PolicyStatement, Role,
  ServicePrincipal,
} from '../../aws-iam';
import type { RemovalPolicy } from '../../core';
import { CfnJson, Names, RemovalPolicies } from '../../core';
// import { FargateCluster } from './index';

/**
 * Enum representing the different identity types that can be used for a Kubernetes service account.
 */
export enum IdentityType {
  /**
   * Use the IAM Roles for Service Accounts (IRSA) identity type.
   * IRSA allows you to associate an IAM role with a Kubernetes service account.
   * This provides a way to grant permissions to Kubernetes pods by associating an IAM role with a Kubernetes service account.
   * The IAM role can then be used to provide AWS credentials to the pods, allowing them to access other AWS resources.
   *
   * When enabled, the openIdConnectProvider of the cluster would be created when you create the ServiceAccount.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html
   */
  IRSA = 'IRSA',

  /**
   * Use the EKS Pod Identities identity type.
   * EKS Pod Identities provide the ability to manage credentials for your applications, similar to the way that Amazon EC2 instance profiles
   * provide credentials to Amazon EC2 instances. Instead of creating and distributing your AWS credentials to the containers or using the
   * Amazon EC2 instance's role, you associate an IAM role with a Kubernetes service account and configure your Pods to use the service account.
   *
   * When enabled, the Pod Identity Agent AddOn of the cluster would be created when you create the ServiceAccount.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html
   */
  POD_IDENTITY = 'POD_IDENTITY',
}

/**
 * Options for `ServiceAccount`
 */
export interface ServiceAccountOptions {
  /**
   * The name of the service account.
   *
   * The name of a ServiceAccount object must be a valid DNS subdomain name.
   * https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/
   * @default - If no name is given, it will use the id of the resource.
   */
  readonly name?: string;

  /**
   * The namespace of the service account.
   *
   * All namespace names must be valid RFC 1123 DNS labels.
   * https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/#namespaces-and-dns
   * @default "default"
   */
  readonly namespace?: string;

  /**
   * Additional annotations of the service account.
   *
   * @default - no additional annotations
   */
  readonly annotations?: { [key: string]: string };

  /**
   * Additional labels of the service account.
   *
   * @default - no additional labels
   */
  readonly labels?: { [key: string]: string };

  /**
   * The identity type to use for the service account.
   * @default IdentityType.IRSA
   */
  readonly identityType?: IdentityType;

  /**
   * Overwrite existing service account.
   *
   * If this is set, we will use `kubectl apply` instead of `kubectl create`
   * when the service account is created. Otherwise, if there is already a service account
   * in the cluster with the same name, the operation will fail.
   *
   * @default false
   */
  readonly overwriteServiceAccount?: boolean;

  /**
   * The removal policy applied to the service account resources.
   *
   * The removal policy controls what happens to the resources if they stop being managed by CloudFormation.
   * This can happen in one of three situations:
   *
   * - The resource is removed from the template, so CloudFormation stops managing it
   * - A change to the resource is made that requires it to be replaced, so CloudFormation stops managing it
   * - The stack is deleted, so CloudFormation stops managing all resources in it
   *
   * @default RemovalPolicy.DESTROY
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * Properties for defining service accounts
 */
export interface ServiceAccountProps extends ServiceAccountOptions {
  /**
   * The cluster to apply the patch to.
   */
  readonly cluster: ICluster;
}

/**
 * Service Account
 */
export class ServiceAccount extends Construct implements IPrincipal {
  /**
   * The role which is linked to the service account.
   */
  public readonly role: IRole;

  public readonly assumeRoleAction: string;
  public readonly grantPrincipal: IPrincipal;
  public readonly policyFragment: PrincipalPolicyFragment;

  /**
   * The name of the service account.
   */
  public readonly serviceAccountName: string;

  /**
   * The namespace where the service account is located in.
   */
  public readonly serviceAccountNamespace: string;

  constructor(scope: Construct, id: string, props: ServiceAccountProps) {
    super(scope, id);

    const { cluster } = props;
    this.serviceAccountName = props.name ?? Names.uniqueId(this).toLowerCase();
    this.serviceAccountNamespace = props.namespace ?? 'default';

    // From K8s docs: https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/
    if (!this.isValidDnsSubdomainName(this.serviceAccountName)) {
      throw RangeError('The name of a ServiceAccount object must be a valid DNS subdomain name.');
    }

    // From K8s docs: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/#namespaces-and-dns
    if (!this.isValidDnsLabelName(this.serviceAccountNamespace)) {
      throw RangeError('All namespace names must be valid RFC 1123 DNS labels.');
    }

    let principal: IPrincipal;
    if (props.identityType !== IdentityType.POD_IDENTITY) {
      /* Add conditions to the role to improve security. This prevents other pods in the same namespace to assume the role.
      * See documentation: https://docs.aws.amazon.com/eks/latest/userguide/create-service-account-iam-policy-and-role.html
      */
      const conditions = new CfnJson(this, 'ConditionJson', {
        value: {
          [`${cluster.openIdConnectProvider.openIdConnectProviderIssuer}:aud`]: 'sts.amazonaws.com',
          [`${cluster.openIdConnectProvider.openIdConnectProviderIssuer}:sub`]: `system:serviceaccount:${this.serviceAccountNamespace}:${this.serviceAccountName}`,
        },
      });
      principal = new OpenIdConnectPrincipal(cluster.openIdConnectProvider).withConditions({
        StringEquals: conditions,
      });
    } else {
      /**
       * Identity type is POD_IDENTITY.
       * Create a service principal with "Service": "pods.eks.amazonaws.com"
       * See https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html
       */

      // EKS Pod Identity does not support Fargate
      // TODO: raise an error when using Fargate
      principal = new ServicePrincipal('pods.eks.amazonaws.com');
    }

    const role = new Role(this, 'Role', { assumedBy: principal });

    // pod identities requires 'sts:TagSession' in its principal actions
    if (props.identityType === IdentityType.POD_IDENTITY) {
      /**
       * EKS Pod Identities requires both assumed role actions otherwise it would fail.
       */
      role.assumeRolePolicy!.addStatements(new PolicyStatement({
        actions: ['sts:AssumeRole', 'sts:TagSession'],
        principals: [new ServicePrincipal('pods.eks.amazonaws.com')],
      }));

      // ensure the pod identity agent
      cluster.eksPodIdentityAgent;

      // associate this service account with the pod role we just created for the cluster
      new CfnPodIdentityAssociation(this, 'Association', {
        clusterName: cluster.clusterName,
        namespace: props.namespace ?? 'default',
        roleArn: role.roleArn,
        serviceAccount: this.serviceAccountName,
      });
    }

    this.role = role;

    this.assumeRoleAction = this.role.assumeRoleAction;
    this.grantPrincipal = this.role.grantPrincipal;
    this.policyFragment = this.role.policyFragment;

    // Note that we cannot use `cluster.addManifest` here because that would create the manifest
    // constrct in the scope of the cluster stack, which might be a different stack than this one.
    // This means that the cluster stack would depend on this stack because of the role,
    // and since this stack inherintely depends on the cluster stack, we will have a circular dependency.
    new KubernetesManifest(this, `manifest-${id}ServiceAccountResource`, {
      cluster,
      overwrite: props.overwriteServiceAccount,
      manifest: [{
        apiVersion: 'v1',
        kind: 'ServiceAccount',
        metadata: {
          name: this.serviceAccountName,
          namespace: this.serviceAccountNamespace,
          labels: {
            'app.kubernetes.io/name': this.serviceAccountName,
            ...props.labels,
          },
          annotations: {
            'eks.amazonaws.com/role-arn': this.role.roleArn,
            ...props.annotations,
          },
        },
      }],
    });

    if (props.removalPolicy) {
      RemovalPolicies.of(this).apply(props.removalPolicy);
    }
  }

  /**
   * @deprecated use `addToPrincipalPolicy()`
   */
  public addToPolicy(statement: PolicyStatement): boolean {
    return this.addToPrincipalPolicy(statement).statementAdded;
  }

  public addToPrincipalPolicy(statement: PolicyStatement): AddToPrincipalPolicyResult {
    return this.role.addToPrincipalPolicy(statement);
  }

  /**
   * If the value is a DNS subdomain name as defined in RFC 1123, from K8s docs.
   *
   * https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
   */
  private isValidDnsSubdomainName(value: string): boolean {
    return value.length <= 253 && /^[a-z0-9]+[a-z0-9-.]*[a-z0-9]+$/.test(value);
  }

  /**
   * If the value follows DNS label standard as defined in RFC 1123, from K8s docs.
   *
   * https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-label-names
   */
  private isValidDnsLabelName(value: string): boolean {
    return value.length <= 63 && /^[a-z0-9]+[a-z0-9-]*[a-z0-9]+$/.test(value);
  }
}
