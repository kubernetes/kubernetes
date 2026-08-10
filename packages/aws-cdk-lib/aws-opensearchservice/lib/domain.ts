import { URL } from 'url';

import type { Construct } from 'constructs';
import { LogGroupResourcePolicy } from './log-group-resource-policy';
import { OpenSearchAccessPolicy } from './opensearch-access-policy';
import { DomainGrants } from './opensearchservice-grants.generated';
import type { DomainReference, IDomainRef } from './opensearchservice.generated';
import { CfnDomain } from './opensearchservice.generated';
import * as perms from './perms';
import type { EngineVersion } from './version';
import * as acm from '../../aws-certificatemanager';
import type { MetricOptions } from '../../aws-cloudwatch';
import { Metric, Statistic } from '../../aws-cloudwatch';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import type * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import { toILogGroup } from '../../aws-logs/lib/private/ref-utils';
import * as route53 from '../../aws-route53';
import * as secretsmanager from '../../aws-secretsmanager';
import * as cdk from '../../core';
import { ValidationError } from '../../core';
import { memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import * as cxapi from '../../cx-api';
import type { ICertificateRef } from '../../interfaces/generated/aws-certificatemanager-interfaces.generated';

/**
 * Configures the capacity of the cluster such as the instance type and the
 * number of instances.
 */
export interface CapacityConfig {
  /**
   * The number of instances to use for the master node.
   *
   * @default - no dedicated master nodes
   */
  readonly masterNodes?: number;

  /**
   * The hardware configuration of the computer that hosts the dedicated master
   * node, such as `m3.medium.search`. For valid values, see [Supported
   * Instance Types](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/supported-instance-types.html)
   * in the Amazon OpenSearch Service Developer Guide.
   *
   * @default - r5.large.search
   */
  readonly masterNodeInstanceType?: string;

  /**
   * The number of data nodes (instances) to use in the Amazon OpenSearch Service domain.
   *
   * @default - 1
   */
  readonly dataNodes?: number;

  /**
   * The instance type for your data nodes, such as
   * `m3.medium.search`. For valid values, see [Supported Instance
   * Types](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/supported-instance-types.html)
   * in the Amazon OpenSearch Service Developer Guide.
   *
   * @default - r5.large.search
   */
  readonly dataNodeInstanceType?: string;

  /**
   * The number of UltraWarm nodes (instances) to use in the Amazon OpenSearch Service domain.
   *
   * @default - no UltraWarm nodes
   */
  readonly warmNodes?: number;

  /**
   * The instance type for your UltraWarm node, such as `ultrawarm1.medium.search`.
   * For valid values, see [UltraWarm Storage
   * Limits](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/limits.html#limits-ultrawarm)
   * in the Amazon OpenSearch Service Developer Guide.
   *
   * @default - ultrawarm1.medium.search
   */
  readonly warmInstanceType?: string;

  /**
   * Indicates whether Multi-AZ with Standby deployment option is enabled.
   * For more information, see [Multi-AZ with
   * Standby](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/managedomains-multiaz.html#managedomains-za-standby)
   *
   * @default - multi-az with standby if the feature flag `ENABLE_OPENSEARCH_MULTIAZ_WITH_STANDBY`
   * is true, no multi-az with standby otherwise
   */
  readonly multiAzWithStandbyEnabled?: boolean;

  /**
   * Additional node options for the domain
   *
   * @default - no additional node options
   */
  readonly nodeOptions?: NodeOptions[];
}

/**
 * Specifies zone awareness configuration options.
 */
export interface ZoneAwarenessConfig {
  /**
   * Indicates whether to enable zone awareness for the Amazon OpenSearch Service domain.
   * When you enable zone awareness, Amazon OpenSearch Service allocates the nodes and replica
   * index shards that belong to a cluster across two Availability Zones (AZs)
   * in the same region to prevent data loss and minimize downtime in the event
   * of node or data center failure. Don't enable zone awareness if your cluster
   * has no replica index shards or is a single-node cluster. For more information,
   * see [Configuring a Multi-AZ Domain](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/managedomains-multiaz.html)
   * in the Amazon OpenSearch Service Developer Guide.
   *
   * @default - false
   */
  readonly enabled?: boolean;

  /**
   * If you enabled multiple Availability Zones (AZs), the number of AZs that you
   * want the domain to use. Valid values are 2 and 3.
   *
   * @default - 2 if zone awareness is enabled.
   */
  readonly availabilityZoneCount?: number;
}

/**
 * The configurations of Amazon Elastic Block Store (Amazon EBS) volumes that
 * are attached to data nodes in the Amazon OpenSearch Service domain. For more information, see
 * [Amazon EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AmazonEBS.html)
 * in the Amazon Elastic Compute Cloud Developer Guide.
 */
export interface EbsOptions {
  /**
   * Specifies whether Amazon EBS volumes are attached to data nodes in the
   * Amazon OpenSearch Service domain.
   *
   * @default - true
   */
  readonly enabled?: boolean;

  /**
   * The number of I/O operations per second (IOPS) that the volume
   * supports. This property applies only to the gp3 and Provisioned IOPS (SSD) EBS
   * volume type.
   *
   * @default - iops are not set.
   */
  readonly iops?: number;

  /**
   * The throughput (in MiB/s) of the EBS volumes attached to data nodes.
   * This property applies only to the gp3 volume type.
   *
   * @default - throughput is not set.
   */
  readonly throughput?: number;

  /**
   * The size (in GiB) of the EBS volume for each data node. The minimum and
   * maximum size of an EBS volume depends on the EBS volume type and the
   * instance type to which it is attached.  For  valid values, see
   * [EBS volume size limits](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/limits.html#ebsresource)
   * in the Amazon OpenSearch Service Developer Guide.
   *
   * @default 10
   */
  readonly volumeSize?: number;

  /**
   * The EBS volume type to use with the Amazon OpenSearch Service domain, such as standard, gp2, io1.
   *
   * @default gp2
   */
  readonly volumeType?: ec2.EbsDeviceVolumeType;
}

/**
 * Configures log settings for the domain.
 */
export interface LoggingOptions {
  /**
   * Specify if slow search logging should be set up.
   * Requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.
   * An explicit `false` is required when disabling it from `true`.
   *
   * @default - false
   */
  readonly slowSearchLogEnabled?: boolean;

  /**
   * Log slow searches to this log group.
   *
   * @default - a new log group is created if slow search logging is enabled
   */
  readonly slowSearchLogGroup?: logs.ILogGroupRef;

  /**
   * Specify if slow index logging should be set up.
   * Requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.
   * An explicit `false` is required when disabling it from `true`.
   *
   * @default - false
   */
  readonly slowIndexLogEnabled?: boolean;

  /**
   * Log slow indices to this log group.
   *
   * @default - a new log group is created if slow index logging is enabled
   */
  readonly slowIndexLogGroup?: logs.ILogGroupRef;

  /**
   * Specify if Amazon OpenSearch Service application logging should be set up.
   * Requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.
   * An explicit `false` is required when disabling it from `true`.
   *
   * @default - false
   */
  readonly appLogEnabled?: boolean;

  /**
   * Log Amazon OpenSearch Service application logs to this log group.
   *
   * @default - a new log group is created if app logging is enabled
   */
  readonly appLogGroup?: logs.ILogGroupRef;

  /**
   * Specify if Amazon OpenSearch Service audit logging should be set up.
   * Requires Elasticsearch version 6.7 or later or OpenSearch version 1.0 or later and fine grained access control to be enabled.
   *
   * @default - false
   */
  readonly auditLogEnabled?: boolean;

  /**
   * Log Amazon OpenSearch Service audit logs to this log group.
   *
   * @default - a new log group is created if audit logging is enabled
   */
  readonly auditLogGroup?: logs.ILogGroupRef;
}

/**
 * Whether the domain should encrypt data at rest, and if so, the AWS Key
 * Management Service (KMS) key to use. Can only be used to create a new domain,
 * not update an existing one. Requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.
 */
export interface EncryptionAtRestOptions {
  /**
   * Specify true to enable encryption at rest.
   *
   * @default - encryption at rest is disabled.
   */
  readonly enabled?: boolean;

  /**
   * Supply if using KMS key for encryption at rest.
   *
   * @default - uses default aws/es KMS key.
   */
  readonly kmsKey?: kms.IKeyRef;
}

/**
 * Configures Amazon OpenSearch Service to use Amazon Cognito authentication for OpenSearch Dashboards.
 * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/cognito-auth.html
 */
export interface CognitoOptions {
  /**
   * The Amazon Cognito identity pool ID that you want Amazon OpenSearch Service to use for OpenSearch Dashboards authentication.
   */
  readonly identityPoolId: string;

  /**
   * A role that allows Amazon OpenSearch Service to configure your user pool and identity pool. It must have the `AmazonESCognitoAccess` policy attached to it.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/cognito-auth.html#cognito-auth-prereq
   */
  readonly role: iam.IRoleRef;

  /**
   * The Amazon Cognito user pool ID that you want Amazon OpenSearch Service to use for OpenSearch Dashboards authentication.
   */
  readonly userPoolId: string;
}

/**
 * The minimum TLS version required for traffic to the domain.
 */
export enum TLSSecurityPolicy {
  /** Cipher suite TLS 1.0 */
  TLS_1_0 = 'Policy-Min-TLS-1-0-2019-07',
  /** Cipher suite TLS 1.2 */
  TLS_1_2 = 'Policy-Min-TLS-1-2-2019-07',
  /** Cipher suite TLS 1.2 to 1.3 with perfect forward secrecy (PFS) */
  TLS_1_2_PFS = 'Policy-Min-TLS-1-2-PFS-2023-10',
}

/**
 * Container for information about the SAML configuration for OpenSearch Dashboards.
 */
export interface SAMLOptionsProperty {
  /**
   * The unique entity ID of the application in the SAML identity provider.
   */
  readonly idpEntityId: string;

  /**
   * The metadata of the SAML application, in XML format.
   */
  readonly idpMetadataContent: string;

  /**
   * The SAML master username, which is stored in the domain's internal user database.
   * This SAML user receives full permission in OpenSearch Dashboards/Kibana.
   * Creating a new master username does not delete any existing master usernames.
   *
   * @default - No master user name is configured
   */
  readonly masterUserName?: string;

  /**
   * The backend role that the SAML master user is mapped to.
   * Any users with this backend role receives full permission in OpenSearch Dashboards/Kibana.
   * To use a SAML master backend role, configure the `rolesKey` property.
   *
   * @default - The master user is not mapped to a backend role
   */
  readonly masterBackendRole?: string;

  /**
   * Element of the SAML assertion to use for backend roles.
   *
   * @default - roles
   */
  readonly rolesKey?: string;

  /**
   * Element of the SAML assertion to use for the user name.
   *
   * @default - NameID element of the SAML assertion fot the user name
   */
  readonly subjectKey?: string;

  /**
   * The duration, in minutes, after which a user session becomes inactive.
   *
   * @default - 60
   */
  readonly sessionTimeoutMinutes?: number;
}

/**
 * Specifies options for fine-grained access control.
 */
export interface AdvancedSecurityOptions {
  /**
   * ARN for the master user. Only specify this or masterUserName, but not both.
   *
   * @default - fine-grained access control is disabled
   */
  readonly masterUserArn?: string;

  /**
   * Username for the master user. Only specify this or masterUserArn, but not both.
   *
   * @default - fine-grained access control is disabled
   */
  readonly masterUserName?: string;

  /**
   * Password for the master user.
   *
   * You can use `SecretValue.unsafePlainText` to specify a password in plain text or
   * use `secretsmanager.Secret.fromSecretAttributes` to reference a secret in
   * Secrets Manager.
   *
   * @default - A Secrets Manager generated password
   */
  readonly masterUserPassword?: cdk.SecretValue;

  /**
   * True to enable SAML authentication for a domain.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/saml.html
   *
   * @default - SAML authentication is disabled. Enabled if `samlAuthenticationOptions` is set.
   */
  readonly samlAuthenticationEnabled?: boolean;

  /**
   * Container for information about the SAML configuration for OpenSearch Dashboards.
   * If set, `samlAuthenticationEnabled` will be enabled.
   *
   * @default - no SAML authentication options
   */
  readonly samlAuthenticationOptions?: SAMLOptionsProperty;
}

/**
 * Configures a custom domain endpoint for the Amazon OpenSearch Service domain
 */
export interface CustomEndpointOptions {
  /**
   * The custom domain name to assign
   */
  readonly domainName: string;

  /**
   * The certificate to use
   * @default - create a new one
   */
  readonly certificate?: ICertificateRef;

  /**
   * The hosted zone in Route53 to create the CNAME record in
   * @default - do not create a CNAME
   */
  readonly hostedZone?: route53.IHostedZone;
}

export interface WindowStartTime {
  /**
   * The start hour of the window in Coordinated Universal Time (UTC), using 24-hour time.
   * For example, 17 refers to 5:00 P.M. UTC.
   *
   * @default - 22
   */
  readonly hours: number;
  /**
   * The start minute of the window, in UTC.
   *
   * @default - 0
   */
  readonly minutes: number;
}

/**
 * The IP address type for the domain.
 */
export enum IpAddressType {
  /**
   * IPv4 addresses only
   */
  IPV4 = 'ipv4',
  /**
   * IPv4 and IPv6 addresses
   */
  DUAL_STACK = 'dualstack',
}

/**
 * Configuration for a specific node type in OpenSearch domain
 */
export interface NodeConfig {
  /**
   * Whether this node type is enabled
   *
   * @default - false
   */
  readonly enabled?: boolean;

  /**
   * The instance type for the nodes
   *
   * @default - m5.large.search
   */
  readonly type?: string;

  /**
   * The number of nodes of this type
   *
   * @default - 1
   */
  readonly count?: number;
}

/**
 * NodeType is a string enum of the node types in OpenSearch domain
 *
 */
export enum NodeType {
  /**
   * Coordinator node type
   */
  COORDINATOR = 'coordinator',
}

/**
 * Configuration for node options in OpenSearch domain
 */
export interface NodeOptions {
  /**
   * The type of node. Currently only 'coordinator' is supported.
   */
  readonly nodeType: NodeType;

  /**
   * Configuration for the node type
   */
  readonly nodeConfig: NodeConfig;
}

/**
 * Properties for an Amazon OpenSearch Service domain.
 */
export interface DomainProps {
  /**
   * Domain access policies.
   *
   * @default - No access policies.
   */
  readonly accessPolicies?: iam.PolicyStatement[];

  /**
   * Additional options to specify for the Amazon OpenSearch Service domain.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/createupdatedomains.html#createdomain-configure-advanced-options
   * @default - no advanced options are specified
   */
  readonly advancedOptions?: { [key: string]: (string) };

  /**
   * Configures Amazon OpenSearch Service to use Amazon Cognito authentication for OpenSearch Dashboards.
   *
   * @default - Cognito not used for authentication to OpenSearch Dashboards.
   */
  readonly cognitoDashboardsAuth?: CognitoOptions;

  /**
   * Enforces a particular physical domain name.
   *
   * @default - A name will be auto-generated.
   */
  readonly domainName?: string;

  /**
   * The configurations of Amazon Elastic Block Store (Amazon EBS) volumes that
   * are attached to data nodes in the Amazon OpenSearch Service domain.
   *
   * @default - 10 GiB General Purpose (SSD) volumes per node.
   */
  readonly ebs?: EbsOptions;

  /**
   * The cluster capacity configuration for the Amazon OpenSearch Service domain.
   *
   * @default - 1 r5.large.search data node; no dedicated master nodes.
   */
  readonly capacity?: CapacityConfig;

  /**
   * The cluster zone awareness configuration for the Amazon OpenSearch Service domain.
   *
   * @default - no zone awareness (1 AZ)
   */
  readonly zoneAwareness?: ZoneAwarenessConfig;

  /**
   * The Elasticsearch/OpenSearch version that your domain will leverage.
   */
  readonly version: EngineVersion;

  /**
   * Encryption at rest options for the cluster.
   *
   * @default - No encryption at rest
   */
  readonly encryptionAtRest?: EncryptionAtRestOptions;

  /**
   * Configuration log publishing configuration options.
   *
   * @default - No logs are published
   */
  readonly logging?: LoggingOptions;

  /**
   * Specify true to enable node to node encryption.
   * Requires Elasticsearch version 6.0 or later or OpenSearch version 1.0 or later.
   *
   * @default - Node to node encryption is not enabled.
   */
  readonly nodeToNodeEncryption?: boolean;

  /**
   * The hour in UTC during which the service takes an automated daily snapshot
   * of the indices in the Amazon OpenSearch Service domain. Only applies for Elasticsearch versions
   * below 5.3.
   *
   * @default - Hourly automated snapshots not used
   */
  readonly automatedSnapshotStartHour?: number;

  /**
   * Place the domain inside this VPC.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/vpc.html
   * @default - Domain is not placed in a VPC.
   */
  readonly vpc?: ec2.IVpc;

  /**
   * The list of security groups that are associated with the VPC endpoints
   * for the domain.
   *
   * Only used if `vpc` is specified.
   *
   * @see https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html
   * @default - One new security group is created.
   */
  readonly securityGroups?: ec2.ISecurityGroup[];

  /**
   * The specific vpc subnets the domain will be placed in. You must provide one subnet for each Availability Zone
   * that your domain uses. For example, you must specify three subnet IDs for a three Availability Zone
   * domain.
   *
   * Only used if `vpc` is specified.
   *
   * @see https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html
   * @default - All private subnets.
   */
  readonly vpcSubnets?: ec2.SubnetSelection[];

  /**
   * True to require that all traffic to the domain arrive over HTTPS.
   *
   * @default - false
   */
  readonly enforceHttps?: boolean;

  /**
   * The minimum TLS version required for traffic to the domain.
   *
   * @default - TLSSecurityPolicy.TLS_1_2
   */
  readonly tlsSecurityPolicy?: TLSSecurityPolicy;

  /**
   * Specifies options for fine-grained access control.
   * Requires Elasticsearch version 6.7 or later or OpenSearch version 1.0 or later. Enabling fine-grained access control
   * also requires encryption of data at rest and node-to-node encryption, along with
   * enforced HTTPS.
   *
   * @default - fine-grained access control is disabled
   */
  readonly fineGrainedAccessControl?: AdvancedSecurityOptions;

  /**
   * Configures the domain so that unsigned basic auth is enabled. If no master user is provided a default master user
   * with username `admin` and a dynamically generated password stored in KMS is created. The password can be retrieved
   * by getting `masterUserPassword` from the domain instance.
   *
   * Setting this to true will also add an access policy that allows unsigned
   * access, enable node to node encryption, encryption at rest. If conflicting
   * settings are encountered (like disabling encryption at rest) enabling this
   * setting will cause a failure.
   *
   * @default - false
   */
  readonly useUnsignedBasicAuth?: boolean;

  /**
   * To upgrade an Amazon OpenSearch Service domain to a new version, rather than replacing the entire
   * domain resource, use the EnableVersionUpgrade update policy.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-attribute-updatepolicy.html#cfn-attributes-updatepolicy-upgradeopensearchdomain
   *
   * @default - false
   */
  readonly enableVersionUpgrade?: boolean;

  /**
   * Policy to apply when the domain is removed from the stack
   *
   * @default RemovalPolicy.RETAIN
   */
  readonly removalPolicy?: cdk.RemovalPolicy;

  /**
   * To configure a custom domain configure these options
   *
   * If you specify a Route53 hosted zone it will create a CNAME record and use DNS validation for the certificate
   *
   * @default - no custom domain endpoint will be configured
   */
  readonly customEndpoint?: CustomEndpointOptions;

  /**
   * Options for enabling a domain's off-peak window, during which OpenSearch Service can perform mandatory
   * configuration changes on the domain.
   *
   * Off-peak windows were introduced on February 16, 2023.
   * All domains created before this date have the off-peak window disabled by default.
   * You must manually enable and configure the off-peak window for these domains.
   * All domains created after this date will have the off-peak window enabled by default.
   * You can't disable the off-peak window for a domain after it's enabled.
   *
   * @see https://docs.aws.amazon.com/it_it/AWSCloudFormation/latest/UserGuide/aws-properties-opensearchservice-domain-offpeakwindow.html
   *
   * @default - Disabled for domains created before February 16, 2023. Enabled for domains created after. Enabled if `offPeakWindowStart` is set.
   */
  readonly offPeakWindowEnabled?: boolean;

  /**
   * Start time for the off-peak window, in Coordinated Universal Time (UTC).
   * The window length will always be 10 hours, so you can't specify an end time.
   * For example, if you specify 11:00 P.M. UTC as a start time, the end time will automatically be set to 9:00 A.M.
   *
   * @default - 10:00 P.M. local time
   */
  readonly offPeakWindowStart?: WindowStartTime;

  /**
   * Specifies whether automatic service software updates are enabled for the domain.
   *
   * @see https://docs.aws.amazon.com/it_it/AWSCloudFormation/latest/UserGuide/aws-properties-opensearchservice-domain-softwareupdateoptions.html
   *
   * @default - false
   */
  readonly enableAutoSoftwareUpdate?: boolean;

  /**
   * Specify either dual stack or IPv4 as your IP address type.
   * Dual stack allows you to share domain resources across IPv4 and IPv6 address types, and is the recommended option.
   *
   * If you set your IP address type to dual stack, you can't change your address type later.
   *
   * @default - IpAddressType.IPV4
   */
  readonly ipAddressType?: IpAddressType;

  /**
   * Specify whether to create a CloudWatch Logs resource policy or not.
   *
   * When logging is enabled for the domain, a CloudWatch Logs resource policy is created by default.
   * However, CloudWatch Logs supports only 10 resource policies per region.
   * If you enable logging for several domains, it may hit the quota and cause an error.
   * By setting this property to true, creating a resource policy is suppressed, allowing you to avoid this problem.
   *
   * If you set this option to true, you must create a resource policy before deployment.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/createdomain-configure-slow-logs.html
   *
   * @default - false
   */
  readonly suppressLogsResourcePolicy?: boolean;

  /**
   * Whether to enable or disable cold storage on the domain. You must enable UltraWarm storage to enable cold storage.
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/cold-storage.html
   *
   * @default - undefined
   */
  readonly coldStorageEnabled?: boolean;

  /**
   * Whether to enable S3 vectors engine.
   * This feature allows you to offload vector data to Amazon S3 while maintaining sub-second vector search capabilities at low cost.
   *
   * Requirements:
   * - OpenSearch version 2.19 or later
   * - OpenSearch Optimized instance types (OR*, OM*, OI*) for data nodes
   * - Encryption at rest must be enabled
   *
   * @see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/s3-vector-opensearch-integration-engine.html
   *
   * @default undefined - AWS OpenSearch Service default is false
   */
  readonly s3VectorsEngineEnabled?: boolean;
}

/**
 * An interface that represents an Amazon OpenSearch Service domain - either created with the CDK, or an existing one.
 */
export interface IDomain extends cdk.IResource, IDomainRef {
  /**
   * Arn of the Amazon OpenSearch Service domain.
   *
   * @attribute
   */
  readonly domainArn: string;

  /**
   * Domain name of the Amazon OpenSearch Service domain.
   *
   * @attribute
   */
  readonly domainName: string;

  /**
   * Identifier of the Amazon OpenSearch Service domain.
   *
   * @attribute
   */
  readonly domainId: string;

  /**
   * Endpoint of the Amazon OpenSearch Service domain.
   *
   * @attribute
   */
  readonly domainEndpoint: string;

  /**
   * Grant read permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * @param identity The principal
   */
  grantRead(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant write permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * @param identity The principal
   */
  grantWrite(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read/write permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * @param identity The principal
   */
  grantReadWrite(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexRead(index: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Grant write permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexWrite(index: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read/write permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexReadWrite(index: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathRead(path: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Grant write permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathWrite(path: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Grant read/write permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathReadWrite(path: string, identity: iam.IGrantable): iam.Grant;

  /**
   * Return the given named metric for this domain.
   */
  metric(metricName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the time the cluster status is red.
   *
   * @default maximum over 5 minutes
   */
  metricClusterStatusRed(props?: MetricOptions): Metric;

  /**
   * Metric for the time the cluster status is yellow.
   *
   * @default maximum over 5 minutes
   */
  metricClusterStatusYellow(props?: MetricOptions): Metric;

  /**
   * Metric for the storage space of nodes in the cluster.
   *
   * @default minimum over 5 minutes
   */
  metricFreeStorageSpace(props?: MetricOptions): Metric;

  /**
   * Metric for the cluster blocking index writes.
   *
   * @default maximum over 1 minute
   */
  metricClusterIndexWritesBlocked(props?: MetricOptions): Metric;

  /**
   * Metric for the number of nodes.
   *
   * @default minimum over 1 hour
   */
  metricNodes(props?: MetricOptions): Metric;

  /**
   * Metric for automated snapshot failures.
   *
   * @default maximum over 5 minutes
   */
  metricAutomatedSnapshotFailure(props?: MetricOptions): Metric;

  /**
   * Metric for CPU utilization.
   *
   * @default maximum over 5 minutes
   */
  metricCPUUtilization(props?: MetricOptions): Metric;

  /**
   * Metric for JVM memory pressure.
   *
   * @default maximum over 5 minutes
   */
  metricJVMMemoryPressure(props?: MetricOptions): Metric;

  /**
   * Metric for master CPU utilization.
   *
   * @default maximum over 5 minutes
   */
  metricMasterCPUUtilization(props?: MetricOptions): Metric;

  /**
   * Metric for master JVM memory pressure.
   *
   * @default maximum over 5 minutes
   */
  metricMasterJVMMemoryPressure(props?: MetricOptions): Metric;

  /**
   * Metric for KMS key errors.
   *
   * @default maximum over 5 minutes
   */
  metricKMSKeyError(props?: MetricOptions): Metric;

  /**
   * Metric for KMS key being inaccessible.
   *
   * @default maximum over 5 minutes
   */
  metricKMSKeyInaccessible(props?: MetricOptions): Metric;

  /**
   * Metric for number of searchable documents.
   *
   * @default maximum over 5 minutes
   */
  metricSearchableDocuments(props?: MetricOptions): Metric;

  /**
   * Metric for search latency.
   *
   * @default p99 over 5 minutes
   */
  metricSearchLatency(props?: MetricOptions): Metric;

  /**
   * Metric for indexing latency.
   *
   * @default p99 over 5 minutes
   */
  metricIndexingLatency(props?: MetricOptions): Metric;
}

/**
 * A new or imported domain.
 */
abstract class DomainBase extends cdk.Resource implements IDomain {
  public abstract readonly domainArn: string;
  public abstract readonly domainName: string;
  public abstract readonly domainId: string;
  public abstract readonly domainEndpoint: string;
  /**
   * Collection of grant methods for a Domain
   */
  public readonly grants = DomainGrants.fromDomain(this);

  public get domainRef(): DomainReference {
    return {
      domainArn: this.domainArn,
      domainName: this.domainName,
    };
  }

  /**
   * Grant read permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   */
  grantRead(identity: iam.IGrantable): iam.Grant {
    return this.grants.read(identity);
  }

  /**
   * Grant write permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   */
  grantWrite(identity: iam.IGrantable): iam.Grant {
    return this.grants.write(identity);
  }

  /**
   * Grant read/write permissions for this domain and its contents to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   */
  grantReadWrite(identity: iam.IGrantable): iam.Grant {
    return this.grants.readWrite(identity);
  }

  /**
   * Grant read permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexRead(index: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_READ_ACTIONS,
      `${this.domainArn}/${index}`,
      `${this.domainArn}/${index}/*`,
    );
  }

  /**
   * Grant write permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexWrite(index: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_WRITE_ACTIONS,
      `${this.domainArn}/${index}`,
      `${this.domainArn}/${index}/*`,
    );
  }

  /**
   * Grant read/write permissions for an index in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param index The index to grant permissions for
   * @param identity The principal
   */
  grantIndexReadWrite(index: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_READ_WRITE_ACTIONS,
      `${this.domainArn}/${index}`,
      `${this.domainArn}/${index}/*`,
    );
  }

  /**
   * Grant read permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathRead(path: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_READ_ACTIONS,
      `${this.domainArn}/${path}`,
    );
  }

  /**
   * Grant write permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathWrite(path: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_WRITE_ACTIONS,
      `${this.domainArn}/${path}`,
    );
  }

  /**
   * Grant read/write permissions for a specific path in this domain to an IAM
   * principal (Role/Group/User).
   *
   * [disable-awslint:no-grants]
   *
   * @param path The path to grant permissions for
   * @param identity The principal
   */
  grantPathReadWrite(path: string, identity: iam.IGrantable): iam.Grant {
    return this.grant(
      identity,
      perms.ES_READ_WRITE_ACTIONS,
      `${this.domainArn}/${path}`,
    );
  }

  /**
   * Return the given named metric for this domain.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    return new Metric({
      namespace: 'AWS/ES',
      metricName,
      dimensionsMap: {
        DomainName: this.domainName,
        ClientId: this.env.account,
      },
      ...props,
    }).attachTo(this);
  }

  /**
   * Metric for the time the cluster status is red.
   *
   * @default maximum over 5 minutes
   */
  public metricClusterStatusRed(props?: MetricOptions): Metric {
    return this.metric('ClusterStatus.red', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for the time the cluster status is yellow.
   *
   * @default maximum over 5 minutes
   */
  public metricClusterStatusYellow(props?: MetricOptions): Metric {
    return this.metric('ClusterStatus.yellow', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for the storage space of nodes in the cluster.
   *
   * @default minimum over 5 minutes
   */
  public metricFreeStorageSpace(props?: MetricOptions): Metric {
    return this.metric('FreeStorageSpace', {
      statistic: Statistic.MINIMUM,
      ...props,
    });
  }

  /**
   * Metric for the cluster blocking index writes.
   *
   * @default maximum over 1 minute
   */
  public metricClusterIndexWritesBlocked(props?: MetricOptions): Metric {
    return this.metric('ClusterIndexWritesBlocked', {
      statistic: Statistic.MAXIMUM,
      period: cdk.Duration.minutes(1),
      ...props,
    });
  }

  /**
   * Metric for the number of nodes.
   *
   * @default minimum over 1 hour
   */
  public metricNodes(props?: MetricOptions): Metric {
    return this.metric('Nodes', {
      statistic: Statistic.MINIMUM,
      period: cdk.Duration.hours(1),
      ...props,
    });
  }

  /**
   * Metric for automated snapshot failures.
   *
   * @default maximum over 5 minutes
   */
  public metricAutomatedSnapshotFailure(props?: MetricOptions): Metric {
    return this.metric('AutomatedSnapshotFailure', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for CPU utilization.
   *
   * @default maximum over 5 minutes
   */
  public metricCPUUtilization(props?: MetricOptions): Metric {
    return this.metric('CPUUtilization', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for JVM memory pressure.
   *
   * @default maximum over 5 minutes
   */
  public metricJVMMemoryPressure(props?: MetricOptions): Metric {
    return this.metric('JVMMemoryPressure', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for master CPU utilization.
   *
   * @default maximum over 5 minutes
   */
  public metricMasterCPUUtilization(props?: MetricOptions): Metric {
    return this.metric('MasterCPUUtilization', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for master JVM memory pressure.
   *
   * @default maximum over 5 minutes
   */
  public metricMasterJVMMemoryPressure(props?: MetricOptions): Metric {
    return this.metric('MasterJVMMemoryPressure', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for KMS key errors.
   *
   * @default maximum over 5 minutes
   */
  public metricKMSKeyError(props?: MetricOptions): Metric {
    return this.metric('KMSKeyError', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for KMS key being inaccessible.
   *
   * @default maximum over 5 minutes
   */
  public metricKMSKeyInaccessible(props?: MetricOptions): Metric {
    return this.metric('KMSKeyInaccessible', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for number of searchable documents.
   *
   * @default maximum over 5 minutes
   */
  public metricSearchableDocuments(props?: MetricOptions): Metric {
    return this.metric('SearchableDocuments', {
      statistic: Statistic.MAXIMUM,
      ...props,
    });
  }

  /**
   * Metric for search latency.
   *
   * @default p99 over 5 minutes
   */
  public metricSearchLatency(props?: MetricOptions): Metric {
    return this.metric('SearchLatency', { statistic: 'p99', ...props });
  }

  /**
   * Metric for indexing latency.
   *
   * @default p99 over 5 minutes
   */
  public metricIndexingLatency(props?: MetricOptions): Metric {
    return this.metric('IndexingLatency', { statistic: 'p99', ...props });
  }

  private grant(
    grantee: iam.IGrantable,
    domainActions: string[],
    resourceArn: string,
    ...otherResourceArns: string[]
  ): iam.Grant {
    const resourceArns = [resourceArn, ...otherResourceArns];

    const grant = iam.Grant.addToPrincipal({
      grantee,
      actions: domainActions,
      resourceArns,
    });

    return grant;
  }
}

/**
 * Reference to an Amazon OpenSearch Service domain.
 */
export interface DomainAttributes {
  /**
   * The ARN of the Amazon OpenSearch Service domain.
   */
  readonly domainArn: string;

  /**
   * The domain endpoint of the Amazon OpenSearch Service domain.
   */
  readonly domainEndpoint: string;
}

/**
 * Provides an Amazon OpenSearch Service domain.
 */
@propertyInjectable
export class Domain extends DomainBase implements IDomain, ec2.IConnectable {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-opensearchservice.Domain';

  /**
   * Creates a domain construct that represents an external domain via domain endpoint.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param domainEndpoint The domain's endpoint.
   */
  public static fromDomainEndpoint(
    scope: Construct,
    id: string,
    domainEndpoint: string,
  ): IDomain {
    const stack = cdk.Stack.of(scope);
    const domainName = extractNameFromEndpoint(domainEndpoint);
    const domainArn = stack.formatArn({
      service: 'es',
      resource: 'domain',
      resourceName: domainName,
    });

    return Domain.fromDomainAttributes(scope, id, {
      domainArn,
      domainEndpoint,
    });
  }

  /**
   * Creates a domain construct that represents an external domain.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param attrs A `DomainAttributes` object.
   */
  public static fromDomainAttributes(scope: Construct, id: string, attrs: DomainAttributes): IDomain {
    const { domainArn, domainEndpoint } = attrs;
    const domainName = cdk.Stack.of(scope).splitArn(domainArn, cdk.ArnFormat.SLASH_RESOURCE_NAME).resourceName
      ?? extractNameFromEndpoint(domainEndpoint);

    return new class extends DomainBase {
      public readonly domainArn = domainArn;
      public readonly domainName = domainName;
      public readonly domainId = domainName;
      public readonly domainEndpoint = domainEndpoint.replace(/^https?:\/\//, '');

      constructor() { super(scope, id); }
    };
  }

  @memoizedGetter
  public get domainArn(): string {
    return this.getResourceArnAttribute(this.domain.attrArn, {
      service: 'es',
      resource: 'domain',
      resourceName: this.physicalName,
    });
  }

  @memoizedGetter
  public get domainName(): string {
    return this.getResourceNameAttribute(this.domain.ref);
  }

  public readonly domainId: string;
  public readonly domainEndpoint: string;

  /**
   * Log group that slow searches are logged to.
   *
   * @attribute
   */
  public get slowSearchLogGroup(): logs.ILogGroup | undefined {
    return this._slowSearchLogGroup ? toILogGroup(this._slowSearchLogGroup) : undefined;
  }

  /**
   * Log group that slow indices are logged to.
   *
   * @attribute
   */
  public get slowIndexLogGroup(): logs.ILogGroup | undefined {
    return this._slowIndexLogGroup ? toILogGroup(this._slowIndexLogGroup) : undefined;
  }

  /**
   * Log group that application logs are logged to.
   *
   * @attribute
   */
  public get appLogGroup(): logs.ILogGroup | undefined {
    return this._appLogGroup ? toILogGroup(this._appLogGroup) : undefined;
  }

  /**
   * Log group that audit logs are logged to.
   *
   * @attribute
   */
  public get auditLogGroup(): logs.ILogGroup | undefined {
    return this._auditLogGroup ? toILogGroup(this._auditLogGroup) : undefined;
  }

  /**
   * Master user password if fine grained access control is configured.
   */
  public readonly masterUserPassword?: cdk.SecretValue;

  private readonly _slowSearchLogGroup?: logs.ILogGroupRef;
  private readonly _slowIndexLogGroup?: logs.ILogGroupRef;
  private readonly _appLogGroup?: logs.ILogGroupRef;
  private readonly _auditLogGroup?: logs.ILogGroupRef;

  private readonly domain: CfnDomain;

  private accessPolicy?: OpenSearchAccessPolicy;

  private encryptionAtRestOptions?: EncryptionAtRestOptions;

  private readonly _connections: ec2.Connections | undefined;

  constructor(scope: Construct, id: string, props: DomainProps) {
    super(scope, id, {
      physicalName: props.domainName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const defaultInstanceType = 'r5.large.search';
    const warmDefaultInstanceType = 'ultrawarm1.medium.search';
    const defaultCoordinatorInstanceType = 'm5.large.search';

    const dedicatedMasterType = initializeInstanceType(defaultInstanceType, props.capacity?.masterNodeInstanceType);
    const dedicatedMasterCount = props.capacity?.masterNodes ?? 0;
    const dedicatedMasterEnabled = cdk.Token.isUnresolved(dedicatedMasterCount) ? true : dedicatedMasterCount > 0;

    const instanceType = initializeInstanceType(defaultInstanceType, props.capacity?.dataNodeInstanceType);
    const instanceCount = props.capacity?.dataNodes ?? 1;

    const warmType = initializeInstanceType(warmDefaultInstanceType, props.capacity?.warmInstanceType);
    const warmCount = props.capacity?.warmNodes ?? 0;
    const warmEnabled = cdk.Token.isUnresolved(warmCount) ? true : warmCount > 0;

    const availabilityZoneCount =
      props.zoneAwareness?.availabilityZoneCount ?? 2;

    if (![2, 3].includes(availabilityZoneCount)) {
      throw new ValidationError(lit`InvalidAvailabilityZoneCount`, 'Invalid zone awareness configuration; availabilityZoneCount must be 2 or 3', this);
    }

    const zoneAwarenessEnabled =
      props.zoneAwareness?.enabled ??
      props.zoneAwareness?.availabilityZoneCount != null;

    let securityGroups: ec2.ISecurityGroup[] | undefined;
    let subnets: ec2.ISubnet[] | undefined;

    let skipZoneAwarenessCheck: boolean = false;
    if (props.vpc) {
      const subnetSelections = props.vpcSubnets ?? [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }];
      subnets = selectSubnets(props.vpc, subnetSelections);
      skipZoneAwarenessCheck = zoneAwarenessCheckShouldBeSkipped(props.vpc, subnetSelections);
      securityGroups = props.securityGroups ?? [new ec2.SecurityGroup(this, 'SecurityGroup', {
        vpc: props.vpc,
        description: `Security group for domain ${this.node.id}`,
      })];
      if (props.enforceHttps) {
        this._connections = new ec2.Connections({ securityGroups, defaultPort: ec2.Port.tcp(443) });
      } else {
        this._connections = new ec2.Connections({ securityGroups });
      }
    }

    // If VPC options are supplied ensure that the number of subnets matches the number AZ (only if the vpc is not imported from another stack)
    if (subnets &&
      zoneAwarenessEnabled &&
      !skipZoneAwarenessCheck &&
      new Set(subnets.map((subnet) => subnet.availabilityZone)).size < availabilityZoneCount
    ) {
      throw new ValidationError(lit`InsufficientSubnetsForZoneAwareness`, 'When providing vpc options you need to provide a subnet for each AZ you are using', this);
    }

    if ([dedicatedMasterType, instanceType, warmType].some(t => (!cdk.Token.isUnresolved(t) && !t.endsWith('.search')))) {
      throw new ValidationError(lit`InvalidInstanceTypeSuffix`, 'Master, data and UltraWarm node instance types must end with ".search".', this);
    }

    if (!cdk.Token.isUnresolved(warmType) && !warmType.startsWith('ultrawarm')) {
      throw new ValidationError(lit`InvalidUltraWarmInstanceType`, 'UltraWarm node instance type must start with "ultrawarm".', this);
    }

    const unsignedBasicAuthEnabled = props.useUnsignedBasicAuth ?? false;

    if (unsignedBasicAuthEnabled) {
      if (props.enforceHttps == false) {
        throw new ValidationError(lit`UnsignedBasicAuthRequiresHttps`, 'You cannot disable HTTPS and use unsigned basic auth', this);
      }
      if (props.nodeToNodeEncryption == false) {
        throw new ValidationError(lit`UnsignedBasicAuthRequiresNodeToNodeEncryption`, 'You cannot disable node to node encryption and use unsigned basic auth', this);
      }
      if (props.encryptionAtRest?.enabled == false) {
        throw new ValidationError(lit`UnsignedBasicAuthRequiresEncryptionAtRest`, 'You cannot disable encryption at rest and use unsigned basic auth', this);
      }
    }

    const unsignedAccessPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['es:ESHttp*'],
      principals: [new iam.AnyPrincipal()],
      resources: [cdk.Lazy.string({ produce: () => `${this.domainArn}/*` })],
    });

    const masterUserArn = props.fineGrainedAccessControl?.masterUserArn;
    const masterUserNameProps = props.fineGrainedAccessControl?.masterUserName;
    // If basic auth is enabled set the user name to admin if no other user info is supplied.
    const masterUserName = unsignedBasicAuthEnabled
      ? (masterUserArn == null ? (masterUserNameProps ?? 'admin') : undefined)
      : masterUserNameProps;

    if (masterUserArn != null && masterUserName != null) {
      throw new ValidationError(lit`ConflictingMasterUserConfiguration`, 'Invalid fine grained access control settings. Only provide one of master user ARN or master user name. Not both.', this);
    }

    const advancedSecurityEnabled = (masterUserArn ?? masterUserName) != null;
    const internalUserDatabaseEnabled = masterUserName != null;
    const masterUserPasswordProp = props.fineGrainedAccessControl?.masterUserPassword;
    const createMasterUserPassword = (): cdk.SecretValue => {
      return new secretsmanager.Secret(this, 'MasterUser', {
        generateSecretString: {
          secretStringTemplate: JSON.stringify({
            username: masterUserName,
          }),
          generateStringKey: 'password',
          excludeCharacters: "{}'\\*[]()`",
        },
      })
        .secretValueFromJson('password');
    };
    this.masterUserPassword = internalUserDatabaseEnabled ?
      (masterUserPasswordProp ?? createMasterUserPassword())
      : undefined;

    const encryptionAtRestEnabled =
      props.encryptionAtRest?.enabled ?? (props.encryptionAtRest?.kmsKey != null || unsignedBasicAuthEnabled);
    const nodeToNodeEncryptionEnabled = props.nodeToNodeEncryption ?? unsignedBasicAuthEnabled;
    const volumeSize = props.ebs?.volumeSize ?? 10;
    const volumeType = props.ebs?.volumeType ?? ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD;
    const ebsEnabled = props.ebs?.enabled ?? true;
    const enforceHttps = props.enforceHttps ?? unsignedBasicAuthEnabled;

    function isInstanceType(t: string): Boolean {
      return dedicatedMasterType.startsWith(t) || instanceType.startsWith(t);
    }

    function isSomeInstanceType(...instanceTypes: string[]): Boolean {
      return instanceTypes.some(isInstanceType);
    }

    function isEveryDatanodeInstanceType(...instanceTypes: string[]): Boolean {
      return instanceTypes.some(t => instanceType.startsWith(t));
    }

    // Validate feature support for the given Elasticsearch/OpenSearch version, per
    // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/features-by-version.html
    const { versionNum: versionNum, isElasticsearchVersion } = parseVersion(this, props.version);
    if (isElasticsearchVersion) {
      if (
        versionNum <= 7.7 &&
        ![
          1.5, 2.3, 5.1, 5.3, 5.5, 5.6, 6.0,
          6.2, 6.3, 6.4, 6.5, 6.7, 6.8, 7.1, 7.4,
          7.7,
        ].includes(versionNum)
      ) {
        throw new ValidationError(lit`UnknownElasticsearchVersion`, `Unknown Elasticsearch version: ${versionNum}`, this);
      }

      if (versionNum < 5.1) {
        if (props.logging?.appLogEnabled) {
          throw new ValidationError(lit`AppLogsRequireMinimumVersion`, 'Error logs publishing requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.', this);
        }
        if (props.encryptionAtRest?.enabled) {
          throw new ValidationError(lit`EncryptionAtRestRequiresMinimumVersion`, 'Encryption of data at rest requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.', this);
        }
        if (props.cognitoDashboardsAuth != null) {
          throw new ValidationError(lit`CognitoAuthRequiresMinimumVersion`, 'Cognito authentication for OpenSearch Dashboards requires Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.', this);
        }
        if (isSomeInstanceType('c5', 'i3', 'm5', 'r5')) {
          throw new ValidationError(lit`InstanceTypesRequireMinimumVersion`, 'C5, I3, M5, and R5 instance types require Elasticsearch version 5.1 or later or OpenSearch version 1.0 or later.', this);
        }
      }

      if (versionNum < 6.0) {
        if (props.nodeToNodeEncryption) {
          throw new ValidationError(lit`NodeToNodeEncryptionRequiresMinimumVersion`, 'Node-to-node encryption requires Elasticsearch version 6.0 or later or OpenSearch version 1.0 or later.', this);
        }
      }

      if (versionNum < 6.7) {
        if (unsignedBasicAuthEnabled) {
          throw new ValidationError(lit`UnsignedBasicAuthRequiresMinimumVersion`, 'Using unsigned basic auth requires Elasticsearch version 6.7 or later or OpenSearch version 1.0 or later.', this);
        }
        if (advancedSecurityEnabled) {
          throw new ValidationError(lit`FineGrainedAccessControlRequiresMinimumVersion`, 'Fine-grained access control requires Elasticsearch version 6.7 or later or OpenSearch version 1.0 or later.', this);
        }
      }

      if (versionNum < 6.8 && warmEnabled) {
        throw new ValidationError(lit`UltraWarmRequiresMinimumVersion`, 'UltraWarm requires Elasticsearch version 6.8 or later or OpenSearch version 1.0 or later.', this);
      }
    }

    const unSupportEbsInstanceType = [
      ec2.InstanceClass.I3,
      ec2.InstanceClass.R6GD,
      ec2.InstanceClass.I4G,
      ec2.InstanceClass.I4I,
      ec2.InstanceClass.I8G,
      ec2.InstanceClass.IM4GN,
      ec2.InstanceClass.R7GD,
      ec2.InstanceClass.R8GD,
      'oi2', // OpenSearch-specific instance type with local NVMe storage
    ];

    const supportInstanceStorageInstanceType = [
      ec2.InstanceClass.R3,
      ...unSupportEbsInstanceType,
    ];

    const unSupportEncryptionAtRestInstanceType=[
      ec2.InstanceClass.M3,
      ec2.InstanceClass.R3,
      ec2.InstanceClass.T2,
    ];

    const unSupportUltraWarmInstanceType=[
      ec2.InstanceClass.T2,
      ec2.InstanceClass.T3,
    ];

    // Validate against instance type restrictions, per
    // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/supported-instance-types.html
    if (isSomeInstanceType(...unSupportEbsInstanceType) && ebsEnabled) {
      throw new ValidationError(lit`InstanceTypesDoNotSupportEbs`, `${formatInstanceTypesList(unSupportEbsInstanceType, 'and')} instance types do not support EBS storage volumes.`, this);
    }

    if (isSomeInstanceType('m3', 'r3', 't2') && encryptionAtRestEnabled) {
      throw new ValidationError(lit`InstanceTypesDoNotSupportEncryption`, `${formatInstanceTypesList(unSupportEncryptionAtRestInstanceType, 'and')} instance types do not support encryption of data at rest.`, this);
    }

    if (isInstanceType('t2.micro') && !(isElasticsearchVersion && versionNum <= 2.3)) {
      throw new ValidationError(lit`T2MicroVersionRestriction`, 'The t2.micro.search instance type supports only Elasticsearch versions 1.5 and 2.3.', this);
    }

    if (isSomeInstanceType('t2', 't3') && warmEnabled) {
      throw new ValidationError(lit`InstanceTypesDoNotSupportUltraWarm`, `${formatInstanceTypesList(unSupportUltraWarmInstanceType, 'and')} instance types do not support UltraWarm storage.`, this);
    }

    // Only R3, I3, R6GD, I4G, I4I, I8G, IM4GN, R7GD, R8GD and OI2 support instance storage, per
    // https://aws.amazon.com/opensearch-service/pricing/
    // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/supported-instance-types.html
    if (!ebsEnabled && !isEveryDatanodeInstanceType(...supportInstanceStorageInstanceType)) {
      throw new ValidationError(lit`EbsVolumesRequiredForInstanceType`, `EBS volumes are required when using instance types other than ${formatInstanceTypesList(supportInstanceStorageInstanceType, 'or')}.`, this);
    }

    // Only for a valid ebs volume configuration, per
    // https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-opensearchservice-domain-ebsoptions.html
    if (ebsEnabled) {
      // Check if iops or throughput if general purpose is configured
      if (volumeType == ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD || volumeType == ec2.EbsDeviceVolumeType.STANDARD) {
        if (props.ebs?.iops !== undefined || props.ebs?.throughput !== undefined) {
          throw new ValidationError(lit`GeneralPurposeVolumesNoIopsThroughput`, 'General Purpose EBS volumes can not be used with Iops or Throughput configuration', this);
        }
      }

      if (
        volumeType &&
        [
          ec2.EbsDeviceVolumeType.PROVISIONED_IOPS_SSD,
        ].includes(volumeType) &&
        !props.ebs?.iops
      ) {
        throw new ValidationError(lit`ProvisionedIopsSsdRequiresIops`, '`iops` must be specified if the `volumeType` is `PROVISIONED_IOPS_SSD`.', this);
      }
      if (props.ebs?.iops) {
        if (
          ![
            ec2.EbsDeviceVolumeType.PROVISIONED_IOPS_SSD,
            ec2.EbsDeviceVolumeType.PROVISIONED_IOPS_SSD_IO2,
            ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD_GP3,
          ].includes(volumeType)
        ) {
          throw new ValidationError(lit`IopsOnlyForSpecificVolumeTypes`, '`iops` may only be specified if the `volumeType` is `PROVISIONED_IOPS_SSD`, `PROVISIONED_IOPS_SSD_IO2` or `GENERAL_PURPOSE_SSD_GP3`.', this);
        }

        // Enforce maximum ratio of IOPS/GiB:
        // https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html
        const maximumRatios: { [key: string]: number } = {};
        maximumRatios[ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD_GP3] = 500;
        maximumRatios[ec2.EbsDeviceVolumeType.PROVISIONED_IOPS_SSD] = 50;
        maximumRatios[ec2.EbsDeviceVolumeType.PROVISIONED_IOPS_SSD_IO2] = 500;
        const maximumRatio = maximumRatios[volumeType];
        if (props.ebs?.volumeSize && (props.ebs?.iops > maximumRatio * props.ebs?.volumeSize)) {
          throw new ValidationError(lit`IopsToVolumeSizeRatioExceeded`, `\`${volumeType}\` volumes iops has a maximum ratio of ${maximumRatio} IOPS/GiB.`, this);
        }

        const maximumThroughputRatios: { [key: string]: number } = {};
        maximumThroughputRatios[ec2.EbsDeviceVolumeType.GP3] = 0.25;
        const maximumThroughputRatio = maximumThroughputRatios[volumeType];
        if (props.ebs?.throughput && props.ebs?.iops) {
          const iopsRatio = (props.ebs?.throughput / props.ebs?.iops);
          if (iopsRatio > maximumThroughputRatio) {
            throw new ValidationError(lit`ThroughputToIopsRatioExceeded`, `Throughput (MiBps) to iops ratio of ${iopsRatio} is too high; maximum is ${maximumThroughputRatio} MiBps per iops.`, this);
          }
        }
      }

      if (props.ebs?.throughput) {
        const throughputRange = { Min: 125, Max: 2000 };
        const { Min, Max } = throughputRange;
        if (volumeType != ec2.EbsDeviceVolumeType.GP3) {
          throw new ValidationError(lit`ThroughputRequiresGp3VolumeType`, '`throughput` property requires volumeType: `EbsDeviceVolumeType.GP3`', this);
        }
        if (props.ebs?.throughput < Min || props.ebs?.throughput > Max) {
          throw new ValidationError(lit`ThroughputOutOfRange`, `throughput property takes a minimum of ${Min} and a maximum of ${Max}.`, this);
        }
      }
    }

    // Fine-grained access control requires node-to-node encryption, encryption at rest,
    // and enforced HTTPS.
    if (advancedSecurityEnabled) {
      if (!nodeToNodeEncryptionEnabled) {
        throw new ValidationError(lit`FineGrainedAccessControlRequiresNodeToNodeEncryption`, 'Node-to-node encryption is required when fine-grained access control is enabled.', this);
      }
      if (!encryptionAtRestEnabled) {
        throw new ValidationError(lit`FineGrainedAccessControlRequiresEncryptionAtRest`, 'Encryption-at-rest is required when fine-grained access control is enabled.', this);
      }
      if (!enforceHttps) {
        throw new ValidationError(lit`FineGrainedAccessControlRequiresHttps`, 'Enforce HTTPS is required when fine-grained access control is enabled.', this);
      }
    }

    // Validate fine grained access control enabled for audit logs, per
    // https://aws.amazon.com/about-aws/whats-new/2020/09/elasticsearch-audit-logs-now-available-on-amazon-elasticsearch-service/
    if (props.logging?.auditLogEnabled && !advancedSecurityEnabled) {
      throw new ValidationError(lit`AuditLogsRequireFineGrainedAccessControl`, 'Fine-grained access control is required when audit logs publishing is enabled.', this);
    }

    // Validate UltraWarm requirement for dedicated master nodes, per
    // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/ultrawarm.html
    if (warmEnabled && !dedicatedMasterEnabled) {
      throw new ValidationError(lit`UltraWarmRequiresDedicatedMasterNodes`, 'Dedicated master node is required when UltraWarm storage is enabled.', this);
    }

    if (props.coldStorageEnabled && !warmEnabled) {
      throw new ValidationError(lit`ColdStorageRequiresUltraWarm`, 'You must enable UltraWarm storage to enable cold storage.', this);
    }

    // Validate S3 Vectors Engine requirements
    // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/s3-vector-opensearch-integration-engine.html
    if (props.s3VectorsEngineEnabled) {
      // S3 Vectors Engine requires OpenSearch version 2.19 or later
      if (isElasticsearchVersion) {
        throw new ValidationError(lit`S3VectorsEngineElasticsearchNotSupported`, 'S3 Vectors Engine requires OpenSearch version 2.19 or later. Elasticsearch versions are not supported.', this);
      }
      if (versionNum < 2.19) {
        throw new ValidationError(lit`S3VectorsEngineVersionTooLow`, `S3 Vectors Engine requires OpenSearch version 2.19 or later. Got version ${versionNum}.`, this);
      }

      // S3 Vectors Engine requires OpenSearch Optimized instance types (OR*, OM*, OI*)
      const isOpenSearchOptimizedInstance = instanceType.startsWith('or') || instanceType.startsWith('om') || instanceType.startsWith('oi');
      if (!cdk.Token.isUnresolved(instanceType) && !isOpenSearchOptimizedInstance) {
        throw new ValidationError(lit`S3VectorsEngineInvalidInstanceType`, `S3 Vectors Engine requires OpenSearch Optimized instance types (OR*, OM*, OI*). Got ${instanceType}.`, this);
      }

      // S3 Vectors Engine requires encryption at rest
      if (!encryptionAtRestEnabled) {
        throw new ValidationError(lit`S3VectorsEngineEncryptionRequired`, 'S3 Vectors Engine requires encryption at rest to be enabled.', this);
      }
    }

    let cfnVpcOptions: CfnDomain.VPCOptionsProperty | undefined;

    if (securityGroups && subnets) {
      cfnVpcOptions = {
        securityGroupIds: securityGroups.map((sg) => sg.securityGroupId),
        subnetIds: subnets.map((subnet) => subnet.subnetId),
      };
    }

    // Setup logging
    const logGroups: logs.ILogGroupRef[] = [];
    const logPublishing: Record<string, any> = {};

    if (props.logging?.slowSearchLogEnabled) {
      this._slowSearchLogGroup = props.logging.slowSearchLogGroup ??
        new logs.LogGroup(this, 'SlowSearchLogs', {
          retention: logs.RetentionDays.ONE_MONTH,
        });

      logGroups.push(this._slowSearchLogGroup);
      logPublishing.SEARCH_SLOW_LOGS = {
        enabled: true,
        cloudWatchLogsLogGroupArn: this._slowSearchLogGroup.logGroupRef.logGroupArn,
      };
    } else if (props.logging?.slowSearchLogEnabled === false) {
      logPublishing.SEARCH_SLOW_LOGS = {
        enabled: false,
      };
    }

    if (props.logging?.slowIndexLogEnabled) {
      this._slowIndexLogGroup = props.logging.slowIndexLogGroup ??
        new logs.LogGroup(this, 'SlowIndexLogs', {
          retention: logs.RetentionDays.ONE_MONTH,
        });

      logGroups.push(this._slowIndexLogGroup);
      logPublishing.INDEX_SLOW_LOGS = {
        enabled: true,
        cloudWatchLogsLogGroupArn: this._slowIndexLogGroup.logGroupRef.logGroupArn,
      };
    } else if (props.logging?.slowIndexLogEnabled === false) {
      logPublishing.INDEX_SLOW_LOGS = {
        enabled: false,
      };
    }

    if (props.logging?.appLogEnabled) {
      this._appLogGroup = props.logging.appLogGroup ??
        new logs.LogGroup(this, 'AppLogs', {
          retention: logs.RetentionDays.ONE_MONTH,
        });

      logGroups.push(this._appLogGroup);
      logPublishing.ES_APPLICATION_LOGS = {
        enabled: true,
        cloudWatchLogsLogGroupArn: this._appLogGroup.logGroupRef.logGroupArn,
      };
    } else if (props.logging?.appLogEnabled === false) {
      logPublishing.ES_APPLICATION_LOGS = {
        enabled: false,
      };
    }

    if (props.logging?.auditLogEnabled) {
      this._auditLogGroup = props.logging.auditLogGroup ??
        new logs.LogGroup(this, 'AuditLogs', {
          retention: logs.RetentionDays.ONE_MONTH,
        });

      logGroups.push(this._auditLogGroup);
      logPublishing.AUDIT_LOGS = {
        enabled: true,
        cloudWatchLogsLogGroupArn: this._auditLogGroup?.logGroupRef.logGroupArn,
      };
    } else if (props.logging?.auditLogEnabled === false) {
      logPublishing.AUDIT_LOGS = {
        enabled: false,
      };
    }

    let logGroupResourcePolicy: LogGroupResourcePolicy | null = null;
    if (logGroups.length > 0 && !props.suppressLogsResourcePolicy) {
      const logPolicyStatement = new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['logs:PutLogEvents', 'logs:CreateLogStream'],
        resources: logGroups.map((lg) => lg.logGroupRef.logGroupArn),
        principals: [new iam.ServicePrincipal('es.amazonaws.com')],
      });

      // Use a custom resource to set the log group resource policy since it is not supported by CDK and cfn.
      // https://github.com/aws/aws-cdk/issues/5343
      logGroupResourcePolicy = new LogGroupResourcePolicy(this, `ESLogGroupPolicy${this.node.addr}`, {
        // create a cloudwatch logs resource policy name that is unique to this domain instance
        policyName: `ESLogPolicy${this.node.addr}`,
        policyStatements: [logPolicyStatement],
      });
    }

    let customEndpointCertificate: ICertificateRef | undefined;
    if (props.customEndpoint) {
      if (props.customEndpoint.certificate) {
        customEndpointCertificate = props.customEndpoint.certificate;
      } else {
        customEndpointCertificate = new acm.Certificate(this, 'CustomEndpointCertificate', {
          domainName: props.customEndpoint.domainName,
          validation: props.customEndpoint.hostedZone ? acm.CertificateValidation.fromDns(props.customEndpoint.hostedZone) : undefined,
        });
      }
    }

    let multiAzWithStandbyEnabled = props.capacity?.multiAzWithStandbyEnabled;
    if (multiAzWithStandbyEnabled === undefined) {
      if (cdk.FeatureFlags.of(this).isEnabled(cxapi.ENABLE_OPENSEARCH_MULTIAZ_WITH_STANDBY)) {
        multiAzWithStandbyEnabled = true;
      }
    }

    if (isSomeInstanceType('t3') && multiAzWithStandbyEnabled) {
      throw new ValidationError(lit`T3InstanceTypeDoesNotSupportMultiAzWithStandby`, 'T3 instance type does not support Multi-AZ with standby feature.', this);
    }

    const offPeakWindowEnabled = props.offPeakWindowEnabled ?? props.offPeakWindowStart !== undefined;
    if (offPeakWindowEnabled) {
      this.validateWindowStartTime(props.offPeakWindowStart);
    }

    const samlAuthenticationEnabled = props.fineGrainedAccessControl?.samlAuthenticationEnabled ??
      props.fineGrainedAccessControl?.samlAuthenticationOptions !== undefined;
    if (samlAuthenticationEnabled) {
      if (!advancedSecurityEnabled) {
        throw new ValidationError(lit`SamlAuthenticationRequiresFineGrainedAccessControl`, 'SAML authentication requires fine-grained access control to be enabled.', this);
      }
      this.validateSamlAuthenticationOptions(props.fineGrainedAccessControl?.samlAuthenticationOptions);
    }

    if (props.capacity?.nodeOptions) {
      // Validate coordinator node configuration
      const coordinatorConfig = props.capacity.nodeOptions.find(opt => opt.nodeType === NodeType.COORDINATOR)?.nodeConfig;
      if (coordinatorConfig?.enabled) {
        const coordinatorType = initializeInstanceType(defaultCoordinatorInstanceType, coordinatorConfig.type);
        if (!cdk.Token.isUnresolved(coordinatorType) && !coordinatorType.endsWith('.search')) {
          throw new ValidationError(lit`CoordinatorNodeInstanceType`, 'Coordinator node instance type must end with ".search".', this);
        }
        if (coordinatorConfig.count !== undefined && coordinatorConfig.count < 1) {
          throw new ValidationError(lit`CoordinatorNodeCountMustBeAtLeastOne`, 'Coordinator node count must be at least 1.', this);
        }
      }
    }

    // Create the domain
    this.domain = new CfnDomain(this, 'Resource', {
      domainName: this.physicalName,
      engineVersion: props.version.version,
      clusterConfig: {
        coldStorageOptions: props.coldStorageEnabled !== undefined ? {
          enabled: props.coldStorageEnabled,
        } : undefined,
        dedicatedMasterEnabled,
        dedicatedMasterCount: dedicatedMasterEnabled
          ? dedicatedMasterCount
          : undefined,
        dedicatedMasterType: dedicatedMasterEnabled
          ? dedicatedMasterType
          : undefined,
        instanceCount,
        instanceType,
        multiAzWithStandbyEnabled,
        warmEnabled: warmEnabled
          ? warmEnabled
          : undefined,
        warmCount: warmEnabled
          ? warmCount
          : undefined,
        warmType: warmEnabled
          ? warmType
          : undefined,
        zoneAwarenessEnabled,
        zoneAwarenessConfig: zoneAwarenessEnabled
          ? { availabilityZoneCount }
          : undefined,
        nodeOptions: props.capacity?.nodeOptions,
      },
      ebsOptions: {
        ebsEnabled,
        volumeSize: ebsEnabled ? volumeSize : undefined,
        volumeType: ebsEnabled ? volumeType : undefined,
        iops: ebsEnabled ? props.ebs?.iops : undefined,
        throughput: ebsEnabled ? props.ebs?.throughput : undefined,
      },
      encryptionAtRestOptions: {
        enabled: encryptionAtRestEnabled,
        kmsKeyId: encryptionAtRestEnabled && props.encryptionAtRest?.kmsKey
          ? this.selectKmsKeyIdentifier(props.encryptionAtRest.kmsKey)
          : undefined,
      },
      nodeToNodeEncryptionOptions: { enabled: nodeToNodeEncryptionEnabled },
      logPublishingOptions: logPublishing,
      cognitoOptions: props.cognitoDashboardsAuth ? {
        enabled: true,
        identityPoolId: props.cognitoDashboardsAuth?.identityPoolId,
        roleArn: props.cognitoDashboardsAuth?.role.roleRef.roleArn,
        userPoolId: props.cognitoDashboardsAuth?.userPoolId,
      } : undefined,
      vpcOptions: cfnVpcOptions,
      snapshotOptions: props.automatedSnapshotStartHour
        ? { automatedSnapshotStartHour: props.automatedSnapshotStartHour }
        : undefined,
      domainEndpointOptions: {
        enforceHttps,
        tlsSecurityPolicy: props.tlsSecurityPolicy ?? TLSSecurityPolicy.TLS_1_2,
        ...props.customEndpoint && {
          customEndpointEnabled: true,
          customEndpoint: props.customEndpoint.domainName,
          customEndpointCertificateArn: customEndpointCertificate!.certificateRef.certificateArn,
        },
      },
      advancedSecurityOptions: advancedSecurityEnabled
        ? {
          enabled: true,
          internalUserDatabaseEnabled,
          masterUserOptions: {
            masterUserArn: masterUserArn,
            masterUserName: masterUserName,
            masterUserPassword: this.masterUserPassword?.unsafeUnwrap(), // Safe usage
          },
          samlOptions: samlAuthenticationEnabled ? {
            enabled: true,
            idp: props.fineGrainedAccessControl && props.fineGrainedAccessControl.samlAuthenticationOptions ? {
              entityId: props.fineGrainedAccessControl.samlAuthenticationOptions.idpEntityId,
              metadataContent: props.fineGrainedAccessControl.samlAuthenticationOptions.idpMetadataContent,
            } : undefined,
            masterUserName: props.fineGrainedAccessControl?.samlAuthenticationOptions?.masterUserName,
            masterBackendRole: props.fineGrainedAccessControl?.samlAuthenticationOptions?.masterBackendRole,
            rolesKey: props.fineGrainedAccessControl?.samlAuthenticationOptions?.rolesKey ?? 'roles',
            subjectKey: props.fineGrainedAccessControl?.samlAuthenticationOptions?.subjectKey,
            sessionTimeoutMinutes: props.fineGrainedAccessControl?.samlAuthenticationOptions?.sessionTimeoutMinutes ?? 60,
          } : undefined,
        }
        : undefined,
      advancedOptions: props.advancedOptions,
      offPeakWindowOptions: offPeakWindowEnabled ? {
        enabled: offPeakWindowEnabled,
        offPeakWindow: {
          windowStartTime: props.offPeakWindowStart ?? {
            hours: 22,
            minutes: 0,
          },
        },
      } : undefined,
      softwareUpdateOptions: props.enableAutoSoftwareUpdate !== undefined ? {
        autoSoftwareUpdateEnabled: props.enableAutoSoftwareUpdate,
      } : undefined,
      ipAddressType: props.ipAddressType,
      aimlOptions: props.s3VectorsEngineEnabled !== undefined ? {
        s3VectorsEngine: {
          enabled: props.s3VectorsEngineEnabled,
        },
      } : undefined,
    });
    this.domain.applyRemovalPolicy(props.removalPolicy);

    if (props.enableVersionUpgrade) {
      this.domain.cfnOptions.updatePolicy = {
        ...this.domain.cfnOptions.updatePolicy,
        enableVersionUpgrade: props.enableVersionUpgrade,
      };
    }

    if (logGroupResourcePolicy) { this.domain.node.addDependency(logGroupResourcePolicy); }

    if (props.domainName) {
      if (!cdk.Token.isUnresolved(props.domainName)) {
        // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/configuration-api.html#configuration-api-datatypes-domainname
        if (!props.domainName.match(/^[a-z0-9\-]+$/)) {
          throw new ValidationError(lit`InvalidDomainNameCharacters`, `Invalid domainName '${props.domainName}'. Valid characters are a-z (lowercase only), 0-9, and – (hyphen).`, this);
        }
        if (props.domainName.length < 3 || props.domainName.length > 28) {
          throw new ValidationError(lit`InvalidDomainNameLength`, `Invalid domainName '${props.domainName}'. It must be between 3 and 28 characters`, this);
        }
        if (props.domainName[0] < 'a' || props.domainName[0] > 'z') {
          throw new ValidationError(lit`InvalidDomainNameStartCharacter`, `Invalid domainName '${props.domainName}'. It must start with a lowercase letter`, this);
        }
      }
      this.node.addMetadata('aws:cdk:hasPhysicalName', props.domainName);
    }

    this.domainId = this.domain.getAtt('Id').toString();

    this.domainEndpoint = this.domain.getAtt('DomainEndpoint').toString();

    if (props.customEndpoint?.hostedZone) {
      new route53.CnameRecord(this, 'CnameRecord', {
        recordName: props.customEndpoint.domainName,
        zone: props.customEndpoint.hostedZone,
        domainName: this.domainEndpoint,
      });
    }

    this.encryptionAtRestOptions = props.encryptionAtRest;
    if (props.accessPolicies) {
      this.addAccessPolicies(...props.accessPolicies);
    }
    if (unsignedBasicAuthEnabled) {
      this.addAccessPolicies(unsignedAccessPolicy);
    }
  }

  /**
   * Validate windowStartTime property according to
   * https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-opensearchservice-domain-windowstarttime.html
   */
  private validateWindowStartTime(windowStartTime?: WindowStartTime) {
    if (!windowStartTime) return;
    if (windowStartTime.hours < 0 || windowStartTime.hours > 23) {
      throw new ValidationError(lit`InvalidWindowStartTimeHours`, `Hours must be a value between 0 and 23, but got ${windowStartTime.hours}.`, this);
    }
    if (windowStartTime.minutes < 0 || windowStartTime.minutes > 59) {
      throw new ValidationError(lit`InvalidWindowStartTimeMinutes`, `Minutes must be a value between 0 and 59, but got ${windowStartTime.minutes}.`, this);
    }
  }

  /**
   * Validate SAML configuration according to
   * https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-opensearchservice-domain-samloptions.html
   */
  private validateSamlAuthenticationOptions(samlAuthenticationOptions?: SAMLOptionsProperty) {
    if (!samlAuthenticationOptions) {
      throw new ValidationError(lit`SamlOptionsRequired`, 'You need to specify at least an Entity ID and Metadata content for the SAML configuration', this);
    }
    if (samlAuthenticationOptions.idpEntityId.length < 8 || samlAuthenticationOptions.idpEntityId.length > 512) {
      throw new ValidationError(lit`InvalidSamlEntityIdLength`, `SAML identity provider entity ID must be between 8 and 512 characters long, received ${samlAuthenticationOptions.idpEntityId.length}.`, this);
    }
    if (samlAuthenticationOptions.idpMetadataContent.length < 1 || samlAuthenticationOptions.idpMetadataContent.length > 1048576) {
      throw new ValidationError(lit`InvalidSamlMetadataContentLength`, `SAML identity provider metadata content must be between 1 and 1048576 characters long, received ${samlAuthenticationOptions.idpMetadataContent.length}.`, this);
    }
    if (
      samlAuthenticationOptions.masterUserName &&
      (samlAuthenticationOptions.masterUserName.length < 1 || samlAuthenticationOptions.masterUserName.length > 64)
    ) {
      throw new ValidationError(lit`InvalidSamlMasterUserNameLength`, `SAML master username must be between 1 and 64 characters long, received ${samlAuthenticationOptions.masterUserName.length}.`, this);
    }
    if (
      samlAuthenticationOptions.masterBackendRole &&
      (samlAuthenticationOptions.masterBackendRole.length < 1 || samlAuthenticationOptions.masterBackendRole.length > 256)
    ) {
      throw new ValidationError(lit`InvalidSamlMasterBackendRoleLength`, `SAML backend role must be between 1 and 256 characters long, received ${samlAuthenticationOptions.masterBackendRole.length}.`, this);
    }
    if (
      samlAuthenticationOptions.sessionTimeoutMinutes &&
      (samlAuthenticationOptions.sessionTimeoutMinutes < 1 || samlAuthenticationOptions.sessionTimeoutMinutes > 1440)
    ) {
      throw new ValidationError(lit`InvalidSamlSessionTimeoutMinutes`, `SAML session timeout must be a value between 1 and 1440, received ${samlAuthenticationOptions.sessionTimeoutMinutes}.`, this);
    }
  }

  /**
   * Manages network connections to the domain. This will throw an error in case the domain
   * is not placed inside a VPC.
   */
  public get connections(): ec2.Connections {
    if (!this._connections) {
      throw new ValidationError(lit`ConnectionsOnlyAvailableOnVpcEnabledDomains`, "Connections are only available on VPC enabled domains. Use the 'vpc' property to place a domain inside a VPC", this);
    }
    return this._connections;
  }

  /**
   * Add policy statements to the domain access policy
   */
  @MethodMetadata()
  public addAccessPolicies(...accessPolicyStatements: iam.PolicyStatement[]) {
    if (accessPolicyStatements.length > 0) {
      if (!this.accessPolicy) {
        // Only create the custom resource after there are statements to set.
        this.accessPolicy = new OpenSearchAccessPolicy(this, 'AccessPolicy', {
          domainName: this.domainName,
          domainArn: this.domainArn,
          accessPolicies: accessPolicyStatements,
        });

        if (this.encryptionAtRestOptions?.kmsKey) {
          // https://docs.aws.amazon.com/opensearch-service/latest/developerguide/encryption-at-rest.html

          // these permissions are documented as required during domain creation.
          // while not strictly documented for updates as well, it stands to reason that an update
          // operation might require these in case the cluster uses a kms key.
          // empircal evidence shows this is indeed required: https://github.com/aws/aws-cdk/issues/11412
          this.accessPolicy.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
            actions: ['kms:List*', 'kms:Describe*', 'kms:CreateGrant'],
            resources: [this.encryptionAtRestOptions.kmsKey.keyRef.keyArn],
            effect: iam.Effect.ALLOW,
          }));
        }
      } else {
        this.accessPolicy.addAccessPolicies(...accessPolicyStatements);
      }
    }
  }

  /**
   * Selects the appropriate KMS key identifier (keyId or keyArn) based on whether
   * the key is from the same account and region as the domain.
   *
   * For cross-account or cross-region KMS keys, OpenSearch requires the full ARN.
   * For same-account, same-region keys, keyId is sufficient.
   */
  private selectKmsKeyIdentifier(key: kms.IKeyRef): string {
    const stack = cdk.Stack.of(this);

    const keyAccount = key.env.account;
    const keyRegion = key.env.region;
    const stackAccount = stack.account;
    const stackRegion = stack.region;

    // If either account or region is different (and not a token), use ARN
    // Tokens are unresolved values that will be determined at deploy time
    const isCrossAccount = !cdk.Token.isUnresolved(keyAccount) &&
                          !cdk.Token.isUnresolved(stackAccount) &&
                          keyAccount !== stackAccount;
    const isCrossRegion = !cdk.Token.isUnresolved(keyRegion) &&
                         !cdk.Token.isUnresolved(stackRegion) &&
                         keyRegion !== stackRegion;

    if (isCrossAccount || isCrossRegion) {
      return key.keyRef.keyArn;
    }

    // For same-account, same-region keys, use keyId (maintains backward compatibility)
    return key.keyRef.keyId;
  }
}

/**
 * Given an Amazon OpenSearch Service domain endpoint, returns a CloudFormation expression that
 * extracts the domain name.
 *
 * Domain endpoints look like this:
 *
 *   https://example-domain-jcjotrt6f7otem4sqcwbch3c4u.us-east-1.es.amazonaws.com
 *   https://<domain-name>-<suffix>.<region>.es.amazonaws.com
 *
 * ..which means that in order to extract the domain name from the endpoint, we can
 * split the endpoint using "-<suffix>" and select the component in index 0.
 *
 * @param domainEndpoint The Amazon OpenSearch Service domain endpoint
 */
function extractNameFromEndpoint(domainEndpoint: string) {
  const { hostname } = new URL(domainEndpoint);
  const domain = hostname.split('.')[0];
  const suffix = '-' + domain.split('-').slice(-1)[0];
  return domain.split(suffix)[0];
}

/**
 * Converts an engine version into a decimal number with major and minor version i.e x.y.
 *
 * @param version The engine version object
 */
function parseVersion(scope: Construct, version: EngineVersion): { versionNum: number; isElasticsearchVersion: boolean } {
  const elasticsearchPrefix = 'Elasticsearch_';
  const openSearchPrefix = 'OpenSearch_';
  const isElasticsearchVersion = version.version.startsWith(elasticsearchPrefix);
  const versionStr = isElasticsearchVersion
    ? version.version.substring(elasticsearchPrefix.length)
    : version.version.substring(openSearchPrefix.length);
  const firstDot = versionStr.indexOf('.');

  if (firstDot < 1) {
    throw new ValidationError(lit`InvalidEngineVersionFormat`, `Invalid engine version: ${versionStr}. Version string needs to start with major and minor version (x.y).`, scope);
  }

  const secondDot = versionStr.indexOf('.', firstDot + 1);

  try {
    if (secondDot == -1) {
      return { versionNum: parseFloat(versionStr), isElasticsearchVersion };
    } else {
      return { versionNum: parseFloat(versionStr.substring(0, secondDot)), isElasticsearchVersion };
    }
  } catch {
    throw new ValidationError(lit`InvalidEngineVersionParsing`, `Invalid engine version: ${versionStr}. Version string needs to start with major and minor version (x.y).`, scope);
  }
}

function selectSubnets(vpc: ec2.IVpc, vpcSubnets: ec2.SubnetSelection[]): ec2.ISubnet[] {
  const selected = [];
  for (const selection of vpcSubnets) {
    selected.push(...vpc.selectSubnets(selection).subnets);
  }
  return selected;
}

/**
 * Check if any of the subnets are pending lookups. If so, the zone awareness check should be skipped, otherwise it will always throw an error
 *
 * @param vpc The vpc to which the subnets apply
 * @param vpcSubnets The vpc subnets that should be checked
 * @returns true if there are pending lookups for the subnets
 */
function zoneAwarenessCheckShouldBeSkipped(vpc: ec2.IVpc, vpcSubnets: ec2.SubnetSelection[]): boolean {
  for (const selection of vpcSubnets) {
    if (vpc.selectSubnets(selection).isPendingLookup) {
      return true;
    }
  }
  return false;
}

/**
 * Initializes an instance type.
 *
 * @param defaultInstanceType Default instance type which is used if no instance type is provided
 * @param instanceType Instance type
 * @returns Instance type in lowercase (if provided) or default instance type
 */
function initializeInstanceType(defaultInstanceType: string, instanceType?: string): string {
  if (instanceType) {
    return cdk.Token.isUnresolved(instanceType) ? instanceType : instanceType.toLowerCase();
  } else {
    return defaultInstanceType;
  }
}

/**
 * Format instance types list for error messages.
 *
 * @param instanceTypes List of instance types to format
 * @param conjunction Word to use as the conjunction (e.g., 'and', 'or')
 * @returns Formatted instance types list for error messages
 */
function formatInstanceTypesList(instanceTypes: string[], conjunction: string): string {
  return instanceTypes.map((type) => type.toUpperCase()).join(', ').replace(/, ([^,]*)$/, ` ${conjunction} $1`);
}
