/* eslint-disable @cdklabs/no-throw-default-error */
import type { FlagInfo } from './private/flag-modeling';
import { FlagType } from './private/flag-modeling';

////////////////////////////////////////////////////////////////////////
//
// This file defines context keys that enable certain features that are
// implemented behind a flag in order to preserve backwards compatibility for
// existing apps. When a new app is initialized through `cdk init`, the CLI will
// automatically add enable these features by adding them to the generated
// `cdk.json` file.
//
////////////////////////////////////////////////////////////////////////
//
//  !!! IMPORTANT !!!
//
//  When you introduce a new flag, set its 'introducedIn.v2' value to the literal string
// 'V2·NEXT', without the dot.
//
//  DO NOT USE A VARIABLE. DO NOT DEFINE A CONSTANT. The actual value will be string-replaced at
//  version bump time.
//
////////////////////////////////////////////////////////////////////////
//
// There are three types of flags: ApiDefault, BugFix, and VisibleContext flags.
//
// - ApiDefault flags: change the behavior or defaults of the construct library.
//   It is still possible to achieve the old behavior via the official API
//   but changes are necessary (e.g. passing a boolean flag).
//
//   Implications for future Major Version:
//   - The recommended value will become the default value.
//   - Flags of this type will be removed (code changes will become mandatory).
//
// - BugFix flags: the old infra we used to generate is no longer recommended.
//   The old behavior cannot be achieved anymore using the official API (only
//   by making sure the feature flag is unset). Mostly used for infra-impacting
//   bugfixes or enhanced security defaults.
//
//   Implications for future Major Version:
//   - The recommended value will become the default value.
//   - Flag will never be removed (no other way to achieve legacy behavior).
//
// - VisibleContext flags: not really a feature flag, but configurable context which is
//   advertised by putting the context in the `cdk.json` file of new projects.
//
// See https://github.com/aws/aws-cdk-rfcs/blob/master/text/0055-feature-flags.md
// --------------------------------------------------------------------------------

export const ENABLE_STACK_NAME_DUPLICATES_CONTEXT = '@aws-cdk/core:enableStackNameDuplicates';
export const ENABLE_DIFF_NO_FAIL_CONTEXT = 'aws-cdk:enableDiffNoFail';
/** @deprecated use `ENABLE_DIFF_NO_FAIL_CONTEXT` */
export const ENABLE_DIFF_NO_FAIL = ENABLE_DIFF_NO_FAIL_CONTEXT;
export const NEW_STYLE_STACK_SYNTHESIS_CONTEXT = '@aws-cdk/core:newStyleStackSynthesis';
export const STACK_RELATIVE_EXPORTS_CONTEXT = '@aws-cdk/core:stackRelativeExports';
export const DOCKER_IGNORE_SUPPORT = '@aws-cdk/aws-ecr-assets:dockerIgnoreSupport';
export const SECRETS_MANAGER_PARSE_OWNED_SECRET_NAME = '@aws-cdk/aws-secretsmanager:parseOwnedSecretName';
export const KMS_DEFAULT_KEY_POLICIES = '@aws-cdk/aws-kms:defaultKeyPolicies';
export const S3_GRANT_WRITE_WITHOUT_ACL = '@aws-cdk/aws-s3:grantWriteWithoutAcl';
export const ECS_REMOVE_DEFAULT_DESIRED_COUNT = '@aws-cdk/aws-ecs-patterns:removeDefaultDesiredCount';
export const RDS_LOWERCASE_DB_IDENTIFIER = '@aws-cdk/aws-rds:lowercaseDbIdentifier';
export const APIGATEWAY_USAGEPLANKEY_ORDERINSENSITIVE_ID = '@aws-cdk/aws-apigateway:usagePlanKeyOrderInsensitiveId';
export const EFS_DEFAULT_ENCRYPTION_AT_REST = '@aws-cdk/aws-efs:defaultEncryptionAtRest';
export const LAMBDA_RECOGNIZE_VERSION_PROPS = '@aws-cdk/aws-lambda:recognizeVersionProps';
export const LAMBDA_RECOGNIZE_LAYER_VERSION = '@aws-cdk/aws-lambda:recognizeLayerVersion';
export const CLOUDFRONT_DEFAULT_SECURITY_POLICY_TLS_V1_2_2021 = '@aws-cdk/aws-cloudfront:defaultSecurityPolicyTLSv1.2_2021';
export const CHECK_SECRET_USAGE = '@aws-cdk/core:checkSecretUsage';
export const TARGET_PARTITIONS = '@aws-cdk/core:target-partitions';
export const ECS_SERVICE_EXTENSIONS_ENABLE_DEFAULT_LOG_DRIVER = '@aws-cdk-containers/ecs-service-extensions:enableDefaultLogDriver';
export const EC2_UNIQUE_IMDSV2_LAUNCH_TEMPLATE_NAME = '@aws-cdk/aws-ec2:uniqueImdsv2TemplateName';
export const ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME = '@aws-cdk/aws-ecs:arnFormatIncludesClusterName';
export const IAM_MINIMIZE_POLICIES = '@aws-cdk/aws-iam:minimizePolicies';
export const IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME = '@aws-cdk/aws-iam:importedRoleStackSafeDefaultPolicyName';
export const VALIDATE_SNAPSHOT_REMOVAL_POLICY = '@aws-cdk/core:validateSnapshotRemovalPolicy';
export const CODEPIPELINE_CROSS_ACCOUNT_KEY_ALIAS_STACK_SAFE_RESOURCE_NAME = '@aws-cdk/aws-codepipeline:crossAccountKeyAliasStackSafeResourceName';
export const S3_CREATE_DEFAULT_LOGGING_POLICY = '@aws-cdk/aws-s3:createDefaultLoggingPolicy';
export const SNS_SUBSCRIPTIONS_SQS_DECRYPTION_POLICY = '@aws-cdk/aws-sns-subscriptions:restrictSqsDescryption';
export const APIGATEWAY_DISABLE_CLOUDWATCH_ROLE = '@aws-cdk/aws-apigateway:disableCloudWatchRole';
export const ENABLE_PARTITION_LITERALS = '@aws-cdk/core:enablePartitionLiterals';
export const EVENTS_TARGET_QUEUE_SAME_ACCOUNT = '@aws-cdk/aws-events:eventsTargetQueueSameAccount';
export const ECS_DISABLE_EXPLICIT_DEPLOYMENT_CONTROLLER_FOR_CIRCUIT_BREAKER = '@aws-cdk/aws-ecs:disableExplicitDeploymentControllerForCircuitBreaker';
export const ECS_PATTERNS_SEC_GROUPS_DISABLES_IMPLICIT_OPEN_LISTENER = '@aws-cdk/aws-ecs-patterns:secGroupsDisablesImplicitOpenListener';
export const S3_SERVER_ACCESS_LOGS_USE_BUCKET_POLICY = '@aws-cdk/aws-s3:serverAccessLogsUseBucketPolicy';
export const ROUTE53_PATTERNS_USE_CERTIFICATE = '@aws-cdk/aws-route53-patters:useCertificate';
export const ROUTE53_PATTERNS_USE_DISTRIBUTION = '@aws-cdk/aws-route53-patterns:useDistribution';
export const AWS_CUSTOM_RESOURCE_LATEST_SDK_DEFAULT = '@aws-cdk/customresources:installLatestAwsSdkDefault';
export const DATABASE_PROXY_UNIQUE_RESOURCE_NAME = '@aws-cdk/aws-rds:databaseProxyUniqueResourceName';
export const CODEDEPLOY_REMOVE_ALARMS_FROM_DEPLOYMENT_GROUP = '@aws-cdk/aws-codedeploy:removeAlarmsFromDeploymentGroup';
export const APIGATEWAY_AUTHORIZER_CHANGE_DEPLOYMENT_LOGICAL_ID = '@aws-cdk/aws-apigateway:authorizerChangeDeploymentLogicalId';
export const EC2_LAUNCH_TEMPLATE_DEFAULT_USER_DATA = '@aws-cdk/aws-ec2:launchTemplateDefaultUserData';
export const SECRETS_MANAGER_TARGET_ATTACHMENT_RESOURCE_POLICY = '@aws-cdk/aws-secretsmanager:useAttachedSecretResourcePolicyForSecretTargetAttachments';
export const REDSHIFT_COLUMN_ID = '@aws-cdk/aws-redshift:columnId';
export const ENABLE_EMR_SERVICE_POLICY_V2 = '@aws-cdk/aws-stepfunctions-tasks:enableEmrServicePolicyV2';
export const EC2_RESTRICT_DEFAULT_SECURITY_GROUP = '@aws-cdk/aws-ec2:restrictDefaultSecurityGroup';
export const APIGATEWAY_REQUEST_VALIDATOR_UNIQUE_ID = '@aws-cdk/aws-apigateway:requestValidatorUniqueId';
export const INCLUDE_PREFIX_IN_UNIQUE_NAME_GENERATION = '@aws-cdk/core:includePrefixInUniqueNameGeneration';
export const KMS_ALIAS_NAME_REF = '@aws-cdk/aws-kms:aliasNameRef';
export const KMS_APPLY_IMPORTED_ALIAS_PERMISSIONS_TO_PRINCIPAL = '@aws-cdk/aws-kms:applyImportedAliasPermissionsToPrincipal';
export const EFS_DENY_ANONYMOUS_ACCESS = '@aws-cdk/aws-efs:denyAnonymousAccess';
export const EFS_MOUNTTARGET_ORDERINSENSITIVE_LOGICAL_ID = '@aws-cdk/aws-efs:mountTargetOrderInsensitiveLogicalId';
export const AUTOSCALING_GENERATE_LAUNCH_TEMPLATE = '@aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig';
export const ENABLE_OPENSEARCH_MULTIAZ_WITH_STANDBY = '@aws-cdk/aws-opensearchservice:enableOpensearchMultiAzWithStandby';
export const LAMBDA_NODEJS_USE_LATEST_RUNTIME = '@aws-cdk/aws-lambda-nodejs:useLatestRuntimeVersion';
export const RDS_PREVENT_RENDERING_DEPRECATED_CREDENTIALS = '@aws-cdk/aws-rds:preventRenderingDeprecatedCredentials';
export const AURORA_CLUSTER_CHANGE_SCOPE_OF_INSTANCE_PARAMETER_GROUP_WITH_EACH_PARAMETERS = '@aws-cdk/aws-rds:auroraClusterChangeScopeOfInstanceParameterGroupWithEachParameters';
export const APPSYNC_ENABLE_USE_ARN_IDENTIFIER_SOURCE_API_ASSOCIATION = '@aws-cdk/aws-appsync:useArnForSourceApiAssociationIdentifier';
export const CODECOMMIT_SOURCE_ACTION_DEFAULT_BRANCH_NAME = '@aws-cdk/aws-codepipeline-actions:useNewDefaultBranchForCodeCommitSource';
export const LAMBDA_PERMISSION_LOGICAL_ID_FOR_LAMBDA_ACTION = '@aws-cdk/aws-cloudwatch-actions:changeLambdaPermissionLogicalIdForLambdaAction';
export const CODEPIPELINE_CROSS_ACCOUNT_KEYS_DEFAULT_VALUE_TO_FALSE = '@aws-cdk/aws-codepipeline:crossAccountKeysDefaultValueToFalse';
export const CODEPIPELINE_DEFAULT_PIPELINE_TYPE_TO_V2 = '@aws-cdk/aws-codepipeline:defaultPipelineTypeToV2';
export const KMS_REDUCE_CROSS_ACCOUNT_REGION_POLICY_SCOPE = '@aws-cdk/aws-kms:reduceCrossAccountRegionPolicyScope';
export const PIPELINE_REDUCE_ASSET_ROLE_TRUST_SCOPE = '@aws-cdk/pipelines:reduceAssetRoleTrustScope';
export const EKS_NODEGROUP_NAME = '@aws-cdk/aws-eks:nodegroupNameAttribute';
export const EKS_USE_NATIVE_OIDC_PROVIDER = '@aws-cdk/aws-eks:useNativeOidcProvider';
export const ECS_PATTERNS_UNIQUE_TARGET_GROUP_ID = '@aws-cdk/aws-ecs-patterns:uniqueTargetGroupId';
export const EBS_DEFAULT_GP3 = '@aws-cdk/aws-ec2:ebsDefaultGp3Volume';
export const ECS_REMOVE_DEFAULT_DEPLOYMENT_ALARM = '@aws-cdk/aws-ecs:removeDefaultDeploymentAlarm';
export const LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT = '@aws-cdk/custom-resources:logApiResponseDataPropertyTrueDefault';
export const S3_KEEP_NOTIFICATION_IN_IMPORTED_BUCKET = '@aws-cdk/aws-s3:keepNotificationInImportedBucket';
export const USE_NEW_S3URI_PARAMETERS_FOR_BEDROCK_INVOKE_MODEL_TASK = '@aws-cdk/aws-stepfunctions-tasks:useNewS3UriParametersForBedrockInvokeModelTask';
export const EXPLICIT_STACK_TAGS = '@aws-cdk/core:explicitStackTags';
export const REDUCE_EC2_FARGATE_CLOUDWATCH_PERMISSIONS = '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions';
export const DYNAMODB_TABLEV2_RESOURCE_POLICY_PER_REPLICA = '@aws-cdk/aws-dynamodb:resourcePolicyPerReplica';
export const EC2_SUM_TIMEOUT_ENABLED = '@aws-cdk/aws-ec2:ec2SumTImeoutEnabled';
export const APPSYNC_GRAPHQLAPI_SCOPE_LAMBDA_FUNCTION_PERMISSION = '@aws-cdk/aws-appsync:appSyncGraphQLAPIScopeLambdaPermission';
export const USE_CORRECT_VALUE_FOR_INSTANCE_RESOURCE_ID_PROPERTY = '@aws-cdk/aws-rds:setCorrectValueForDatabaseInstanceReadReplicaInstanceResourceId';
export const CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS = '@aws-cdk/core:cfnIncludeRejectComplexResourceUpdateCreatePolicyIntrinsics';
export const LAMBDA_NODEJS_SDK_V3_EXCLUDE_SMITHY_PACKAGES = '@aws-cdk/aws-lambda-nodejs:sdkV3ExcludeSmithyPackages';
export const STEPFUNCTIONS_TASKS_FIX_RUN_ECS_TASK_POLICY = '@aws-cdk/aws-stepfunctions-tasks:fixRunEcsTaskPolicy';
export const STEPFUNCTIONS_USE_DISTRIBUTED_MAP_RESULT_WRITER_V2 = '@aws-cdk/aws-stepfunctions:useDistributedMapResultWriterV2';
export const BASTION_HOST_USE_AMAZON_LINUX_2023_BY_DEFAULT = '@aws-cdk/aws-ec2:bastionHostUseAmazonLinux2023ByDefault';
export const ASPECT_STABILIZATION = '@aws-cdk/core:aspectStabilization';
export const SIGNER_PROFILE_NAME_PASSED_TO_CFN = '@aws-cdk/aws-signer:signingProfileNamePassedToCfn';
export const USER_POOL_DOMAIN_NAME_METHOD_WITHOUT_CUSTOM_RESOURCE = '@aws-cdk/aws-route53-targets:userPoolDomainNameMethodWithoutCustomResource';
export const ALB_DUALSTACK_WITHOUT_PUBLIC_IPV4_SECURITY_GROUP_RULES_DEFAULT = '@aws-cdk/aws-elasticloadbalancingV2:albDualstackWithoutPublicIpv4SecurityGroupRulesDefault';
export const IAM_OIDC_REJECT_UNAUTHORIZED_CONNECTIONS = '@aws-cdk/aws-iam:oidcRejectUnauthorizedConnections';
export const ENABLE_ADDITIONAL_METADATA_COLLECTION = '@aws-cdk/core:enableAdditionalMetadataCollection';
export const LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY = '@aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy';
export const SET_UNIQUE_REPLICATION_ROLE_NAME = '@aws-cdk/aws-s3:setUniqueReplicationRoleName';
export const PIPELINE_REDUCE_STAGE_ROLE_TRUST_SCOPE = '@aws-cdk/pipelines:reduceStageRoleTrustScope';
export const EVENTBUS_POLICY_SID_REQUIRED = '@aws-cdk/aws-events:requireEventBusPolicySid';
export const ASPECT_PRIORITIES_MUTATING = '@aws-cdk/core:aspectPrioritiesMutating';
export const DYNAMODB_TABLE_RETAIN_TABLE_REPLICA = '@aws-cdk/aws-dynamodb:retainTableReplica';
export const LOG_USER_POOL_CLIENT_SECRET_VALUE = '@aws-cdk/cognito:logUserPoolClientSecretValue';
export const PIPELINE_REDUCE_CROSS_ACCOUNT_ACTION_ROLE_TRUST_SCOPE = '@aws-cdk/pipelines:reduceCrossAccountActionRoleTrustScope';
export const S3_TRUST_KEY_POLICY_FOR_SNS_SUBSCRIPTIONS = '@aws-cdk/s3-notifications:addS3TrustKeyPolicyForSnsSubscriptions';
export const EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY = '@aws-cdk/aws-ec2:requirePrivateSubnetsForEgressOnlyInternetGateway';
export const USE_RESOURCEID_FOR_VPCV2_MIGRATION = '@aws-cdk/aws-ec2-alpha:useResourceIdForVpcV2Migration';
export const S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT = '@aws-cdk/aws-s3:publicAccessBlockedByDefault';
export const USE_CDK_MANAGED_LAMBDA_LOGGROUP = '@aws-cdk/aws-lambda:useCdkManagedLogGroup';
export const NETWORK_LOAD_BALANCER_WITH_SECURITY_GROUP_BY_DEFAULT = '@aws-cdk/aws-elasticloadbalancingv2:networkLoadBalancerWithSecurityGroupByDefault';
export const STEPFUNCTIONS_TASKS_HTTPINVOKE_DYNAMIC_JSONPATH_ENDPOINT = '@aws-cdk/aws-stepfunctions-tasks:httpInvokeDynamicJsonPathEndpoint';
export const CLOUDFRONT_FUNCTION_DEFAULT_RUNTIME_V2_0 = '@aws-cdk/aws-cloudfront:defaultFunctionRuntimeV2_0';
export const ELB_USE_POST_QUANTUM_TLS_POLICY = '@aws-cdk/aws-elasticloadbalancingv2:usePostQuantumTlsPolicy';
export const AUTOMATIC_L1_TRAITS = '@aws-cdk/core:automaticL1Traits';
export const BATCH_DEFAULT_AL2023 = '@aws-cdk/aws-batch:defaultToAL2023';
export const EKS_DEFAULT_AL2023 = '@aws-cdk/aws-eks:defaultToAL2023';
export const ANNOTATIONS_IN_VALIDATION_REPORT = '@aws-cdk/core:annotationsInValidationReport';
export const DEFAULT_CROSS_STACK_REFERENCES = '@aws-cdk/core:defaultCrossStackReferences';
export const VALIDATE_AGAINST_DEFAULT_RULES = '@aws-cdk/core:validateAgainstDefaultRules';

export const FLAGS: Record<string, FlagInfo> = {
  //////////////////////////////////////////////////////////////////////
  [SIGNER_PROFILE_NAME_PASSED_TO_CFN]: {
    type: FlagType.BugFix,
    summary: 'Pass signingProfileName to CfnSigningProfile',
    detailsMd: `
      When enabled, the \`signingProfileName\` property is passed to the L1 \`CfnSigningProfile\` construct,
      which ensures that the AWS Signer profile is created with the specified name.

      When disabled, the \`signingProfileName\` is not passed to CloudFormation, maintaining backward
      compatibility with existing deployments where CloudFormation auto-generated profile names.

      This feature flag is needed because enabling it can cause existing signing profiles to be
      replaced during deployment if a \`signingProfileName\` was specified but not previously used
      in the CloudFormation template.`,
    introducedIn: { v2: '2.212.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: false },
  },
  //////////////////////////////////////////////////////////////////////
  [ENABLE_STACK_NAME_DUPLICATES_CONTEXT]: {
    type: FlagType.ApiDefault,
    summary: 'Allow multiple stacks with the same name',
    detailsMd: `
      If this is set, multiple stacks can use the same stack name (e.g. deployed to
      different environments). This means that the name of the synthesized template
      file will be based on the construct path and not on the defined \`stackName\`
      of the stack.`,
    recommendedValue: true,
    introducedIn: { v1: '1.16.0' },
    unconfiguredBehavesLike: { v2: true },
    compatibilityWithOldBehaviorMd: 'Pass stack identifiers to the CLI instead of stack names.',
  },

  //////////////////////////////////////////////////////////////////////
  [ENABLE_DIFF_NO_FAIL_CONTEXT]: {
    type: FlagType.ApiDefault,
    summary: 'Make `cdk diff` not fail when there are differences',
    detailsMd: `
      Determines what status code \`cdk diff\` should return when the specified stack
      differs from the deployed stack or the local CloudFormation template:

      * \`aws-cdk:enableDiffNoFail=true\` => status code == 0
      * \`aws-cdk:enableDiffNoFail=false\` => status code == 1

      You can override this behavior with the --fail flag:

      * \`--fail\` => status code == 1
      * \`--no-fail\` => status code == 0`,
    introducedIn: { v1: '1.19.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Specify `--fail` to the CLI.',
  },

  //////////////////////////////////////////////////////////////////////
  [NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: {
    type: FlagType.BugFix,
    summary: 'Switch to new stack synthesis method which enables CI/CD',
    detailsMd: `
      If this flag is specified, all \`Stack\`s will use the \`DefaultStackSynthesizer\` by
      default. If it is not set, they will use the \`LegacyStackSynthesizer\`.`,
    introducedIn: { v1: '1.39.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [STACK_RELATIVE_EXPORTS_CONTEXT]: {
    type: FlagType.BugFix,
    summary: 'Name exports based on the construct paths relative to the stack, rather than the global construct path',
    detailsMd: `
      Combined with the stack name this relative construct path is good enough to
      ensure uniqueness, and makes the export names robust against refactoring
      the location of the stack in the construct tree (specifically, moving the Stack
      into a Stage).`,
    introducedIn: { v1: '1.58.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [DOCKER_IGNORE_SUPPORT]: {
    type: FlagType.ApiDefault,
    summary: 'DockerImageAsset properly supports `.dockerignore` files by default',
    detailsMd: `
      If this flag is not set, the default behavior for \`DockerImageAsset\` is to use
      glob semantics for \`.dockerignore\` files. If this flag is set, the default behavior
      is standard Docker ignore semantics.

      This is a feature flag as the old behavior was technically incorrect but
      users may have come to depend on it.`,
    introducedIn: { v1: '1.73.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Update your `.dockerignore` file to match standard Docker ignore rules, if necessary.',
  },

  //////////////////////////////////////////////////////////////////////
  [SECRETS_MANAGER_PARSE_OWNED_SECRET_NAME]: {
    type: FlagType.ApiDefault,
    summary: 'Fix the referencing of SecretsManager names from ARNs',
    detailsMd: `
      Secret.secretName for an "owned" secret will attempt to parse the secretName from the ARN,
      rather than the default full resource name, which includes the SecretsManager suffix.

      If this flag is not set, Secret.secretName will include the SecretsManager suffix, which cannot be directly
      used by SecretsManager.DescribeSecret, and must be parsed by the user first (e.g., Fn:Join, Fn:Select, Fn:Split).`,
    introducedIn: { v1: '1.77.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Use `parseArn(secret.secretName).resourceName` to emulate the incorrect old parsing.',
  },

  //////////////////////////////////////////////////////////////////////
  [KMS_DEFAULT_KEY_POLICIES]: {
    type: FlagType.ApiDefault,
    summary: 'Tighten default KMS key policies',
    detailsMd: `
      KMS Keys start with a default key policy that grants the account access to administer the key,
      mirroring the behavior of the KMS SDK/CLI/Console experience. Users may override the default key
      policy by specifying their own.

      If this flag is not set, the default key policy depends on the setting of the \`trustAccountIdentities\`
      flag. If false (the default, for backwards-compatibility reasons), the default key policy somewhat
      resembles the default admin key policy, but with the addition of 'GenerateDataKey' permissions. If
      true, the policy matches what happens when this feature flag is set.

      Additionally, if this flag is not set and the user supplies a custom key policy, this will be appended
      to the key's default policy (rather than replacing it).`,
    introducedIn: { v1: '1.78.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `trustAccountIdentities: false` to `Key` construct to restore the old behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [S3_GRANT_WRITE_WITHOUT_ACL]: {
    type: FlagType.ApiDefault,
    summary: 'Remove `PutObjectAcl` from Bucket.grantWrite',
    detailsMd: `
      Change the old 's3:PutObject*' permission to 's3:PutObject' on Bucket,
      as the former includes 's3:PutObjectAcl',
      which could be used to grant read/write object access to IAM principals in other accounts.
      Use a feature flag to make sure existing customers who might be relying
      on the overly-broad permissions are not broken.`,
    introducedIn: { v1: '1.85.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Call `bucket.grantPutAcl()` in addition to `bucket.grantWrite()` to grant ACL permissions.',
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_REMOVE_DEFAULT_DESIRED_COUNT]: {
    type: FlagType.ApiDefault,
    summary: 'Do not specify a default DesiredCount for ECS services',
    detailsMd: `
      ApplicationLoadBalancedServiceBase, ApplicationMultipleTargetGroupServiceBase,
      NetworkLoadBalancedServiceBase, NetworkMultipleTargetGroupServiceBase, and
      QueueProcessingServiceBase currently determine a default value for the desired count of
      a CfnService if a desiredCount is not provided. The result of this is that on every
      deployment, the service count is reset to the fixed value, even if it was autoscaled.

      If this flag is not set, the default behaviour for CfnService.desiredCount is to set a
      desiredCount of 1, if one is not provided. If true, a default will not be defined for
      CfnService.desiredCount and as such desiredCount will be undefined, if one is not provided.`,
    introducedIn: { v1: '1.92.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'You can pass `desiredCount: 1` explicitly, but you should never need this.',
  },

  [ECS_PATTERNS_SEC_GROUPS_DISABLES_IMPLICIT_OPEN_LISTENER]: {
    type: FlagType.ApiDefault,
    summary: 'Disable implicit openListener when custom security groups are provided',
    detailsMd: `
      ApplicationLoadBalancedServiceBase currently defaults openListener to true, which creates
      security group rules allowing ingress from 0.0.0.0/0. This can be a security risk when
      users provide custom security groups on their load balancer, expecting those to be the
      only ingress rules.

      If this flag is not set, openListener will always default to true for backward compatibility.
      If true, openListener will default to false when custom security groups are detected on the
      load balancer, and true otherwise. Users can still explicitly set openListener: true to
      override this behavior.`,
    introducedIn: { v2: '2.214.0' },
    unconfiguredBehavesLike: { v2: false },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'You can pass `openListener: true` explicitly to maintain the old behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [RDS_LOWERCASE_DB_IDENTIFIER]: {
    type: FlagType.BugFix,
    summary: 'Force lowercasing of RDS Cluster names in CDK',
    detailsMd: `
      Cluster names must be lowercase, and the service will lowercase the name when the cluster
      is created. However, CDK did not use to know about this, and would use the user-provided name
      referencing the cluster, which would fail if it happened to be mixed-case.

      With this flag, lowercase the name in CDK so we can reference it properly.

      Must be behind a permanent flag because changing a name from mixed case to lowercase between deployments
      would lead CloudFormation to think the name was changed and would trigger a cluster replacement
      (losing data!).`,
    introducedIn: { v1: '1.97.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [APIGATEWAY_USAGEPLANKEY_ORDERINSENSITIVE_ID]: {
    type: FlagType.BugFix,
    summary: 'Allow adding/removing multiple UsagePlanKeys independently',
    detailsMd: `
      The UsagePlanKey resource connects an ApiKey with a UsagePlan. API Gateway does not allow more than one UsagePlanKey
      for any given UsagePlan and ApiKey combination. For this reason, CloudFormation cannot replace this resource without
      either the UsagePlan or ApiKey changing.

      The feature addition to support multiple UsagePlanKey resources - 142bd0e2 - recognized this and attempted to keep
      existing UsagePlanKey logical ids unchanged.
      However, this intentionally caused the logical id of the UsagePlanKey to be sensitive to order. That is, when
      the 'first' UsagePlanKey resource is removed, the logical id of the 'second' assumes what was originally the 'first',
      which again is disallowed.

      In effect, there is no way to get out of this mess in a backwards compatible way, while supporting existing stacks.
      This flag changes the logical id layout of UsagePlanKey to not be sensitive to order.`,
    introducedIn: { v1: '1.98.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EFS_DEFAULT_ENCRYPTION_AT_REST]: {
    type: FlagType.ApiDefault,
    summary: 'Enable this feature flag to have elastic file systems encrypted at rest by default.',
    detailsMd: `
      Encryption can also be configured explicitly using the \`encrypted\` property.
      `,
    introducedIn: { v1: '1.98.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass the `encrypted: false` property to the `FileSystem` construct to disable encryption.',
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_RECOGNIZE_VERSION_PROPS]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature flag to opt in to the updated logical id calculation for Lambda Version created using the  `fn.currentVersion`.',
    detailsMd: `
      The previous calculation incorrectly considered properties of the \`AWS::Lambda::Function\` resource that did
      not constitute creating a new Version.

      See 'currentVersion' section in the aws-lambda module's README for more details.`,
    introducedIn: { v1: '1.106.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_RECOGNIZE_LAYER_VERSION]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature flag to opt in to the updated logical id calculation for Lambda Version created using the `fn.currentVersion`.',
    detailsMd: `
      This flag correct incorporates Lambda Layer properties into the Lambda Function Version.

      See 'currentVersion' section in the aws-lambda module's README for more details.`,
    introducedIn: { v1: '1.159.0', v2: '2.27.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [CLOUDFRONT_DEFAULT_SECURITY_POLICY_TLS_V1_2_2021]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature flag to have cloudfront distributions use the security policy TLSv1.2_2021 by default.',
    detailsMd: `
      The security policy can also be configured explicitly using the \`minimumProtocolVersion\` property.`,
    introducedIn: { v1: '1.117.0', v2: '2.0.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [CHECK_SECRET_USAGE]: {
    type: FlagType.VisibleContext,
    summary: 'Enable this flag to make it impossible to accidentally use SecretValues in unsafe locations',
    detailsMd: `
      With this flag enabled, \`SecretValue\` instances can only be passed to
      constructs that accept \`SecretValue\`s; otherwise, \`unsafeUnwrap()\` must be
      called to use it as a regular string.`,
    introducedIn: { v1: '1.153.0', v2: '2.21.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [TARGET_PARTITIONS]: {
    type: FlagType.VisibleContext,
    summary: 'What regions to include in lookup tables of environment agnostic stacks',
    detailsMd: `
      Has no effect on stacks that have a defined region, but will limit the amount
      of unnecessary regions included in stacks without a known region.

      The type of this value should be a list of strings.`,
    introducedIn: { v1: '1.137.0', v2: '2.4.0' },
    recommendedValue: ['aws', 'aws-cn'],
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_SERVICE_EXTENSIONS_ENABLE_DEFAULT_LOG_DRIVER]: {
    type: FlagType.ApiDefault,
    summary: 'ECS extensions will automatically add an `awslogs` driver if no logging is specified',
    detailsMd: `
      Enable this feature flag to configure default logging behavior for the ECS Service Extensions. This will enable the
      \`awslogs\` log driver for the application container of the service to send the container logs to CloudWatch Logs.

      This is a feature flag as the new behavior provides a better default experience for the users.`,
    introducedIn: { v1: '1.140.0', v2: '2.8.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Specify a log driver explicitly.',
  },

  //////////////////////////////////////////////////////////////////////
  [EC2_UNIQUE_IMDSV2_LAUNCH_TEMPLATE_NAME]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature flag to have Launch Templates generated by the `InstanceRequireImdsv2Aspect` use unique names.',
    detailsMd: `
      Previously, the generated Launch Template names were only unique within a stack because they were based only on the
      \`Instance\` construct ID. If another stack that has an \`Instance\` with the same construct ID is deployed in the same
      account and region, the deployments would always fail as the generated Launch Template names were the same.

      The new implementation addresses this issue by generating the Launch Template name with the \`Names.uniqueId\` method.`,
    introducedIn: { v1: '1.140.0', v2: '2.8.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: {
    type: FlagType.BugFix,
    summary: 'ARN format used by ECS. In the new ARN format, the cluster name is part of the resource ID.',
    detailsMd: `
      If this flag is not set, the old ARN format (without cluster name) for ECS is used.
      If this flag is set, the new ARN format (with cluster name) for ECS is used.

      This is a feature flag as the old format is still valid for existing ECS clusters.

      See https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids
      `,
    introducedIn: { v2: '2.35.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [IAM_MINIMIZE_POLICIES]: {
    type: FlagType.VisibleContext,
    summary: 'Minimize IAM policies by combining Statements',
    detailsMd: `
      Minimize IAM policies by combining Principals, Actions and Resources of two
      Statements in the policies, as long as it doesn't change the meaning of the
      policy.`,
    introducedIn: { v1: '1.150.0', v2: '2.18.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [VALIDATE_SNAPSHOT_REMOVAL_POLICY]: {
    type: FlagType.ApiDefault,
    summary: 'Error on snapshot removal policies on resources that do not support it.',
    detailsMd: `
      Makes sure we do not allow snapshot removal policy on resources that do not support it.
      If supplied on an unsupported resource, CloudFormation ignores the policy altogether.
      This flag will reduce confusion and unexpected loss of data when erroneously supplying
      the snapshot removal policy.`,
    introducedIn: { v2: '2.28.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'The old behavior was incorrect. Update your source to not specify SNAPSHOT policies on resources that do not support it.',
  },

  //////////////////////////////////////////////////////////////////////
  [CODEPIPELINE_CROSS_ACCOUNT_KEY_ALIAS_STACK_SAFE_RESOURCE_NAME]: {
    type: FlagType.BugFix,
    summary: 'Generate key aliases that include the stack name',
    detailsMd: `
      Enable this feature flag to have CodePipeline generate a unique cross account key alias name using the stack name.

      Previously, when creating multiple pipelines with similar naming conventions and when crossAccountKeys is true,
      the KMS key alias name created for these pipelines may be the same due to how the uniqueId is generated.

      This new implementation creates a stack safe resource name for the alias using the stack name instead of the stack ID.
      `,
    introducedIn: { v2: '2.29.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [S3_CREATE_DEFAULT_LOGGING_POLICY]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature flag to create an S3 bucket policy by default in cases where an AWS service would automatically create the Policy if one does not exist.',
    detailsMd: `
      For example, in order to send VPC flow logs to an S3 bucket, there is a specific Bucket Policy
      that needs to be attached to the bucket. If you create the bucket without a policy and then add the
      bucket as the flow log destination, the service will automatically create the bucket policy with the
      necessary permissions. If you were to then try and add your own bucket policy CloudFormation will throw
      and error indicating that a bucket policy already exists.

      In cases where we know what the required policy is we can go ahead and create the policy so we can
      remain in control of it.

      @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-and-resource-policy.html#AWS-logs-infrastructure-S3
      `,
    introducedIn: { v2: '2.31.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [SNS_SUBSCRIPTIONS_SQS_DECRYPTION_POLICY]: {
    type: FlagType.BugFix,
    summary: 'Restrict KMS key policy for encrypted Queues a bit more',
    detailsMd: `
      Enable this feature flag to restrict the decryption of a SQS queue, which is subscribed to a SNS topic, to
      only the topic which it is subscribed to and not the whole SNS service of an account.

      Previously the decryption was only restricted to the SNS service principal. To make the SQS subscription more
      secure, it is a good practice to restrict the decryption further and only allow the connected SNS topic to decryption
      the subscribed queue.`,
    introducedIn: { v2: '2.32.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [APIGATEWAY_DISABLE_CLOUDWATCH_ROLE]: {
    type: FlagType.BugFix,
    summary: 'Make default CloudWatch Role behavior safe for multiple API Gateways in one environment',
    detailsMd: `
      Enable this feature flag to change the default behavior for aws-apigateway.RestApi and aws-apigateway.SpecRestApi
      to _not_ create a CloudWatch role and Account. There is only a single ApiGateway account per AWS
      environment which means that each time you create a RestApi in your account the ApiGateway account
      is overwritten. If at some point the newest RestApi is deleted, the ApiGateway Account and CloudWatch
      role will also be deleted, breaking any existing ApiGateways that were depending on them.

      When this flag is enabled you should either create the ApiGateway account and CloudWatch role
      separately _or_ only enable the cloudWatchRole on a single RestApi.
      `,
    introducedIn: { v2: '2.38.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ENABLE_PARTITION_LITERALS]: {
    type: FlagType.BugFix,
    summary: 'Make ARNs concrete if AWS partition is known',
    // eslint-disable-next-line @cdklabs/no-literal-partition
    detailsMd: `
      Enable this feature flag to get partition names as string literals in Stacks with known regions defined in
      their environment, such as "aws" or "aws-cn".  Previously the CloudFormation intrinsic function
      "Ref: AWS::Partition" was used.  For example:

      \`\`\`yaml
      Principal:
        AWS:
          Fn::Join:
            - ""
            - - "arn:"
              - Ref: AWS::Partition
              - :iam::123456789876:root
      \`\`\`

      becomes:

      \`\`\`yaml
      Principal:
        AWS: "arn:aws:iam::123456789876:root"
      \`\`\`

      The intrinsic function will still be used in Stacks where no region is defined or the region's partition
      is unknown.
      `,
    introducedIn: { v2: '2.38.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EVENTS_TARGET_QUEUE_SAME_ACCOUNT]: {
    type: FlagType.BugFix,
    summary: 'Event Rules may only push to encrypted SQS queues in the same account',
    detailsMd: `
      This flag applies to SQS Queues that are used as the target of event Rules. When enabled, only principals
      from the same account as the Rule can send messages. If a queue is unencrypted, this restriction will
      always apply, regardless of the value of this flag.
      `,
    introducedIn: { v2: '2.51.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_DISABLE_EXPLICIT_DEPLOYMENT_CONTROLLER_FOR_CIRCUIT_BREAKER]: {
    type: FlagType.BugFix,
    summary: 'Avoid setting the "ECS" deployment controller when adding a circuit breaker',
    detailsMd: `
      Enable this feature flag to avoid setting the "ECS" deployment controller when adding a circuit breaker to an
      ECS Service, as this will trigger a full replacement which fails to deploy when using set service names.
      This does not change any behaviour as the default deployment controller when it is not defined is ECS.

      This is a feature flag as the new behavior provides a better default experience for the users.
      `,
    introducedIn: { v2: '2.51.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME]: {
    type: FlagType.BugFix,
    summary: 'Enable this feature to create default policy names for imported roles that depend on the stack the role is in.',
    detailsMd: `
      Without this, importing the same role in multiple places could lead to the permissions given for one version of the imported role
      to overwrite permissions given to the role at a different place where it was imported. This was due to all imported instances
      of a role using the same default policy name.

      This new implementation creates default policy names based on the constructs node path in their stack.
      `,
    introducedIn: { v2: '2.60.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [S3_SERVER_ACCESS_LOGS_USE_BUCKET_POLICY]: {
    type: FlagType.BugFix,
    summary: 'Use S3 Bucket Policy instead of ACLs for Server Access Logging',
    detailsMd: `
      Enable this feature flag to use S3 Bucket Policy for granting permission for Server Access Logging
      rather than using the canned \`LogDeliveryWrite\` ACL. ACLs do not work when Object Ownership is
      enabled on the bucket.

      This flag uses a Bucket Policy statement to allow Server Access Log delivery, following best
      practices for S3.

      @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-server-access-logging.html
    `,
    introducedIn: { v2: '2.60.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ROUTE53_PATTERNS_USE_CERTIFICATE]: {
    type: FlagType.ApiDefault,
    summary: 'Use the official `Certificate` resource instead of `DnsValidatedCertificate`',
    detailsMd: `
      Enable this feature flag to use the official CloudFormation supported \`Certificate\` resource instead
      of the deprecated \`DnsValidatedCertificate\` construct. If this flag is enabled and you are creating
      the stack in a region other than us-east-1 then you must also set \`crossRegionReferences=true\` on the
      stack.
      `,
    introducedIn: { v2: '2.61.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Define a `DnsValidatedCertificate` explicitly and pass in the `certificate` property',
  },

  //////////////////////////////////////////////////////////////////////
  [AWS_CUSTOM_RESOURCE_LATEST_SDK_DEFAULT]: {
    type: FlagType.ApiDefault,
    summary: 'Whether to install the latest SDK by default in AwsCustomResource',
    detailsMd: `
      This was originally introduced and enabled by default to not be limited by the SDK version
      that's installed on AWS Lambda. However, it creates issues for Lambdas bound to VPCs that
      do not have internet access, or in environments where 'npmjs.com' is not available.

      The recommended setting is to disable the default installation behavior, and pass the
      flag on a resource-by-resource basis to enable it if necessary.
    `,
    compatibilityWithOldBehaviorMd: 'Set installLatestAwsSdk: true on all resources that need it.',
    introducedIn: { v2: '2.60.0' },
    recommendedValue: false,
  },

  //////////////////////////////////////////////////////////////////////
  [DATABASE_PROXY_UNIQUE_RESOURCE_NAME]: {
    type: FlagType.BugFix,
    summary: 'Use unique resource name for Database Proxy',
    detailsMd: `
      If this flag is not set, the default behavior for \`DatabaseProxy\` is
      to use \`id\` of the constructor for \`dbProxyName\` when it's not specified in the argument.
      In this case, users can't deploy \`DatabaseProxy\`s that have the same \`id\` in the same region.

      If this flag is set, the default behavior is to use unique resource names for each \`DatabaseProxy\`.

      This is a feature flag as the old behavior was technically incorrect, but users may have come to depend on it.
    `,
    introducedIn: { v2: '2.65.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [CODEDEPLOY_REMOVE_ALARMS_FROM_DEPLOYMENT_GROUP]: {
    type: FlagType.BugFix,
    summary: 'Remove CloudWatch alarms from deployment group',
    detailsMd: `
      Enable this flag to be able to remove all CloudWatch alarms from a deployment group by removing
      the alarms from the construct. If this flag is not set, removing all alarms from the construct
      will still leave the alarms configured for the deployment group.
    `,
    introducedIn: { v2: '2.65.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [APIGATEWAY_AUTHORIZER_CHANGE_DEPLOYMENT_LOGICAL_ID]: {
    type: FlagType.BugFix,
    summary: 'Include authorizer configuration in the calculation of the API deployment logical ID.',
    detailsMd: `
      The logical ID of the AWS::ApiGateway::Deployment resource is calculated by hashing
      the API configuration, including methods, and resources, etc. Enable this feature flag
      to also include the configuration of any authorizer attached to the API in the
      calculation, so any changes made to an authorizer will create a new deployment.
      `,
    introducedIn: { v2: '2.66.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EC2_LAUNCH_TEMPLATE_DEFAULT_USER_DATA]: {
    type: FlagType.BugFix,
    summary: 'Define user data for a launch template by default when a machine image is provided.',
    detailsMd: `
      The ec2.LaunchTemplate construct did not define user data when a machine image is
      provided despite the document. If this is set, a user data is automatically defined
      according to the OS of the machine image.
      `,
    recommendedValue: true,
    introducedIn: { v2: '2.67.0' },
  },

  //////////////////////////////////////////////////////////////////////
  [SECRETS_MANAGER_TARGET_ATTACHMENT_RESOURCE_POLICY]: {
    type: FlagType.BugFix,
    summary: 'SecretTargetAttachments uses the ResourcePolicy of the attached Secret.',
    detailsMd: `
      Enable this feature flag to make SecretTargetAttachments use the ResourcePolicy of the attached Secret.
      SecretTargetAttachments are created to connect a Secret to a target resource.
      In CDK code, they behave like regular Secret and can be used as a stand-in in most situations.
      Previously, adding to the ResourcePolicy of a SecretTargetAttachment did attempt to create a separate ResourcePolicy for the same Secret.
      However Secrets can only have a single ResourcePolicy, causing the CloudFormation deployment to fail.

      When enabling this feature flag for an existing Stack, ResourcePolicies created via a SecretTargetAttachment will need replacement.
      This won't be possible without intervention due to limitation outlined above.
      First remove all permissions granted to the Secret and deploy without the ResourcePolicies.
      Then you can re-add the permissions and deploy again.
      `,
    recommendedValue: true,
    introducedIn: { v2: '2.67.0' },
  },

  //////////////////////////////////////////////////////////////////////
  [REDSHIFT_COLUMN_ID]: {
    type: FlagType.BugFix,
    summary: 'Whether to use an ID to track Redshift column changes',
    detailsMd: `
      Redshift columns are identified by their \`name\`. If a column is renamed, the old column
      will be dropped and a new column will be created. This can cause data loss.

      This flag enables the use of an \`id\` attribute for Redshift columns. If this flag is enabled, the
      internal CDK architecture will track changes of Redshift columns through their \`id\`, rather
      than their \`name\`. This will prevent data loss when columns are renamed.

      **NOTE** - Enabling this flag comes at a **risk**. When enabled, update the \`id\`s of all columns,
      **however** do not change the \`names\`s of the columns. If the \`name\`s of the columns are changed during
      initial deployment, the columns will be dropped and recreated, causing data loss. After the initial deployment
      of the \`id\`s, the \`name\`s of the columns can be changed without data loss.
      `,
    introducedIn: { v2: '2.68.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ENABLE_EMR_SERVICE_POLICY_V2]: {
    type: FlagType.BugFix,
    summary: 'Enable AmazonEMRServicePolicy_v2 managed policies',
    detailsMd: `
      If this flag is not set, the default behavior for \`EmrCreateCluster\` is
      to use \`AmazonElasticMapReduceRole\` managed policies.

      If this flag is set, the default behavior is to use the new \`AmazonEMRServicePolicy_v2\`
      managed policies.

      This is a feature flag as the old behavior will be deprecated, but some resources may require manual
      intervention since they might not have the appropriate tags propagated automatically.
      `,
    introducedIn: { v2: '2.72.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EC2_RESTRICT_DEFAULT_SECURITY_GROUP]: {
    type: FlagType.ApiDefault,
    summary: 'Restrict access to the VPC default security group',
    detailsMd: `
      Enable this feature flag to remove the default ingress/egress rules from the
      VPC default security group.

      When a VPC is created, a default security group is created as well and this cannot
      be deleted. The default security group is created with ingress/egress rules that allow
      _all_ traffic. [AWS Security best practices recommend](https://docs.aws.amazon.com/securityhub/latest/userguide/ec2-controls.html#ec2-2)
      removing these ingress/egress rules in order to restrict access to the default security group.
    `,
    introducedIn: { v2: '2.78.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: `
      To allow all ingress/egress traffic to the VPC default security group you
      can set the \`restrictDefaultSecurityGroup: false\`.
    `,
  },

  //////////////////////////////////////////////////////////////////////
  [APIGATEWAY_REQUEST_VALIDATOR_UNIQUE_ID]: {
    type: FlagType.BugFix,
    summary: 'Generate a unique id for each RequestValidator added to a method',
    detailsMd: `
      This flag allows multiple RequestValidators to be added to a RestApi when
      providing the \`RequestValidatorOptions\` in the \`addMethod()\` method.

      If the flag is not set then only a single RequestValidator can be added in this way.
      Any additional RequestValidators have to be created directly with \`new RequestValidator\`.
    `,
    introducedIn: { v2: '2.78.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [KMS_ALIAS_NAME_REF]: {
    type: FlagType.BugFix,
    summary: 'KMS Alias name and keyArn will have implicit reference to KMS Key',
    detailsMd: `
      This flag allows an implicit dependency to be created between KMS Alias and KMS Key
      when referencing key.aliasName or key.keyArn.

      If the flag is not set then a raw string is passed as the Alias name and no
      implicit dependencies will be set.
    `,
    introducedIn: { v2: '2.83.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [KMS_APPLY_IMPORTED_ALIAS_PERMISSIONS_TO_PRINCIPAL]: {
    type: FlagType.BugFix,
    summary: 'Enable grant methods on Aliases imported by name to use kms:ResourceAliases condition',
    detailsMd: `
      This flag enables the grant methods (grant, grantDecrypt, grantEncrypt, etc.) on Aliases imported
      by name to grant permissions based on the 'kms:ResourceAliases' condition rather than no-op grants.
      When disabled, grant calls on imported aliases will be dropped (no-op) to maintain compatibility.
    `,
    introducedIn: { v2: '2.202.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Remove calls to the grant* methods on the aliases referenced by name',
  },

  //////////////////////////////////////////////////////////////////////
  [AUTOSCALING_GENERATE_LAUNCH_TEMPLATE]: {
    type: FlagType.BugFix,
    summary: 'Generate a launch template when creating an AutoScalingGroup',
    detailsMd: `
      Enable this flag to allow AutoScalingGroups to generate a launch template when being created.
      Launch configurations have been deprecated and cannot be created in AWS Accounts created after
      December 31, 2023. Existing 'AutoScalingGroup' properties used for creating a launch configuration
      will now create an equivalent 'launchTemplate'. Alternatively, users can provide an explicit
      'launchTemplate' or 'mixedInstancesPolicy'. When this flag is enabled a 'launchTemplate' will
      attempt to set user data according to the OS of the machine image if explicit user data is not
      provided.
    `,
    introducedIn: { v2: '2.88.0' },
    compatibilityWithOldBehaviorMd: `
      If backwards compatibility needs to be maintained due to an existing autoscaling group
      using a launch config, set this flag to false.
    `,
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [INCLUDE_PREFIX_IN_UNIQUE_NAME_GENERATION]: {
    type: FlagType.BugFix,
    summary: 'Include the stack prefix in the stack name generation process',
    detailsMd: `
      This flag prevents the prefix of a stack from making the stack's name longer than the 128 character limit.

      If the flag is set, the prefix is included in the stack name generation process.
      If the flag is not set, then the prefix of the stack is prepended to the generated stack name.

      **NOTE** - Enabling this flag comes at a **risk**. If you have already deployed stacks, changing the status of this
      feature flag can lead to a change in stacks' name. Changing a stack name mean recreating the whole stack, which
      is not viable in some productive setups.
    `,
    introducedIn: { v2: '2.84.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EFS_DENY_ANONYMOUS_ACCESS]: {
    type: FlagType.ApiDefault,
    summary: 'EFS denies anonymous clients accesses',
    detailsMd: `
      This flag adds the file system policy that denies anonymous clients
      access to \`efs.FileSystem\`.

      If this flag is not set, \`efs.FileSystem\` will allow all anonymous clients
      that can access over the network.`,
    introducedIn: { v2: '2.93.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'You can pass `allowAnonymousAccess: true` so allow anonymous clients access.',
  },

  //////////////////////////////////////////////////////////////////////
  [ENABLE_OPENSEARCH_MULTIAZ_WITH_STANDBY]: {
    type: FlagType.ApiDefault,
    summary: 'Enables support for Multi-AZ with Standby deployment for opensearch domains',
    detailsMd: `
      If this is set, an opensearch domain will automatically be created with
      multi-az with standby enabled.
    `,
    introducedIn: { v2: '2.88.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `capacity.multiAzWithStandbyEnabled: false` to `Domain` construct to restore the old behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_NODEJS_USE_LATEST_RUNTIME]: {
    type: FlagType.ApiDefault,
    summary: 'Enables aws-lambda-nodejs.Function to use the latest available NodeJs runtime as the default',
    detailsMd: `
      If this is set, and a \`runtime\` prop is not passed to, Lambda NodeJs
      functions will use the latest version of the runtime provided by the Lambda
      service. Do not use this if you your lambda function is reliant on dependencies
      shipped as part of the runtime environment.
    `,
    introducedIn: { v2: '2.93.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `runtime: lambda.Runtime.NODEJS_16_X` to `Function` construct to restore the previous behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [EFS_MOUNTTARGET_ORDERINSENSITIVE_LOGICAL_ID]: {
    type: FlagType.BugFix,
    summary: 'When enabled, mount targets will have a stable logicalId that is linked to the associated subnet.',
    detailsMd: `
      When this feature flag is enabled, each mount target will have a stable
      logicalId that is linked to the associated subnet. If the flag is set to
      false then the logicalIds of the mount targets can change if the number of
      subnets changes.

      Set this flag to false for existing mount targets.
    `,
    introducedIn: { v2: '2.93.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [AURORA_CLUSTER_CHANGE_SCOPE_OF_INSTANCE_PARAMETER_GROUP_WITH_EACH_PARAMETERS]: {
    type: FlagType.BugFix,
    summary: 'When enabled, a scope of InstanceParameterGroup for AuroraClusterInstance with each parameters will change.',
    detailsMd: `
      When this feature flag is enabled, a scope of \`InstanceParameterGroup\` for
      \`AuroraClusterInstance\` with each parameters will change to AuroraClusterInstance
      from AuroraCluster.

      If the flag is set to false then it can only make one \`AuroraClusterInstance\`
      with each \`InstanceParameterGroup\` in the AuroraCluster.
    `,
    introducedIn: { v2: '2.97.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [APPSYNC_ENABLE_USE_ARN_IDENTIFIER_SOURCE_API_ASSOCIATION]: {
    type: FlagType.BugFix,
    summary: 'When enabled, will always use the arn for identifiers for CfnSourceApiAssociation in the GraphqlApi construct rather than id.',
    detailsMd: `
      When this feature flag is enabled, we use the IGraphqlApi ARN rather than ID when creating or updating CfnSourceApiAssociation in
      the GraphqlApi construct. Using the ARN allows the association to support an association with a source api or merged api in another account.
      Note that for existing source api associations created with this flag disabled, enabling the flag will lead to a resource replacement.
    `,
    introducedIn: { v2: '2.97.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [RDS_PREVENT_RENDERING_DEPRECATED_CREDENTIALS]: {
    type: FlagType.BugFix,
    summary: 'When enabled, creating an RDS database cluster from a snapshot will only render credentials for snapshot credentials.',
    detailsMd: `
      The \`credentials\` property on the \`DatabaseClusterFromSnapshotProps\`
      interface was deprecated with the new \`snapshotCredentials\` property being
      recommended. Before deprecating \`credentials\`, a secret would be generated
      while rendering credentials if the \`credentials\` property was undefined or
      if a secret wasn't provided via the \`credentials\` property. This behavior
      is replicated with the new \`snapshotCredentials\` property, but the original
      \`credentials\` secret can still be created resulting in an extra database
      secret.

      Set this flag to prevent rendering deprecated \`credentials\` and creating an
      extra database secret when only using \`snapshotCredentials\` to create an RDS
      database cluster from a snapshot.
    `,
    introducedIn: { v2: '2.98.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [CODECOMMIT_SOURCE_ACTION_DEFAULT_BRANCH_NAME]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the CodeCommit source action is using the default branch name \'main\'.',
    detailsMd: `
      When setting up a CodeCommit source action for the source stage of a pipeline, please note that the
      default branch is \'master\'.
      However, with the activation of this feature flag, the default branch is updated to \'main\'.
    `,
    introducedIn: { v2: '2.103.1' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_PERMISSION_LOGICAL_ID_FOR_LAMBDA_ACTION]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the logical ID of a Lambda permission for a Lambda action includes an alarm ID.',
    detailsMd: `
      When this feature flag is enabled, a logical ID of \`LambdaPermission\` for a
      \`LambdaAction\` will include an alarm ID. Therefore multiple alarms for the same Lambda
      can be created with \`LambdaAction\`.

      If the flag is set to false then it can only make one alarm for the Lambda with
      \`LambdaAction\`.
    `,
    introducedIn: { v2: '2.124.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [CODEPIPELINE_CROSS_ACCOUNT_KEYS_DEFAULT_VALUE_TO_FALSE]: {
    type: FlagType.ApiDefault,
    summary: 'Enables Pipeline to set the default value for crossAccountKeys to false.',
    detailsMd: `
      When this feature flag is enabled, and the \`crossAccountKeys\` property is not provided in a \`Pipeline\`
      construct, the construct automatically defaults the value of this property to false.
    `,
    introducedIn: { v2: '2.127.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `crossAccountKeys: true` to `Pipeline` construct to restore the previous behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [CODEPIPELINE_DEFAULT_PIPELINE_TYPE_TO_V2]: {
    type: FlagType.ApiDefault,
    summary: 'Enables Pipeline to set the default pipeline type to V2.',
    detailsMd: `
      When this feature flag is enabled, and the \`pipelineType\` property is not provided in a \`Pipeline\`
      construct, the construct automatically defaults the value of this property to \`PipelineType.V2\`.
    `,
    introducedIn: { v2: '2.133.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `pipelineType: PipelineType.V1` to `Pipeline` construct to restore the previous behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [KMS_REDUCE_CROSS_ACCOUNT_REGION_POLICY_SCOPE]: {
    type: FlagType.BugFix,
    summary: 'When enabled, IAM Policy created from KMS key grant will reduce the resource scope to this key only.',
    detailsMd: `
      When this feature flag is enabled and calling KMS key grant method, the created IAM policy will reduce the resource scope from
      '*' to this specific granting KMS key.
    `,
    introducedIn: { v2: '2.134.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [PIPELINE_REDUCE_ASSET_ROLE_TRUST_SCOPE]: {
    type: FlagType.ApiDefault,
    summary: 'Remove the root account principal from PipelineAssetsFileRole trust policy',
    detailsMd: `
      When this feature flag is enabled, the root account principal will not be added to the trust policy of asset role.
      When this feature flag is disabled, it will keep the root account principal in the trust policy.
    `,
    introducedIn: { v2: '2.141.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to add the root account principal back',
  },

  //////////////////////////////////////////////////////////////////////
  [EKS_NODEGROUP_NAME]: {
    type: FlagType.BugFix,
    summary: 'When enabled, nodegroupName attribute of the provisioned EKS NodeGroup will not have the cluster name prefix.',
    detailsMd: `
      When this feature flag is enabled, the nodegroupName attribute will be exactly the name of the nodegroup without
      any prefix.
    `,
    introducedIn: { v2: '2.139.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EKS_USE_NATIVE_OIDC_PROVIDER]: {
    type: FlagType.BugFix,
    summary: 'When enabled, EKS V2 clusters will use the native OIDC provider resource AWS::IAM::OIDCProvider instead of creating the OIDCProvider with a custom resource (iam.OpenIDConnectProvider).',
    detailsMd: `
      When this feature flag is enabled, EKS clusters will use the native AWS::IAM::OIDCProvider
      CloudFormation resource instead of the custom resource provider for creating OIDC providers.

			WARNING: Enabling this flag on a cluster with an existing OIDC provider created by the custom resource (iam.OpenIDConnectProvider)
			will cause the OIDC provider to be replaced with the native resource, which may lead to disruption.

			To migrate in place without disruption, follow the guide at: https://github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-eks/README.md#migrating-from-the-deprecated-eksopenidconnectprovider-to-eksoidcprovidernative
    `,
    introducedIn: { v2: '2.237.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to use the custom resource provider.',
  },

  //////////////////////////////////////////////////////////////////////
  [EBS_DEFAULT_GP3]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, the default volume type of the EBS volume will be GP3',
    detailsMd: `
      When this featuer flag is enabled, the default volume type of the EBS volume will be \`EbsDeviceVolumeType.GENERAL_PURPOSE_SSD_GP3\`.
    `,
    introducedIn: { v2: '2.140.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Pass `volumeType: EbsDeviceVolumeType.GENERAL_PURPOSE_SSD` to `Volume` construct to restore the previous behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_REMOVE_DEFAULT_DEPLOYMENT_ALARM]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, remove default deployment alarm settings',
    detailsMd: `
      When this feature flag is enabled, remove the default deployment alarm settings when creating a AWS ECS service.
    `,
    introducedIn: { v2: '2.143.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Set AWS::ECS::Service \'DeploymentAlarms\' manually to restore the previous behavior.',
  },

  //////////////////////////////////////////////////////////////////////
  [LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the custom resource used for `AwsCustomResource` will configure the `logApiResponseData` property as true by default',
    detailsMd: `
      This results in 'logApiResponseData' being passed as true to the custom resource provider. This will cause the custom resource handler to receive an 'Update' event. If you don't
      have an SDK call configured for the 'Update' event and you're dependent on specific SDK call response data, you will see this error from CFN:

      CustomResource attribute error: Vendor response doesn't contain <attribute-name> attribute in object. See https://github.com/aws/aws-cdk/issues/29949) for more details.

      Unlike most feature flags, we don't recommend setting this feature flag to true. However, if you're using the 'AwsCustomResource' construct with 'logApiResponseData' as true in
      the event object, then setting this feature flag will keep this behavior. Otherwise, setting this feature flag to false will trigger an 'Update' event by removing the 'logApiResponseData'
      property from the event object.
    `,
    introducedIn: { v2: '2.145.0' },
    recommendedValue: false,
  },

  //////////////////////////////////////////////////////////////////////
  [S3_KEEP_NOTIFICATION_IN_IMPORTED_BUCKET]: {
    type: FlagType.BugFix,
    summary: 'When enabled, Adding notifications to a bucket in the current stack will not remove notification from imported stack.',
    detailsMd: `
      Currently, adding notifications to a bucket where it was created by ourselves will override notification added where it is imported.

      When this feature flag is enabled, adding notifications to a bucket in the current stack will only update notification defined in this stack.
      Other notifications that are not managed by this stack will be kept.
    `,
    introducedIn: { v2: '2.155.0' },
    recommendedValue: false,
  },

  //////////////////////////////////////////////////////////////////////
  [USE_NEW_S3URI_PARAMETERS_FOR_BEDROCK_INVOKE_MODEL_TASK]: {
    type: FlagType.BugFix,
    summary: 'When enabled, use new props for S3 URI field in task definition of state machine for bedrock invoke model.',
    detailsMd: `
    Currently, 'inputPath' and 'outputPath' from the TaskStateBase Props is being used under BedrockInvokeModelProps to define S3URI under 'input' and 'output' fields
    of State Machine Task definition.

    When this feature flag is enabled, specify newly introduced props 's3InputUri' and
    's3OutputUri' to populate S3 uri under input and output fields in state machine task definition for Bedrock invoke model.

    `,
    introducedIn: { v2: '2.156.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to use input and output path fields for s3 URI',
  },

  //////////////////////////////////////////////////////////////////////
  [EXPLICIT_STACK_TAGS]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, stack tags need to be assigned explicitly on a Stack.',
    detailsMd: `
    Without this feature flag enabled, if tags are added to a Stack using
    \`Tags.of(scope).add(...)\`, they will be added to both the stack and all resources
    in the stack template.

    That leads to the tags being applied twice: once in the template, and once
    again automatically by CloudFormation, which will apply all stack tags to
    all resources in the stack. This leads to loss of control, as the
    \`excludeResourceTypes\` option of the Tags API will not have any effect.

    With this flag enabled, tags added to a stack using \`Tags.of(...)\` are ignored,
    and Stack tags must be configured explicitly on the Stack object.
    `,
    introducedIn: { v2: '2.205.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Configure stack-level tags using `new Stack(..., { tags: { ... } })`.',
  },

  //////////////////////////////////////////////////////////////////////
  [REDUCE_EC2_FARGATE_CLOUDWATCH_PERMISSIONS]: {
    type: FlagType.BugFix,
    summary: 'When enabled, we will only grant the necessary permissions when users specify cloudwatch log group through logConfiguration',
    detailsMd: `
    Currently, we automatically add a number of cloudwatch permissions to the task role when no cloudwatch log group is
    specified as logConfiguration and it will grant 'Resources': ['*'] to the task role.

    When this feature flag is enabled, we will only grant the necessary permissions when users specify cloudwatch log group.
    `,
    introducedIn: { v2: '2.159.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to continue grant permissions to log group when no log group is specified',
  },

  //////////////////////////////////////////////////////////////////////
  [DYNAMODB_TABLEV2_RESOURCE_POLICY_PER_REPLICA]: {
    type: FlagType.BugFix,
    summary: 'When enabled will allow you to specify a resource policy per replica, and not copy the source table policy to all replicas',
    detailsMd: `
      If this flag is not set, the default behavior for \`TableV2\` is to use a different \`resourcePolicy\` for each replica.

      If this flag is set to false, the behavior is that each replica shares the same \`resourcePolicy\` as the source table.
      This will prevent you from creating a new table which has an additional replica and a resource policy.

      This is a feature flag as the old behavior was technically incorrect but users may have come to depend on it.`,
    introducedIn: { v2: '2.164.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EC2_SUM_TIMEOUT_ENABLED]: {
    type: FlagType.BugFix,
    summary: 'When enabled, initOptions.timeout and resourceSignalTimeout values will be summed together.',
    detailsMd: `
      Currently is both initOptions.timeout and resourceSignalTimeout are both specified in the options for creating an EC2 Instance,
      only the value from 'resourceSignalTimeout' will be used.

      When this feature flag is enabled, if both initOptions.timeout and resourceSignalTimeout are specified, the values will to be summed together.
      `,
    recommendedValue: true,
    introducedIn: { v2: '2.160.0' },
  },

  //////////////////////////////////////////////////////////////////////
  [APPSYNC_GRAPHQLAPI_SCOPE_LAMBDA_FUNCTION_PERMISSION]: {
    type: FlagType.BugFix,
    summary: 'When enabled, a Lambda authorizer Permission created when using GraphqlApi will be properly scoped with a SourceArn.',
    detailsMd: `
        Currently, when using a Lambda authorizer with an AppSync GraphQL API, the AWS CDK automatically generates the necessary AWS::Lambda::Permission
        to allow the AppSync API to invoke the Lambda authorizer. This permission is overly permissive because it lacks a SourceArn, meaning
        it allows invocations from any source.

        When this feature flag is enabled, the AWS::Lambda::Permission will be properly scoped with the SourceArn corresponding to the
        specific AppSync GraphQL API.
        `,
    recommendedValue: true,
    introducedIn: { v2: '2.161.0' },
  },

  //////////////////////////////////////////////////////////////////////
  [USE_CORRECT_VALUE_FOR_INSTANCE_RESOURCE_ID_PROPERTY]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the value of property `instanceResourceId` in construct `DatabaseInstanceReadReplica` will be set to the correct value which is `DbiResourceId` instead of currently `DbInstanceArn`',
    detailsMd: `
      Currently, the value of the property 'instanceResourceId' in construct 'DatabaseInstanceReadReplica' is not correct, and set to 'DbInstanceArn' which is not correct when it is used to create the IAM Policy in the grantConnect method.

      When this feature flag is enabled, the value of that property will be as expected set to 'DbiResourceId' attribute, and that will fix the grantConnect method.
    `,
    introducedIn: { v2: '2.161.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to use `DbInstanceArn` as value for property `instanceResourceId`',
  },

  //////////////////////////////////////////////////////////////////////
  [CFN_INCLUDE_REJECT_COMPLEX_RESOURCE_UPDATE_CREATE_POLICY_INTRINSICS]: {
    type: FlagType.BugFix,
    summary: 'When enabled, CFN templates added with `cfn-include` will error if the template contains Resource Update or Create policies with CFN Intrinsics that include non-primitive values.',
    detailsMd: `
    Without enabling this feature flag, \`cfn-include\` will silently drop resource update or create policies that contain CFN Intrinsics if they include non-primitive values.

    Enabling this feature flag will make \`cfn-include\` throw on these templates, unless you specify the logical ID of the resource in the 'unhydratedResources' property.
    `,
    recommendedValue: true,
    introducedIn: { v2: '2.161.0' },
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_NODEJS_SDK_V3_EXCLUDE_SMITHY_PACKAGES]: {
    type: FlagType.BugFix,
    summary: 'When enabled, both `@aws-sdk` and `@smithy` packages will be excluded from the Lambda Node.js 18.x runtime to prevent version mismatches in bundled applications.',
    detailsMd: `
      Currently, when bundling Lambda functions with the non-latest runtime that supports AWS SDK JavaScript (v3), only the '@aws-sdk/*' packages are excluded by default.
      However, this can cause version mismatches between the '@aws-sdk/*' and '@smithy/*' packages, as they are tightly coupled dependencies in AWS SDK v3.

      When this feature flag is enabled, both '@aws-sdk/*' and '@smithy/*' packages will be excluded during the bundling process. This ensures that no mismatches
      occur between these tightly coupled dependencies when using the AWS SDK v3 in Lambda functions.
    `,
    introducedIn: { v2: '2.161.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [STEPFUNCTIONS_TASKS_FIX_RUN_ECS_TASK_POLICY]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the resource of IAM Run Ecs policy generated by SFN EcsRunTask will reference the definition, instead of constructing ARN.',
    detailsMd: `
      Currently, in the IAM Run Ecs policy generated by SFN EcsRunTask(), CDK will construct the ARN with wildcard attached at the end.
      The revision number at the end will be replaced with a wildcard which it shouldn't.

      When this feature flag is enabled, if the task definition is created in the stack, the 'Resource' section will 'Ref' the taskDefinition.
    `,
    introducedIn: { v2: '2.163.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [BASTION_HOST_USE_AMAZON_LINUX_2023_BY_DEFAULT]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, the BastionHost construct will use the latest Amazon Linux 2023 AMI, instead of Amazon Linux 2.',
    detailsMd: `
      Currently, if the machineImage property of the BastionHost construct defaults to using the latest Amazon Linux 2
      AMI. Amazon Linux 2 hits end-of-life in June 2025, so using Amazon Linux 2023 by default is a more future-proof
      and secure option.

      When this feature flag is enabled, if you do not pass the machineImage property to the BastionHost construct,
      the latest Amazon Linux 2023 version will be used instead of Amazon Linux 2.
    `,
    introducedIn: { v2: '2.172.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag or explicitly pass an Amazon Linux 2 machine image to the BastionHost construct.',
  },

  //////////////////////////////////////////////////////////////////////
  [ASPECT_STABILIZATION]: {
    type: FlagType.VisibleContext,
    summary: 'When enabled, a stabilization loop will be run when invoking Aspects during synthesis.',
    detailsMd: `
      Previously, Aspects were invoked in a single pass of the construct tree.
      This meant that Aspects which created other Aspects were not run, and Aspects that created new nodes in the tree sometimes did not inherit their parent Aspects.

      When this feature flag is enabled, a stabilization loop is run to recurse the construct tree multiple times when invoking Aspects.
    `,
    unconfiguredBehavesLike: { v2: true },
    introducedIn: { v2: '2.172.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [USER_POOL_DOMAIN_NAME_METHOD_WITHOUT_CUSTOM_RESOURCE]: {
    type: FlagType.BugFix,
    summary: 'When enabled, use a new method for DNS Name of user pool domain target without creating a custom resource.',
    detailsMd: `
    When this feature flag is enabled, a new method will be used to get the DNS Name of the user pool domain target. The old method
    creates a custom resource internally, but the new method doesn't need a custom resource.

    If the flag is set to false then a custom resource will be created when using \`UserPoolDomainTarget\`.
    `,
    introducedIn: { v2: '2.174.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ALB_DUALSTACK_WITHOUT_PUBLIC_IPV4_SECURITY_GROUP_RULES_DEFAULT]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the default security group ingress rules will allow IPv6 ingress from anywhere',
    detailsMd: `
      For internet facing ALBs with 'dualstack-without-public-ipv4' IP address type, the default security group rules
      will allow IPv6 ingress from anywhere (::/0). Previously, the default security group rules would only allow IPv4 ingress.

      Using a feature flag to make sure existing customers who might be relying
      on the overly restrictive permissions are not broken.`,
    introducedIn: { v2: '2.176.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to only allow IPv4 ingress in the default security group rules.',
  },

  //////////////////////////////////////////////////////////////////////
  [IAM_OIDC_REJECT_UNAUTHORIZED_CONNECTIONS]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the default behaviour of OIDC provider will reject unauthorized connections',
    detailsMd: `
      When this feature flag is enabled, the default behaviour of OIDC Provider's custom resource handler will
      default to reject unauthorized connections when downloading CA Certificates.

      When this feature flag is disabled, the behaviour will be the same as current and will allow downloading
      thumbprints from unsecure connections.`,
    introducedIn: { v2: '2.177.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to allow unsecure OIDC connection.',
  },

  //////////////////////////////////////////////////////////////////////
  [ENABLE_ADDITIONAL_METADATA_COLLECTION]: {
    type: FlagType.VisibleContext,
    summary: 'When enabled, CDK will expand the scope of usage data collected to better inform CDK development and improve communication for security concerns and emerging issues.',
    detailsMd: `
    When this feature flag is enabled, CDK expands the scope of usage data collection to include the following:
      * L2 construct property keys - Collect which property keys you use from the L2 constructs in your app. This includes property keys nested in dictionary objects.
      * L2 construct property values of BOOL and ENUM types - Collect property key values of only BOOL and ENUM types. All other types, such as string values or construct references will be redacted.
      * L2 construct method usage - Collection method name, parameter keys and parameter values of BOOL and ENUM type.
    `,
    introducedIn: { v2: '2.178.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: {
    type: FlagType.BugFix,
    summary: '[Deprecated] When enabled, Lambda will create new inline policies with AddToRolePolicy instead of adding to the Default Policy Statement',
    detailsMd: `
      [Deprecated default feature] When this feature flag is enabled, Lambda will create new inline policies with AddToRolePolicy.
      The purpose of this is to prevent lambda from creating a dependency on the Default Policy Statement.
      This solves an issue where a circular dependency could occur if adding lambda to something like a Cognito Trigger, then adding the User Pool to the lambda execution role permissions.
      However in the current implementation, we have removed a dependency of the lambda function on the policy. In addition to this, a Role will be attached to the Policy instead of an inline policy being attached to the role.
      This will create a data race condition in the CloudFormation template because the creation of the Lambda function no longer waits for the policy to be created. Having said that, we are not deprecating the feature (we are defaulting the feature flag to false for new stacks) since this feature can still be used to get around the circular dependency issue (issue-7016) particularly in cases where the lambda resource creation doesnt need to depend on the policy resource creation.
      We recommend to unset the feature flag if already set which will restore the original behavior.
    `,
    introducedIn: { v2: '2.180.0' },
    recommendedValue: false,
  },

  //////////////////////////////////////////////////////////////////////
  [SET_UNIQUE_REPLICATION_ROLE_NAME]: {
    type: FlagType.BugFix,
    summary: 'When enabled, CDK will automatically generate a unique role name that is used for s3 object replication.',
    detailsMd: `
      When performing cross-account S3 replication, we need to explicitly specify a role name for the replication execution role.
      When this feature flag is enabled, a unique role name is specified only when performing cross-account replication.
      When disabled, 'CDKReplicationRole' is always specified.
    `,
    introducedIn: { v2: '2.182.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [PIPELINE_REDUCE_STAGE_ROLE_TRUST_SCOPE]: {
    type: FlagType.ApiDefault,
    summary: 'Remove the root account principal from Stage addActions trust policy',
    detailsMd: `
      When this feature flag is enabled, the root account principal will not be added to the trust policy of stage role.
      When this feature flag is disabled, it will keep the root account principal in the trust policy.

      For cross-account cases, when this feature flag is enabled the trust policy will be scoped to the role only.
      If you are providing a custom role, you will need to ensure 'roleName' is specified or set to PhysicalName.GENERATE_IF_NEEDED.
    `,
    introducedIn: { v2: '2.184.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to add the root account principal back',
  },

  //////////////////////////////////////////////////////////////////////
  [EVENTBUS_POLICY_SID_REQUIRED]: {
    type: FlagType.BugFix,
    summary: 'When enabled, grantPutEventsTo() will use resource policies with Statement IDs for service principals.',
    detailsMd: `
      Currently, when granting permissions to service principals using grantPutEventsTo(), the operation silently fails
      because service principals require resource policies with Statement IDs.

      When this flag is enabled:
      - Resource policies will be created with Statement IDs for service principals
      - The operation will succeed as expected

      When this flag is disabled:
      - A warning will be emitted
      - The grant operation will be dropped
      - No permissions will be added

      This fixes the issue where permissions were silently not being added for service principals.
    `,
    introducedIn: { v2: '2.186.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ASPECT_PRIORITIES_MUTATING]: {
    type: FlagType.ApiDefault,
    summary: 'When set to true, Aspects added by the construct library on your behalf will be given a priority of MUTATING.',
    detailsMd: `
      Custom Aspects you add have a priority of DEFAULT (500) if you don't
      assign a more specific priority, which is higher than MUTATING (200). This
      is relevant if a custom Aspect you add and an Aspect added by CDK try to
      configure the same value.

      If this flag is set to false (old behavior), Aspects added by CDK are also
      added with a priority of DEFAULT; because their priorities are equal, the
      Aspects that is closest to the target construct executes last (either
      yours or the Aspect added by the CDK).

      If this flag is set to true (recommended behavior), Aspects added by CDK are added
      with a priority of MUTATING, and custom Aspects you add with DEFAULT
      priority will always execute last and "win" the write. If you need Aspects
      added by CDK to run after yours, your Aspect needs to have a priority of
      MUTATING or lower.

      This setting only applies to Aspects that were already being added for you
      before version 2.172.0. Aspects introduced since that version will always
      be added with a priority of MUTATING, independent of this feature flag.
    `,
    introducedIn: { v2: '2.189.1' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: `
      To add mutating Aspects controlling construct values that can be overridden
      by Aspects added by CDK, give them MUTATING priority:

      \`\`\`
      Aspects.of(stack).add(new MyCustomAspect(), {
        priority: AspectPriority.MUTATING,
      });
      \`\`\`
    `,
  },
  [DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: {
    type: FlagType.BugFix,
    summary: 'When enabled, table replica will be default to the removal policy of source table unless specified otherwise.',
    detailsMd: `
      Currently, table replica will always be deleted when stack deletes regardless of source table's deletion policy.
      When enabled, table replica will be default to the removal policy of source table unless specified otherwise.
    `,
    introducedIn: { v2: '2.187.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [LOG_USER_POOL_CLIENT_SECRET_VALUE]: {
    type: FlagType.ApiDefault,
    summary: 'When disabled, the value of the user pool client secret will not be logged in the custom resource lambda function logs.',
    detailsMd: `
      When this feature flag is enabled, the SDK API call response to describe user pool client values will be logged in the custom
      resource lambda function logs.

      When this feature flag is disabled, the SDK API call response to describe user pool client values will not be logged in the custom
      resource lambda function logs.
    `,
    introducedIn: { v2: '2.187.0' },
    unconfiguredBehavesLike: { v2: false },
    recommendedValue: false,
    compatibilityWithOldBehaviorMd: 'Enable the feature flag to keep the old behavior and log the client secret values',
  },

  //////////////////////////////////////////////////////////////////////
  [PIPELINE_REDUCE_CROSS_ACCOUNT_ACTION_ROLE_TRUST_SCOPE]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, scopes down the trust policy for the cross-account action role',
    detailsMd: `
        When this feature flag is enabled, the trust policy of the cross-account action role will be scoped to the pipeline role.
        If you are providing a custom role, you will need to ensure 'roleName' is specified or set to PhysicalName.GENERATE_IF_NEEDED.
        When this feature flag is disabled, it will keep the root account principal in the trust policy.
      `,
    introducedIn: { v2: '2.189.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to add the root account principal back',
  },

  //////////////////////////////////////////////////////////////////////
  [STEPFUNCTIONS_USE_DISTRIBUTED_MAP_RESULT_WRITER_V2]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, the resultWriterV2 property of DistributedMap will be used insted of resultWriter',
    detailsMd: `
      When this feature flag is enabled, the resultWriterV2 property is used instead of resultWriter in DistributedMap class.
      resultWriterV2 uses ResultWriterV2 class in StepFunctions ASL and can have either Bucket/Prefix or WriterConfig or both.
    `,
    introducedIn: { v2: '2.188.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag and set resultWriter in DistributedMap',
  },

  //////////////////////////////////////////////////////////////////////
  [S3_TRUST_KEY_POLICY_FOR_SNS_SUBSCRIPTIONS]: {
    type: FlagType.BugFix,
    summary: 'Add an S3 trust policy to a KMS key resource policy for SNS subscriptions.',
    detailsMd: `
      When this feature flag is enabled, a S3 trust policy will be added to the KMS key resource policy for encrypted SNS subscriptions.
          `,
    introducedIn: { v2: '2.195.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [EC2_REQUIRE_PRIVATE_SUBNETS_FOR_EGRESSONLYINTERNETGATEWAY]: {
    type: FlagType.BugFix,
    summary: 'When enabled, the EgressOnlyGateway resource is only created if private subnets are defined in the dual-stack VPC.',
    detailsMd: `
      When this feature flag is enabled, EgressOnlyGateway resource will not be created when you create a vpc with only public subnets.
          `,
    introducedIn: { v2: '2.196.0' },
    recommendedValue: true,
  },

  /// ///////////////////////////////////////////////////////////////////
  [USE_RESOURCEID_FOR_VPCV2_MIGRATION]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, use resource IDs for VPC V2 migration',
    detailsMd: `
        When this feature flag is enabled, the VPC V2 migration will use resource IDs instead of getAtt references
        for migrating resources from VPC V1 to VPC V2. This helps ensure a smoother migration path between
        the two versions.
      `,
    introducedIn: { v2: '2.196.0' },
    recommendedValue: false,
    unconfiguredBehavesLike: { v2: false },
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to use getAtt references for VPC V2 migration',
  },

  //////////////////////////////////////////////////////////////////////
  [S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT]: {
    type: FlagType.BugFix,
    summary: 'When enabled, setting any combination of options for BlockPublicAccess will automatically set true for any options not defined.',
    detailsMd: `
      When BlockPublicAccess is not set at all, s3's default behavior will be to set all options to true in aws console.
      The previous behavior in cdk before this feature was; if only some of the BlockPublicAccessOptions were set (not all 4), then the ones undefined would default to false.
      This is counter intuitive to the console behavior where the options would start in true state and a user would uncheck the boxes as needed.
      The new behavior from this feature will allow a user, for example, to set 1 of the 4 BlockPublicAccessOpsions to false, and on deployment the other 3 will remain true.
    `,
    introducedIn: { v2: '2.196.0' },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [USE_CDK_MANAGED_LAMBDA_LOGGROUP]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, CDK creates and manages loggroup for the lambda function',
    detailsMd: `
        When this feature flag is enabled, CDK will create a loggroup for lambda function with default properties
        which supports CDK features Tag propagation, Property Injectors, Aspects
        if the cdk app doesnt pass a 'logRetention' or 'logGroup' explicitly.
        LogGroups created via 'logRetention' do not support Tag propagation, Property Injectors, Aspects.
        LogGroups created via 'logGroup' created in CDK support Tag propagation, Property Injectors, Aspects.

        When this feature flag is disabled, a loggroup is created by Lambda service on first invocation
        of the function (existing behavior).
        LogGroups created in this way do not support Tag propagation, Property Injectors, Aspects.

        DO NOT ENABLE: If you have an existing app defining a lambda function and
        have not supplied a logGroup or logRetention prop and your lambda function has
        executed at least once, the logGroup has been already created with the same name
        so your deployment will start failing.
        Refer aws-lambda/README.md for more details on Customizing Log Group creation.
      `,
    introducedIn: { v2: '2.200.0' },
    unconfiguredBehavesLike: { v2: false },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to let lambda service create logGroup or specify logGroup or logRetention',
  },

  //////////////////////////////////////////////////////////////////////
  [NETWORK_LOAD_BALANCER_WITH_SECURITY_GROUP_BY_DEFAULT]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, Network Load Balancer will be created with a security group by default.',
    detailsMd: `
      When this feature flag is enabled, Network Load Balancer will be created with a security group by default.
    `,
    introducedIn: { v2: '2.222.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable the feature flag to create Network Load Balancer without a security group by default.',
  },

  //////////////////////////////////////////////////////////////////////
  [STEPFUNCTIONS_TASKS_HTTPINVOKE_DYNAMIC_JSONPATH_ENDPOINT]: {
    type: FlagType.BugFix,
    summary: 'When enabled, allows using a dynamic apiEndpoint with JSONPath format in HttpInvoke tasks.',
    detailsMd: `
      When this feature flag is enabled, the JSONPath apiEndpoint value will be resolved dynamically at runtime, while slightly increasing the size of the state machine definition.
      When disabled, the JSONPath apiEndpoint property will only support a static string value.
    `,
    introducedIn: { v2: '2.221.0' },
    unconfiguredBehavesLike: { v2: true },
    recommendedValue: true,
  },

  //////////////////////////////////////////////////////////////////////
  [ECS_PATTERNS_UNIQUE_TARGET_GROUP_ID]: {
    type: FlagType.BugFix,
    summary: 'When enabled, ECS patterns will generate unique target group IDs to prevent conflicts during load balancer replacement',
    detailsMd: `
      When this feature flag is enabled, ECS patterns will generate unique target group IDs that include
      both the load balancer type (public/private) and load balancer name. This prevents CloudFormation
      conflicts when switching between public and private load balancers or when changing load balancer names.

      Without this flag, target groups use generic IDs like 'ECS' which can cause conflicts when the
      underlying load balancer is replaced due to changes in internetFacing or loadBalancerName properties.

      This is a breaking change as it will cause target group replacement when the flag is enabled.
    `,
    introducedIn: { v2: '2.221.0' },
    recommendedValue: true,
  },

  [ROUTE53_PATTERNS_USE_DISTRIBUTION]: {
    type: FlagType.ApiDefault,
    summary: 'Use the `Distribution` resource instead of `CloudFrontWebDistribution`',
    detailsMd: `
      Enable this feature flag to use the new \`Distribution\` resource instead
      of the deprecated \`CloudFrontWebDistribution\` construct.
      `,
    introducedIn: { v2: '2.233.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Define a `CloudFrontWebDistribution` explicitly',
  },

  //////////////////////////////////////////////////////////////////////
  [CLOUDFRONT_FUNCTION_DEFAULT_RUNTIME_V2_0]: {
    type: FlagType.ApiDefault,
    summary: 'Use cloudfront-js-2.0 as the default runtime for CloudFront Functions',
    detailsMd: `
      When enabled, CloudFront Functions will use cloudfront-js-2.0 runtime by default instead of cloudfront-js-1.0.
      The runtime can still be configured explicitly using the \`runtime\` property.

      If \`keyValueStore\` is specified, the runtime will always be cloudfront-js-2.0 regardless of this flag.`,
    introducedIn: { v2: '2.245.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Set `runtime: FunctionRuntime.JS_1_0` explicitly to use the v1.0 runtime.',
  },

  //////////////////////////////////////////////////////////////////////
  [ELB_USE_POST_QUANTUM_TLS_POLICY]: {
    type: FlagType.ApiDefault,
    summary: 'When enabled, HTTPS/TLS listeners use post-quantum TLS policy by default',
    detailsMd: `
      When this feature flag is enabled, HTTPS and TLS listeners that do not have an explicit
      \`sslPolicy\` will use the post-quantum cryptography policy
      \`ELBSecurityPolicy-TLS13-1-2-PQ-2025-09\` by default.

      This policy uses the non-restricted variant (without -Res-) to maintain AES-CBC cipher support
      for TLS 1.2 clients, ensuring nearly 100% backward compatibility with the previous CDK default.
      Post-quantum policies provide protection against "Harvest Now, Decrypt Later" attacks using
      hybrid ML-KEM key exchange.

      When disabled (default), no explicit SSL policy is set, preserving the existing CDK behavior
      where \`RECOMMENDED_TLS\` (\`ELBSecurityPolicy-TLS13-1-2-2021-06\`) is used.
    `,
    introducedIn: { v2: '2.245.0' },
    recommendedValue: true,
    compatibilityWithOldBehaviorMd: 'Disable this feature flag to preserve existing behavior where no explicit SSL policy is set.',
  },

  [AUTOMATIC_L1_TRAITS]: {
    type: FlagType.ApiDefault,
    summary: 'Automatically use the default L1 traits for L1 constructs`',
    detailsMd: `
      When enabled, the construct library will apply default L1 traits for types that
      have no traits defined yet. Traits regulate behaviors such as how to create
      resource policies, or how to find an encryption key for a given L1 construct.
      `,
    introducedIn: { v2: '2.239.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: true },
    compatibilityWithOldBehaviorMd: 'Register traits explicitly for each resource type',
  },

  //////////////////////////////////////////////////////////////////////
  [BATCH_DEFAULT_AL2023]: {
    type: FlagType.ApiDefault,
    summary: 'Use AL2023 as the default imageType for EC2 Batch compute environments instead of the deprecated AL2',
    detailsMd: `
      When enabled, EC2 Batch compute environments (both ECS and EKS) that do not specify an \`imageType\`
      will default to \`ECS_AL2023\` or \`EKS_AL2023\` instead of the deprecated \`ECS_AL2\` or \`EKS_AL2\`
      (Amazon Linux 2, reaching EOL June 2026 for ECS; already EOL for EKS).

      For EKS compute environments with a launch template, \`userdataType\` will automatically be set
      to \`EKS_NODEADM\` when an AL2023 image type is used, as required by the AWS Batch API.

      When disabled, the default \`imageType\` remains \`ECS_AL2\` / \`EKS_AL2\` for backward compatibility.`,
    introducedIn: { v2: '2.249.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: false },
    compatibilityWithOldBehaviorMd: `Explicitly set \`imageType\` to \`ECS_AL2\` or \`EKS_AL2\` in your compute environment images configuration.

**Warning**: Enabling this flag on existing stacks may cause compute environment replacement, which terminates running jobs. To migrate safely, first pin existing environments to their current imageType explicitly, then enable the flag.`,
  },

  //////////////////////////////////////////////////////////////////////
  [EKS_DEFAULT_AL2023]: {
    type: FlagType.ApiDefault,
    summary: 'Use AL2023 as the default AMI type for EKS managed node groups using non-GPU instance types instead of the deprecated AL2',
    detailsMd: `
      When enabled, EKS managed node groups that do not specify an \`amiType\` will default to
      AL2023 AMI types (AL2023_x86_64_STANDARD, AL2023_ARM_64_STANDARD) instead of the deprecated
      AL2 types (AL2_x86_64, AL2_ARM_64).

      This only affects non-GPU instance types. GPU instances continue to default to AL2_x86_64_GPU
      because AL2023 splits GPU support into separate NVIDIA and Neuron AMI variants.

      Amazon Linux 2 reached end of support on November 26, 2025. AL2023 is the AWS-recommended default.

      When disabled, the default AMI types remain AL2 for backward compatibility.`,
    introducedIn: { v2: '2.259.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: false },
    compatibilityWithOldBehaviorMd: `Explicitly set \`amiType\` to the desired AL2 type (e.g., \`NodegroupAmiType.AL2_X86_64\`) in your nodegroup configuration.

**Warning**: Enabling this flag on existing stacks will cause node group replacement, which terminates running pods. To migrate safely, first pin existing node groups to their current amiType explicitly, then enable the flag for new node groups.`,
  },

  //////////////////////////////////////////////////////////////////////
  [ANNOTATIONS_IN_VALIDATION_REPORT]: {
    type: FlagType.VisibleContext,
    summary: 'Include construct annotations (warnings and errors) in the policy validation report',
    detailsMd: `
      When enabled, construct annotations added via \`Annotations.of()\` or \`Validations.of()\`
      are collected post-synthesis and included in the policy validation report alongside
      plugin violations. Annotations appear under a "Construct Annotations" source entry.

      When disabled, annotations are only displayed through the CLI's standard metadata
      output (e.g. \`[Warning at /path] message\`) and do not appear in the validation report.

      Note: enabling this flag may cause annotations to appear twice — once in the CLI's
      standard output and once in the validation report — until the CLI is updated to
      consolidate both displays.`,
    introducedIn: { v2: '2.253.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: false },
  },

  //////////////////////////////////////////////////////////////////////
  [DEFAULT_CROSS_STACK_REFERENCES]: {
    type: FlagType.VisibleContext,
    summary: 'Controls whether cross-stack references are strong, weak, or both',
    detailsMd: `
      Controls the default type of cross-stack references. Accepted values are
      \`"strong"\`, \`"weak"\`, and \`"both"\`.

      The flag is read from the **consumer** stack's context, not the producer's.

      - \`"strong"\` (default): Uses ExportWriter/ExportReader custom resources that
        write values to SSM Parameters in the consuming region. This prevents the
        producing stack from being deleted while consumers exist.
      - \`"weak"\`: Uses Fn::GetStackOutput to read an output directly from the
        producing stack. Simpler (no extra infrastructure), but the producing stack
        can be deleted independently of consumers.
      - \`"both"\`: A transitional state for migrating from strong to weak. The producer
        keeps the ExportWriter (continues writing to SSM) and also adds an Output. The
        consumer switches to Fn::GetStackOutput. This allows removing the ExportReader
        without breaking anything.

      **Migration from strong to weak**: set to \`"both"\` and deploy, then set to
      \`"weak"\` and deploy again.

      **Migration from weak to strong**: set directly to \`"strong"\` (single deployment).`,
    introducedIn: { v2: '2.254.0' },
    recommendedValue: 'weak',
    unconfiguredBehavesLike: { v2: 'strong' },
  },

  //////////////////////////////////////////////////////////////////////
  [VALIDATE_AGAINST_DEFAULT_RULES]: {
    type: FlagType.VisibleContext,
    summary: 'Treat CloudFormation Validate findings as errors',
    detailsMd: `
      The CDK always validates synthesized templates against a default set of CloudFormation
      rules during synthesis. These rules include schema validation, best-practice linting,
      and common misconfiguration detection.

      When this flag is explicitly set to \`true\`, violations are treated as errors and will
      fail synthesis. When unconfigured, violations are reported as warnings only.`,
    introducedIn: { v2: '2.262.0' },
    recommendedValue: true,
    unconfiguredBehavesLike: { v2: false },
  },
};

export const CURRENT_MV = 'v2';

/**
 * The list of future flags that are now expired. This is going to be used to identify
 * and block usages of old feature flags in the new major version of CDK.
 */
export const CURRENT_VERSION_EXPIRED_FLAGS: string[] = Object.entries(FLAGS)
  .filter(([_, flag]) => flag.introducedIn[CURRENT_MV] === undefined)
  .map(([name, _]) => name).sort();

/**
 * Flag values that should apply for new projects
 *
 * This contains flags that satisfy both criteria of:
 *
 * - They are configurable for the current major version line.
 * - The recommended value is different from the unconfigured value (i.e.,
 *   configuring a flag is useful)
 */
export const CURRENTLY_RECOMMENDED_FLAGS = Object.fromEntries(
  Object.entries(FLAGS)
    .filter(([_, flag]) => flag.introducedIn[CURRENT_MV] !== undefined)
    .filter(([_, flag]) => flag.recommendedValue !== flag.unconfiguredBehavesLike?.[CURRENT_MV])
    .map(([name, flag]) => [name, flag.recommendedValue]),
);

/**
 * The default values of each of these flags in the current major version.
 *
 * This is the effective value of the flag, unless it's overridden via
 * context.
 *
 * Adding new flags here is only allowed during the pre-release period of a new
 * major version!
 */
export const CURRENT_VERSION_FLAG_DEFAULTS = Object.fromEntries(Object.entries(FLAGS)
  .filter(([_, flag]) => flag.unconfiguredBehavesLike?.[CURRENT_MV] !== undefined)
  .map(([name, flag]) => [name, flag.unconfiguredBehavesLike?.[CURRENT_MV]]));

export function futureFlagDefault(flag: string): boolean {
  const value = CURRENT_VERSION_FLAG_DEFAULTS[flag] ?? false;
  if (typeof value !== 'boolean') {
    // This should never happen, if this error is thrown it's a bug

    throw new Error(`futureFlagDefault: default type of flag '${flag}' should be boolean, got '${typeof value}'`);
  }
  return value;
}

// Nobody should have been using any of this, but you never know

/** @deprecated use CURRENT_VERSION_EXPIRED_FLAGS instead */
export const FUTURE_FLAGS_EXPIRED = CURRENT_VERSION_EXPIRED_FLAGS;

/** @deprecated do not use at all! */
export const FUTURE_FLAGS = Object.fromEntries(Object.entries(CURRENTLY_RECOMMENDED_FLAGS)
  .filter(([_, v]) => typeof v === 'boolean'));

/** @deprecated do not use at all! */
export const NEW_PROJECT_DEFAULT_CONTEXT = Object.fromEntries(Object.entries(CURRENTLY_RECOMMENDED_FLAGS)
  .filter(([_, v]) => typeof v !== 'boolean'));
