import { EOL } from 'os';
import type { Construct, IConstruct } from 'constructs';
import { BucketGrants } from './bucket-grants';
import { BucketPolicy } from './bucket-policy';
import { BucketReflection } from './bucket-reflection';
import type { IBucketNotificationDestination } from './destination';
import { BucketAutoDeleteObjects } from './mixins';
import { BucketNotifications } from './notifications-resource';
import * as perms from './perms';
import type { LifecycleRule, StorageClass } from './rule';
import type { BucketReference, IBucketRef } from './s3.generated';
import { CfnBucket } from './s3.generated';
import { parseBucketArn, parseBucketName } from './util';
import * as events from '../../aws-events';
import * as iam from '../../aws-iam';
import type { GrantOnKeyResult, IEncryptedResource, IGrantable } from '../../aws-iam';
import * as kms from '../../aws-kms';
import type {
  Duration,
  IResource,
  ResourceProps,
} from '../../core';
import {
  Annotations,
  FeatureFlags,
  Fn,
  Lazy,
  PhysicalName,
  RemovalPolicy,
  Resource,
  Stack,
  Token,
  Tokenization,
  Validations,
} from '../../core';
import { UnscopedValidationError, ValidationError } from '../../core/lib/errors';
import type { IArrayBox, IBox } from '../../core/lib/helpers-internal';
import { Box, memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { CfnReference } from '../../core/lib/private/cfn-reference';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import * as cxapi from '../../cx-api';
import * as regionInformation from '../../region-info';

export interface IBucket extends IResource, IBucketRef {
  /**
   * The ARN of the bucket.
   * @attribute
   */
  readonly bucketArn: string;

  /**
   * The name of the bucket.
   * @attribute
   */
  readonly bucketName: string;

  /**
   * The URL of the static website.
   * @attribute
   */
  readonly bucketWebsiteUrl: string;

  /**
   * The Domain name of the static website.
   * @attribute
   */
  readonly bucketWebsiteDomainName: string;

  /**
   * The IPv4 DNS name of the specified bucket.
   * @attribute
   */
  readonly bucketDomainName: string;

  /**
   * The IPv6 DNS name of the specified bucket.
   * @attribute
   */
  readonly bucketDualStackDomainName: string;

  /**
   * The regional domain name of the specified bucket.
   * @attribute
   */
  readonly bucketRegionalDomainName: string;

  /**
   * If this bucket has been configured for static website hosting.
   */
  readonly isWebsite?: boolean;

  /**
   * Optional KMS encryption key associated with this bucket.
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * The resource policy associated with this bucket.
   *
   * If `autoCreatePolicy` is true, a `BucketPolicy` will be created upon the
   * first call to addToResourcePolicy(s).
   */
  policy?: BucketPolicy;

  /**
   * Role used to set up permissions on this bucket for replication
   */
  replicationRoleArn?: string;

  /**
   * Adds a statement to the resource policy for a principal (i.e.
   * account/role/service) to perform actions on this bucket and/or its
   * contents. Use `bucketArn` and `arnForObjects(keys)` to obtain ARNs for
   * this bucket or objects.
   *
   * Note that the policy statement may or may not be added to the policy.
   * For example, when an `IBucket` is created from an existing bucket,
   * it's not possible to tell whether the bucket already has a policy
   * attached, let alone to re-use that policy to add more statements to it.
   * So it's safest to do nothing in these cases.
   *
   * @param permission the policy statement to be added to the bucket's
   * policy.
   * @returns metadata about the execution of this method. If the policy
   * was not added, the value of `statementAdded` will be `false`. You
   * should always check this value to make sure that the operation was
   * actually carried out. Otherwise, synthesis and deploy will terminate
   * silently, which may be confusing.
   */
  addToResourcePolicy(permission: iam.PolicyStatement): iam.AddToResourcePolicyResult;

  /**
   * The https URL of an S3 object. For example:
   *
   * - `https://s3.us-west-1.amazonaws.com/onlybucket`
   * - `https://s3.us-west-1.amazonaws.com/bucket/key`
   * - `https://s3.cn-north-1.amazonaws.com.cn/china-bucket/mykey`
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @returns an ObjectS3Url token
   */
  urlForObject(key?: string): string;

  /**
   * The https Transfer Acceleration URL of an S3 object. Specify `dualStack: true` at the options
   * for dual-stack endpoint (connect to the bucket over IPv6). For example:
   *
   * - `https://bucket.s3-accelerate.amazonaws.com`
   * - `https://bucket.s3-accelerate.amazonaws.com/key`
   *
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @param options Options for generating URL.
   * @returns an TransferAccelerationUrl token
   */
  transferAccelerationUrlForObject(key?: string, options?: TransferAccelerationUrlOptions): string;

  /**
   * The virtual hosted-style URL of an S3 object. Specify `regional: false` at
   * the options for non-regional URL. For example:
   *
   * - `https://only-bucket.s3.us-west-1.amazonaws.com`
   * - `https://bucket.s3.us-west-1.amazonaws.com/key`
   * - `https://bucket.s3.amazonaws.com/key`
   * - `https://china-bucket.s3.cn-north-1.amazonaws.com.cn/mykey`
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @param options Options for generating URL.
   * @returns an ObjectS3Url token
   */
  virtualHostedUrlForObject(key?: string, options?: VirtualHostedStyleUrlOptions): string;

  /**
   * The S3 URL of an S3 object. For example:
   * - `s3://onlybucket`
   * - `s3://bucket/key`
   * @param key The S3 key of the object. If not specified, the S3 URL of the
   *      bucket is returned.
   * @returns an ObjectS3Url token
   */
  s3UrlForObject(key?: string): string;

  /**
   * Returns an ARN that represents all objects within the bucket that match
   * the key pattern specified. To represent all keys, specify ``"*"``.
   */
  arnForObjects(keyPattern: string): string;

  /**
   * Grant read permissions for this bucket and its contents to an IAM
   * principal (Role/Group/User).
   *
   * If encryption is used, permission to use the key to decrypt the contents
   * of the bucket will also be granted to the same principal.
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  grantRead(identity: iam.IGrantable, objectsKeyPattern?: any): iam.Grant;

  /**
   * Grant write permissions to this bucket to an IAM principal.
   *
   * If encryption is used, permission to use the key to encrypt the contents
   * of written files will also be granted to the same principal.
   *
   * Before CDK version 1.85.0, this method granted the `s3:PutObject*` permission that included `s3:PutObjectAcl`,
   * which could be used to grant read/write object access to IAM principals in other accounts.
   * If you want to get rid of that behavior, update your CDK version to 1.85.0 or later,
   * and make sure the `@aws-cdk/aws-s3:grantWriteWithoutAcl` feature flag is set to `true`
   * in the `context` key of your cdk.json file.
   * If you've already updated, but still need the principal to have permissions to modify the ACLs,
   * use the `grantPutAcl` method.
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   * @param allowedActionPatterns Restrict the permissions to certain list of action patterns
   */
  grantWrite(identity: iam.IGrantable, objectsKeyPattern?: any, allowedActionPatterns?: string[]): iam.Grant;

  /**
   * Grants s3:PutObject* and s3:Abort* permissions for this bucket to an IAM principal.
   *
   * If encryption is used, permission to use the key to encrypt the contents
   * of written files will also be granted to the same principal.
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  grantPut(identity: iam.IGrantable, objectsKeyPattern?: any): iam.Grant;

  /**
   * Grant the given IAM identity permissions to modify the ACLs of objects in the given Bucket.
   *
   * If your application has the '@aws-cdk/aws-s3:grantWriteWithoutAcl' feature flag set,
   * calling `grantWrite` or `grantReadWrite` no longer grants permissions to modify the ACLs of the objects;
   * in this case, if you need to modify object ACLs, call this method explicitly.
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*')
   */
  grantPutAcl(identity: iam.IGrantable, objectsKeyPattern?: string): iam.Grant;

  /**
   * Grants s3:DeleteObject* permission to an IAM principal for objects
   * in this bucket.
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  grantDelete(identity: iam.IGrantable, objectsKeyPattern?: any): iam.Grant;

  /**
   * Grants read/write permissions for this bucket and its contents to an IAM
   * principal (Role/Group/User).
   *
   * If an encryption key is used, permission to use the key for
   * encrypt/decrypt will also be granted.
   *
   * Before CDK version 1.85.0, this method granted the `s3:PutObject*` permission that included `s3:PutObjectAcl`,
   * which could be used to grant read/write object access to IAM principals in other accounts.
   * If you want to get rid of that behavior, update your CDK version to 1.85.0 or later,
   * and make sure the `@aws-cdk/aws-s3:grantWriteWithoutAcl` feature flag is set to `true`
   * in the `context` key of your cdk.json file.
   * If you've already updated, but still need the principal to have permissions to modify the ACLs,
   * use the `grantPutAcl` method.
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  grantReadWrite(identity: iam.IGrantable, objectsKeyPattern?: any): iam.Grant;

  /**
   * Allows permissions for replication operation to bucket replication role.
   *
   * If an encryption key is used, permission to use the key for
   * encrypt/decrypt will also be granted.
   *
   * @param identity The principal
   * @param props The properties of the replication source and destination buckets.
   * @returns The `iam.Grant` object, which represents the grant of permissions.
   */
  grantReplicationPermission(identity: iam.IGrantable, props: GrantReplicationPermissionProps): iam.Grant;

  /**
   * Allows unrestricted access to objects from this bucket.
   *
   * IMPORTANT: This permission allows anyone to perform actions on S3 objects
   * in this bucket, which is useful for when you configure your bucket as a
   * website and want everyone to be able to read objects in the bucket without
   * needing to authenticate.
   *
   * Without arguments, this method will grant read ("s3:GetObject") access to
   * all objects ("*") in the bucket.
   *
   * The method returns the `iam.Grant` object, which can then be modified
   * as needed. For example, you can add a condition that will restrict access only
   * to an IPv4 range like this:
   *
   *     const grant = bucket.grantPublicAccess();
   *     grant.resourceStatement!.addCondition(‘IpAddress’, { “aws:SourceIp”: “54.240.143.0/24” });
   *
   *
   * @param keyPrefix the prefix of S3 object keys (e.g. `home/*`). Default is "*".
   * @param allowedActions the set of S3 actions to allow. Default is "s3:GetObject".
   * @returns The `iam.PolicyStatement` object, which can be used to apply e.g. conditions.
   */
  grantPublicAccess(keyPrefix?: string, ...allowedActions: string[]): iam.Grant;

  /**
   * Defines a CloudWatch event that triggers when something happens to this bucket
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  onCloudTrailEvent(id: string, options?: OnCloudTrailBucketEventOptions): events.Rule;

  /**
   * Defines an AWS CloudWatch event that triggers when an object is uploaded
   * to the specified paths (keys) in this bucket using the PutObject API call.
   *
   * Note that some tools like `aws s3 cp` will automatically use either
   * PutObject or the multipart upload API depending on the file size,
   * so using `onCloudTrailWriteObject` may be preferable.
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  onCloudTrailPutObject(id: string, options?: OnCloudTrailBucketEventOptions): events.Rule;

  /**
   * Defines an AWS CloudWatch event that triggers when an object at the
   * specified paths (keys) in this bucket are written to.  This includes
   * the events PutObject, CopyObject, and CompleteMultipartUpload.
   *
   * Note that some tools like `aws s3 cp` will automatically use either
   * PutObject or the multipart upload API depending on the file size,
   * so using this method may be preferable to `onCloudTrailPutObject`.
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  onCloudTrailWriteObject(id: string, options?: OnCloudTrailBucketEventOptions): events.Rule;

  /**
   * Adds a bucket notification event destination.
   * @param event The event to trigger the notification
   * @param dest The notification destination (Lambda, SNS Topic or SQS Queue)
   *
   * @param filters S3 object key filter rules to determine which objects
   * trigger this event. Each filter must include a `prefix` and/or `suffix`
   * that will be matched against the s3 object key. Refer to the S3 Developer Guide
   * for details about allowed filter rules.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-filtering.html
   *
   * @example
   *
   *    declare const myLambda: lambda.Function;
   *    const bucket = new s3.Bucket(this, 'MyBucket');
   *    const filter: s3.NotificationKeyFilter = { prefix: 'home/myusername/*' };
   *    bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.LambdaDestination(myLambda), filter);
   *
   * @see
   * https://docs.aws.amazon.com/AmazonS3/latest/dev/NotificationHowTo.html
   */
  addEventNotification(event: EventType, dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]): void;

  /**
   * Subscribes a destination to receive notifications when an object is
   * created in the bucket. This is identical to calling
   * `onEvent(s3.EventType.OBJECT_CREATED)`.
   *
   * @param dest The notification destination (see onEvent)
   * @param filters Filters (see onEvent)
   */
  addObjectCreatedNotification(dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]): void;

  /**
   * Subscribes a destination to receive notifications when an object is
   * removed from the bucket. This is identical to calling
   * `onEvent(EventType.OBJECT_REMOVED)`.
   *
   * @param dest The notification destination (see onEvent)
   * @param filters Filters (see onEvent)
   */
  addObjectRemovedNotification(dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]): void;

  /**
   * Enables event bridge notification, causing all events below to be sent to EventBridge:
   *
   * - Object Created
   * - Object Deleted (DeleteObject)
   * - Object Deleted (Lifecycle expiration)
   * - Object Restore Initiated
   * - Object Restore Completed
   * - Object Restore Expired
   * - Object Storage Class Changed
   * - Object Access Tier Changed
   * - Object ACL Updated
   * - Object Tags Added
   * - Object Tags Deleted
   */
  enableEventBridgeNotification(): void;

  /**
   * Function to add required permissions to the destination bucket for cross account
   * replication. These permissions will be added as a resource based policy on the bucket.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-walkthrough-2.html
   * If owner of the bucket needs to be overridden, set accessControlTransition to true and provide
   * account ID in which destination bucket is hosted. For more information on accessControlTransition
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3-bucket-accesscontroltranslation.html
   */
  addReplicationPolicy(roleArn: string, accessControlTransition?: boolean, account?: string): void;
}

/**
 * A reference to a bucket outside this stack
 */
export interface BucketAttributes {
  /**
   * The ARN of the bucket. At least one of bucketArn or bucketName must be
   * defined in order to initialize a bucket ref.
   */
  readonly bucketArn?: string;

  /**
   * The name of the bucket. If the underlying value of ARN is a string, the
   * name will be parsed from the ARN. Otherwise, the name is optional, but
   * some features that require the bucket name such as auto-creating a bucket
   * policy, won't work.
   */
  readonly bucketName?: string;

  /**
   * The domain name of the bucket.
   *
   * @default - Inferred from bucket name
   */
  readonly bucketDomainName?: string;

  /**
   * The website URL of the bucket (if static web hosting is enabled).
   *
   * @default - Inferred from bucket name and region
   */
  readonly bucketWebsiteUrl?: string;

  /**
   * The regional domain name of the specified bucket.
   */
  readonly bucketRegionalDomainName?: string;

  /**
   * The IPv6 DNS name of the specified bucket.
   */
  readonly bucketDualStackDomainName?: string;

  /**
   * Force the format of the website URL of the bucket. This should be true for
   * regions launched since 2014.
   *
   * @default - inferred from available region information, `false` otherwise
   *
   * @deprecated The correct website url format can be inferred automatically from the bucket `region`.
   * Always provide the bucket region if the `bucketWebsiteUrl` will be used.
   * Alternatively provide the full `bucketWebsiteUrl` manually.
   */
  readonly bucketWebsiteNewUrlFormat?: boolean;

  /**
   * KMS encryption key associated with this bucket.
   *
   * @default - no encryption key
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * If this bucket has been configured for static website hosting.
   *
   * @default false
   */
  readonly isWebsite?: boolean;

  /**
   * The account this existing bucket belongs to.
   *
   * @default - it's assumed the bucket belongs to the same account as the scope it's being imported into
   */
  readonly account?: string;

  /**
   * The region this existing bucket is in.
   * Features that require the region (e.g. `bucketWebsiteUrl`) won't fully work
   * if the region cannot be correctly inferred.
   *
   * @default - it's assumed the bucket is in the same region as the scope it's being imported into
   */
  readonly region?: string;

  /**
   * The role to be used by the notifications handler
   *
   * @default - a new role will be created.
   */
  readonly notificationsHandlerRole?: iam.IRole;
}

/**
 * The properties for the destination bucket for granting replication permission.
 */
export interface GrantReplicationPermissionDestinationProps {
  /**
   * The destination bucket
   */
  readonly bucket: IBucket;

  /**
   * The KMS key to use for encryption if a destination bucket needs to be encrypted with a customer-managed KMS key.
   *
   * @default - no KMS key is used for replication.
   */
  readonly encryptionKey?: kms.IKey;
}

/**
 * The properties for the destination bucket for granting replication permission.
 */
export interface GrantReplicationPermissionProps {
  /**
   * The KMS key used to decrypt objects in the source bucket for replication.
   * **Required if** the source bucket is encrypted with a customer-managed KMS key.
   *
   * @default - it's assumed the source bucket is not encrypted with a customer-managed KMS key.
   */
  readonly sourceDecryptionKey?: kms.IKey;

  /**
   * The destination buckets for replication.
   * Specify the KMS key to use for encryption if a destination bucket needs to be encrypted with a customer-managed KMS key.
   * One or more destination buckets are required if replication configuration is enabled (i.e., `replicationRole` is specified).
   *
   * @default - empty array (valid only if the `replicationRole` property is NOT specified)
   */
  readonly destinations: GrantReplicationPermissionDestinationProps[];
}

/**
 * Represents an S3 Bucket.
 *
 * Buckets can be either defined within this stack:
 *
 *   new Bucket(this, 'MyBucket', { props });
 *
 * Or imported from an existing bucket:
 *
 *   Bucket.import(this, 'MyImportedBucket', { bucketArn: ... });
 *
 * You can also export a bucket and import it into another stack:
 *
 *   const ref = myBucket.export();
 *   Bucket.import(this, 'MyImportedBucket', ref);
 *
 */
export abstract class BucketBase extends Resource implements IBucket, IEncryptedResource {
  public abstract readonly bucketArn: string;
  public abstract readonly bucketName: string;
  public abstract readonly bucketDomainName: string;
  public abstract readonly bucketWebsiteUrl: string;
  public abstract readonly bucketWebsiteDomainName: string;
  public abstract readonly bucketRegionalDomainName: string;
  public abstract readonly bucketDualStackDomainName: string;

  /**
   * Optional KMS encryption key associated with this bucket.
   */
  public abstract readonly encryptionKey?: kms.IKey;

  /**
   * If this bucket has been configured for static website hosting.
   */
  public abstract readonly isWebsite?: boolean;

  /**
   * The resource policy associated with this bucket.
   *
   * If `autoCreatePolicy` is true, a `BucketPolicy` will be created upon the
   * first call to addToResourcePolicy(s).
   */
  public abstract policy?: BucketPolicy;

  /**
   * Role used to set up permissions on this bucket for replication
   */
  public abstract replicationRoleArn?: string;

  /**
   * Indicates if a bucket resource policy should automatically created upon
   * the first call to `addToResourcePolicy`.
   */
  protected abstract autoCreatePolicy: boolean;

  /**
   * Whether to disallow public access
   */
  public abstract disallowPublicAccess?: boolean;

  private notifications?: BucketNotifications;

  protected notificationsHandlerRole?: iam.IRole;

  protected notificationsSkipDestinationValidation?: boolean;

  protected objectOwnership?: ObjectOwnership;

  constructor(scope: Construct, id: string, props: ResourceProps = {}) {
    super(scope, id, props);

    this.node.addValidation({ validate: () => this.policy?.document.validateForResourcePolicy() ?? [] });
  }

  public grantOnKey(grantee: IGrantable, ...actions: string[]): GrantOnKeyResult {
    return {
      grant: this.encryptionKey?.grant(grantee, ...actions),
    };
  }

  /**
   * Collection of grant methods for a Bucket
   */
  @memoizedGetter
  public get grants() {
    return BucketGrants.fromBucket(this);
  }

  /**
   * Define a CloudWatch event that triggers when something happens to this repository
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  public onCloudTrailEvent(id: string, options: OnCloudTrailBucketEventOptions = {}): events.Rule {
    const rule = new events.Rule(this, id, options);
    rule.addTarget(options.target);
    rule.addEventPattern({
      source: ['aws.s3'],
      detailType: ['AWS API Call via CloudTrail'],
      detail: {
        resources: {
          ARN: options.paths?.map(p => this.arnForObjects(p)) ?? [this.bucketArn],
        },
      },
    });
    return rule;
  }

  /**
   * Defines an AWS CloudWatch event that triggers when an object is uploaded
   * to the specified paths (keys) in this bucket using the PutObject API call.
   *
   * Note that some tools like `aws s3 cp` will automatically use either
   * PutObject or the multipart upload API depending on the file size,
   * so using `onCloudTrailWriteObject` may be preferable.
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  public onCloudTrailPutObject(id: string, options: OnCloudTrailBucketEventOptions = {}): events.Rule {
    const rule = this.onCloudTrailEvent(id, options);
    rule.addEventPattern({
      detail: {
        eventName: ['PutObject'],
      },
    });
    return rule;
  }

  /**
   * Defines an AWS CloudWatch event that triggers when an object at the
   * specified paths (keys) in this bucket are written to.  This includes
   * the events PutObject, CopyObject, and CompleteMultipartUpload.
   *
   * Note that some tools like `aws s3 cp` will automatically use either
   * PutObject or the multipart upload API depending on the file size,
   * so using this method may be preferable to `onCloudTrailPutObject`.
   *
   * Requires that there exists at least one CloudTrail Trail in your account
   * that captures the event. This method will not create the Trail.
   *
   * @param id The id of the rule
   * @param options Options for adding the rule
   */
  public onCloudTrailWriteObject(id: string, options: OnCloudTrailBucketEventOptions = {}): events.Rule {
    const rule = this.onCloudTrailEvent(id, options);
    rule.addEventPattern({
      detail: {
        eventName: [
          'CompleteMultipartUpload',
          'CopyObject',
          'PutObject',
        ],
        requestParameters: {
          bucketName: [this.bucketName],
          key: options.paths,
        },
      },
    });
    return rule;
  }

  /**
   * Adds a statement to the resource policy for a principal (i.e.
   * account/role/service) to perform actions on this bucket and/or its
   * contents. Use `bucketArn` and `arnForObjects(keys)` to obtain ARNs for
   * this bucket or objects.
   *
   * Note that the policy statement may or may not be added to the policy.
   * For example, when an `IBucket` is created from an existing bucket,
   * it's not possible to tell whether the bucket already has a policy
   * attached, let alone to re-use that policy to add more statements to it.
   * So it's safest to do nothing in these cases.
   *
   * @param permission the policy statement to be added to the bucket's
   * policy.
   * @returns metadata about the execution of this method. If the policy
   * was not added, the value of `statementAdded` will be `false`. You
   * should always check this value to make sure that the operation was
   * actually carried out. Otherwise, synthesis and deploy will terminate
   * silently, which may be confusing.
   */
  public addToResourcePolicy(permission: iam.PolicyStatement): iam.AddToResourcePolicyResult {
    this.maybeAutoCreatePolicy();

    if (this.policy) {
      this.policy.document.addStatements(permission);
      return { statementAdded: true, policyDependable: this.policy };
    }

    return { statementAdded: false };
  }

  /**
   * Ensures a bucket policy exists on the L2 if `autoCreatePolicy` is set.
   */
  protected maybeAutoCreatePolicy() {
    if (!this.policy && this.autoCreatePolicy) {
      this.policy = new BucketPolicy(this, 'Policy', { bucket: this });
    }
  }

  /**
   * The https URL of an S3 object. Specify `regional: false` at the options
   * for non-regional URLs. For example:
   *
   * - `https://s3.us-west-1.amazonaws.com/onlybucket`
   * - `https://s3.us-west-1.amazonaws.com/bucket/key`
   * - `https://s3.cn-north-1.amazonaws.com.cn/china-bucket/mykey`
   *
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @returns an ObjectS3Url token
   */
  public urlForObject(key?: string): string {
    const stack = Stack.of(this);
    const prefix = `https://s3.${this.env.region}.${stack.urlSuffix}/`;
    if (typeof key !== 'string') {
      return this.urlJoin(prefix, this.bucketName);
    }
    return this.urlJoin(prefix, this.bucketName, key);
  }

  /**
   * The https Transfer Acceleration URL of an S3 object. Specify `dualStack: true` at the options
   * for dual-stack endpoint (connect to the bucket over IPv6). For example:
   *
   * - `https://bucket.s3-accelerate.amazonaws.com`
   * - `https://bucket.s3-accelerate.amazonaws.com/key`
   *
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @param options Options for generating URL.
   * @returns an TransferAccelerationUrl token
   */
  public transferAccelerationUrlForObject(key?: string, options?: TransferAccelerationUrlOptions): string {
    const dualStack = options?.dualStack ? '.dualstack' : '';
    const prefix = `https://${this.bucketName}.s3-accelerate${dualStack}.amazonaws.com/`;
    if (typeof key !== 'string') {
      return this.urlJoin(prefix);
    }
    return this.urlJoin(prefix, key);
  }

  /**
   * The virtual hosted-style URL of an S3 object. Specify `regional: false` at
   * the options for non-regional URL. For example:
   *
   * - `https://only-bucket.s3.us-west-1.amazonaws.com`
   * - `https://bucket.s3.us-west-1.amazonaws.com/key`
   * - `https://bucket.s3.amazonaws.com/key`
   * - `https://china-bucket.s3.cn-north-1.amazonaws.com.cn/mykey`
   *
   * @param key The S3 key of the object. If not specified, the URL of the
   *      bucket is returned.
   * @param options Options for generating URL.
   * @returns an ObjectS3Url token
   */
  public virtualHostedUrlForObject(key?: string, options?: VirtualHostedStyleUrlOptions): string {
    const domainName = options?.regional ?? true ? this.bucketRegionalDomainName : this.bucketDomainName;
    const prefix = `https://${domainName}`;
    if (typeof key !== 'string') {
      return prefix;
    }
    return this.urlJoin(prefix, key);
  }

  /**
   * The S3 URL of an S3 object. For example:
   *
   * - `s3://onlybucket`
   * - `s3://bucket/key`
   *
   * @param key The S3 key of the object. If not specified, the S3 URL of the
   *      bucket is returned.
   * @returns an ObjectS3Url token
   */
  public s3UrlForObject(key?: string): string {
    const prefix = 's3://';
    if (typeof key !== 'string') {
      return this.urlJoin(prefix, this.bucketName);
    }
    return this.urlJoin(prefix, this.bucketName, key);
  }

  /**
   * Returns an ARN that represents all objects within the bucket that match
   * the key pattern specified. To represent all keys, specify ``"*"``.
   *
   * If you need to specify a keyPattern with multiple components, concatenate them into a single string, e.g.:
   *
   *   arnForObjects(`home/${team}/${user}/*`)
   *
   */
  public arnForObjects(keyPattern: string): string {
    return perms.arnForObjects(this.bucketArn, keyPattern);
  }

  /**
   * Grant read permissions for this bucket and its contents to an IAM
   * principal (Role/Group/User).
   *
   * If encryption is used, permission to use the key to decrypt the contents
   * of the bucket will also be granted to the same principal.
   *
   *
   * The use of this method is discouraged. Please use `grants.read()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  public grantRead(identity: iam.IGrantable, objectsKeyPattern: any = '*') {
    return this.grants.read(identity, objectsKeyPattern);
  }

  /**
   *
   * The use of this method is discouraged. Please use `grants.write()` instead.
   *
   * [disable-awslint:no-grants]
   */
  public grantWrite(identity: iam.IGrantable, objectsKeyPattern: any = '*', allowedActionPatterns: string[] = []) {
    return this.grants.write(identity, objectsKeyPattern, allowedActionPatterns);
  }

  /**
   * Grants s3:PutObject* and s3:Abort* permissions for this bucket to an IAM principal.
   *
   * If encryption is used, permission to use the key to encrypt the contents
   * of written files will also be granted to the same principal.
   *
   *
   * The use of this method is discouraged. Please use `grants.put()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  public grantPut(identity: iam.IGrantable, objectsKeyPattern: any = '*') {
    return this.grants.put(identity, objectsKeyPattern);
  }

  /**
   *
   * The use of this method is discouraged. Please use `grants.putAcl()` instead.
   *
   * [disable-awslint:no-grants]
   */
  public grantPutAcl(identity: iam.IGrantable, objectsKeyPattern: string = '*') {
    return this.grants.putAcl(identity, objectsKeyPattern);
  }

  /**
   * Grants s3:DeleteObject* permission to an IAM principal for objects
   * in this bucket.
   *
   *
   * The use of this method is discouraged. Please use `grants.delete()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal
   * @param objectsKeyPattern Restrict the permission to a certain key pattern (default '*'). Parameter type is `any` but `string` should be passed in.
   */
  public grantDelete(identity: iam.IGrantable, objectsKeyPattern: any = '*') {
    return this.grants.delete(identity, objectsKeyPattern);
  }

  /**
   *
   * The use of this method is discouraged. Please use `grants.readWrite()` instead.
   *
   * [disable-awslint:no-grants]
   */
  public grantReadWrite(identity: iam.IGrantable, objectsKeyPattern: any = '*') {
    return this.grants.readWrite(identity, objectsKeyPattern);
  }

  /**
   * Grant replication permission to a principal.
   * This method allows the principal to perform replication operations on this bucket.
   *
   * Note that when calling this function for source or destination buckets that support KMS encryption,
   * you need to specify the KMS key for encryption and the KMS key for decryption, respectively.
   *
   *
   * The use of this method is discouraged. Please use `grants.replicationPermission()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param identity The principal to grant replication permission to.
   * @param props The properties of the replication source and destination buckets.
   */
  public grantReplicationPermission(identity: iam.IGrantable, props: GrantReplicationPermissionProps): iam.Grant {
    return this.grants.replicationPermission(identity, props);
  }

  /**
   * Allows unrestricted access to objects from this bucket.
   *
   * IMPORTANT: This permission allows anyone to perform actions on S3 objects
   * in this bucket, which is useful for when you configure your bucket as a
   * website and want everyone to be able to read objects in the bucket without
   * needing to authenticate.
   *
   * Without arguments, this method will grant read ("s3:GetObject") access to
   * all objects ("*") in the bucket.
   *
   * The method returns the `iam.Grant` object, which can then be modified
   * as needed. For example, you can add a condition that will restrict access only
   * to an IPv4 range like this:
   *
   *     const grant = bucket.grantPublicAccess();
   *     grant.resourceStatement!.addCondition(‘IpAddress’, { “aws:SourceIp”: “54.240.143.0/24” });
   *
   * Note that if this `IBucket` refers to an existing bucket, possibly not
   * managed by CloudFormation, this method will have no effect, since it's
   * impossible to modify the policy of an existing bucket.
   *
   *
   * The use of this method is discouraged. Please use `grants.publicAccess()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param keyPrefix the prefix of S3 object keys (e.g. `home/*`). Default is "*".
   * @param allowedActions the set of S3 actions to allow. Default is "s3:GetObject".
   */
  public grantPublicAccess(keyPrefix = '*', ...allowedActions: string[]) {
    return this.grants.publicAccess(keyPrefix, ...allowedActions);
  }

  /**
   * Adds a bucket notification event destination.
   * @param event The event to trigger the notification
   * @param dest The notification destination (Lambda, SNS Topic or SQS Queue)
   *
   * @param filters S3 object key filter rules to determine which objects
   * trigger this event. Each filter must include a `prefix` and/or `suffix`
   * that will be matched against the s3 object key. Refer to the S3 Developer Guide
   * for details about allowed filter rules.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-filtering.html
   *
   * @example
   *
   *    declare const myLambda: lambda.Function;
   *    const bucket = new s3.Bucket(this, 'MyBucket');
   *    const filter: s3.NotificationKeyFilter = { prefix: 'home/myusername/*' };
   *    bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.LambdaDestination(myLambda), filter);
   *
   * @see
   * https://docs.aws.amazon.com/AmazonS3/latest/dev/NotificationHowTo.html
   */
  public addEventNotification(event: EventType, dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]) {
    this.withNotifications(notifications => notifications.addNotification(event, dest, ...filters));
  }

  private withNotifications(cb: (notifications: BucketNotifications) => void) {
    if (!this.notifications) {
      this.notifications = new BucketNotifications(this, 'Notifications', {
        bucket: this,
        handlerRole: this.notificationsHandlerRole,
        skipDestinationValidation: this.notificationsSkipDestinationValidation ?? false,
      });
    }
    cb(this.notifications);
  }

  /**
   * Subscribes a destination to receive notifications when an object is
   * created in the bucket. This is identical to calling
   * `onEvent(EventType.OBJECT_CREATED)`.
   *
   * @param dest The notification destination (see onEvent)
   * @param filters Filters (see onEvent)
   */
  public addObjectCreatedNotification(dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]) {
    return this.addEventNotification(EventType.OBJECT_CREATED, dest, ...filters);
  }

  /**
   * Subscribes a destination to receive notifications when an object is
   * removed from the bucket. This is identical to calling
   * `onEvent(EventType.OBJECT_REMOVED)`.
   *
   * @param dest The notification destination (see onEvent)
   * @param filters Filters (see onEvent)
   */
  public addObjectRemovedNotification(dest: IBucketNotificationDestination, ...filters: NotificationKeyFilter[]) {
    return this.addEventNotification(EventType.OBJECT_REMOVED, dest, ...filters);
  }

  /**
   * Enables event bridge notification, causing all events below to be sent to EventBridge:
   *
   * - Object Deleted (DeleteObject)
   * - Object Deleted (Lifecycle expiration)
   * - Object Restore Initiated
   * - Object Restore Completed
   * - Object Restore Expired
   * - Object Storage Class Changed
   * - Object Access Tier Changed
   * - Object ACL Updated
   * - Object Tags Added
   * - Object Tags Deleted
   */
  public enableEventBridgeNotification() {
    this.withNotifications(notifications => notifications.enableEventBridgeNotification());
  }

  /**
   * Function to add required permissions to the destination bucket for cross account
   * replication. These permissions will be added as a resource based policy on the bucket
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-walkthrough-2.html
   * If owner of the bucket needs to be overridden, set accessControlTransition to true and provide
   * account ID in which destination bucket is hosted. For more information on accessControlTransition
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3-bucket-accesscontroltranslation.html
   */
  public addReplicationPolicy(roleArn: string, accessControlTransition?: boolean, account?: string) {
    const results: boolean[] = [];
    results.push(this.addToResourcePolicy(new iam.PolicyStatement({
      actions: ['s3:GetBucketVersioning', 's3:PutBucketVersioning'],
      resources: [this.bucketArn],
      principals: [new iam.ArnPrincipal(roleArn)],
    })).statementAdded);
    results.push(this.addToResourcePolicy(new iam.PolicyStatement({
      actions: ['s3:ReplicateObject', 's3:ReplicateDelete'],
      resources: [this.arnForObjects('*')],
      principals: [new iam.ArnPrincipal(roleArn)],
    })).statementAdded);
    if (accessControlTransition) {
      if (!account) {
        throw new ValidationError(lit`AccountSpecifiedOverrideOwnershipAccess`, 'account must be specified to override ownership access control transition', this);
      }
      results.push(this.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:ObjectOwnerOverrideToBucketOwner'],
        resources: [this.arnForObjects('*')],
        principals: [new iam.AccountPrincipal(account)],
      })).statementAdded);
    }

    if (results.includes(false)) {
      Annotations.of(this).addInfo(`Cross-account S3 replication for a referenced destination bucket is set up. In the destination bucket's bucket policy, please grant access permissions from ${this.stack.resolve(roleArn)}.`);
    }
  }

  private urlJoin(...components: string[]): string {
    return components.reduce((result, component) => {
      if (result.endsWith('/')) {
        result = result.slice(0, -1);
      }
      if (component.startsWith('/')) {
        component = component.slice(1);
      }
      return `${result}/${component}`;
    });
  }

  public get bucketRef(): BucketReference {
    return {
      bucketArn: this.bucketArn,
      bucketName: this.bucketName,
    };
  }
}

export interface BlockPublicAccessOptions {
  /**
   * Whether to block public ACLs
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html#access-control-block-public-access-options
   */
  readonly blockPublicAcls?: boolean;

  /**
   * Whether to block public policy
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html#access-control-block-public-access-options
   */
  readonly blockPublicPolicy?: boolean;

  /**
   * Whether to ignore public ACLs
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html#access-control-block-public-access-options
   */
  readonly ignorePublicAcls?: boolean;

  /**
   * Whether to restrict public access
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html#access-control-block-public-access-options
   */
  readonly restrictPublicBuckets?: boolean;
}

export class BlockPublicAccess {
  /**
   * Use this option if you want to ensure every public access method is blocked.
   * However keep in mind that this is the default state of an S3 bucket, and leaving blockPublicAccess undefined would also work.
   */
  public static readonly BLOCK_ALL = new BlockPublicAccess({
    blockPublicAcls: true,
    blockPublicPolicy: true,
    ignorePublicAcls: true,
    restrictPublicBuckets: true,
  });

  /**
   *
   * @deprecated Use `BLOCK_ACLS_ONLY` instead.
   */
  public static readonly BLOCK_ACLS = new BlockPublicAccess({
    blockPublicAcls: true,
    ignorePublicAcls: true,
  });

  /**
   * Use this option if you want to only block the ACLs, using this will set blockPublicPolicy and restrictPublicBuckets to false.
   */
  public static readonly BLOCK_ACLS_ONLY = new BlockPublicAccess({
    blockPublicAcls: true,
    blockPublicPolicy: false,
    ignorePublicAcls: true,
    restrictPublicBuckets: false,
  });

  public blockPublicAcls: boolean | undefined;
  public blockPublicPolicy: boolean | undefined;
  public ignorePublicAcls: boolean | undefined;
  public restrictPublicBuckets: boolean | undefined;

  constructor(options: BlockPublicAccessOptions) {
    this.blockPublicAcls = options.blockPublicAcls;
    this.blockPublicPolicy = options.blockPublicPolicy;
    this.ignorePublicAcls = options.ignorePublicAcls;
    this.restrictPublicBuckets = options.restrictPublicBuckets;
  }
}

/**
 * Specifies a metrics configuration for the CloudWatch request metrics from an Amazon S3 bucket.
 */
export interface BucketMetrics {
  /**
   * The ID used to identify the metrics configuration.
   */
  readonly id: string;
  /**
   * The prefix that an object must have to be included in the metrics results.
   */
  readonly prefix?: string;
  /**
   * Specifies a list of tag filters to use as a metrics configuration filter.
   * The metrics configuration includes only objects that meet the filter's criteria.
   */
  readonly tagFilters?: { [tag: string]: any };
}

/**
 * All http request methods
 */
export enum HttpMethods {
  /**
   * The GET method requests a representation of the specified resource.
   */
  GET = 'GET',
  /**
   * The PUT method replaces all current representations of the target resource with the request payload.
   */
  PUT = 'PUT',
  /**
   * The HEAD method asks for a response identical to that of a GET request, but without the response body.
   */
  HEAD = 'HEAD',
  /**
   * The POST method is used to submit an entity to the specified resource, often causing a change in state or side effects on the server.
   */
  POST = 'POST',
  /**
   * The DELETE method deletes the specified resource.
   */
  DELETE = 'DELETE',
}

/**
 * Specifies a cross-origin access rule for an Amazon S3 bucket.
 */
export interface CorsRule {
  /**
   * A unique identifier for this rule.
   *
   * @default - No id specified.
   */
  readonly id?: string;
  /**
   * The time in seconds that your browser is to cache the preflight response for the specified resource.
   *
   * @default - No caching.
   */
  readonly maxAge?: number;
  /**
   * Headers that are specified in the Access-Control-Request-Headers header.
   *
   * @default - No headers allowed.
   */
  readonly allowedHeaders?: string[];
  /**
   * An HTTP method that you allow the origin to execute.
   */
  readonly allowedMethods: HttpMethods[];
  /**
   * One or more origins you want customers to be able to access the bucket from.
   */
  readonly allowedOrigins: string[];
  /**
   * One or more headers in the response that you want customers to be able to access from their applications.
   *
   * @default - No headers exposed.
   */
  readonly exposedHeaders?: string[];
}

/**
 * All http request methods
 */
export enum RedirectProtocol {
  HTTP = 'http',
  HTTPS = 'https',
}

/**
 * Specifies a redirect behavior of all requests to a website endpoint of a bucket.
 */
export interface RedirectTarget {
  /**
   * Name of the host where requests are redirected
   */
  readonly hostName: string;

  /**
   * Protocol to use when redirecting requests
   *
   * @default - The protocol used in the original request.
   */
  readonly protocol?: RedirectProtocol;
}

/**
 * All supported inventory list formats.
 */
export enum InventoryFormat {
  /**
   * Generate the inventory list as CSV.
   */
  CSV = 'CSV',
  /**
   * Generate the inventory list as Parquet.
   */
  PARQUET = 'Parquet',
  /**
   * Generate the inventory list as ORC.
   */
  ORC = 'ORC',
}

/**
 * All supported inventory frequencies.
 */
export enum InventoryFrequency {
  /**
   * A report is generated every day.
   */
  DAILY = 'Daily',
  /**
   * A report is generated every Sunday (UTC timezone) after the initial report.
   */
  WEEKLY = 'Weekly',
}

/**
 * Inventory version support.
 */
export enum InventoryObjectVersion {
  /**
   * Includes all versions of each object in the report.
   */
  ALL = 'All',
  /**
   * Includes only the current version of each object in the report.
   */
  CURRENT = 'Current',
}

/**
 * The destination of the inventory.
 */
export interface InventoryDestination {
  /**
   * Bucket where all inventories will be saved in.
   */
  readonly bucket: IBucket;
  /**
   * The prefix to be used when saving the inventory.
   *
   * @default - No prefix.
   */
  readonly prefix?: string;
  /**
   * The account ID that owns the destination S3 bucket.
   * If no account ID is provided, the owner is not validated before exporting data.
   * It's recommended to set an account ID to prevent problems if the destination bucket ownership changes.
   *
   * @default - No account ID.
   */
  readonly bucketOwner?: string;
}

/**
 * Specifies the inventory configuration of an S3 Bucket.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/storage-inventory.html
 */
export interface Inventory {
  /**
   * The destination of the inventory.
   */
  readonly destination: InventoryDestination;
  /**
   * The inventory will only include objects that meet the prefix filter criteria.
   *
   * @default - No objects prefix
   */
  readonly objectsPrefix?: string;
  /**
   * The format of the inventory.
   *
   * @default InventoryFormat.CSV
   */
  readonly format?: InventoryFormat;
  /**
   * Whether the inventory is enabled or disabled.
   *
   * @default true
   */
  readonly enabled?: boolean;
  /**
   * The inventory configuration ID.
   * Should be limited to 64 characters and can only contain letters, numbers, periods, dashes, and underscores.
   *
   * @default - generated ID.
   */
  readonly inventoryId?: string;
  /**
   * Frequency at which the inventory should be generated.
   *
   * @default InventoryFrequency.WEEKLY
   */
  readonly frequency?: InventoryFrequency;
  /**
   * If the inventory should contain all the object versions or only the current one.
   *
   * @default InventoryObjectVersion.ALL
   */
  readonly includeObjectVersions?: InventoryObjectVersion;
  /**
   * A list of optional fields to be included in the inventory result.
   *
   * @default - No optional fields.
   */
  readonly optionalFields?: string[];
}
/**
 * The ObjectOwnership of the bucket.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/about-object-ownership.html
 *
 */
export enum ObjectOwnership {
  /**
   * ACLs are disabled, and the bucket owner automatically owns
   * and has full control over every object in the bucket.
   * ACLs no longer affect permissions to data in the S3 bucket.
   * The bucket uses policies to define access control.
   */
  BUCKET_OWNER_ENFORCED = 'BucketOwnerEnforced',
  /**
   * The bucket owner will own the object if the object is uploaded with
   * the bucket-owner-full-control canned ACL. Without this setting and
   * canned ACL, the object is uploaded and remains owned by the uploading account.
   */
  BUCKET_OWNER_PREFERRED = 'BucketOwnerPreferred',
  /**
   * The uploading account will own the object.
   */
  OBJECT_WRITER = 'ObjectWriter',
}
/**
 * The intelligent tiering configuration.
 */
export interface IntelligentTieringConfiguration {
  /**
   * Configuration name
   */
  readonly name: string;

  /**
   * Add a filter to limit the scope of this configuration to a single prefix.
   *
   * @default this configuration will apply to **all** objects in the bucket.
   */
  readonly prefix?: string;

  /**
   * You can limit the scope of this rule to the key value pairs added below.
   *
   * @default No filtering will be performed on tags
   */
  readonly tags?: Tag[];

  /**
   * When enabled, Intelligent-Tiering will automatically move objects that
   * haven’t been accessed for a minimum of 90 days to the Archive Access tier.
   *
   * @default Objects will not move to Glacier
   */
  readonly archiveAccessTierTime?: Duration;

  /**
   * When enabled, Intelligent-Tiering will automatically move objects that
   * haven’t been accessed for a minimum of 180 days to the Deep Archive Access
   * tier.
   *
   * @default Objects will not move to Glacier Deep Access
   */
  readonly deepArchiveAccessTierTime?: Duration;
}

/**
 * The date source for the partitioned prefix.
 */
export enum PartitionDateSource {
  /**
   * The year, month, and day will be based on the timestamp of the S3 event in the file that's been delivered.
   */
  EVENT_TIME = 'EventTime',

  /**
   * The year, month, and day will be based on the time when the log file was delivered to S3.
   */
  DELIVERY_TIME = 'DeliveryTime',
}

/**
 * The key format for the log object.
 */
export abstract class TargetObjectKeyFormat {
  /**
   * Use partitioned prefix for log objects.
   * If you do not specify the dateSource argument, the default is EventTime.
   *
   * The partitioned prefix format as follow:
   * [DestinationPrefix][SourceAccountId]/​[SourceRegion]/​[SourceBucket]/​[YYYY]/​[MM]/​[DD]/​[YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]-[UniqueString]
   */
  public static partitionedPrefix(dateSource?: PartitionDateSource): TargetObjectKeyFormat {
    return new class extends TargetObjectKeyFormat {
      public _render(): CfnBucket.LoggingConfigurationProperty['targetObjectKeyFormat'] {
        return {
          partitionedPrefix: {
            partitionDateSource: dateSource,
          },
        };
      }
    }();
  }

  /**
   * Use the simple prefix for log objects.
   *
   * The simple prefix format as follow:
   * [DestinationPrefix][YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]-[UniqueString]
   */
  public static simplePrefix(): TargetObjectKeyFormat {
    return new class extends TargetObjectKeyFormat {
      public _render(): CfnBucket.LoggingConfigurationProperty['targetObjectKeyFormat'] {
        return {
          simplePrefix: {},
        };
      }
    }();
  }

  /**
   * Render the log object key format.
   *
   * @internal
   */
  public abstract _render(): CfnBucket.LoggingConfigurationProperty['targetObjectKeyFormat'];
}

/**
 * The replication time value used for S3 Replication Time Control (S3 RTC).
 */
export class ReplicationTimeValue {
  /**
   * Fifteen minutes.
   */
  public static readonly FIFTEEN_MINUTES = new ReplicationTimeValue(15);

  /**
   * @param minutes the time in minutes
   */
  private constructor(public readonly minutes: number) {}
}

/**
 * Specifies which Amazon S3 objects to replicate and where to store the replicas.
 */
export interface ReplicationRule {
  /**
   * The destination bucket for the replicated objects.
   *
   * The destination can be either in the same AWS account or a cross account.
   *
   * If you want to configure cross-account replication,
   * the destination bucket must have a policy that allows the source bucket to replicate objects to it.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-walkthrough-2.html
   */
  readonly destination: IBucket;

  /**
   * Whether to want to change replica ownership to the AWS account that owns the destination bucket.
   *
   * This can only be specified if the source bucket and the destination bucket are not in the same AWS account.
   *
   * @default - The replicas are owned by same AWS account that owns the source object
   */
  readonly accessControlTransition?: boolean;

  /**
   * Specifying S3 Replication Time Control (S3 RTC),
   * including whether S3 RTC is enabled and the time when all objects and operations on objects must be replicated.
   *
   * @default - S3 Replication Time Control is not enabled
   */
  readonly replicationTimeControl?: ReplicationTimeValue;

  /**
   * A container specifying replication metrics-related settings enabling replication metrics and events.
   *
   * When a value is set, metrics will be output to indicate whether the replication took longer than the specified time.
   *
   * @default - Replication metrics are not enabled
   */
  readonly metrics?: ReplicationTimeValue;

  /**
   * The customer managed AWS KMS key stored in AWS Key Management Service (KMS) for the destination bucket.
   * Amazon S3 uses this key to encrypt replica objects.
   *
   * Amazon S3 only supports symmetric encryption KMS keys.
   *
   * @see https://docs.aws.amazon.com/kms/latest/developerguide/symmetric-asymmetric.html
   *
   * @default - Amazon S3 uses the AWS managed KMS key for encryption
   */
  readonly kmsKey?: kms.IKey;

  /**
   * The storage class to use when replicating objects, such as S3 Standard or reduced redundancy.
   *
   * @default - The storage class of the source object
   */
  readonly storageClass?: StorageClass;

  /**
   * Specifies whether Amazon S3 replicates objects created with server-side encryption using an AWS KMS key stored in AWS Key Management Service.
   *
   * @default false
   */
  readonly sseKmsEncryptedObjects?: boolean;

  /**
   * Specifies whether Amazon S3 replicates modifications on replicas.
   *
   * @default false
   */
  readonly replicaModifications?: boolean;

  /**
   * The priority indicates which rule has precedence whenever two or more replication rules conflict.
   *
   * Amazon S3 will attempt to replicate objects according to all replication rules.
   * However, if there are two or more rules with the same destination bucket,
   * then objects will be replicated according to the rule with the highest priority.
   *
   * The higher the number, the higher the priority.
   *
   * It is essential to specify priority explicitly when the replication configuration has multiple rules.
   *
   * @default 0
   */
  readonly priority?: number;

  /**
   * Specifies whether Amazon S3 replicates delete markers.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/delete-marker-replication.html
   *
   * @default - delete markers in source bucket is not replicated to destination bucket
   */
  readonly deleteMarkerReplication?: boolean;

  /**
   * A unique identifier for the rule.
   *
   * The maximum value is 255 characters.
   *
   * @default - auto generated random ID
   */
  readonly id?: string;

  /**
   * A filter that identifies the subset of objects to which the replication rule applies.
   *
   * @default - applies to all objects
   */
  readonly filter?: Filter;
}

/**
 * A filter that identifies the subset of objects to which the replication rule applies.
 */
export interface Filter {
  /**
   * An object key name prefix that identifies the object or objects to which the rule applies.
   *
   * @default - applies to all objects
   */
  readonly prefix?: string;

  /**
   * The tag array used for tag filters.
   *
   * The rule applies only to objects that have the tag in this set.
   *
   * @default - applies to all objects
   */
  readonly tags?: Tag[];
}

/**
 * The transition default minimum object size for lifecycle
 */
export enum TransitionDefaultMinimumObjectSize {
  /**
   * Objects smaller than 128 KB will not transition to any storage class by default.
   */
  ALL_STORAGE_CLASSES_128_K = 'all_storage_classes_128K',

  /**
   * Objects smaller than 128 KB will transition to Glacier Flexible Retrieval or Glacier
   * Deep Archive storage classes.
   *
   * By default, all other storage classes will prevent transitions smaller than 128 KB.
   */
  VARIES_BY_STORAGE_CLASS = 'varies_by_storage_class',
}

/**
 * The namespace for the bucket name when using `bucketNamePrefix`.
 *
 * Determines how CloudFormation generates the unique portion of the bucket name.
 */
export enum BucketNamespace {
  /**
   * The bucket name is globally unique.
   */
  GLOBAL = 'global',

  /**
   * The bucket name is unique within the account and region.
   */
  ACCOUNT_REGIONAL = 'account-regional',
}

export interface BucketProps {
  /**
   * The kind of server-side encryption to apply to this bucket.
   *
   * If you choose KMS, you can specify a KMS key via `encryptionKey`. If
   * encryption key is not specified, a key will automatically be created.
   *
   * @default - `KMS` if `encryptionKey` is specified, or `S3_MANAGED` otherwise.
   */
  readonly encryption?: BucketEncryption;

  /**
   * External KMS key to use for bucket encryption.
   *
   * The `encryption` property must be either not specified or set to `KMS` or `DSSE`.
   * An error will be emitted if `encryption` is set to `UNENCRYPTED` or `S3_MANAGED`.
   *
   * @default - If `encryption` is set to `KMS` and this property is undefined,
   * a new KMS key will be created and associated with this bucket.
   */
  readonly encryptionKey?: kms.IKey;

  /**
   * Encryption types that should be blocked for this bucket. Use `NONE` to allow all
   * encryption types.
   *
   * At least one `BlockedEncryptionType` must be given. If `NONE` is given, it must be
   * the only `BlockedEncryptionType` in the list.
   *
   * @default - Amazon S3 determines which encryption types to block.
   */
  readonly blockedEncryptionTypes?: BlockedEncryptionType[];

  /**
   * Enforces SSL for requests. S3.5 of the AWS Foundational Security Best Practices Regarding S3.
   * @see https://docs.aws.amazon.com/config/latest/developerguide/s3-bucket-ssl-requests-only.html
   *
   * @default false
   */
  readonly enforceSSL?: boolean;

  /**
   * Whether Amazon S3 should use its own intermediary key to generate data keys.
   *
   * Only relevant when using KMS for encryption.
   *
   * - If not enabled, every object GET and PUT will cause an API call to KMS (with the
   *   attendant cost implications of that).
   * - If enabled, S3 will use its own time-limited key instead.
   *
   * Only relevant, when Encryption is not set to `BucketEncryption.UNENCRYPTED`.
   *
   * @default - false
   */
  readonly bucketKeyEnabled?: boolean;

  /**
   * Physical name of this bucket.
   *
   * Cannot be used together with `bucketNamePrefix` or `bucketNamespace`.
   *
   * @default - Assigned by CloudFormation (recommended).
   */
  readonly bucketName?: string;

  /**
   * A prefix for the bucket name in the account-regional namespace.
   *
   * Requires `bucketNamespace` to be set to `ACCOUNT_REGIONAL`.
   * Cannot be used together with `bucketName`.
   *
   * CloudFormation appends `-<accountId>-<region>-an` to form the full name.
   * For example, `my-app` becomes `my-app-123456789012-us-east-1-an`.
   *
   * Must contain only lowercase letters, numbers, and hyphens.
   * Must start and end with a lowercase letter or number.
   *
   * @default - No prefix.
   */
  readonly bucketNamePrefix?: string;

  /**
   * The namespace for the bucket name.
   *
   * AWS recommends `ACCOUNT_REGIONAL` for improved security, as bucket names
   * are scoped to your account and cannot be claimed by other accounts.
   * When set to `ACCOUNT_REGIONAL`, `bucketNamePrefix` is required.
   * When set to `GLOBAL`, it can be used standalone to explicitly specify the default namespace.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/gpbucketnamespaces.html
   * @default - Global namespace.
   */
  readonly bucketNamespace?: BucketNamespace;

  /**
   * Policy to apply when the bucket is removed from this stack.
   *
   * @default - The bucket will be orphaned.
   */
  readonly removalPolicy?: RemovalPolicy;

  /**
   * Whether all objects should be automatically deleted when the bucket is
   * removed from the stack or when the stack is deleted.
   *
   * Requires the `removalPolicy` to be set to `RemovalPolicy.DESTROY`.
   *
   * **Warning** if you have deployed a bucket with `autoDeleteObjects: true`,
   * switching this to `false` in a CDK version *before* `1.126.0` will lead to
   * all objects in the bucket being deleted. Be sure to update your bucket resources
   * by deploying with CDK version `1.126.0` or later **before** switching this value to `false`.
   *
   * Setting `autoDeleteObjects` to true on a bucket will add `s3:PutBucketPolicy` to the
   * bucket policy. This is because during bucket deletion, the custom resource provider
   * needs to update the bucket policy by adding a deny policy for `s3:PutObject` to
   * prevent race conditions with external bucket writers.
   *
   * @default false
   */
  readonly autoDeleteObjects?: boolean;

  /**
   * Whether this bucket should have versioning turned on or not.
   *
   * @default false (unless object lock is enabled, then true)
   */
  readonly versioned?: boolean;

  /**
   * Enable object lock on the bucket.
   *
   * Enabling object lock for existing buckets is not supported. Object lock must be
   * enabled when the bucket is created.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-bucket-config-enable
   *
   * @default false, unless objectLockDefaultRetention is set (then, true)
   */
  readonly objectLockEnabled?: boolean;

  /**
   * Enables Amazon S3 to evaluate the ABAC policy in the request.
   * Set to true to enable ABAC, false to explicitly disable it.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-s3-bucket.html#cfn-s3-bucket-abacstatus
   *
   * @default - The ABAC status is not set
   */
  readonly abacStatus?: boolean;

  /**
   * The default retention mode and rules for S3 Object Lock.
   *
   * Default retention can be configured after a bucket is created if the bucket already
   * has object lock enabled. Enabling object lock for existing buckets is not supported.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-bucket-config-enable
   *
   * @default no default retention period
   */
  readonly objectLockDefaultRetention?: ObjectLockRetention;

  /**
   * Whether this bucket should send notifications to Amazon EventBridge or not.
   *
   * @default false
   */
  readonly eventBridgeEnabled?: boolean;

  /**
   * Rules that define how Amazon S3 manages objects during their lifetime.
   *
   * @default - No lifecycle rules.
   */
  readonly lifecycleRules?: LifecycleRule[];

  /**
   * Indicates which default minimum object size behavior is applied to the lifecycle configuration.
   *
   * To customize the minimum object size for any transition you can add a filter that specifies a custom
   * `objectSizeGreaterThan` or `objectSizeLessThan` for `lifecycleRules` property. Custom filters always
   * take precedence over the default transition behavior.
   *
   * @default - TransitionDefaultMinimumObjectSize.VARIES_BY_STORAGE_CLASS before September 2024,
   * otherwise TransitionDefaultMinimumObjectSize.ALL_STORAGE_CLASSES_128_K.
   */
  readonly transitionDefaultMinimumObjectSize?: TransitionDefaultMinimumObjectSize;

  /**
   * The name of the index document (e.g. "index.html") for the website. Enables static website
   * hosting for this bucket.
   *
   * @default - No index document.
   */
  readonly websiteIndexDocument?: string;

  /**
   * The name of the error document (e.g. "404.html") for the website.
   * `websiteIndexDocument` must also be set if this is set.
   *
   * @default - No error document.
   */
  readonly websiteErrorDocument?: string;

  /**
   * Specifies the redirect behavior of all requests to a website endpoint of a bucket.
   *
   * If you specify this property, you can't specify "websiteIndexDocument", "websiteErrorDocument" nor , "websiteRoutingRules".
   *
   * @default - No redirection.
   */
  readonly websiteRedirect?: RedirectTarget;

  /**
   * Rules that define when a redirect is applied and the redirect behavior
   *
   * @default - No redirection rules.
   */
  readonly websiteRoutingRules?: RoutingRule[];

  /**
   * Specifies a canned ACL that grants predefined permissions to the bucket.
   *
   * @default BucketAccessControl.PRIVATE
   */
  readonly accessControl?: BucketAccessControl;

  /**
   * Grants public read access to all objects in the bucket.
   * Similar to calling `bucket.grantPublicAccess()`
   *
   * @default false
   */
  readonly publicReadAccess?: boolean;

  /**
   * The block public access configuration of this bucket.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html
   *
   *
   * @default - CloudFormation defaults will apply. New buckets and objects don't allow public access, but users can modify bucket policies or object permissions to allow public access
   */
  readonly blockPublicAccess?: BlockPublicAccess;

  /**
   * The metrics configuration of this bucket.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3-bucket-metricsconfiguration.html
   *
   * @default - No metrics configuration.
   */
  readonly metrics?: BucketMetrics[];

  /**
   * The CORS configuration of this bucket.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-s3-bucket-cors.html
   *
   * @default - No CORS configuration.
   */
  readonly cors?: CorsRule[];

  /**
   * Destination bucket for the server access logs.
   * @default - If "serverAccessLogsPrefix" undefined - access logs disabled, otherwise - log to current bucket.
   */
  readonly serverAccessLogsBucket?: IBucket;

  /**
   * Optional log file prefix to use for the bucket's access logs.
   * If defined without "serverAccessLogsBucket", enables access logs to current bucket with this prefix.
   * @default - No log file prefix
   */
  readonly serverAccessLogsPrefix?: string;

  /**
   * Optional key format for log objects.
   *
   * @default - the default key format is: [DestinationPrefix][YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]-[UniqueString]
   */
  readonly targetObjectKeyFormat?: TargetObjectKeyFormat;

  /**
   * The inventory configuration of the bucket.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/storage-inventory.html
   *
   * @default - No inventory configuration
   */
  readonly inventories?: Inventory[];
  /**
   * The objectOwnership of the bucket.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/about-object-ownership.html
   *
   * @default - No ObjectOwnership configuration. By default, Amazon S3 sets Object Ownership to `Bucket owner enforced`.
   * This means ACLs are disabled and the bucket owner will own every object.
   *
   */
  readonly objectOwnership?: ObjectOwnership;

  /**
   * Whether this bucket should have transfer acceleration turned on or not.
   *
   * @default false
   */
  readonly transferAcceleration?: boolean;

  /**
   * The role to be used by the notifications handler
   *
   * @default - a new role will be created.
   */
  readonly notificationsHandlerRole?: iam.IRole;

  /**
   * Skips notification validation of Amazon SQS, Amazon SNS, and Lambda destinations.
   *
   * @default false
   */
  readonly notificationsSkipDestinationValidation?: boolean;

  /**
   * Intelligent Tiering Configurations
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/intelligent-tiering.html
   *
   * @default No Intelligent Tiering Configurations.
   */
  readonly intelligentTieringConfigurations?: IntelligentTieringConfiguration[];

  /**
   * Enforces minimum TLS version for requests.
   *
   * Requires `enforceSSL` to be enabled.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/amazon-s3-policy-keys.html#example-object-tls-version
   *
   * @default No minimum TLS version is enforced.
   */
  readonly minimumTLSVersion?: number;

  /**
   * The role to be used by the replication.
   *
   * When setting this property, you must also set `replicationRules`.
   *
   * @default - a new role will be created.
   */
  readonly replicationRole?: iam.IRole;

  /**
   * A container for one or more replication rules.
   *
   * @default - No replication
   */
  readonly replicationRules?: ReplicationRule[];
}

/**
 * Tag
 */
export interface Tag {

  /**
   * key to e tagged
   */
  readonly key: string;
  /**
   * additional value
   */
  readonly value: string;
}

/**
 * An S3 bucket with associated policy objects
 *
 * This bucket does not yet have all features that exposed by the underlying
 * BucketResource.
 *
 * @example
 * import { RemovalPolicy } from 'aws-cdk-lib';
 *
 * new s3.Bucket(scope, 'Bucket', {
 *   blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
 *   encryption: s3.BucketEncryption.S3_MANAGED,
 *   enforceSSL: true,
 *   versioned: true,
 *   removalPolicy: RemovalPolicy.RETAIN,
 * });
 *
 */
@propertyInjectable
@noBoxStackTraces
export class Bucket extends BucketBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-s3.Bucket';

  public static fromBucketArn(scope: Construct, id: string, bucketArn: string): IBucket {
    return Bucket.fromBucketAttributes(scope, id, { bucketArn });
  }

  public static fromBucketName(scope: Construct, id: string, bucketName: string): IBucket {
    return Bucket.fromBucketAttributes(scope, id, { bucketName });
  }

  /**
   * Creates a Bucket construct that represents an external bucket.
   *
   * @param scope The parent creating construct (usually `this`).
   * @param id The construct's name.
   * @param attrs A `BucketAttributes` object. Can be obtained from a call to
   * `bucket.export()` or manually created.
   */
  public static fromBucketAttributes(scope: Construct, id: string, attrs: BucketAttributes): IBucket {
    const stack = Stack.of(scope);
    const region = attrs.region ?? stack.region;
    const regionInfo = regionInformation.RegionInfo.get(region);
    const urlSuffix = regionInfo.domainSuffix ?? stack.urlSuffix;

    const bucketName = parseBucketName(scope, attrs);
    if (!bucketName) {
      throw new ValidationError(lit`BucketNameRequired`, 'Bucket name is required', scope);
    }

    const oldEndpoint = `s3-website-${region}.${urlSuffix}`;
    const newEndpoint = `s3-website.${region}.${urlSuffix}`;

    let staticDomainEndpoint = regionInfo.s3StaticWebsiteEndpoint
        ?? Lazy.string({ produce: () => stack.regionalFact(regionInformation.FactName.S3_STATIC_WEBSITE_ENDPOINT, newEndpoint) });

    // Deprecated use of bucketWebsiteNewUrlFormat
    if (attrs.bucketWebsiteNewUrlFormat !== undefined) {
      staticDomainEndpoint = attrs.bucketWebsiteNewUrlFormat ? newEndpoint : oldEndpoint;
    }

    const websiteDomain = `${bucketName}.${staticDomainEndpoint}`;

    const ret = new ReferencedBucket(scope, id, {
      account: attrs.account,
      region: attrs.region,
      bucketName: bucketName!,
      bucketArn: parseBucketArn(scope, attrs),
      bucketDomainName: attrs.bucketDomainName || `${bucketName}.s3.${urlSuffix}`,
      bucketWebsiteUrl: attrs.bucketWebsiteUrl || `http://${websiteDomain}`,
      bucketWebsiteDomainName: attrs.bucketWebsiteUrl ? Fn.select(2, Fn.split('/', attrs.bucketWebsiteUrl)) : websiteDomain,
      bucketRegionalDomainName: attrs.bucketRegionalDomainName || `${bucketName}.s3.${region}.${urlSuffix}`,
      bucketDualStackDomainName: attrs.bucketDualStackDomainName || `${bucketName}.s3.dualstack.${region}.${urlSuffix}`,
      encryptionKey: attrs.encryptionKey,
      isWebsite: attrs.isWebsite ?? false,
      autoCreatePolicy: false,
      disallowPublicAccess: false,
      notificationsHandlerRole: attrs.notificationsHandlerRole,
    });
    Bucket.validateBucketNameScoped(ret, bucketName, true);
    return ret;
  }

  /**
   * Create a mutable `IBucket` based on a low-level `CfnBucket`.
   */
  public static fromCfnBucket(cfnBucket: CfnBucket): IBucket {
    // use a "weird" id that has a higher chance of being unique
    const id = '@FromCfnBucket';

    // if fromCfnBucket() was already called on this cfnBucket,
    // return the same L2
    // (as different L2s would conflict, because of the mutation of the policy property of the L1 below)
    const existing = cfnBucket.node.tryFindChild(id);
    if (existing) {
      return <IBucket>existing;
    }

    // handle the KMS Key if the Bucket references one
    let encryptionKey: kms.IKey | undefined;
    if (cfnBucket.bucketEncryption) {
      const serverSideEncryptionConfiguration = (cfnBucket.bucketEncryption as any).serverSideEncryptionConfiguration;
      if (Array.isArray(serverSideEncryptionConfiguration) && serverSideEncryptionConfiguration.length === 1) {
        const serverSideEncryptionRuleProperty = serverSideEncryptionConfiguration[0];
        const serverSideEncryptionByDefault = serverSideEncryptionRuleProperty.serverSideEncryptionByDefault;
        if (serverSideEncryptionByDefault && Token.isUnresolved(serverSideEncryptionByDefault.kmsMasterKeyId)) {
          const kmsIResolvable = Tokenization.reverse(serverSideEncryptionByDefault.kmsMasterKeyId);
          if (kmsIResolvable instanceof CfnReference) {
            const cfnElement = kmsIResolvable.target;
            if (cfnElement instanceof kms.CfnKey) {
              encryptionKey = kms.Key.fromCfnKey(cfnElement);
            }
          }
        }
      }
    }

    return new class extends BucketBase {
      public readonly bucketArn = cfnBucket.attrArn;
      public readonly bucketName = cfnBucket.ref;
      public readonly bucketDomainName = cfnBucket.attrDomainName;
      public readonly bucketDualStackDomainName = cfnBucket.attrDualStackDomainName;
      public readonly bucketRegionalDomainName = cfnBucket.attrRegionalDomainName;
      public readonly bucketWebsiteUrl = cfnBucket.attrWebsiteUrl;

      public readonly encryptionKey = encryptionKey;
      public policy = undefined;
      public replicationRoleArn = undefined;
      protected autoCreatePolicy = true;

      private readonly reflection: BucketReflection;

      constructor() {
        super(cfnBucket, id);

        this.node.defaultChild = cfnBucket;
        this.reflection = BucketReflection.of(this);
      }

      public get bucketWebsiteDomainName() {
        return this.reflection.bucketWebsiteDomainName;
      }

      public get isWebsite(): boolean | undefined {
        return this.reflection.isWebsite;
      }

      public get disallowPublicAccess(): boolean | undefined {
        return this.reflection.disallowPublicAccess;
      }
      public set disallowPublicAccess(_value: boolean | undefined) {
        // Ignored — value is derived from the CfnBucket resource via reflection
      }
    }();
  }

  /**
   * Thrown an exception if the given bucket name is not valid.
   *
   * @param physicalName name of the bucket.
   * @param allowLegacyBucketNaming allow legacy bucket naming style, default is false.
   */
  public static validateBucketName(physicalName: string, allowLegacyBucketNaming: boolean = false): void {
    const errors = Bucket._validateBucketName(physicalName, allowLegacyBucketNaming);
    if (errors.length > 0) {
      // Since this is public and can be called statically, we have no object instance, so we throw an unscoped error.
      throw new UnscopedValidationError(lit`InvalidBucketNameValue`, `Invalid S3 bucket name (value: ${physicalName})${EOL}${errors.join(EOL)}`);
    }
  }

  /**
   * Maximum length for the bucket name prefix in the account-regional namespace.
   *
   * The account-regional suffix format is `-<accountId(12)>-<region>-an`.
   * For the shortest region codes (9 chars, e.g. `us-east-1`), the suffix
   * is 26 chars, leaving 37 chars for the prefix within the 63-char bucket
   * name limit. Longer region codes (e.g. `ap-southeast-1`) will have a
   * shorter effective limit enforced by CloudFormation at deploy time.
   */
  private static readonly MAX_BUCKET_NAME_PREFIX_LENGTH = 37;

  /**
   * Return any errors against the bucket name
   */
  private static _validateBucketName(physicalName: string, allowLegacyBucketNaming: boolean = false): string[] {
    const bucketName = physicalName;
    if (!bucketName || Token.isUnresolved(bucketName)) {
      // the name is a late-bound value, not a defined string,
      // so skip validation
      return [];
    }

    const errors: string[] = [];

    // Rules codified from https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
    if (bucketName.length < 3 || bucketName.length > 63) {
      errors.push('Bucket name must be at least 3 and no more than 63 characters');
    }

    const illegalCharsetRegEx = allowLegacyBucketNaming ? /[^A-Za-z0-9._-]/ : /[^a-z0-9.-]/;
    const allowedEdgeCharsetRegEx = allowLegacyBucketNaming ? /[A-Za-z0-9]/ : /[a-z0-9]/;

    const illegalCharMatch = bucketName.match(illegalCharsetRegEx);
    if (illegalCharMatch) {
      errors.push(allowLegacyBucketNaming
        ? 'Bucket name must only contain uppercase or lowercase characters and the symbols, period (.), underscore (_), and dash (-)'
        : 'Bucket name must only contain lowercase characters and the symbols, period (.) and dash (-)'
          + ` (offset: ${illegalCharMatch.index})`,
      );
    }
    if (!allowedEdgeCharsetRegEx.test(bucketName.charAt(0))) {
      errors.push(allowLegacyBucketNaming
        ? 'Bucket name must start with an uppercase, lowercase character or number'
        : 'Bucket name must start with a lowercase character or number'
          + ' (offset: 0)',
      );
    }
    if (!allowedEdgeCharsetRegEx.test(bucketName.charAt(bucketName.length - 1))) {
      errors.push(allowLegacyBucketNaming
        ? 'Bucket name must end with an uppercase, lowercase character or number'
        : 'Bucket name must end with a lowercase character or number'
          + ` (offset: ${bucketName.length - 1})`,
      );
    }

    const consecSymbolMatch = bucketName.match(/\.-|-\.|\.\./);
    if (consecSymbolMatch) {
      errors.push('Bucket name must not have dash next to period, or period next to dash, or consecutive periods'
          + ` (offset: ${consecSymbolMatch.index})`);
    }
    if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(bucketName)) {
      errors.push('Bucket name must not resemble an IP address');
    }

    return errors;
  }

  /**
   * Like 'validateBucketName', but has an instance to throw a scoped ValidationError against
   */
  private static validateBucketNameScoped(scope: IConstruct, physicalName: string, allowLegacyBucketNaming: boolean = false): void {
    const errors = Bucket._validateBucketName(physicalName, allowLegacyBucketNaming);
    if (errors.length > 0) {
      throw new ValidationError(lit`InvalidBucketNameValue`, `Invalid S3 bucket name (value: ${physicalName})${EOL}${errors.join(EOL)}`, scope);
    }
  }

  /**
   * Return any errors against the bucket name prefix.
   */
  private static _validateBucketNamePrefix(prefix?: string): string[] {
    if (!prefix || Token.isUnresolved(prefix)) {
      return [];
    }

    const errors: string[] = [];

    if (prefix.length > Bucket.MAX_BUCKET_NAME_PREFIX_LENGTH) {
      errors.push(`Bucket name prefix must be ${Bucket.MAX_BUCKET_NAME_PREFIX_LENGTH} characters or fewer`);
    }
    if (!/^[a-z0-9]/.test(prefix)) {
      errors.push('Bucket name prefix must start with a lowercase letter or number');
    }
    if (!/[a-z0-9]$/.test(prefix)) {
      errors.push('Bucket name prefix must end with a lowercase letter or number');
    }
    if (/[^a-z0-9-]/.test(prefix)) {
      errors.push('Bucket name prefix must only contain lowercase letters, numbers, and hyphens');
    }

    return errors;
  }

  /**
   * Like '_validateBucketNamePrefix', but throws a scoped ValidationError
   */
  private static validateBucketNamePrefixScoped(scope: IConstruct, prefix?: string): void {
    const errors = Bucket._validateBucketNamePrefix(prefix);
    if (errors.length > 0) {
      throw new ValidationError(lit`InvalidBucketNamePrefixValue`, `Invalid S3 bucket name prefix (value: ${prefix})${EOL}${errors.join(EOL)}`, scope);
    }
  }

  private static validateBucketNamingCombination(scope: IConstruct, props: BucketProps): void {
    if (props.bucketName && props.bucketNamePrefix) {
      throw new ValidationError(
        lit`BucketNameConflictsWithPrefix`,
        '\'bucketName\' and \'bucketNamePrefix\' cannot be used together',
        scope,
      );
    }
    if (props.bucketName && props.bucketNamespace && props.bucketNamespace !== BucketNamespace.GLOBAL) {
      throw new ValidationError(
        lit`BucketNameConflictsWithNamespace`,
        '\'bucketName\' cannot be used with \'bucketNamespace\' (except GLOBAL). Use \'bucketNamePrefix\' with \'bucketNamespace\' instead',
        scope,
      );
    }
    if (props.bucketNamePrefix && props.bucketNamespace !== BucketNamespace.ACCOUNT_REGIONAL) {
      throw new ValidationError(
        lit`BucketNamePrefixRequiresAccountRegional`,
        '\'bucketNamePrefix\' requires \'bucketNamespace\' to be set to ACCOUNT_REGIONAL',
        scope,
      );
    }
    if (props.bucketNamespace === BucketNamespace.ACCOUNT_REGIONAL && !props.bucketNamePrefix) {
      throw new ValidationError(
        lit`AccountRegionalNamespaceRequiresPrefix`,
        '\'bucketNamespace\' ACCOUNT_REGIONAL requires \'bucketNamePrefix\' to be specified',
        scope,
      );
    }
  }

  @memoizedGetter
  public get bucketArn(): string {
    return this.getResourceArnAttribute(this._resource.attrArn, {
      region: '',
      account: '',
      service: 's3',
      resource: this.physicalName,
    });
  }

  @memoizedGetter
  public get bucketName(): string {
    return this.getResourceNameAttribute(this._resource.ref);
  }
  public readonly bucketDomainName: string;
  public readonly bucketWebsiteUrl: string;
  public get bucketWebsiteDomainName(): string {
    return this.reflection.bucketWebsiteDomainName;
  }
  public readonly bucketDualStackDomainName: string;
  public readonly bucketRegionalDomainName: string;

  public readonly encryptionKey?: kms.IKey;
  public get isWebsite(): boolean | undefined {
    return this.reflection.isWebsite;
  }
  public policy?: BucketPolicy;

  public replicationRoleArn?: string;
  protected autoCreatePolicy = true;
  public get disallowPublicAccess(): boolean | undefined {
    return this.reflection.disallowPublicAccess || undefined;
  }
  public set disallowPublicAccess(_value: boolean | undefined) {
    // Ignored — value is derived from the CfnBucket resource via reflection
  }
  private readonly accessControl: IBox<BucketAccessControl | undefined>;
  private readonly ownershipControls: IBox<CfnBucket.OwnershipControlsProperty | undefined>;
  private readonly lifecycleRules: IArrayBox<LifecycleRule> = Box.fromArray();
  private readonly transitionDefaultMinimumObjectSize?: TransitionDefaultMinimumObjectSize;
  private readonly eventBridgeEnabled?: boolean;
  private readonly metrics: IArrayBox<BucketMetrics> = Box.fromArray();
  private readonly cors: IArrayBox<CorsRule> = Box.fromArray();
  private readonly inventories: IArrayBox<Inventory> = Box.fromArray();
  private readonly _resource: CfnBucket;
  private readonly reflection: BucketReflection;
  private _suppressedTypeCheck = false;

  constructor(scope: Construct, id: string, props: BucketProps = {}) {
    super(scope, id, {
      physicalName: props.bucketName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.notificationsHandlerRole = props.notificationsHandlerRole;
    this.notificationsSkipDestinationValidation = props.notificationsSkipDestinationValidation;

    const { bucketEncryption, encryptionKey } = this.parseEncryption(props);
    this.encryptionKey = encryptionKey;

    Bucket.validateBucketNamingCombination(this, props);
    Bucket.validateBucketNameScoped(this, this.physicalName);
    Bucket.validateBucketNamePrefixScoped(this, props.bucketNamePrefix);

    let publicAccessBlockConfig: BlockPublicAccessOptions | undefined = props.blockPublicAccess;
    if (props.blockPublicAccess && FeatureFlags.of(this).isEnabled(cxapi.S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT)) {
      publicAccessBlockConfig = this.setDefaultPublicAccessBlockConfig(props.blockPublicAccess);
    }

    const websiteConfiguration = this.renderWebsiteConfiguration(props);

    const objectLockConfiguration = this.parseObjectLockConfig(props);
    const replicationConfiguration = this.renderReplicationConfiguration(props);
    this.replicationRoleArn = replicationConfiguration?.role;
    this.objectOwnership = props.objectOwnership;
    this.transitionDefaultMinimumObjectSize = props.transitionDefaultMinimumObjectSize;
    this.accessControl = Box.fromValue<BucketAccessControl | undefined>(props.accessControl);
    this.ownershipControls = Box.fromValue<CfnBucket.OwnershipControlsProperty | undefined>(
      this.parseOwnershipControls(props.accessControl),
    );

    this.acknowledgeAccessControl();

    const resource = new CfnBucket(this, 'Resource', {
      bucketName: this.physicalName,
      bucketNamePrefix: props.bucketNamePrefix,
      bucketNamespace: props.bucketNamespace,
      bucketEncryption,
      versioningConfiguration: props.versioned ? { status: 'Enabled' } : undefined,
      lifecycleConfiguration: this.lifecycleRules.derive(rules => this.parseLifecycleConfiguration(rules)),
      websiteConfiguration,
      publicAccessBlockConfiguration: publicAccessBlockConfig,
      metricsConfigurations: this.metrics.derive(m => this.parseMetricConfiguration(m)),
      corsConfiguration: this.cors.derive(c => this.parseCorsConfiguration(c)),
      // eslint-disable-next-line @cdklabs/no-unconditional-token-allocation
      accessControl: Token.asString(this.accessControl),
      loggingConfiguration: this.parseServerAccessLogs(props),
      inventoryConfigurations: this.inventories.derive(inv => this.parseInventoryConfiguration(inv)),
      ownershipControls: this.ownershipControls,
      accelerateConfiguration: props.transferAcceleration ? { accelerationStatus: 'Enabled' } : undefined,
      intelligentTieringConfigurations: this.parseTieringConfig(props),
      objectLockEnabled: objectLockConfiguration ? true : props.objectLockEnabled,
      objectLockConfiguration: objectLockConfiguration,
      replicationConfiguration,
      abacStatus: props.abacStatus !== undefined ? (props.abacStatus ? 'Enabled' : 'Disabled') : undefined,
    });
    this._resource = resource;
    this.reflection = BucketReflection.of(this);

    resource.applyRemovalPolicy(props.removalPolicy);

    this.eventBridgeEnabled = props.eventBridgeEnabled;

    this.bucketDomainName = resource.attrDomainName;
    this.bucketWebsiteUrl = resource.attrWebsiteUrl;
    this.bucketDualStackDomainName = resource.attrDualStackDomainName;
    this.bucketRegionalDomainName = resource.attrRegionalDomainName;

    // Enforce AWS Foundational Security Best Practice
    if (props.enforceSSL) {
      this.enforceSSLStatement();
      this.minimumTLSVersionStatement(props.minimumTLSVersion);
    } else if (props.minimumTLSVersion) {
      throw new ValidationError(lit`EnforceEnabledMinimumVersionApplied`, '\'enforceSSL\' must be enabled for \'minimumTLSVersion\' to be applied', this);
    }

    if (props.serverAccessLogsBucket instanceof Bucket) {
      props.serverAccessLogsBucket.allowLogDelivery(this, props.serverAccessLogsPrefix);
      // It is possible that `serverAccessLogsBucket` was specified but is some other `IBucket`
      // that cannot have the ACLs or bucket policy applied. In that scenario, we should only
      // setup log delivery permissions to `this` if a bucket was not specified at all, as documented.
      // For example, we should not allow log delivery to `this` if given an imported bucket or
      // another situation that causes `instanceof` to fail
    } else if (!props.serverAccessLogsBucket && props.serverAccessLogsPrefix) {
      this.allowLogDelivery(this, props.serverAccessLogsPrefix);
    } else if (props.serverAccessLogsBucket) {
      // A `serverAccessLogsBucket` was provided but it is not a concrete `Bucket` and it
      // may not be possible to configure the ACLs or bucket policy as required.
      Annotations.of(this).addWarningV2('@aws-cdk/aws-s3:accessLogsPolicyNotAdded',
        `Unable to add necessary logging permissions to imported target bucket: ${props.serverAccessLogsBucket}`,
      );
    }

    for (const inventory of props.inventories ?? []) {
      this.addInventory(inventory);
    }

    // Add all bucket metric configurations rules
    (props.metrics || []).forEach(this.addMetric.bind(this));
    // Add all cors configuration rules
    (props.cors || []).forEach(this.addCorsRule.bind(this));

    // Add all lifecycle rules
    (props.lifecycleRules || []).forEach(this.addLifecycleRule.bind(this));

    if (props.publicReadAccess) {
      if (props.blockPublicAccess === undefined) {
        throw new ValidationError(lit`CannotPublicReadAccessProperty`, 'Cannot use \'publicReadAccess\' property on a bucket without allowing bucket-level public access through \'blockPublicAccess\' property.', this);
      }

      this.grantPublicAccess();
    }

    if (props.autoDeleteObjects) {
      if (props.removalPolicy !== RemovalPolicy.DESTROY) {
        throw new ValidationError(lit`CannotAutoDeleteObjectsProperty`, 'Cannot use \'autoDeleteObjects\' property on a bucket without setting removal policy to \'DESTROY\'.', this);
      }

      this.enableAutoDeleteObjects();
    }

    if (this.eventBridgeEnabled) {
      this.enableEventBridgeNotification();
    }
  }

  /**
   * If we are using the accessControl property (for historical reasons), silence the warning about it.
   */
  private acknowledgeAccessControl() {
    if (this.accessControl.get() !== undefined) {
      Validations.of(this).acknowledge({
        id: 'CloudFormation-Validate::W3045',
        reason: 'accessControl is deprecated, but we are still using it for historical reasons.',
      });
    }
  }

  /**
   * Add a lifecycle rule to the bucket
   *
   * @param rule The rule to add
   */
  @MethodMetadata()
  public addLifecycleRule(rule: LifecycleRule) {
    this.lifecycleRules.push(rule);

    if ((rule.objectSizeLessThan !== undefined || rule.objectSizeLessThan !== undefined) && !this._suppressedTypeCheck) {
      // These are typed as numbers by CDK, but as strings by CloudFormation. The validation plugin is going to complain
      // about the type mismatch, so suppress the warning for this construct if it's applicable.
      Validations.of(this).acknowledge({
        id: 'CloudFormation-Validate::W9003',
        reason: 'LifecycleRule.objectSizeLessThan and LifecycleRule.objectSizeGreaterThan are numbers for historical reasons',
      });
      this._suppressedTypeCheck = true;
    }
  }

  /**
   * Adds a metrics configuration for the CloudWatch request metrics from the bucket.
   *
   * @param metric The metric configuration to add
   */
  @MethodMetadata()
  public addMetric(metric: BucketMetrics) {
    this.metrics.push(metric);
  }

  /**
   * Adds a cross-origin access configuration for objects in an Amazon S3 bucket
   *
   * @param rule The CORS configuration rule to add
   */
  @MethodMetadata()
  public addCorsRule(rule: CorsRule) {
    this.cors.push(rule);
  }

  /**
   * Add an inventory configuration.
   *
   * @param inventory configuration to add
   */
  @MethodMetadata()
  public addInventory(inventory: Inventory): void {
    this.inventories.push(inventory);
  }

  /**
   * Adds an iam statement to enforce SSL requests only.
   */
  private enforceSSLStatement() {
    const statement = new iam.PolicyStatement({
      actions: ['s3:*'],
      conditions: {
        Bool: { 'aws:SecureTransport': 'false' },
      },
      effect: iam.Effect.DENY,
      resources: [
        this.bucketArn,
        this.arnForObjects('*'),
      ],
      principals: [new iam.AnyPrincipal()],
    });
    this.addToResourcePolicy(statement);
  }

  /**
   * Adds an iam statement to allow requests with a minimum TLS
   * version only.
   */
  private minimumTLSVersionStatement(minimumTLSVersion?: number) {
    if (!minimumTLSVersion) return;
    const statement = new iam.PolicyStatement({
      actions: ['s3:*'],
      conditions: {
        NumericLessThan: { 's3:TlsVersion': minimumTLSVersion },
      },
      effect: iam.Effect.DENY,
      resources: [
        this.bucketArn,
        this.arnForObjects('*'),
      ],
      principals: [new iam.AnyPrincipal()],
    });
    this.addToResourcePolicy(statement);
  }

  /**
   * Set up key properties and return the Bucket encryption property from the
   * user's configuration, according to the following table:
   *
   * | props.encryption | props.encryptionKey | props.bucketKeyEnabled | bucketEncryption (return value) | encryptionKey (return value) |
   * |------------------|---------------------|------------------------|---------------------------------|------------------------------|
   * | undefined        | undefined           | e                      | undefined                       | undefined                    |
   * | UNENCRYPTED      | undefined           | false                  | undefined                       | undefined                    |
   * | undefined        | k                   | e                      | SSE-KMS, bucketKeyEnabled = e   | k                            |
   * | KMS              | k                   | e                      | SSE-KMS, bucketKeyEnabled = e   | k                            |
   * | KMS              | undefined           | e                      | SSE-KMS, bucketKeyEnabled = e   | new key                      |
   * | KMS_MANAGED      | undefined           | e                      | SSE-KMS, bucketKeyEnabled = e   | undefined                    |
   * | S3_MANAGED       | undefined           | false                  | SSE-S3                          | undefined                    |
   * | S3_MANAGED       | undefined           | e                      | SSE-S3, bucketKeyEnabled = e    | undefined                    |
   * | UNENCRYPTED      | undefined           | true                   | ERROR!                          | ERROR!                       |
   * | UNENCRYPTED      | k                   | e                      | ERROR!                          | ERROR!                       |
   * | KMS_MANAGED      | k                   | e                      | ERROR!                          | ERROR!                       |
   * | S3_MANAGED       | undefined           | true                   | ERROR!                          | ERROR!                       |
   * | S3_MANAGED       | k                   | e                      | ERROR!                          | ERROR!                       |
   */
  private parseEncryption(props: BucketProps): {
    bucketEncryption?: CfnBucket.BucketEncryptionProperty;
    encryptionKey?: kms.IKey;
  } {
    // default based on whether encryptionKey is specified
    let encryptionType = props.encryption;
    if (encryptionType === undefined) {
      encryptionType = props.encryptionKey ? BucketEncryption.KMS : BucketEncryption.UNENCRYPTED;
    }

    // if encryption key is set, encryption must be set to KMS or DSSE.
    if (encryptionType !== BucketEncryption.DSSE && encryptionType !== BucketEncryption.KMS && props.encryptionKey) {
      throw new ValidationError(lit`EncryptionkeySpecified`, `encryptionKey is specified, so 'encryption' must be set to KMS or DSSE (value: ${encryptionType})`, this);
    }

    // if bucketKeyEnabled is set, encryption cannot be BucketEncryption.UNENCRYPTED
    if (props.bucketKeyEnabled && encryptionType === BucketEncryption.UNENCRYPTED) {
      throw new ValidationError(lit`BucketKeyEnabledSpecifiedEncryption`, `bucketKeyEnabled is specified, so 'encryption' must be set to KMS, DSSE or S3 (value: ${encryptionType})`, this);
    }

    let blockedEncryptionTypes: CfnBucket.BlockedEncryptionTypesProperty | undefined;
    if (props.blockedEncryptionTypes === undefined) {
      blockedEncryptionTypes = undefined;
    } else if (props.blockedEncryptionTypes.length === 0) {
      throw new ValidationError(lit`EmptyBlockedEncryptionTypes`, 'At least one blocked encryption type must be specified', this);
    } else {
      const typeNames = props.blockedEncryptionTypes.map(type => type.name);
      if (typeNames.includes(BlockedEncryptionType.NONE.name) && props.blockedEncryptionTypes.length > 1) {
        throw new ValidationError(lit`ConflictingBlockedEncryptionTypes`, 'If NONE is specified as the blocked encryption type, no other encryption types may be specified', this);
      }
      blockedEncryptionTypes = {
        encryptionType: typeNames,
      };
    }

    if (encryptionType === BucketEncryption.UNENCRYPTED) {
      if (blockedEncryptionTypes === undefined) {
        return { bucketEncryption: undefined, encryptionKey: undefined };
      } else {
        return {
          bucketEncryption: {
            serverSideEncryptionConfiguration: [{
              blockedEncryptionTypes,
            }],
          },
          encryptionKey: undefined,
        };
      }
    }

    if (encryptionType === BucketEncryption.KMS) {
      const encryptionKey = props.encryptionKey || new kms.Key(this, 'Key', {
        description: `Created by ${this.node.path}`,
        enableKeyRotation: true,
      });

      const bucketEncryption = {
        serverSideEncryptionConfiguration: [
          {
            bucketKeyEnabled: props.bucketKeyEnabled,
            serverSideEncryptionByDefault: {
              sseAlgorithm: 'aws:kms',
              kmsMasterKeyId: encryptionKey.keyArn,
            },
            blockedEncryptionTypes,
          },
        ],
      };
      return { encryptionKey, bucketEncryption };
    }

    if (encryptionType === BucketEncryption.S3_MANAGED) {
      const bucketEncryption = {
        serverSideEncryptionConfiguration: [
          {
            bucketKeyEnabled: props.bucketKeyEnabled,
            serverSideEncryptionByDefault: { sseAlgorithm: 'AES256' },
            blockedEncryptionTypes,
          },
        ],
      };

      return { bucketEncryption };
    }

    if (encryptionType === BucketEncryption.KMS_MANAGED) {
      const bucketEncryption = {
        serverSideEncryptionConfiguration: [
          {
            bucketKeyEnabled: props.bucketKeyEnabled,
            serverSideEncryptionByDefault: { sseAlgorithm: 'aws:kms' },
            blockedEncryptionTypes,
          },
        ],
      };
      return { bucketEncryption };
    }

    if (encryptionType === BucketEncryption.DSSE) {
      const encryptionKey = props.encryptionKey || new kms.Key(this, 'Key', {
        description: `Created by ${this.node.path}`,
      });

      const bucketEncryption = {
        serverSideEncryptionConfiguration: [
          {
            bucketKeyEnabled: props.bucketKeyEnabled,
            serverSideEncryptionByDefault: {
              sseAlgorithm: 'aws:kms:dsse',
              kmsMasterKeyId: encryptionKey.keyArn,
            },
            blockedEncryptionTypes,
          },
        ],
      };
      return { encryptionKey, bucketEncryption };
    }

    if (encryptionType === BucketEncryption.DSSE_MANAGED) {
      const bucketEncryption = {
        serverSideEncryptionConfiguration: [
          {
            bucketKeyEnabled: props.bucketKeyEnabled,
            serverSideEncryptionByDefault: { sseAlgorithm: 'aws:kms:dsse' },
            blockedEncryptionTypes,
          },
        ],
      };
      return { bucketEncryption };
    }

    throw new ValidationError(lit`UnexpectedEncryptionType`, `Unexpected 'encryptionType': ${encryptionType}`, this);
  }

  /**
   * Parse the lifecycle configuration out of the bucket props
   * @param props Par
   */
  private parseLifecycleConfiguration(lifecycleRules: readonly LifecycleRule[]): CfnBucket.LifecycleConfigurationProperty | undefined {
    if (!lifecycleRules || lifecycleRules.length === 0) {
      return undefined;
    }

    const self = this;

    return {
      rules: lifecycleRules.map(parseLifecycleRule),
      transitionDefaultMinimumObjectSize: this.transitionDefaultMinimumObjectSize,
    };

    function parseLifecycleRule(rule: LifecycleRule): CfnBucket.RuleProperty {
      const enabled = rule.enabled ?? true;
      if ((rule.expiredObjectDeleteMarker)
          && (rule.expiration || rule.expirationDate || self.parseTagFilters(rule.tagFilters))) {
        // ExpiredObjectDeleteMarker cannot be specified with ExpirationInDays, ExpirationDate, or TagFilters.
        throw new ValidationError(lit`ExpiredObjectDeleteMarkerCannot`, 'ExpiredObjectDeleteMarker cannot be specified with expiration, ExpirationDate, or TagFilters.', self);
      }

      if (
        rule.abortIncompleteMultipartUploadAfter === undefined &&
          rule.expiration === undefined &&
          rule.expirationDate === undefined &&
          rule.expiredObjectDeleteMarker === undefined &&
          rule.noncurrentVersionExpiration === undefined &&
          rule.noncurrentVersionsToRetain === undefined &&
          rule.noncurrentVersionTransitions === undefined &&
          rule.transitions === undefined
      ) {
        throw new ValidationError(lit`RulesLeastFollowingProperties`, 'All rules for `lifecycleRules` must have at least one of the following properties: `abortIncompleteMultipartUploadAfter`, `expiration`, `expirationDate`, `expiredObjectDeleteMarker`, `noncurrentVersionExpiration`, `noncurrentVersionsToRetain`, `noncurrentVersionTransitions`, or `transitions`', self);
      }

      // Validate transitions: exactly one of transitionDate or transitionAfter must be specified
      if (rule.transitions) {
        for (const transition of rule.transitions) {
          const hasTransitionDate = transition.transitionDate !== undefined;
          const hasTransitionAfter = transition.transitionAfter !== undefined;

          if (!hasTransitionDate && !hasTransitionAfter) {
            throw new ValidationError(lit`ExactlyOneTransitionDateTransition`, 'Exactly one of transitionDate or transitionAfter must be specified in lifecycle rule transition', self);
          }

          if (hasTransitionDate && hasTransitionAfter) {
            throw new ValidationError(lit`ExactlyOneTransitionDateTransition`, 'Exactly one of transitionDate or transitionAfter must be specified in lifecycle rule transition', self);
          }
        }
      }

      const x: CfnBucket.RuleProperty = {
        // eslint-disable-next-line max-len
        abortIncompleteMultipartUpload: rule.abortIncompleteMultipartUploadAfter !== undefined ? { daysAfterInitiation: rule.abortIncompleteMultipartUploadAfter.toDays() } : undefined,
        expirationDate: rule.expirationDate,
        expirationInDays: rule.expiration?.toDays(),
        id: rule.id,
        noncurrentVersionExpiration: rule.noncurrentVersionExpiration && {
          noncurrentDays: rule.noncurrentVersionExpiration.toDays(),
          newerNoncurrentVersions: rule.noncurrentVersionsToRetain,
        },
        noncurrentVersionTransitions: mapOrUndefined(rule.noncurrentVersionTransitions, t => ({
          storageClass: t.storageClass.value,
          transitionInDays: t.transitionAfter.toDays(),
          newerNoncurrentVersions: t.noncurrentVersionsToRetain,
        })),
        prefix: rule.prefix,
        status: enabled ? 'Enabled' : 'Disabled',
        transitions: mapOrUndefined(rule.transitions, t => ({
          storageClass: t.storageClass.value,
          transitionDate: t.transitionDate,
          transitionInDays: t.transitionAfter && t.transitionAfter.toDays(),
        })),
        expiredObjectDeleteMarker: rule.expiredObjectDeleteMarker,
        tagFilters: self.parseTagFilters(rule.tagFilters),
        objectSizeLessThan: rule.objectSizeLessThan,
        objectSizeGreaterThan: rule.objectSizeGreaterThan,
      };

      return x;
    }
  }

  private parseServerAccessLogs(props: BucketProps): CfnBucket.LoggingConfigurationProperty | undefined {
    if (!props.serverAccessLogsBucket && !props.serverAccessLogsPrefix) {
      return undefined;
    }

    // KMS_MANAGED can't be used for logging since the account can't access the logging service key - account can't read logs
    if (
      !props.serverAccessLogsBucket &&
        props.encryption &&
        [BucketEncryption.KMS_MANAGED, BucketEncryption.DSSE_MANAGED].includes(props.encryption)
    ) {
      throw new ValidationError(lit`KmsManagedKeyNotSupportedForAccessLogs`, 'Default bucket encryption with KMS managed or DSSE managed key is not supported for Server Access Logging target buckets', this);
    }

    // When there is an encryption key exists for the server access logs bucket, grant permission to the S3 logging SP.
    if (props.serverAccessLogsBucket?.encryptionKey) {
      props.serverAccessLogsBucket.encryptionKey.grantEncryptDecrypt(new iam.ServicePrincipal('logging.s3.amazonaws.com'));
    }

    return {
      destinationBucketName: props.serverAccessLogsBucket?.bucketName,
      logFilePrefix: props.serverAccessLogsPrefix,
      targetObjectKeyFormat: props.targetObjectKeyFormat?._render(),
    };
  }

  private parseMetricConfiguration(metrics: readonly BucketMetrics[]): CfnBucket.MetricsConfigurationProperty[] | undefined {
    if (!metrics || metrics.length === 0) {
      return undefined;
    }

    const self = this;

    return metrics.map(parseMetric);

    function parseMetric(metric: BucketMetrics): CfnBucket.MetricsConfigurationProperty {
      return {
        id: metric.id,
        prefix: metric.prefix,
        tagFilters: self.parseTagFilters(metric.tagFilters),
      };
    }
  }

  private parseCorsConfiguration(cors: readonly CorsRule[]): CfnBucket.CorsConfigurationProperty | undefined {
    if (!cors || cors.length === 0) {
      return undefined;
    }

    return { corsRules: cors.map(parseCors) };

    function parseCors(rule: CorsRule): CfnBucket.CorsRuleProperty {
      return {
        id: rule.id,
        maxAge: rule.maxAge,
        allowedHeaders: rule.allowedHeaders,
        allowedMethods: rule.allowedMethods,
        allowedOrigins: rule.allowedOrigins,
        exposedHeaders: rule.exposedHeaders,
      };
    }
  }

  private parseTagFilters(tagFilters?: { [tag: string]: any }) {
    if (!tagFilters || tagFilters.length === 0) {
      return undefined;
    }

    return Object.keys(tagFilters).map(tag => ({
      key: tag,
      value: tagFilters[tag],
    }));
  }

  private parseOwnershipControls(accessControl: BucketAccessControl | undefined): CfnBucket.OwnershipControlsProperty | undefined {
    // Enabling an ACL explicitly is required for all new buckets.
    // https://aws.amazon.com/about-aws/whats-new/2022/12/amazon-s3-automatically-enable-block-public-access-disable-access-control-lists-buckets-april-2023/
    const aclsThatDoNotRequireObjectOwnership = [
      BucketAccessControl.PRIVATE,
      BucketAccessControl.BUCKET_OWNER_READ,
      BucketAccessControl.BUCKET_OWNER_FULL_CONTROL,
    ];
    const accessControlRequiresObjectOwnership = (accessControl && !aclsThatDoNotRequireObjectOwnership.includes(accessControl));
    if (!this.objectOwnership && !accessControlRequiresObjectOwnership) {
      return undefined;
    }

    if (accessControlRequiresObjectOwnership && this.objectOwnership === ObjectOwnership.BUCKET_OWNER_ENFORCED) {
      throw new ValidationError(lit`MustBeObjectownershipAccesscontrol`, `objectOwnership must be set to "${ObjectOwnership.OBJECT_WRITER}" when accessControl is "${accessControl}"`, this);
    }

    return {
      rules: [{
        objectOwnership: this.objectOwnership ?? ObjectOwnership.OBJECT_WRITER,
      }],
    };
  }

  private parseTieringConfig({ intelligentTieringConfigurations }: BucketProps): CfnBucket.IntelligentTieringConfigurationProperty[] | undefined {
    if (!intelligentTieringConfigurations) {
      return undefined;
    }

    return intelligentTieringConfigurations.map(config => {
      const tierings = [];
      if (config.archiveAccessTierTime) {
        tierings.push({
          accessTier: 'ARCHIVE_ACCESS',
          days: config.archiveAccessTierTime.toDays({ integral: true }),
        });
      }
      if (config.deepArchiveAccessTierTime) {
        tierings.push({
          accessTier: 'DEEP_ARCHIVE_ACCESS',
          days: config.deepArchiveAccessTierTime.toDays({ integral: true }),
        });
      }
      return {
        id: config.name,
        prefix: config.prefix,
        status: 'Enabled',
        tagFilters: config.tags,
        tierings: tierings,
      };
    });
  }

  private parseObjectLockConfig(props: BucketProps): CfnBucket.ObjectLockConfigurationProperty | undefined {
    const { objectLockEnabled, objectLockDefaultRetention } = props;

    if (!objectLockDefaultRetention) {
      return undefined;
    }
    if (objectLockEnabled === false && objectLockDefaultRetention) {
      throw new ValidationError(lit`ObjectLockEnabledConfigureDefault`, 'Object Lock must be enabled to configure default retention settings', this);
    }

    return {
      objectLockEnabled: 'Enabled',
      rule: {
        defaultRetention: {
          days: objectLockDefaultRetention.duration.toDays(),
          mode: objectLockDefaultRetention.mode,
        },
      },
    };
  }

  private renderWebsiteConfiguration(props: BucketProps): CfnBucket.WebsiteConfigurationProperty | undefined {
    if (!props.websiteErrorDocument && !props.websiteIndexDocument && !props.websiteRedirect && !props.websiteRoutingRules) {
      return undefined;
    }

    if (props.websiteErrorDocument && !props.websiteIndexDocument) {
      throw new ValidationError(lit`WebsiteIndexDocumentRequiredWebsite`, '"websiteIndexDocument" is required if "websiteErrorDocument" is set', this);
    }

    if (props.websiteRedirect && (props.websiteErrorDocument || props.websiteIndexDocument || props.websiteRoutingRules)) {
      throw new ValidationError(lit`WebsiteIndexDocumentWebsiteError`, '"websiteIndexDocument", "websiteErrorDocument" and, "websiteRoutingRules" cannot be set if "websiteRedirect" is used', this);
    }

    const routingRules = props.websiteRoutingRules ? props.websiteRoutingRules.map<CfnBucket.RoutingRuleProperty>((rule) => {
      if (rule.condition && rule.condition.httpErrorCodeReturnedEquals == null && rule.condition.keyPrefixEquals == null) {
        throw new ValidationError(lit`ConditionPropertyCannotEmptyObject`, 'The condition property cannot be an empty object', this);
      }

      return {
        redirectRule: {
          hostName: rule.hostName,
          httpRedirectCode: rule.httpRedirectCode,
          protocol: rule.protocol,
          replaceKeyWith: rule.replaceKey && rule.replaceKey.withKey,
          replaceKeyPrefixWith: rule.replaceKey && rule.replaceKey.prefixWithKey,
        },
        routingRuleCondition: rule.condition,
      };
    }) : undefined;

    return {
      indexDocument: props.websiteIndexDocument,
      errorDocument: props.websiteErrorDocument,
      redirectAllRequestsTo: props.websiteRedirect,
      routingRules,
    };
  }

  private renderReplicationConfiguration(props: BucketProps): CfnBucket.ReplicationConfigurationProperty | undefined {
    const replicationRulesIsEmpty = !props.replicationRules || props.replicationRules.length === 0;
    if (replicationRulesIsEmpty && props.replicationRole) {
      throw new ValidationError(lit`CannotSpecifyReplicationRoleReplication`, 'cannot specify replicationRole when replicationRules is empty', this);
    }
    if (replicationRulesIsEmpty) {
      return undefined;
    }

    if (!props.versioned) {
      throw new ValidationError(lit`ReplicationRulesRequireVersioningEnabled`, 'Replication rules require versioning to be enabled on the bucket', this);
    }
    if (props.replicationRules.length > 1 && props.replicationRules.some(rule => rule.priority === undefined)) {
      throw new ValidationError(lit`MustBePrioritySpecifiedReplication`, '\'priority\' must be specified for all replication rules when there are multiple rules', this);
    }
    props.replicationRules.forEach(rule => {
      if (rule.replicationTimeControl && !rule.metrics) {
        throw new ValidationError(lit`ReplicationTimeControlMetricsEnabled`, '\'replicationTimeControlMetrics\' must be enabled when \'replicationTimeControl\' is enabled.', this);
      }
      if (rule.deleteMarkerReplication && rule.filter?.tags) {
        throw new ValidationError(lit`TagFilterCannotSpecifiedDelete`, 'tag filter cannot be specified when \'deleteMarkerReplication\' is enabled.', this);
      }
    });

    let replicationRole: iam.IRole;
    if (!props.replicationRole) {
      replicationRole = new iam.Role(this, 'ReplicationRole', {
        assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
        roleName: FeatureFlags.of(this).isEnabled(cxapi.SET_UNIQUE_REPLICATION_ROLE_NAME) ? PhysicalName.GENERATE_IF_NEEDED : 'CDKReplicationRole',
      });

      this.grantReplicationPermission(replicationRole, {
        sourceDecryptionKey: props.encryptionKey,
        destinations: props.replicationRules.map(rule => ({
          encryptionKey: rule.kmsKey,
          bucket: rule.destination,
        })),
      });
    } else {
      replicationRole = props.replicationRole;
    }

    return {
      role: replicationRole.roleArn,
      rules: props.replicationRules.map((rule) => {
        const sourceSelectionCriteria = (rule.replicaModifications !== undefined || rule.sseKmsEncryptedObjects !== undefined) ? {
          replicaModifications: rule.replicaModifications !== undefined ? {
            status: rule.replicaModifications ? 'Enabled' : 'Disabled',
          } : undefined,
          sseKmsEncryptedObjects: rule.sseKmsEncryptedObjects !== undefined ? {
            status: rule.sseKmsEncryptedObjects ? 'Enabled' : 'Disabled',
          } : undefined,
        } : undefined;

        // Whether to configure filter settings by And property.
        const isAndFilter = rule.filter?.tags && rule.filter.tags.length > 0;
        // To avoid deploy error when there are multiple replication rules with undefined prefix,
        // CDK set the prefix to an empty string if it is undefined.
        const prefix = rule.filter?.prefix ?? '';
        const filter = isAndFilter ? {
          and: {
            prefix,
            tagFilters: rule.filter?.tags,
          },
        } : {
          prefix,
        };

        const sourceAccount = Stack.of(this).account;
        const destinationAccount = rule.destination.env.account;
        const isCrossAccount = sourceAccount !== destinationAccount;

        if (isCrossAccount) {
          Annotations.of(this).addInfo('For Cross-account S3 replication, ensure to set up permissions on destination bucket using method addReplicationPolicy() ');
        } else if (rule.accessControlTransition) {
          throw new ValidationError(lit`AccessControlTranslationSupportedCross`, 'accessControlTranslation is only supported for cross-account replication', this);
        }

        return {
          id: rule.id,
          priority: rule.priority,
          status: 'Enabled',
          destination: {
            bucket: rule.destination.bucketArn,
            account: isCrossAccount ? destinationAccount : undefined,
            storageClass: rule.storageClass?.toString(),
            accessControlTranslation: rule.accessControlTransition ? {
              owner: 'Destination',
            } : undefined,
            encryptionConfiguration: rule.kmsKey ? {
              replicaKmsKeyId: rule.kmsKey.keyArn,
            } : undefined,
            replicationTime: rule.replicationTimeControl !== undefined ? {
              status: 'Enabled',
              time: {
                minutes: rule.replicationTimeControl.minutes,
              },
            } : undefined,
            metrics: rule.metrics !== undefined ? {
              status: 'Enabled',
              eventThreshold: {
                minutes: rule.metrics.minutes,
              },
            } : undefined,
          },
          filter,
          // To avoid deploy error when there are multiple replication rules with undefined deleteMarkerReplication,
          // CDK explicitly set the deleteMarkerReplication if it is undefined.
          deleteMarkerReplication: {
            status: rule.deleteMarkerReplication ? 'Enabled' : 'Disabled',
          },
          sourceSelectionCriteria,
        };
      }),
    };
  }

  /**
   * Allows Log Delivery to the S3 bucket, using a Bucket Policy if the relevant feature
   * flag is enabled, otherwise the canned ACL is used.
   *
   * If log delivery is to be allowed using the ACL and an ACL has already been set, this fails.
   *
   * @see
   * https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-server-access-logging.html
   */
  private allowLogDelivery(from: IBucket, prefix?: string) {
    if (FeatureFlags.of(this).isEnabled(cxapi.S3_SERVER_ACCESS_LOGS_USE_BUCKET_POLICY)) {
      let conditions = undefined;
      // The conditions for the bucket policy can be applied only when the buckets are in
      // the same stack and a concrete bucket instance (not imported). Otherwise, the
      // necessary imports may result in a cyclic dependency between the stacks.
      if (from instanceof Bucket && Stack.of(this) === Stack.of(from)) {
        conditions = {
          ArnLike: {
            'aws:SourceArn': from.bucketArn,
          },
          StringEquals: {
            'aws:SourceAccount': from.env.account,
          },
        };
      }
      this.addToResourcePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal('logging.s3.amazonaws.com')],
        actions: ['s3:PutObject'],
        resources: [this.arnForObjects(prefix ? `${prefix}*` : '*')],
        conditions: conditions,
      }));
    } else if (this.accessControl.get() && this.accessControl.get() !== BucketAccessControl.LOG_DELIVERY_WRITE) {
      throw new ValidationError(lit`CannotEnableLogDeliveryBucket`, "Cannot enable log delivery to this bucket because the bucket's ACL has been set and can't be changed", this);
    } else {
      this.accessControl.set(BucketAccessControl.LOG_DELIVERY_WRITE);
      this.ownershipControls.set(this.parseOwnershipControls(BucketAccessControl.LOG_DELIVERY_WRITE));
      this.acknowledgeAccessControl();
    }
  }

  private parseInventoryConfiguration(inventories: readonly Inventory[]): CfnBucket.InventoryConfigurationProperty[] | undefined {
    if (!inventories || inventories.length === 0) {
      return undefined;
    }
    const inventoryIdValidationRegex = /[^\w\.\-]/g;

    return inventories.map((inventory, index) => {
      const format = inventory.format ?? InventoryFormat.CSV;
      const frequency = inventory.frequency ?? InventoryFrequency.WEEKLY;
      if (inventory.inventoryId !== undefined && (inventory.inventoryId.length > 64 || inventoryIdValidationRegex.test(inventory.inventoryId))) {
        throw new ValidationError(lit`InventoryIdExceedCharactersContain`, `inventoryId should not exceed 64 characters and should not contain special characters except . and -, got ${inventory.inventoryId}`, this);
      }
      const id = inventory.inventoryId ?? `${this.node.id}Inventory${index}`.replace(inventoryIdValidationRegex, '').slice(-64);

      if (inventory.destination.bucket instanceof Bucket) {
        inventory.destination.bucket.addToResourcePolicy(new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ['s3:PutObject'],
          resources: [
            inventory.destination.bucket.bucketArn,
            inventory.destination.bucket.arnForObjects(`${inventory.destination.prefix ?? ''}*`),
          ],
          principals: [new iam.ServicePrincipal('s3.amazonaws.com')],
          conditions: {
            ArnLike: {
              'aws:SourceArn': this.bucketArn,
            },
          },
        }));
      }

      return {
        id,
        destination: {
          bucketArn: inventory.destination.bucket.bucketArn,
          bucketAccountId: inventory.destination.bucketOwner,
          prefix: inventory.destination.prefix,
          format,
        },
        enabled: inventory.enabled ?? true,
        includedObjectVersions: inventory.includeObjectVersions ?? InventoryObjectVersion.ALL,
        scheduleFrequency: frequency,
        optionalFields: inventory.optionalFields,
        prefix: inventory.objectsPrefix,
      };
    });
  }

  private enableAutoDeleteObjects() {
    // When called from the constructor, this is the first time the bucket policy
    // might be auto created. For backwards compatibility, we need to make sure the
    // L2 owned policy already exists, so it will be used by the Mixin.
    this.maybeAutoCreatePolicy();
    this.with(new BucketAutoDeleteObjects());
  }

  /**
   * Function to set the blockPublicAccessOptions to a true default if not defined.
   * If no blockPublicAccessOptions are specified at all, this is already the case as an s3 default in aws
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html
   */
  private setDefaultPublicAccessBlockConfig(blockPublicAccessOptions: BlockPublicAccessOptions) {
    return {
      blockPublicAcls: blockPublicAccessOptions.blockPublicAcls ?? true,
      blockPublicPolicy: blockPublicAccessOptions.blockPublicPolicy ?? true,
      ignorePublicAcls: blockPublicAccessOptions.ignorePublicAcls ?? true,
      restrictPublicBuckets: blockPublicAccessOptions.restrictPublicBuckets ?? true,
    };
  }
}

/**
 * What kind of server-side encryption to apply to this bucket
 */
export enum BucketEncryption {
  /**
   * Previous option. Buckets cannot be unencrypted now.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/serv-side-encryption.html
   * @deprecated S3 applies server-side encryption with SSE-S3 for every bucket
   * that default encryption is not configured.
   */
  UNENCRYPTED = 'UNENCRYPTED',

  /**
   * Server-side KMS encryption with a master key managed by KMS.
   */
  KMS_MANAGED = 'KMS_MANAGED',

  /**
   * Server-side encryption with a master key managed by S3.
   */
  S3_MANAGED = 'S3_MANAGED',

  /**
   * Server-side encryption with a KMS key managed by the user.
   * If `encryptionKey` is specified, this key will be used, otherwise, one will be defined.
   */
  KMS = 'KMS',

  /**
   * Double server-side KMS encryption with a master key managed by KMS.
   */
  DSSE_MANAGED = 'DSSE_MANAGED',

  /**
   * Double server-side encryption with a KMS key managed by the user.
   * If `encryptionKey` is specified, this key will be used, otherwise, one will be defined.
   */
  DSSE = 'DSSE',
}

/**
 * Encryption types that can be blocked on an S3 bucket.
 */
export class BlockedEncryptionType {
  /** Special value - all encryption types are allowed */
  public static readonly NONE = new BlockedEncryptionType('NONE');
  /** Server-Side Encryption with customer-provided keys (SSE-C) is blocked */
  public static readonly SSE_C = new BlockedEncryptionType('SSE-C');

  /**
   * Use this constructor only if S3 releases a new BlockedEncryptionType
   * that is unknown to CDK. Otherwise, use this class's static constants.
   */
  public static custom(name: string) {
    return new BlockedEncryptionType(name);
  }

  /**
   * @param name The name for this blocked encryption type used in the API
   */
  private constructor(public readonly name: string) {}
}

/**
 * Notification event types.
 * @link https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-event-types-and-destinations.html#supported-notification-event-types
 */
export enum EventType {
  /**
   * Amazon S3 APIs such as PUT, POST, and COPY can create an object. Using
   * these event types, you can enable notification when an object is created
   * using a specific API, or you can use the s3:ObjectCreated:* event type to
   * request notification regardless of the API that was used to create an
   * object.
   */
  OBJECT_CREATED = 's3:ObjectCreated:*',

  /**
   * Amazon S3 APIs such as PUT, POST, and COPY can create an object. Using
   * these event types, you can enable notification when an object is created
   * using a specific API, or you can use the s3:ObjectCreated:* event type to
   * request notification regardless of the API that was used to create an
   * object.
   */
  OBJECT_CREATED_PUT = 's3:ObjectCreated:Put',

  /**
   * Amazon S3 APIs such as PUT, POST, and COPY can create an object. Using
   * these event types, you can enable notification when an object is created
   * using a specific API, or you can use the s3:ObjectCreated:* event type to
   * request notification regardless of the API that was used to create an
   * object.
   */
  OBJECT_CREATED_POST = 's3:ObjectCreated:Post',

  /**
   * Amazon S3 APIs such as PUT, POST, and COPY can create an object. Using
   * these event types, you can enable notification when an object is created
   * using a specific API, or you can use the s3:ObjectCreated:* event type to
   * request notification regardless of the API that was used to create an
   * object.
   */
  OBJECT_CREATED_COPY = 's3:ObjectCreated:Copy',

  /**
   * Amazon S3 APIs such as PUT, POST, and COPY can create an object. Using
   * these event types, you can enable notification when an object is created
   * using a specific API, or you can use the s3:ObjectCreated:* event type to
   * request notification regardless of the API that was used to create an
   * object.
   */
  OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD = 's3:ObjectCreated:CompleteMultipartUpload',

  /**
   * By using the ObjectRemoved event types, you can enable notification when
   * an object or a batch of objects is removed from a bucket.
   *
   * You can request notification when an object is deleted or a versioned
   * object is permanently deleted by using the s3:ObjectRemoved:Delete event
   * type. Or you can request notification when a delete marker is created for
   * a versioned object by using s3:ObjectRemoved:DeleteMarkerCreated. For
   * information about deleting versioned objects, see Deleting Object
   * Versions. You can also use a wildcard s3:ObjectRemoved:* to request
   * notification anytime an object is deleted.
   *
   * You will not receive event notifications from automatic deletes from
   * lifecycle policies or from failed operations.
   */
  OBJECT_REMOVED = 's3:ObjectRemoved:*',

  /**
   * By using the ObjectRemoved event types, you can enable notification when
   * an object or a batch of objects is removed from a bucket.
   *
   * You can request notification when an object is deleted or a versioned
   * object is permanently deleted by using the s3:ObjectRemoved:Delete event
   * type. Or you can request notification when a delete marker is created for
   * a versioned object by using s3:ObjectRemoved:DeleteMarkerCreated. For
   * information about deleting versioned objects, see Deleting Object
   * Versions. You can also use a wildcard s3:ObjectRemoved:* to request
   * notification anytime an object is deleted.
   *
   * You will not receive event notifications from automatic deletes from
   * lifecycle policies or from failed operations.
   */
  OBJECT_REMOVED_DELETE = 's3:ObjectRemoved:Delete',

  /**
   * By using the ObjectRemoved event types, you can enable notification when
   * an object or a batch of objects is removed from a bucket.
   *
   * You can request notification when an object is deleted or a versioned
   * object is permanently deleted by using the s3:ObjectRemoved:Delete event
   * type. Or you can request notification when a delete marker is created for
   * a versioned object by using s3:ObjectRemoved:DeleteMarkerCreated. For
   * information about deleting versioned objects, see Deleting Object
   * Versions. You can also use a wildcard s3:ObjectRemoved:* to request
   * notification anytime an object is deleted.
   *
   * You will not receive event notifications from automatic deletes from
   * lifecycle policies or from failed operations.
   */
  OBJECT_REMOVED_DELETE_MARKER_CREATED = 's3:ObjectRemoved:DeleteMarkerCreated',

  /**
   * Using restore object event types you can receive notifications for
   * initiation and completion when restoring objects from the S3 Glacier
   * storage class.
   *
   * You use s3:ObjectRestore:Post to request notification of object restoration
   * initiation.
   */
  OBJECT_RESTORE_POST = 's3:ObjectRestore:Post',

  /**
   * Using restore object event types you can receive notifications for
   * initiation and completion when restoring objects from the S3 Glacier
   * storage class.
   *
   * You use s3:ObjectRestore:Completed to request notification of
   * restoration completion.
   */
  OBJECT_RESTORE_COMPLETED = 's3:ObjectRestore:Completed',

  /**
   * Using restore object event types you can receive notifications for
   * initiation and completion when restoring objects from the S3 Glacier
   * storage class.
   *
   * You use s3:ObjectRestore:Delete to request notification of
   * restoration completion.
   */
  OBJECT_RESTORE_DELETE = 's3:ObjectRestore:Delete',

  /**
   * You can use this event type to request Amazon S3 to send a notification
   * message when Amazon S3 detects that an object of the RRS storage class is
   * lost.
   */
  REDUCED_REDUNDANCY_LOST_OBJECT = 's3:ReducedRedundancyLostObject',

  /**
   * You receive this notification event when an object that was eligible for
   * replication using Amazon S3 Replication Time Control failed to replicate.
   */
  REPLICATION_OPERATION_FAILED_REPLICATION = 's3:Replication:OperationFailedReplication',

  /**
   * You receive this notification event when an object that was eligible for
   * replication using Amazon S3 Replication Time Control exceeded the 15-minute
   * threshold for replication.
   */
  REPLICATION_OPERATION_MISSED_THRESHOLD = 's3:Replication:OperationMissedThreshold',

  /**
   * You receive this notification event for an object that was eligible for
   * replication using the Amazon S3 Replication Time Control feature replicated
   * after the 15-minute threshold.
   */
  REPLICATION_OPERATION_REPLICATED_AFTER_THRESHOLD = 's3:Replication:OperationReplicatedAfterThreshold',

  /**
   * You receive this notification event for an object that was eligible for
   * replication using Amazon S3 Replication Time Control but is no longer tracked
   * by replication metrics.
   */
  REPLICATION_OPERATION_NOT_TRACKED = 's3:Replication:OperationNotTracked',

  /**
   * By using the LifecycleExpiration event types, you can receive a notification
   * when Amazon S3 deletes an object based on your S3 Lifecycle configuration.
   */
  LIFECYCLE_EXPIRATION = 's3:LifecycleExpiration:*',

  /**
   * The s3:LifecycleExpiration:Delete event type notifies you when an object
   * in an unversioned bucket is deleted.
   * It also notifies you when an object version is permanently deleted by an
   * S3 Lifecycle configuration.
   */
  LIFECYCLE_EXPIRATION_DELETE = 's3:LifecycleExpiration:Delete',

  /**
   * The s3:LifecycleExpiration:DeleteMarkerCreated event type notifies you
   * when S3 Lifecycle creates a delete marker when a current version of an
   * object in versioned bucket is deleted.
   */
  LIFECYCLE_EXPIRATION_DELETE_MARKER_CREATED = 's3:LifecycleExpiration:DeleteMarkerCreated',

  /**
   * You receive this notification event when an object is transitioned to
   * another Amazon S3 storage class by an S3 Lifecycle configuration.
   */
  LIFECYCLE_TRANSITION = 's3:LifecycleTransition',

  /**
   * You receive this notification event when an object within the
   * S3 Intelligent-Tiering storage class moved to the Archive Access tier or
   * Deep Archive Access tier.
   */
  INTELLIGENT_TIERING = 's3:IntelligentTiering',

  /**
   * By using the ObjectTagging event types, you can enable notification when
   * an object tag is added or deleted from an object.
   */
  OBJECT_TAGGING = 's3:ObjectTagging:*',

  /**
   * The s3:ObjectTagging:Put event type notifies you when a tag is PUT on an
   * object or an existing tag is updated.
   */
  OBJECT_TAGGING_PUT = 's3:ObjectTagging:Put',

  /**
   * The s3:ObjectTagging:Delete event type notifies you when a tag is removed
   * from an object.
   */
  OBJECT_TAGGING_DELETE = 's3:ObjectTagging:Delete',

  /**
   * You receive this notification event when an ACL is PUT on an object or when
   * an existing ACL is changed.
   * An event is not generated when a request results in no change to an
   * object’s ACL.
   */
  OBJECT_ACL_PUT = 's3:ObjectAcl:Put',

  /**
   * Using restore object event types you can receive notifications for
   * initiation and completion when restoring objects from the S3 Glacier
   * storage class.
   *
   * You use s3:ObjectRestore:* to request notification of
   * any restoration event.
   */
  OBJECT_RESTORE = 's3:ObjectRestore:*',

  /**
   * You receive this notification event for any object replication event.
   */
  REPLICATION = 's3:Replication:*',
}

export interface NotificationKeyFilter {
  /**
   * S3 keys must have the specified prefix.
   */
  readonly prefix?: string;

  /**
   * S3 keys must have the specified suffix.
   */
  readonly suffix?: string;
}

/**
 * Options for the onCloudTrailPutObject method
 */
export interface OnCloudTrailBucketEventOptions extends events.OnEventOptions {
  /**
   * Only watch changes to these object paths
   *
   * @default - Watch changes to all objects
   */
  readonly paths?: string[];
}

/**
 * Default bucket access control types.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/acl-overview.html
 */
export enum BucketAccessControl {
  /**
   * Owner gets FULL_CONTROL. No one else has access rights.
   */
  PRIVATE = 'Private',

  /**
   * Owner gets FULL_CONTROL. The AllUsers group gets READ access.
   */
  PUBLIC_READ = 'PublicRead',

  /**
   * Owner gets FULL_CONTROL. The AllUsers group gets READ and WRITE access.
   * Granting this on a bucket is generally not recommended.
   */
  PUBLIC_READ_WRITE = 'PublicReadWrite',

  /**
   * Owner gets FULL_CONTROL. The AuthenticatedUsers group gets READ access.
   */
  AUTHENTICATED_READ = 'AuthenticatedRead',

  /**
   * The LogDelivery group gets WRITE and READ_ACP permissions on the bucket.
   * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/ServerLogs.html
   */
  LOG_DELIVERY_WRITE = 'LogDeliveryWrite',

  /**
   * Object owner gets FULL_CONTROL. Bucket owner gets READ access.
   * If you specify this canned ACL when creating a bucket, Amazon S3 ignores it.
   */
  BUCKET_OWNER_READ = 'BucketOwnerRead',

  /**
   * Both the object owner and the bucket owner get FULL_CONTROL over the object.
   * If you specify this canned ACL when creating a bucket, Amazon S3 ignores it.
   */
  BUCKET_OWNER_FULL_CONTROL = 'BucketOwnerFullControl',

  /**
   * Owner gets FULL_CONTROL. Amazon EC2 gets READ access to GET an Amazon Machine Image (AMI) bundle from Amazon S3.
   */
  AWS_EXEC_READ = 'AwsExecRead',
}

export interface RoutingRuleCondition {
  /**
   * The HTTP error code when the redirect is applied
   *
   * In the event of an error, if the error code equals this value, then the specified redirect is applied.
   *
   * If both condition properties are specified, both must be true for the redirect to be applied.
   *
   * @default - The HTTP error code will not be verified
   */
  readonly httpErrorCodeReturnedEquals?: string;

  /**
   * The object key name prefix when the redirect is applied
   *
   * If both condition properties are specified, both must be true for the redirect to be applied.
   *
   * @default - The object key name will not be verified
   */
  readonly keyPrefixEquals?: string;
}

export class ReplaceKey {
  /**
   * The specific object key to use in the redirect request
   */
  public static with(keyReplacement: string) {
    return new this(keyReplacement);
  }

  /**
   * The object key prefix to use in the redirect request
   */
  public static prefixWith(keyReplacement: string) {
    return new this(undefined, keyReplacement);
  }

  private constructor(public readonly withKey?: string, public readonly prefixWithKey?: string) {
  }
}

/**
 * Rule that define when a redirect is applied and the redirect behavior.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/dev/how-to-page-redirect.html
 */
export interface RoutingRule {
  /**
   * The host name to use in the redirect request
   *
   * @default - The host name used in the original request.
   */
  readonly hostName?: string;

  /**
   * The HTTP redirect code to use on the response
   *
   * @default "301" - Moved Permanently
   */
  readonly httpRedirectCode?: string;

  /**
   * Protocol to use when redirecting requests
   *
   * @default - The protocol used in the original request.
   */
  readonly protocol?: RedirectProtocol;

  /**
   * Specifies the object key prefix to use in the redirect request
   *
   * @default - The key will not be replaced
   */
  readonly replaceKey?: ReplaceKey;

  /**
   * Specifies a condition that must be met for the specified redirect to apply.
   *
   * @default - No condition
   */
  readonly condition?: RoutingRuleCondition;
}

/**
 * Modes in which S3 Object Lock retention can be configured.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-retention-modes
 */
export enum ObjectLockMode {
  /**
   * The Governance retention mode.
   *
   * With governance mode, you protect objects against being deleted by most users, but you can
   * still grant some users permission to alter the retention settings or delete the object if
   * necessary. You can also use governance mode to test retention-period settings before
   * creating a compliance-mode retention period.
   */
  GOVERNANCE = 'GOVERNANCE',

  /**
   * The Compliance retention mode.
   *
   * When an object is locked in compliance mode, its retention mode can't be changed, and
   * its retention period can't be shortened. Compliance mode helps ensure that an object
   * version can't be overwritten or deleted for the duration of the retention period.
   */
  COMPLIANCE = 'COMPLIANCE',
}

/**
 * The default retention settings for an S3 Object Lock configuration.
 *
 * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html
 */
export class ObjectLockRetention {
  /**
   * Configure for Governance retention for a specified duration.
   *
   * With governance mode, you protect objects against being deleted by most users, but you can
   * still grant some users permission to alter the retention settings or delete the object if
   * necessary. You can also use governance mode to test retention-period settings before
   * creating a compliance-mode retention period.
   *
   * @param duration the length of time for which objects should retained
   * @returns the ObjectLockRetention configuration
   */
  public static governance(duration: Duration): ObjectLockRetention {
    return new ObjectLockRetention(ObjectLockMode.GOVERNANCE, duration);
  }

  /**
   * Configure for Compliance retention for a specified duration.
   *
   * When an object is locked in compliance mode, its retention mode can't be changed, and
   * its retention period can't be shortened. Compliance mode helps ensure that an object
   * version can't be overwritten or deleted for the duration of the retention period.
   *
   * @param duration the length of time for which objects should be retained
   * @returns the ObjectLockRetention configuration
   */
  public static compliance(duration: Duration): ObjectLockRetention {
    return new ObjectLockRetention(ObjectLockMode.COMPLIANCE, duration);
  }

  /**
   * The default period for which objects should be retained.
   */
  public readonly duration: Duration;

  /**
   * The retention mode to use for the object lock configuration.
   *
   * @see https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-retention-modes
   */
  public readonly mode: ObjectLockMode;

  private constructor(mode: ObjectLockMode, duration: Duration) {
    // https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-managing.html#object-lock-managing-retention-limits
    if (duration.toDays() > 365 * 100) {
      throw new UnscopedValidationError(lit`MustBeObjectLockRetention`, 'Object Lock retention duration must be less than 100 years');
    }
    if (duration.toDays() < 1) {
      throw new UnscopedValidationError(lit`MustBeObjectLockRetention`, 'Object Lock retention duration must be at least 1 day');
    }

    this.mode = mode;
    this.duration = duration;
  }
}

/**
 * Options for creating Virtual-Hosted style URL.
 */
export interface VirtualHostedStyleUrlOptions {
  /**
   * Specifies the URL includes the region.
   *
   * @default - true
   */
  readonly regional?: boolean;
}

/**
 * Options for creating a Transfer Acceleration URL.
 */
export interface TransferAccelerationUrlOptions {
  /**
   * Dual-stack support to connect to the bucket over IPv6.
   *
   * @default - false
   */
  readonly dualStack?: boolean;
}

function mapOrUndefined<T, U>(list: T[] | undefined, callback: (element: T) => U): U[] | undefined {
  if (!list || list.length === 0) {
    return undefined;
  }

  return list.map(callback);
}

/**
 * An instance of a referenced Bucket
 *
 * An explicit class declaration to save memory; previously,
 * `fromBucketAttributes()` etc. used to declare an anonymous subclass of
 * `BucketBase`, which would lead to a prototype and constructor for every
 * invocation (but they are all the same).
 *
 * Use a single class instance.
 */
@propertyInjectable
class ReferencedBucket extends BucketBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-s3.ReferencedBucket';
  public bucketArn: string;
  public bucketName: string;
  public bucketDomainName: string;
  public bucketWebsiteUrl: string;
  public bucketWebsiteDomainName: string;
  public bucketRegionalDomainName: string;
  public bucketDualStackDomainName: string;
  public encryptionKey?: kms.IKey | undefined;
  public isWebsite?: boolean | undefined;
  public policy?: BucketPolicy | undefined;
  public replicationRoleArn?: string | undefined;
  protected autoCreatePolicy: boolean;
  public disallowPublicAccess?: boolean | undefined;
  protected notificationsHandlerRole?: iam.IRole;

  constructor(scope: Construct, id: string, props: {
    account?: string;
    region?: string;
    bucketArn: string;
    bucketName: string;
    bucketDomainName: string;
    bucketWebsiteUrl: string;
    bucketWebsiteDomainName: string;
    bucketRegionalDomainName: string;
    bucketDualStackDomainName: string;
    encryptionKey?: kms.IKey | undefined;
    isWebsite?: boolean | undefined;
    policy?: BucketPolicy | undefined;
    replicationRoleArn?: string | undefined;
    autoCreatePolicy: boolean;
    disallowPublicAccess?: boolean | undefined;
    notificationsHandlerRole?: iam.IRole;
  }) {
    super(scope, id, {
      account: props.account,
      region: props.region,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);
    this.bucketArn = props.bucketArn;
    this.bucketName = props.bucketName;
    this.bucketDomainName = props.bucketDomainName;
    this.bucketWebsiteUrl = props.bucketWebsiteUrl;
    this.bucketWebsiteDomainName = props.bucketWebsiteDomainName;
    this.bucketRegionalDomainName = props.bucketRegionalDomainName;
    this.bucketDualStackDomainName = props.bucketDualStackDomainName;
    this.encryptionKey = props.encryptionKey;
    this.isWebsite = props.isWebsite;
    this.policy = props.policy;
    this.replicationRoleArn = props.replicationRoleArn;
    this.autoCreatePolicy = props.autoCreatePolicy;
    this.disallowPublicAccess = props.disallowPublicAccess;
    this.notificationsHandlerRole = props.notificationsHandlerRole;
  }
}
