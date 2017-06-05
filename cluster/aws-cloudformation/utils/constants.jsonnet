/**
 * Constants defined for a particular incarnation of this set of CF templates.
 * Each fork of this code must setup their own values below.
 */
{
  /**
   * AWS project ID for the project hosting the Kubernetes cluster(s).
   */
  AwsProjectId:: "",

  /**
   * Array of IAM ARNs of administrative users who will have root access to sensitive objects such as certifcates
   * and private keys.
   * These look like "arn:aws:iam::<project id>:user/<user name>"
   */
  AdminUsers:: [],

  /**
   * Email address to send alerts to.
   */
  AlertsEmail:: "alerts@mycorp.com",

  /**
   * A company- or use-case-specific prefix for all S3 storage buckets. Required as all S3 bucket names share
   * a global namespace among customers.
   */
  StorageBucketPrefix:: "my-corp",

  /**
   * Name of the keypair which will be used for SSH access to instances created by CloudFormation.
   * Note: This needs to be created manually by the admin prior to spinning up these templates.
   */
  InstanceKeyName:: "",

  /**
   * VPC-Internal DNS suffix where cluster internal services will be exposed.
   */
  // FIXME: Need to actually create this...
  InternalDNSSuffix:: ".internal.my-corp.com",

  /**
   * The DNS zone name where kubernetes master and services will be exposed.
   */
  // FIXME: Need to actually create this...
  ExternalOpsDNSZoneName:: "ops.my-corp.com.",

  // FIXME: Document or skip?
  // These look like "arn:aws:iam::<project id>:server-certificate/<name>"
  KubeServicesCertARN:: "",
}

// FIXME: Also need to layout instructions for setup, including:
// Keypair creation via cmdline, certificate generation & upload via TinyCert, push tool?

