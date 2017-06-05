/**
 * Template for a regional S3 bucket, roles, and policies used to store and distribute certificates.
 */

local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";
local iam_config = import "../global/iam_config.jsonnet";
local network_template = import "network_template.jsonnet";

local etcd_path = "etcd/*";
local kubelet_path = "kubelet/*";
local kube_master_path = "kubemaster/*";

// Define how this stack's parameters are filled from the output of other stacks.
local parameter_map(region) = [
  network_template.vpc_id_output(region),
  iam_config.iam_output("CertificateAdminRoleArn"),
  iam_config.iam_output("EtcdNodeRoleArn"),
  iam_config.iam_output("KubeNodeRoleArn"),
  iam_config.iam_output("KubeMasterRoleArn"),
];

{
  /**
   * Common template for the regional certificate storage system.
   *
   * Derive from this for each region we run in.
   */
  regional_cert_storage_definition(aws_region):: cf_utils.cf_template(
    aws_region + " regional certificate storage",
    "cert-storage-" + aws_region,
    aws_region,
    ["CAPABILITY_IAM"],
    parameter_map(aws_region)) {

    // Input parameters to this stack, fed in from the map provided above.
    Parameters: {
      VpcId: {
        Type: "String",
        AllowedPattern: "vpc-.*",
        Description: "Vpc ID of the underlying pre-defined regional VPC.",
      },
      CertificateAdminRoleArn: {
        Type: "String",
        AllowedPattern: "arn:aws:iam::.*",
      },
      EtcdNodeRoleArn: {
        Type: "String",
        AllowedPattern: "arn:aws:iam::.*",
      },
      KubeNodeRoleArn: {
        Type: "String",
        AllowedPattern: "arn:aws:iam::.*",
      },
      KubeMasterRoleArn: {
        Type: "String",
        AllowedPattern: "arn:aws:iam::.*",
      },
    },

    Resources: {

      // A bucket to store our certs in.  Certs are populated outside of this stack by an admin.
      CertBucket: {
        Type: "AWS::S3::Bucket",
        DeletionPolicy: "Retain",
        Properties: {
          BucketName: $.bucket_name(aws_region),
        },
      },

      // Access policy for the bucket, using the roles above.
      CertBucketPolicy: {
        Type: "AWS::S3::BucketPolicy",
        DependsOn: ["CertBucket"],
        Properties: {
          Bucket: { Ref: "CertBucket" },
          PolicyDocument: {
            Id: "Grants for the cert bucket.",
            Statement: [
              // Allow administrator role and admin users full access, no access for others.
              {
                Sid: "certs-full-access-from-admin-role",
                Action: "s3:*",
                Principal: {
                  AWS: [{ Ref: "CertificateAdminRoleArn" }] + constants.AdminUsers,
                },
                Effect: "Allow",
                Resource: [
                  { "Fn::Join": ["", ["arn:aws:s3:::", { Ref: "CertBucket" }]] },
                  { "Fn::Join": ["", ["arn:aws:s3:::", { Ref: "CertBucket" }, "/*"]] },
                ],
                Condition: { Bool: { "aws:SecureTransport": "true" } },
              },

              $.define_reader_statement("etcd", "EtcdNodeRoleArn", etcd_path),
              $.define_reader_statement("kubelet", "KubeNodeRoleArn", kubelet_path),
              $.define_reader_statement("kubemaster", "KubeMasterRoleArn", kube_master_path),

              $.define_lister_statement("etcd", "EtcdNodeRoleArn", etcd_path),
              $.define_lister_statement("kubelet", "KubeNodeRoleArn", kubelet_path),
              $.define_lister_statement("kubemaster", "KubeMasterRoleArn", kube_master_path),
            ],
          },
        },
      },
    },
  },

  /**
   * Retrieve the bucket name for the given region.
   */
  bucket_name(region)::
    "%s-certs-%s" % [constants.StorageBucketPrefix, region],

  /**
   * Define a statement which allows the given reader role to get from the given path in the cert bucket
   * from within the VPC.
   */
  define_reader_statement(name, reader_role_param, path):: {
    Sid: "certs-%s-read-access" % name,
    Action: "s3:GetObject",
    Principal: { AWS: { Ref: reader_role_param } },
    Effect: "Allow",
    Resource: { "Fn::Join": ["", ["arn:aws:s3:::", { Ref: "CertBucket" }, "/", path]] },
    Condition: {
      Bool: { "aws:SecureTransport": "true" },
      StringEquals: { "aws:sourceVpc": { Ref: "VpcId" } },
    },
  },

  /**
   * Define a statement which allows the given reader role to list from the given path in the cert bucket
   * from within the VPC.
   */
  define_lister_statement(name, reader_role_param, path):: {
    Sid: "certs-%s-list-access" % name,
    Action: "s3:ListBucket",
    Principal: { AWS: { Ref: reader_role_param } },
    Effect: "Allow",
    Resource: { "Fn::Join": ["", ["arn:aws:s3:::", { Ref: "CertBucket" }]] },
    Condition: {
      Bool: { "aws:SecureTransport": "true" },
      StringEquals: { "aws:sourceVpc": { Ref: "VpcId" } },
      StringLike: { "s3:prefix": [path] },
    },
  },
}
