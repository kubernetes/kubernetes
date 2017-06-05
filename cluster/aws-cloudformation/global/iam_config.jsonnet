/**
 * Global IAM configuration.  Includes role and instance profile definitions for cluster infrastructure.
 */

local cf_utils = import "../utils/cf_utils.jsonnet";

cf_utils.cf_template(
  "Global IAM configuration stack",
  "iam-global",
  "us-west-2",
  ["CAPABILITY_IAM"],
  []) {

  local certificate_path = "/infra/admin/",
  local etcd_path = "/infra/etcd/",
  local kube_path = "/infra/kube/",

  Resources: {
    /**
     * Managed policy which allows writing cloudwatch metrics.
     */
    CloudWatchPutMetricsManagedPolicy: {
      Type: "AWS::IAM::ManagedPolicy",
      Properties: {
        Description: "Enable writing metrics data to CloudWatch",
        PolicyDocument: {
          Version: "2012-10-17",
          Statement: [{
            Sid: "GrantPutMetricsToCloudWatch",
            Effect: "Allow",
            Action: [
              "cloudwatch:PutMetricData",
            ],
            Resource: [
              "*",
            ],
          }],
        },
      },
    },

    // IAM Certificate Admin role - gets full access to the root certicate volume and write access to regional
    // storage buckets.
    CertificateAdminRole: $.define_ec2_role(certificate_path),
    CertificateAdminInstanceProfile: $.define_instance_profile("CertificateAdminRole", certificate_path),

    // ETCD server role
    EtcdNodeRole: $.define_ec2_role(etcd_path),
    EtcdNodeInstanceProfile: $.define_instance_profile("EtcdNodeRole", etcd_path),

    // Kubernetes node role
    KubeNodeRole: $.define_ec2_role(kube_path),
    KubeNodeInstanceProfile: $.define_instance_profile("KubeNodeRole", kube_path),

    // Kubernetes master role
    KubeMasterRole: $.define_ec2_role(kube_path),
    KubeMasterInstanceProfile: $.define_instance_profile("KubeMasterRole", kube_path),
  },

  // Export AWS chosen values so other stacks can depend on these objects.
  Outputs: {
    CloudWatchPutMetricsManagedPolicyArn: {
      Description: "ARN of the managed policy which allows writing to cloudwatch.",
      Value: { Ref: "CloudWatchPutMetricsManagedPolicy" },
    },

    CertificateAdminRoleArn: $.define_role_arn_output("CertificateAdminRole", "certificate admin"),
    EtcdNodeRoleArn: $.define_role_arn_output("EtcdNodeRole", "etcd node"),
    KubeNodeRoleArn: $.define_role_arn_output("KubeNodeRole", "kube node"),
    KubeMasterRoleArn: $.define_role_arn_output("KubeMasterRole", "kube master"),

    CertificateAdminRoleName: $.define_role_name_output("CertificateAdminRole", "certificate admin"),
    EtcdNodeRoleName: $.define_role_name_output("EtcdNodeRole", "etcd node"),
    KubeNodeRoleName: $.define_role_name_output("KubeNodeRole", "kube node"),
    KubeMasterRoleName: $.define_role_name_output("KubeMasterRole", "kube master"),

    CertificateAdminInstanceProfileName: $.define_instance_profile_name_output(
      "CertificateAdminInstanceProfile", "certificate admin"),
    EtcdNodeInstanceProfileName: $.define_instance_profile_name_output("EtcdNodeInstanceProfile", "etcd node"),
    KubeNodeInstanceProfileName: $.define_instance_profile_name_output("KubeNodeInstanceProfile", "kube node"),
    KubeMasterInstanceProfileName: $.define_instance_profile_name_output("KubeMasterInstanceProfile", "kube master"),
  },

  /**
   * Define an IAM role for use by EC2 instances.
   */
  define_ec2_role(path):: {
    Type: "AWS::IAM::Role",
    Properties: {
      AssumeRolePolicyDocument: {
        Version: "2012-10-17",
        Statement: [{
          Effect: "Allow",
          Principal: { Service: ["ec2.amazonaws.com"] },
          Action: ["sts:AssumeRole"],
        }],
      },
      Path: path,
      ManagedPolicyArns: [
        { Ref: "CloudWatchPutMetricsManagedPolicy" },
      ],
    },
  },

  /**
   * Define an EC2 instance profile for the given IAM role.
   */
  define_instance_profile(logical_role_name, path):: {
    Type: "AWS::IAM::InstanceProfile",
    DependsOn: [logical_role_name],
    Properties: {
      Path: path,
      Roles: [{ Ref: logical_role_name }],
    },
  },

  /**
   * Output the ARN for a role.
   */
  define_role_arn_output(logical_role_name, role_description):: {
    Description: "ARN of the %s role." % role_description,
    Value: { "Fn::GetAtt": [logical_role_name, "Arn"] },
  },

  /**
   * Output the name for a role.
   */
  define_role_name_output(logical_role_name, role_description):: {
    Description: "Name of the %s role." % role_description,
    Value: { Ref: logical_role_name },
  },

  /**
   * Output the name for an instance profile.
   */
  define_instance_profile_name_output(logical_instance_profile_name, profile_description):: {
    Description: "Name of the %s instance profile." % profile_description,
    Value: { Ref: logical_instance_profile_name },
  },

  // Helpers for retriving output fields from this stack in other stacks.
  iam_output(input_output_param_name):: {
    Param: input_output_param_name,
    Source: { Stack: "iam-global", Output: input_output_param_name },
  },
}
