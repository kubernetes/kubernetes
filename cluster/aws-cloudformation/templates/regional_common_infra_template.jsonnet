/**
 * Template for common infra running in an AWS region.
 *
 */

local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";
local etcd_cluster_template = import "etcd_cluster_template.jsonnet";
local iam_config = import "../global/iam_config.jsonnet";
local network_cidrs = import "../utils/network_cidrs.jsonnet";
local network_template = import "network_template.jsonnet";

// Define how this stack's parameters are filled from the output of other stacks.
local parameter_map(region, zones) = [
  network_template.vpc_id_output(region),
  iam_config.iam_output("KubeNodeRoleName"),
  iam_config.iam_output("KubeMasterRoleName"),
];

{
  stack_name(aws_region)::
    "common-infra" + "-" + aws_region,

  // The template definition
  regional_common_infra_environment_template(aws_region, aws_zones):: cf_utils.cf_template(
    aws_region + " regional infra environment for common assets.",
    $.stack_name(aws_region),
    aws_region,
    ["CAPABILITY_IAM"],
    parameter_map(aws_region, aws_zones)) {

    Parameters: {
      VpcId: {
        Type: "String",
        AllowedPattern: "vpc-.*",
        Description: "Vpc ID of the underlying pre-defined regional VPC.",
      },
      KubeMasterRoleName: {
        Type: "String",
        Description: "Name of the IAM role for the master node.",
      },
      KubeNodeRoleName: {
        Type: "String",
        Description: "Name of the IAM role for the kubernetes nodes.",
      },
    },

    // Export AWS chosen values so other stacks can depend on these objects.
    Outputs: {
      AuditLogBucketName: {
        Description: "Name of the audit log bucket created by this stack.",
        Value: { Ref: "AuditLogBucket" },
      },
      KubernetesMasterAWSAccessPolicy: {
        Description: "Policy for the Kubernetes master role to read EC2 state and setup new ELBs group for kubelet nodes within each Kube cluster.",
        Value: { Ref: "KubernetesMasterAWSAccessPolicy" },
      },
      KubernetesMasterELBSecurityGroup: {
        Description: "Security group for the Kubernetes master API ELB.",
        Value: { Ref: "KubernetesMasterELBSecurityGroup" },
      },
      KubernetesNodeSecurityGroup: {
        Description: "Security group for kubelet nodes within each Kube cluster.",
        Value: { Ref: "KubernetesNodeSecurityGroup" },
      },
      KubernetesNodeSecurityGroupId: {
        Description: "Security groupId for kubelet nodes within each Kube cluster.",
        Value: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
      },
      KubernetesNodeAWSAccessPolicy: {
        Description: "kubelet node AWS access policy.",
        Value: { Ref: "KubernetesNodeAWSAccessPolicy" },
      },
      KubernetesServicesELBSecurityGroup: {
        Description: "Security group for the Kubernetes core services ELB.",
        Value: { Ref: "KubernetesServicesELBSecurityGroup" },
      },
      KubernetesServicesELBSecurityGroupId: {
        Description: "Security groupId for the Kubernetes core services ELB.",
        Value: { "Fn::GetAtt": ["KubernetesServicesELBSecurityGroup", "GroupId"] },
      },
    },

    Resources: {
      // Define a regional S3 bucket which stores audit logs for other resources in the region.
      AuditLogBucket: {
        // Don't delete this bucket when this CloudFormation stack is deleted.
        DeletionPolicy: "Retain",
        Type: "AWS::S3::Bucket",
        Properties: {
          BucketName: $.audit_bucket_name(aws_region),
          AccessControl: "LogDeliveryWrite",
        },
      },

      // Define a policy on the bucket which allows CloudTrail to write API audit logs
      BucketPolicy: {
        local resource_arn = "arn:aws:s3:::" + $.audit_bucket_name(aws_region),

        Type: "AWS::S3::BucketPolicy",
        DependsOn: ["AuditLogBucket"],
        Properties: {
          Bucket: { Ref: "AuditLogBucket" },
          PolicyDocument: {
            Version: "2012-10-17",
            Statement: [
              {
                Sid: "AWSCloudTrailAclCheck20150319",
                Effect: "Allow",
                Principal: {
                  Service: "cloudtrail.amazonaws.com",
                },
                Action: "s3:GetBucketAcl",
                Resource: resource_arn,
              },
              {
                Sid: "AWSCloudTrailWrite20150319",
                Effect: "Allow",
                Principal: {
                  Service: "cloudtrail.amazonaws.com",
                },
                Action: "s3:PutObject",
                Resource: "%s/AWSLogs/%s/*" % [resource_arn, constants.AwsProjectId],
                Condition: {
                  StringEquals: {
                    "s3:x-amz-acl": "bucket-owner-full-control",
                  },
                },
              },
            ],
          },
        },
      },

      AuditingCloudTrail: {
        Type: "AWS::CloudTrail::Trail",
        DependsOn: ["BucketPolicy"],
        Properties: {
          S3BucketName: { Ref: "AuditLogBucket" },
          IncludeGlobalServiceEvents: "true",
          IsLogging: "true",
        },
      },

      // Create a Security Group for any Kubernetes master ELB.
      KubernetesMasterELBSecurityGroup: {
        Type: "AWS::EC2::SecurityGroup",
        Properties: {
          GroupDescription: "Security group for the Kubernetes master API ELB.",
          Tags: cf_utils.make_tags({ Name: "kubernetes-master-elb-security-group" }),
          VpcId: { Ref: "VpcId" },
          SecurityGroupEgress: [
            // Allow communication to the Kubernetes master, defined below due to circular dependency.
          ],
          SecurityGroupIngress: [
            // Allow HTTPS from corp IPs
            {
              CidrIp: corp_cidr,
              IpProtocol: "TCP",
              FromPort: "443",
              ToPort: "443",
            } for corp_cidr in network_cidrs.corp_cidr_blocks_list()
          ],
        },
      },
      KubernetesMasterELBSecurityGroupEgress: {
        Type: "AWS::EC2::SecurityGroupEgress",
        DependsOn: ["KubernetesMasterELBSecurityGroup", "KubernetesNodeSecurityGroup"],
        Properties: {
          GroupId: { "Fn::GetAtt": ["KubernetesMasterELBSecurityGroup", "GroupId"] },
          DestinationSecurityGroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          IpProtocol: "-1",
        },
      },

      // Create a Security Group for the Kubernetes core services ELB.
      KubernetesServicesELBSecurityGroup: {
        Type: "AWS::EC2::SecurityGroup",
        Properties: {
          GroupDescription: "Security group for the Kubernetes core services ELB.",
          Tags: cf_utils.make_tags({ Name: "kubernetes-services-elb-security-group" }),
          VpcId: { Ref: "VpcId" },
          SecurityGroupEgress: [
            // Allow communication to the Kubernetes nodes, defined below due to circular dependency.
          ],
          SecurityGroupIngress: [
            // Allow all traffic from corp IPs
            {
              CidrIp: corp_cidr,
              IpProtocol: "-1",
            } for corp_cidr in network_cidrs.corp_cidr_blocks_list()
          ],
        },
      },

      KubernetesServicesELBSecurityGroupEgress: {
        Type: "AWS::EC2::SecurityGroupEgress",
        DependsOn: ["KubernetesServicesELBSecurityGroup", "KubernetesNodeSecurityGroup"],
        Properties: {
          GroupId: { "Fn::GetAtt": ["KubernetesServicesELBSecurityGroup", "GroupId"] },
          DestinationSecurityGroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          IpProtocol: "-1",
        },
      },

      // Create a Security Group to house the Kubernetes nodes.
      KubernetesNodeSecurityGroup: {
        Type: "AWS::EC2::SecurityGroup",
        DependsOn: ["KubernetesMasterELBSecurityGroup", "KubernetesServicesELBSecurityGroup"],
        Properties: {
          GroupDescription: "Security group for Kubernetes nodes.",
          Tags: cf_utils.make_tags({ Name: "kubernetes-node-security-group" }),
          VpcId: { Ref: "VpcId" },
          SecurityGroupEgress: [
            // HTTP for CoreOS updates
            {
              CidrIp: "0.0.0.0/0",
              IpProtocol: "TCP",
              FromPort: "80",
              ToPort: "80",
            },
            // HTTPS for CoreOS updates
            {
              CidrIp: "0.0.0.0/0",
              IpProtocol: "TCP",
              FromPort: "443",
              ToPort: "443",
            },
            // ETCD cluster
            {
              CidrIp: network_cidrs.vpc_cidr_block(aws_region),
              IpProtocol: "TCP",
              FromPort: etcd_cluster_template.etcd_client_port(),
              ToPort: etcd_cluster_template.etcd_client_port(),
            },
            // ICMP for debugging
            {
              CidrIp: "0.0.0.0/0",
              IpProtocol: "ICMP",
              FromPort: "-1",
              ToPort: "-1",
            },
          ],
          SecurityGroupIngress: [
            // Allow HTTPS incoming for serving API traffic from inside the cluster.
            {
              CidrIp: network_cidrs.vpc_cidr_block(aws_region),
              IpProtocol: "TCP",
              FromPort: "443",
              ToPort: "443",
            },
            // Allow HTTPS incoming from the public ELB for serving API traffic from outside the cluster.
            {
              SourceSecurityGroupId: { "Fn::GetAtt": ["KubernetesMasterELBSecurityGroup", "GroupId"] },
              IpProtocol: "TCP",
              FromPort: "443",
              ToPort: "443",
            },
            // Allow TCP on 8080 incoming from the public ELB for serving healthz.
            {
              SourceSecurityGroupId: { "Fn::GetAtt": ["KubernetesMasterELBSecurityGroup", "GroupId"] },
              IpProtocol: "TCP",
              FromPort: "8080",
              ToPort: "8080",
            },
            // Allow incoming debugging SSH connections from inside the VPC.
            {
              CidrIp: network_cidrs.vpc_cidr_block(aws_region),
              IpProtocol: "TCP",
              FromPort: "22",
              ToPort: "22",
            },
            // ICMP for debugging
            {
              CidrIp: "0.0.0.0/0",
              IpProtocol: "ICMP",
              FromPort: "-1",
              ToPort: "-1",
            },
            // Kubernetes core services ELB
            {
              SourceSecurityGroupId: { "Fn::GetAtt": ["KubernetesServicesELBSecurityGroup", "GroupId"] },
              IpProtocol: "-1",
            },
          ] + [
            // Allow incoming debugging SSH connections from corp VPN blocks.
            {
              CidrIp: vpn_block,
              IpProtocol: "TCP",
              FromPort: "22",
              ToPort: "22",
            } for vpn_block in network_cidrs.corp_internal_cidr_blocks_list()
          ],
        },
      },

      // Allow nodes to speak with each other on any ports.
      KubernetesInternalIngress: {
        Type: "AWS::EC2::SecurityGroupIngress",
        DependsOn: ["KubernetesNodeSecurityGroup"],
        Properties: {
          GroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          SourceSecurityGroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          IpProtocol: "-1",
        },
      },
      KubernetesInternalEgress: {
        Type: "AWS::EC2::SecurityGroupEgress",
        DependsOn: ["KubernetesNodeSecurityGroup"],
        Properties: {
          GroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          DestinationSecurityGroupId: { "Fn::GetAtt": ["KubernetesNodeSecurityGroup", "GroupId"] },
          IpProtocol: "-1",
        },
      },

      // Create an IAM User, available to Kubernetes, for the kube-system namespace.
      // We don't use a role as that would enable any pod in the cluster to access any environment resources
      // with impunity, so we'll need to manually pull a secret key out of the AWS console and stick it into
      // an encrypted Kubernetes secret where needed.
      NodeUser: {
        Type: "AWS::IAM::User",
        Properties: {
          Path: "/infra/kube/",
        },
      },

      // Give the Kubernetes node role the ability to modify cluster resources required for managing pods,
      // volumes, etc.
      KubernetesNodeAWSAccessPolicy: {
        Type: "AWS::IAM::Policy",
        Properties: {
          PolicyName: "KubernetesNodeAWSAccessPolicy",
          Roles: [{ Ref: "KubeNodeRoleName" }],
          Users: [{ Ref: "NodeUser" }],
          PolicyDocument: {
            Version: "2012-10-17",
            Statement: [
              // Enables kubelet to query metadata about its environment.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: "ec2:Describe*",
              },
              // Allows the instance to self-modify the source/dest checking parameter for overlay networking.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: "ec2:ModifyInstanceAttribute",
              },
              // Allows the instance to mount EBS volumes for use by containers needing persistent storage.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: "ec2:AttachVolume",
              },
              // Allows the instance to unmount EBS volumes for use by containers needing persistent storage.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: "ec2:DetachVolume",
              },
              // Allow the instance to manage EBS snapshots.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: [
                  "ec2:CreateSnapshot",
                  "ec2:DeleteSnapshot",
                  "ec2:ModifySnapshotAttribute",
                  "ec2:ResetSnapshotAttribute",
                  "ec2:CreateTags",
                  "ec2:DeleteTags",
                ],
              },
            ],
          },
        },
      },

      // Give the Kubernetes master role the ability to read EC2 state and setup new ELBs.
      KubernetesMasterAWSAccessPolicy: {
        Type: "AWS::IAM::Policy",
        Properties: {
          PolicyName: "KubernetesMasterAWSAccessPolicy",
          Roles: [{ Ref: "KubeMasterRoleName" }],
          PolicyDocument: {
            Version: "2012-10-17",
            Statement: [
              // TODO: This is too broad - narrow this in scope.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: ["ec2:*"],
              },
              // This enables the master to create and manage load balancers to back Kubernetes service objects.
              {
                Effect: "Allow",
                Resource: ["*"],
                Action: ["elasticloadbalancing:*"],
              },
            ],
          },
        },
      },
    },
  },

  // Helpers for retrieving output fields from this stack in other stacks.
  common_infra_output_named(region, input_param_name):: {
    Param: input_param_name,
    Source: {
      Stack: $.stack_name(region),
      Output: input_param_name,
    },
  },

  audit_bucket_name(region)::
    "%s-%s-aws-audit-logs" % [constants.StorageBucketPrefix, region],
}
