/*
 * Regional AWS network specification template.
 *
 * This should be used to stamp out regional network infrastructure for each of our regions.
 */

local alert_utils = import "../utils/alert_utils.jsonnet";
local aws_images = import "../utils/aws_images.jsonnet";
local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";
local dns_utils = import "../utils/dns_utils.jsonnet";
local network_cidrs = import "../utils/network_cidrs.jsonnet";

{
  /**
   * Common template for the regional AWS-level network.
   *
   * Derive from this for each region we run in.
   */
  regional_network_definition(aws_region, aws_zones)::
    local stack_name = "vpc-" + aws_region;

    cf_utils.cf_template(
      aws_region + " regional network stack",
      stack_name,
      aws_region,
      [],
      []) {

      // Export AWS chosen values so other stacks can depend on these objects.
      Outputs: {
        // VPC ID
        VpcId: {
          Description: "Id of the VPC created by this stack.",
          Value: { Ref: "VPC" },
        },

        // VPCBreakGlassSecurityGroup
        VPCBreakGlassSecurityGroup: {
          Description: "Id of the VPCBreakGlassSecurityGroup created by this stack.",
          Value: { Ref: "VPCBreakGlassSecurityGroup" },
        },

        // VPCNATInstanceSecurityGroup
        VPCNATInstanceSecurityGroup: {
          Description: "Id of the VPCNATInstanceSecurityGroup created by this stack.",
          Value: { Ref: "VPCNATInstanceSecurityGroup" },
        },
      }
               // Zone to subnet map (private/public)
               + {
        ["PublicSubnet" + cf_utils.logical_zone_name(zone)]: {
          Description: "Id of the public subnet for the " + zone + " zone.",
          Value: { Ref: $.logical_public_subnet_name(zone) },
        } for zone in aws_zones
      } + {
        ["PrivateSubnet" + cf_utils.logical_zone_name(zone)]: {
          Description: "Id of the private subnet for the " + zone + " zone.",
          Value: { Ref: $.logical_private_subnet_name(zone) },
        } for zone in aws_zones
      },

      Resources: {
        // Create a single VPC for the region.
        VPC: {
          Type: "AWS::EC2::VPC",
          Properties: {
            CidrBlock: network_cidrs.vpc_cidr_block(aws_region),
            EnableDnsSupport: "true",
            EnableDnsHostnames: "true",
            InstanceTenancy: "default",
            Tags: cf_utils.make_tags({
              // The current Kubernetes release (v1.0.6) uses the hardcoded name "kubernetes-vpc" in its VPC lookup.
              // Temporarily rename our VPC to match.
              Name: "kubernetes-vpc",
            }),
          },
        },

        // Create a public hosted DNS zone for ops endpoints.  This will be where we expose the Kubernetes master
        // and embedded services.
        PublicHostedDNSZone: {
          Type: "AWS::Route53::HostedZone",
          DependsOn: ["VPC"],
          Properties: {
            Name: dns_utils.relative_public_ops_zone_name(),
            HostedZoneConfig: {
              Comment: "Public Hosted DNS Zone for ops endpoints.",
            },
          },
        },

        // Create a private hosted DNS zone for this VPC.  We can use this to provide A and CNAME records
        // for internal routing.
        PrivateHostedDNSZone: {
          Type: "AWS::Route53::HostedZone",
          DependsOn: ["VPC"],
          Properties: {
            Name: dns_utils.relative_hosted_zone_name(aws_region),
            VPCs: [{
              VPCId: { Ref: "VPC" },
              VPCRegion: aws_region,
            }],
            HostedZoneConfig: {
              Comment: "Private Hosted DNS Zone for the " + aws_region + " region.",
            },
          },
        },

        // A security group per-region providing manual override for "breaking glass" access.
        // Typically used for employee SSH or KUBECTL access to resources when off VPC.
        VPCBreakGlassSecurityGroup: {
          Type: "AWS::EC2::SecurityGroup",
          DependsOn: ["VPC"],
          Properties: {
            GroupDescription: "Security group for controlling Bastian and KUBECTL access overrides.",
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({ Name: "vpc-break-glass-security-group" }),
            SecurityGroupIngress: [
              // Manual maintenance for ingress entries
            ],
            SecurityGroupEgress: [
              // No restrictions on out-bound traffic
              {
                CidrIp: "0.0.0.0/0",
                IpProtocol: "-1",
                FromPort: "-1",
                ToPort: "-1",
              },
            ],
          },
        },

        // A security group representing VPC NAT instances running in the public subnets.
        VPCNATInstanceSecurityGroup: {
          Type: "AWS::EC2::SecurityGroup",
          DependsOn: ["VPC"],
          Properties: {
            GroupDescription: "Security group for NAT instances in public subnets.",
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({ Name: "vpc-nat-instance-security-group" }),
            SecurityGroupIngress: [
              // Allow SSH from corp IPs over the Internet
              {
                CidrIp: cidr,
                IpProtocol: "TCP",
                FromPort: "22",
                ToPort: "22",
              } for cidr in network_cidrs.corp_cidr_blocks_list()]
                                  // Allow SSH from VPCBreakGlassSecurityGroup
                                  + [{
              IpProtocol: "TCP",
              FromPort: "22",
              ToPort: "22",
              SourceSecurityGroupId: { Ref: "VPCBreakGlassSecurityGroup" },
            }]
                                  // Allow HTTP from each private subnet
                                  + [{
              CidrIp: network_cidrs.private_subnet_cidr_block(zone),
              IpProtocol: "TCP",
              FromPort: "80",
              ToPort: "80",
            } for zone in aws_zones]
                                  // Allow HTTPS from each private subnet
                                  + [{
              CidrIp: network_cidrs.private_subnet_cidr_block(zone),
              IpProtocol: "TCP",
              FromPort: "443",
              ToPort: "443",
            } for zone in aws_zones],
            SecurityGroupEgress: [
              // Allow outbound HTTP traffic to the Internet, so instances in private subnets can get updates.
              {
                CidrIp: "0.0.0.0/0",
                IpProtocol: "TCP",
                FromPort: "80",
                ToPort: "80",
              },
              // Allow outbound HTTPS traffic to the Internet, so instances in private subnets can get updates.
              {
                CidrIp: "0.0.0.0/0",
                IpProtocol: "TCP",
                FromPort: "443",
                ToPort: "443",
              },
              // Allow SSH from NAT instances to the private subnets, used by developers to debug machines in
              // the private subnets.
              {
                CidrIp: network_cidrs.vpc_cidr_block(aws_region),
                IpProtocol: "TCP",
                FromPort: "22",
                ToPort: "22",
              },
            ],
          },
        },

        // Create a public Internet routing table with an attached Internet gateway used by public subnets.
        InternetFacingRouteTable: {
          Type: "AWS::EC2::RouteTable",
          DependsOn: ["VPC"],
          Properties: {
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({ Name: "public-route-table" }),
          },
        },
        InternetRoute: {
          Type: "AWS::EC2::Route",
          DependsOn: ["InternetFacingRouteTable", "InternetGatewayAttachment"],
          Properties: {
            DestinationCidrBlock: "0.0.0.0/0",
            GatewayId: { Ref: "InternetGateway" },
            RouteTableId: { Ref: "InternetFacingRouteTable" },
          },
        },
        InternetGateway: {
          Type: "AWS::EC2::InternetGateway",
          Properties: {
            Tags: cf_utils.make_tags({ Name: "public-internet-gateway" }),
          },
        },
        InternetGatewayAttachment: {
          Type: "AWS::EC2::VPCGatewayAttachment",
          DependsOn: ["VPC", "InternetGateway"],
          Properties: {
            InternetGatewayId: { Ref: "InternetGateway" },
            VpcId: { Ref: "VPC" },
          },
        },

        // Allow direct access to S3 from within the VPC, without having to transit the Internet.
        // This endpoint is non-restrictive at the moment - we'll control bucket policies individually as needed.
        S3Endpoint: {
          Type: "AWS::EC2::VPCEndpoint",
          DependsOn: ["InternetRoute"] + [$.logical_private_route_table_name(zone) for zone in aws_zones],
          Properties: {
            PolicyDocument: {
              Version: "2012-10-17",
              Statement: [{
                Effect: "Allow",
                Principal: "*",
                Action: ["s3:*"],
                Resource: ["arn:aws:s3:::*"],
              }],
            },
            RouteTableIds: [
              { Ref: "InternetFacingRouteTable" },
            ] + [
              { Ref: $.logical_private_route_table_name(zone) } for zone in aws_zones
            ],
            ServiceName: "com.amazonaws.%s.s3" % aws_region,
            VpcId: { Ref: "VPC" },
          },
        },
      }

                 /*
                  * Public Subnet definitions, one per zone.
                  * Each public subnet is connected to the shared Internet facing routing table.
                  */
                 + {
        [$.logical_public_subnet_name(zone)]: {
          Type: "AWS::EC2::Subnet",
          DependsOn: ["VPC"],
          Properties: {
            AvailabilityZone: zone,
            CidrBlock: network_cidrs.public_subnet_cidr_block(zone),
            MapPublicIpOnLaunch: "true",
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({
              Name: $.friendly_public_subnet_name(zone),

              // Kubernetes looks up the subnet to insert new ELBs (to front external services) though matching
              // on VPC and this tag.  Ensure the public subnet, and not the private subnet, contains this tag to
              // force Kubernetes controlled ELBs to appear there correctly.
              KubernetesCluster: zone,
            }),
          },
        } for zone in aws_zones
      } + {
        [$.logical_public_subnet_route_assoc(zone)]: {
          local subnet_name = $.logical_public_subnet_name(zone),
          Type: "AWS::EC2::SubnetRouteTableAssociation",
          DependsOn: ["InternetFacingRouteTable", subnet_name],
          Properties: {
            RouteTableId: { Ref: "InternetFacingRouteTable" },
            SubnetId: { Ref: subnet_name },
          },
        } for zone in aws_zones
      }

                 /*
                  * VPC NAT instance definitions.  These instances serve two purposes:
                  * 1. Allow instances in the private subnets to contact the Internet for updates & reporting
                  * 2. Provide a NAT so devs can ssh to instances in the private subnet (until a VPN connection is established)
                  *
                  * AWS docs: http://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_NAT_Instance.html
                  *
                  * Migrate these to NAT Gateways once support is in CloudFormation.
                  */
                 + {
        [$.logical_vpc_nat_instance(zone)]: {
          Type: "AWS::EC2::Instance",
          DependsOn: [
            "VPCNATInstanceSecurityGroup",
            "VPCBreakGlassSecurityGroup",
            $.logical_public_subnet_name(zone),
          ],
          Properties: {
            AvailabilityZone: zone,

            ImageId: aws_images.amazon_vpc_nat_ami_id(aws_region),

            // Sizing choice from: http://www.azavea.com/blogs/labs/2015/01/selecting-a-nat-instance-size-on-ec2/
            InstanceType: "t2.medium",
            KeyName: constants.InstanceKeyName,
            Monitoring: "true",
            SecurityGroupIds: [
              { "Fn::GetAtt": ["VPCNATInstanceSecurityGroup", "GroupId"] },
              { "Fn::GetAtt": ["VPCBreakGlassSecurityGroup", "GroupId"] },
            ],

            // We'll be routing and forwarding traffic through this instance.
            SourceDestCheck: "false",
            SubnetId: { Ref: $.logical_public_subnet_name(zone) },
            Tags: cf_utils.make_tags({ Name: "vpc-public-nat-instance" }),
          },
        } for zone in aws_zones
      }

                 // Setup alerts for each ETC node instance
                 // CPU
                 + {
        [$.logical_vpc_nat_instance(zone) + "AlertsCPU"]:
          alert_utils.define_cpu_alert_with_default_sns_topic(stack_name, $.logical_vpc_nat_instance(zone))
        for zone in aws_zones
      }
                 // Instance Health
                 + {
        [$.logical_vpc_nat_instance(zone) + "AlertsInstanceHealth"]:
          alert_utils.define_instance_alert_with_default_sns_topic(stack_name, $.logical_vpc_nat_instance(zone))
        for zone in aws_zones
      }
                 // Instance Healing
                 + {
        [$.logical_vpc_nat_instance(zone) + "AlertsInstanceRecovery"]:
          alert_utils.define_instance_recovery_action_with_default_sns_topic(stack_name, $.logical_vpc_nat_instance(zone))
        for zone in aws_zones
      }

                 /*
                  * Private Subnet definitions.  Each subnet is connected to its own private routing table.
                  */
                 + {
        // The private per-zone subnets
        [$.logical_private_subnet_name(zone)]: {
          Type: "AWS::EC2::Subnet",
          DependsOn: ["VPC"],
          Properties: {
            AvailabilityZone: zone,
            CidrBlock: network_cidrs.private_subnet_cidr_block(zone),
            MapPublicIpOnLaunch: "false",
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({ Name: $.friendly_private_subnet_name(zone) }),
          },
        } for zone in aws_zones
      } + {
        // The private per-zone routing table.
        [$.logical_private_route_table_name(zone)]: {
          Type: "AWS::EC2::RouteTable",
          DependsOn: ["VPC"],
          Properties: {
            VpcId: { Ref: "VPC" },
            Tags: cf_utils.make_tags({ Name: $.friendly_private_route_table_name(zone) }),
          },
        } for zone in aws_zones
      } + {
        // The private per-zone route entry mapping Internet bound traffic to the public subnet's VPC NAT instance.
        [$.logical_private_forwarding_route_name(zone)]: {
          Type: "AWS::EC2::Route",
          DependsOn: [$.logical_vpc_nat_instance(zone)],
          Properties: {
            DestinationCidrBlock: "0.0.0.0/0",
            InstanceId: { Ref: $.logical_vpc_nat_instance(zone) },
            RouteTableId: { Ref: $.logical_private_route_table_name(zone) },
          },
        } for zone in aws_zones
      } + {
        // The private per-zone association of subnet to private routing table.
        [$.logical_private_subnet_route_assoc(zone)]: {
          local subnet_name = $.logical_private_subnet_name(zone),
          local route_table_name = $.logical_private_route_table_name(zone),

          Type: "AWS::EC2::SubnetRouteTableAssociation",
          DependsOn: [route_table_name, subnet_name],
          Properties: {
            RouteTableId: { Ref: route_table_name },
            SubnetId: { Ref: subnet_name },
          },
        } for zone in aws_zones
      },
    },

  /**
   * Generates the logical name of the public subnet for the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PublicSubnetuswesta
   */
  logical_public_subnet_name(zone)::
    "PublicSubnet" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the friendly name of the public subnet for the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding resource name; e.g., public-subnet-us-west-2a
   */
  friendly_public_subnet_name(zone)::
    "public-subnet-" + zone,

  /**
   * Generates the logical name of the object which associates a public subnet to its public route table.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PublicSubnetRouteAssocuswest2a
   */
  logical_public_subnet_route_assoc(zone)::
    "PublicSubnetRouteAssoc" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the logical name of the private subnet for the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PrivateSubnetuswest2a
   */
  logical_private_subnet_name(zone)::
    "PrivateSubnet" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the friendly name of the private subnet for the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding resource name; e.g., private-subnet-us-west-2a
   */
  friendly_private_subnet_name(zone)::
    "private-subnet-" + zone,

  /**
   * Generates the logical name of the object which associates a private subnet to its private route table.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PrivateSubnetRouteAssocuswest2a
   */
  logical_private_subnet_route_assoc(zone)::
    "PrivateSubnetRouteAssoc" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the logical name of the per-zone VPC NAT instance.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., NATInstanceuswest2a
   */
  logical_vpc_nat_instance(zone)::
    "NATInstance" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the logical name of the routing table for the private subnet in the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PrivateRouteTableuswest2a
   */
  logical_private_route_table_name(zone)::
    "PrivateRouteTable" + cf_utils.logical_zone_name(zone),

  /**
   * Generates the friendly name of the routing table for the private subnet in the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding resource name; e.g., private-route-table-us-west-2a
   */
  friendly_private_route_table_name(zone)::
    "private-route-table-" + zone,

  /**
   * Generates the logical name of the routing table entry for the private subnet in the given zone.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return a corresponding CloudFormation-friendly logical resource id; e.g., PrivateForwardingRouteuswest2a
   */
  logical_private_forwarding_route_name(zone)::
    "PrivateForwardingRoute" + cf_utils.logical_zone_name(zone),

  // Helpers for retrieving output fields from this stack in other stacks.
  vpc_id_output(region):: {
    Param: "VpcId",
    Source: { Stack: "vpc-" + region, Output: "VpcId" } },

  vpc_public_subnet_output_named(region, zone, input_param_name):: {
    local paramName = "PublicSubnet" + cf_utils.logical_zone_name(zone),
    Param: input_param_name,
    Source: { Stack: "vpc-" + region, Output: paramName } },

  vpc_public_subnet_output(region, zone)::
    $.vpc_public_subnet_output_named(region, zone, "PublicSubnet" + cf_utils.logical_zone_name(zone)),

  vpc_private_subnet_output_named(region, zone, input_param_name):: {
    local paramName = "PrivateSubnet" + cf_utils.logical_zone_name(zone),
    Param: input_param_name,
    Source: { Stack: "vpc-" + region, Output: paramName } },

  vpc_private_subnet_output(region, zone)::
    $.vpc_private_subnet_output_named(region, zone, "PrivateSubnet" + cf_utils.logical_zone_name(zone)),

  vpc_break_glass_security_group_output(region):: {
    Param: "VPCBreakGlassSecurityGroup",
    Source: { Stack: "vpc-" + region, Output: "VPCBreakGlassSecurityGroup" },
  },

  vpc_security_group_output(region, group):: {
    Param: group,
    Source: { Stack: "vpc-" + region, Output: group },
  },
}
