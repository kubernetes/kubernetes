/**
 * Template for an ETCD cluster running in an AWS region.
 */

local alert_utils = import "../utils/alert_utils.jsonnet";
local aws_images = import "../utils/aws_images.jsonnet";
local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";
local dns_utils = import "../utils/dns_utils.jsonnet";
local iam_config = import "../global/iam_config.jsonnet";
local network_cidrs = import "../utils/network_cidrs.jsonnet";
local network_template = import "network_template.jsonnet";

local etcd_peer_port = "2380";
local etcd_client_port = "2379";

// Define how this stack's parameters are filled from the output of other stacks.
local parameter_map(region, zones) = [
  network_template.vpc_id_output(region),
  iam_config.iam_output("EtcdNodeInstanceProfileName"),
] + [
  network_template.vpc_private_subnet_output(region, zone) for zone in zones
];

{
  regional_etcd_cluster_template(aws_region, aws_zones)::
    local stack_name = "etcd-cluster-" + aws_region;

    cf_utils.cf_template(
      aws_region + " regional ETCD cluster stack",
      stack_name,
      aws_region,
      [],
      parameter_map(aws_region, aws_zones)) {

      Parameters: {
        VpcId: {
          Type: "String",
          AllowedPattern: "vpc-.*",
          Description: "Vpc ID of the underlying pre-defined regional VPC.",
        },
        EtcdNodeInstanceProfileName: {
          Type: "String",
          Description: "Name of the IAM instance profile to run the ETCD nodes under.",
        },
      } + {
        ["PrivateSubnet" + cf_utils.logical_zone_name(zone)]: {
          Type: "String",
          Description: "Id of the private subnet for the " + zone + " zone.",
        } for zone in aws_zones
      },

      Resources: {
        // Create a Security Group to house the ETCD instance nodes.
        EtcdNodeSecurityGroup: {
          Type: "AWS::EC2::SecurityGroup",
          Properties: {
            GroupDescription: "SecurityGroup for the regional ETCD instance nodes",
            Tags: cf_utils.make_tags({ Name: "etcd-node-security-group" }),
            VpcId: { Ref: "VpcId" },
            SecurityGroupEgress: [
              // ETCD Peer Port - specified below as it is self-referential.
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
            ],
            SecurityGroupIngress: [
              // ETCD Peer Port - specified below as it is self-referential.
              // ETCD Client Port
              {
                CidrIp: network_cidrs.vpc_cidr_block(aws_region),
                IpProtocol: "TCP",
                FromPort: etcd_client_port,
                ToPort: etcd_client_port,
              },
              // Allow incoming debugging SSH connections from inside the VPC.
              {
                CidrIp: network_cidrs.vpc_cidr_block(aws_region),
                IpProtocol: "TCP",
                FromPort: "22",
                ToPort: "22",
              },
            ],
          },
        },
        EtcdPeerPortIngressRule: {
          Type: "AWS::EC2::SecurityGroupIngress",
          DependsOn: ["EtcdNodeSecurityGroup"],
          Properties: {
            GroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            SourceSecurityGroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            IpProtocol: "TCP",
            FromPort: etcd_peer_port,
            ToPort: etcd_peer_port,
          },
        },
        EtcdPeerPortEgressRule: {
          Type: "AWS::EC2::SecurityGroupEgress",
          DependsOn: ["EtcdNodeSecurityGroup"],
          Properties: {
            GroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            DestinationSecurityGroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            IpProtocol: "TCP",
            FromPort: etcd_peer_port,
            ToPort: etcd_peer_port,
          },
        },
        EtcdClientPortEgressRule: {
          Type: "AWS::EC2::SecurityGroupEgress",
          DependsOn: ["EtcdNodeSecurityGroup"],
          Properties: {
            GroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            DestinationSecurityGroupId: { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            IpProtocol: "TCP",
            FromPort: etcd_client_port,
            ToPort: etcd_client_port,
          },
        },

        // Create internal DNS records for each ETCD node.
        EtcdNodeDNSRecordSetGroup: {
          Type: "AWS::Route53::RecordSetGroup",
          DependsOn: [$.logical_etcd_instance_name(zone) for zone in aws_zones],
          Properties: {
            HostedZoneName: dns_utils.absolute_hosted_zone_name(aws_region),
            Comment: "'A' records for the ETCD cluster nodes.",
            RecordSets: [
              {
                Name: dns_utils.absolute_instance_dns_name(aws_region, $.etcd_instance_host_name(zone)),
                Type: "A",
                TTL: "900",
                ResourceRecords: [
                  { "Fn::GetAtt": [$.logical_etcd_instance_name(zone), "PrivateIp"] },
                ],
              } for zone in aws_zones
            ],
          },
        },

        // Create internal DNS SRV record which contains all nodes.  This is used during the bootstrap
        // sequence so the nodes can find each other.
        EtcdBootstrapDNSRecordSet: {
          Type: "AWS::Route53::RecordSet",
          Properties: {
            HostedZoneName: dns_utils.absolute_hosted_zone_name(aws_region),
            Comment: "DNS SRV record for ETCD cluster nodes; used for cluster bootstrapping.",
            // TODO(chris): Consider adding an SSL endpoint when we add cert support.
            Name: dns_utils.absolute_instance_dns_name(aws_region, "_etcd-server._tcp"),
            Type: "SRV",
            TTL: "900",
            // SRV Format: <priority> <weight> <port> <domain name>
            ResourceRecords: [
              "0 0 %s %s" % [etcd_peer_port, $.relative_etcd_instance_dns_name(aws_region, zone)] for zone in aws_zones
            ],
          },
        },
      }

                 // Create one ETCD node per AZ.
                 // TODO: Consider driving this with an autoscaling group and getting the cluster to self-repair on instance
                 // failure.
                 + {
        [$.logical_etcd_instance_name(zone)]: {
          Type: "AWS::EC2::Instance",
          DependsOn: ["EtcdNodeSecurityGroup", "EtcdBootstrapDNSRecordSet"],
          Properties: {
            AvailabilityZone: zone,
            // Make the root disk 100 GB.
            BlockDeviceMappings: [{
              DeviceName: "/dev/xvda",
              Ebs: {
                DeleteOnTermination: "true",
                VolumeType: "gp2",
                VolumeSize: "100",
              },
            }],
            IamInstanceProfile: { Ref: "EtcdNodeInstanceProfileName" },
            ImageId: aws_images.coreos_ami_id(aws_region),
            InstanceType: "m3.medium",
            KeyName: constants.InstanceKeyName,
            Monitoring: "true",
            SecurityGroupIds: [
              { "Fn::GetAtt": ["EtcdNodeSecurityGroup", "GroupId"] },
            ],
            SubnetId: $.private_subnet_id_for_zone(zone),
            Tags: cf_utils.make_tags({ Name: "etcd-cluster-node" }),
            UserData: $.etcd_cloud_config(aws_region, zone),
          },
        } for zone in aws_zones
      }

                 // Setup alerts for each ETC node instance
                 // CPU
                 + {
        [$.logical_etcd_instance_name(zone) + "AlertsCPU"]:
          alert_utils.define_cpu_alert_with_default_sns_topic(stack_name, $.logical_etcd_instance_name(zone))
        for zone in aws_zones
      }
                 // Instance Health
                 + {
        [$.logical_etcd_instance_name(zone) + "AlertsInstanceHealth"]:
          alert_utils.define_instance_alert_with_default_sns_topic(stack_name, $.logical_etcd_instance_name(zone))
        for zone in aws_zones
      }
                 // Instance Healing
                 + {
        [$.logical_etcd_instance_name(zone) + "AlertsInstanceRecovery"]:
          alert_utils.define_instance_recovery_action_with_default_sns_topic(stack_name, $.logical_etcd_instance_name(zone))
        for zone in aws_zones
      },
    },

  etcd_cloud_config(region, zone):: std.base64(|||
    #cloud-config
    hostname: %(fqdn)s
    coreos:
      update:
        # Only allows one node in the cluster to reboot at a time.
        reboot-strategy: "etcd-lock"
        group: "stable"
      etcd2:
        name: %(host_name)s
        discovery-srv: %(domain_name)s
        initial-advertise-peer-urls: "http://$private_ipv4:%(etcd_peer_port)s"
        initial-cluster-token: "etcd-cluster-1"
        initial-cluster-state: "new"
        advertise-client-urls: "http://$private_ipv4:%(etcd_client_port)s"
        # Choose 0.0.0.0 here to bind to all interfaces, including localhost.
        listen-client-urls: "http://0.0.0.0:%(etcd_client_port)s"
        listen-peer-urls: "http://$private_ipv4:%(etcd_peer_port)s"
      units:
      - name: etcd.service
        mask: true
      - name: etcd2.service
        command: restart
  ||| % {
    host_name: $.etcd_instance_host_name(zone),
    fqdn: $.relative_etcd_instance_dns_name(region, zone),
    domain_name: dns_utils.relative_hosted_zone_name(region),
    etcd_peer_port: etcd_peer_port,
    etcd_client_port: etcd_client_port,
  }),

  private_subnet_id_for_zone(zone)::
    { Ref: "PrivateSubnet" + cf_utils.logical_zone_name(zone) },

  logical_etcd_instance_name(zone)::
    "EtcdNodeInstance" + cf_utils.logical_zone_name(zone),

  relative_etcd_instance_dns_name(region, zone)::
    dns_utils.relative_instance_dns_name(region, $.etcd_instance_host_name(zone)),

  etcd_instance_host_name(zone)::
    "etcd-" + cf_utils.zone_suffix(zone),

  // Export the client port for use by the Kubernetes template for use in external security groups, etc.
  etcd_client_port()::
    etcd_client_port,

  // Expose a list of the endpoints for this region for dependencies like flanneld.
  etcd_endpoints(region, zones):: [
    "http://%(hostname)s:%(port)s" % {
      hostname: dns_utils.relative_instance_dns_name(region, $.etcd_instance_host_name(zone)),
      port: etcd_client_port,
    } for zone in zones
  ],
}
