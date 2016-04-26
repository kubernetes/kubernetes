/**
 * Template for a Kubernetes cluster running in an AWS availability zone.
 *
 * Depends on a running regional network fabric and etcd cluster.
 */

local alert_utils = import "../utils/alert_utils.jsonnet";
local aws_images = import "../utils/aws_images.jsonnet";
local cert_storage_template = import "cert_storage_template.jsonnet";
local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";
local dns_utils = import "../utils/dns_utils.jsonnet";
local etcd_cluster_template = import "etcd_cluster_template.jsonnet";
local iam_config = import "../global/iam_config.jsonnet";
local network_cidrs = import "../utils/network_cidrs.jsonnet";
local network_template = import "network_template.jsonnet";
local regional_common_infra_template = import "regional_common_infra_template.jsonnet";

local kube_version = "v1.2.1";

// Define how this stack's parameters are filled from the output of other stacks.
local parameter_map(region, zone) = [
  network_template.vpc_id_output(region),
  network_template.vpc_public_subnet_output_named(region, zone, "PublicSubnetId"),
  network_template.vpc_private_subnet_output_named(region, zone, "PrivateSubnetId"),
  network_template.vpc_break_glass_security_group_output(region),
  iam_config.iam_output("KubeMasterInstanceProfileName"),
  iam_config.iam_output("KubeNodeInstanceProfileName"),
  regional_common_infra_template.common_infra_output_named(region, "KubernetesServicesELBSecurityGroup"),
  regional_common_infra_template.common_infra_output_named(region, "KubernetesNodeSecurityGroupId"),
  regional_common_infra_template.common_infra_output_named(region, "KubernetesMasterELBSecurityGroup"),
];

{
  // TODO(chris): Eliminate the peer_zones parameter once the etcd client supports
  // SRV DNS resolution.
  kube_cluster_template(aws_region, aws_zone, peer_zones, node_count)::
    local stack_name = "kube-cluster-" + aws_zone;

    cf_utils.cf_template(
      aws_zone + " kubernetes cluster stack",
      stack_name,
      aws_region,
      ["CAPABILITY_IAM"],
      parameter_map(aws_region, aws_zone)) {

      Parameters: {
        VpcId: {
          Type: "String",
          AllowedPattern: "vpc-.*",
          Description: "Vpc ID of the underlying pre-defined regional VPC.",
        },
        PrivateSubnetId: {
          Type: "String",
          AllowedPattern: "subnet-.*",
          Description: "Private subnet ID for this zone.",
        },
        PublicSubnetId: {
          Type: "String",
          AllowedPattern: "subnet-.*",
          Description: "Public subnet ID for this zone.",
        },
        VPCBreakGlassSecurityGroup: {
          Type: "String",
          Description: "Regional security group used to break glass for non-corpnet operator access.",
        },
        KubeMasterInstanceProfileName: {
          Type: "String",
          Description: "Name of the IAM instance profile for the master node.",
        },
        KubeNodeInstanceProfileName: {
          Type: "String",
          Description: "Name of the IAM instance profile for the kubernetes nodes.",
        },
        KubernetesServicesELBSecurityGroup: {
          Type: "String",
          Description: "Name of the IAM role for the kubernetes nodes.",
        },
        KubernetesMasterELBSecurityGroup: {
          Type: "String",
          Description: "Security group for the Kubernetes master API ELB.",
        },
        KubernetesNodeSecurityGroupId: {
          Type: "String",
          Description: "Security groupId for kubelet nodes within each Kube cluster.",
        },
      },

      Outputs: {
        MetricsPersistentVolumeId: {
          Description: "EC2 volume ID of the EBS volume used to store persistent metrics data for influxdb.",
          Value: { Ref: "MetricsStorageVolume" },
        },

        LogsPersistentVolumeId: {
          Description: "EC2 volume ID of the EBS volume used to store persistent logs data for elasticsearch.",
          Value: { Ref: "LogsStorageVolume" },
        },
      },

      Resources: {
        // Create the Kubernetes master
        KubernetesMasterInstance: {
          Type: "AWS::EC2::Instance",
          Properties: {
            AvailabilityZone: aws_zone,
            // Make the root disk 100 GB.
            BlockDeviceMappings: [{
              DeviceName: "/dev/xvda",
              Ebs: {
                DeleteOnTermination: "true",
                VolumeType: "gp2",
                VolumeSize: "100",
              },
            }],
            IamInstanceProfile: { Ref: "KubeMasterInstanceProfileName" },
            ImageId: aws_images.coreos_ami_id(aws_region),
            InstanceType: "m3.large",
            KeyName: constants.InstanceKeyName,
            Monitoring: "true",
            SecurityGroupIds: [{ Ref: "KubernetesNodeSecurityGroupId" }],
            SourceDestCheck: "false",
            SubnetId: { Ref: "PrivateSubnetId" },
            Tags: cf_utils.make_tags({
              Name: "kubernetes-master",

              // The Kubernetes stack looks for this Tag in the EC2 metadata server as a way of isolating logically
              // separated clusters running in the same AZ.
              // See pkg/cloudprovider/providers/aws/aws.go#L50 in the kubernetes source for more details.
              KubernetesCluster: aws_zone,
            }),
            UserData: std.base64($.kube_master_cloud_config(aws_region, aws_zone, peer_zones)),
          },
        },

        // Create default alerts for the Kubernets master instance
        // CPU
        KubernetesMasterInstanceAlertsCPU:
          alert_utils.define_cpu_alert_with_default_sns_topic(stack_name, "KubernetesMasterInstance"),

        // Instance Health
        KubernetesMasterInstanceAlertsInstanceHealth:
          alert_utils.define_instance_alert_with_default_sns_topic(stack_name, "KubernetesMasterInstance"),

        // Instance Recovery
        KubernetesMasterInstanceAlertsInstanceRecovery:
          alert_utils.define_instance_recovery_action_with_default_sns_topic(stack_name, "KubernetesMasterInstance"),

        // Create an ELB for the master API.
        KubernetesMasterELBTarget: {
          Type: "AWS::ElasticLoadBalancing::LoadBalancer",
          DependsOn: ["KubernetesMasterInstance"],
          Properties: {
            ConnectionDrainingPolicy: {
              Enabled: "true",
              Timeout: "15",
            },
            ConnectionSettings: {
              IdleTimeout: "120",
            },
            LoadBalancerName: "kube-master-api-lb-" + aws_zone,
            HealthCheck: {
              HealthyThreshold: "2",
              Interval: "30",
              Target: "HTTP:8080/healthz",
              Timeout: "10",
              UnhealthyThreshold: "3",
            },
            Instances: [{ Ref: "KubernetesMasterInstance" }],
            Listeners: [{
              InstancePort: "443",
              InstanceProtocol: "TCP",
              LoadBalancerPort: "443",
              Protocol: "TCP",
            }],
            Scheme: "internet-facing",
            SecurityGroups: [
              { Ref: "KubernetesMasterELBSecurityGroup" },
              { Ref: "VPCBreakGlassSecurityGroup" },
            ],
            Subnets: [{ Ref: "PublicSubnetId" }],
            Tags: cf_utils.make_tags({ Name: "kube-master-api-lb" }),
          },
        },

        // Create a friendly DNS A record for the Master's private IP, for use within the VPC.
        KubernetesMasterDNSRecord: {
          Type: "AWS::Route53::RecordSet",
          DependsOn: ["KubernetesMasterInstance"],
          Properties: {
            Name: $.absolute_kube_master_internal_dns_alias(aws_region, aws_zone),
            Comment: "Internal 'A' record for the Kubernetes master.",
            HostedZoneName: dns_utils.absolute_hosted_zone_name(aws_region),
            TTL: "900",
            Type: "A",
            ResourceRecords: [
              { "Fn::GetAtt": ["KubernetesMasterInstance", "PrivateIp"] },
            ],
          },
        },

        // Create a friendly DNS alias for the ELB endpoint.
        KubernetesMasterELBAlias: {
          Type: "AWS::Route53::RecordSet",
          DependsOn: ["KubernetesMasterELBTarget"],
          Properties: {
            Name: $.absolute_kube_master_api_dns_alias(aws_zone),
            Comment: "Friendly DNS alias for the Kubernetes Master API ELB.",
            Type: "A",
            HostedZoneName: dns_utils.absolute_public_ops_zone_name(),
            AliasTarget: {
              DNSName: { "Fn::GetAtt": ["KubernetesMasterELBTarget", "CanonicalHostedZoneName"] },
              HostedZoneId: { "Fn::GetAtt": ["KubernetesMasterELBTarget", "CanonicalHostedZoneNameID"] },
            },
          },
        },

        // Create an ELB for the core services.  We cannot use LoadBalancer type services in Kubernetes yet
        // because they're open to the world by default.
        KubernetesServicesELBTarget: {
          Type: "AWS::ElasticLoadBalancing::LoadBalancer",
          Properties: {
            ConnectionDrainingPolicy: {
              Enabled: "true",
              Timeout: "15",
            },
            ConnectionSettings: {
              IdleTimeout: "120",
            },
            LoadBalancerName: "kube-services-lb-" + aws_zone,
            HealthCheck: {
              HealthyThreshold: "2",
              Interval: "30",
              Target: "HTTP:32100/",
              Timeout: "10",
              UnhealthyThreshold: "3",
            },
            Listeners: [
              // Kube Dashboard
              {
                InstancePort: "32100",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "443",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
              // Kube UI
              {
                InstancePort: "32101",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "444",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
              // Kibana
              {
                InstancePort: "32102",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "445",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
              // Kibana doesn't work properly behind SSL at the moment, so push this through port 80 as well until
              // we figure it out.
              {
                InstancePort: "32102",
                InstanceProtocol: "HTTP",
                LoadBalancerPort: "80",
                Protocol: "HTTP",
              },
              // InfluxDB HTTP
              {
                InstancePort: "32103",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "446",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
              // Grafana
              {
                InstancePort: "32104",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "447",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
              // InfluxDB API
              {
                InstancePort: "32105",
                InstanceProtocol: "TCP",
                LoadBalancerPort: "448",
                PolicyNames: ["ELBSecurityPolicy-2015-05"],
                Protocol: "SSL",
                SSLCertificateId: constants.KubeServicesCertARN,
              },
            ],
            Scheme: "internet-facing",
            SecurityGroups: [{ Ref: "KubernetesServicesELBSecurityGroup" }],
            Subnets: [{ Ref: "PublicSubnetId" }],
            Tags: cf_utils.make_tags({ Name: "kube-services-lb" }),
          },
        },

        // Create a friendly DNS alias for the ELB endpoint.
        KubernetesServicesELBAlias: {
          Type: "AWS::Route53::RecordSet",
          DependsOn: ["KubernetesServicesELBTarget"],
          Properties: {
            Name: $.absolute_kube_services_dns_alias(aws_zone),
            Comment: "Friendly DNS alias for the Kubernetes Services ELB.",
            Type: "A",
            HostedZoneName: dns_utils.absolute_public_ops_zone_name(),
            AliasTarget: {
              DNSName: { "Fn::GetAtt": ["KubernetesServicesELBTarget", "CanonicalHostedZoneName"] },
              HostedZoneId: { "Fn::GetAtt": ["KubernetesServicesELBTarget", "CanonicalHostedZoneNameID"] },
            },
          },
        },

        // Create the Kubernetes node auto-scaler.  We'll control this manually for now.
        KubernetesNodeLaunchConfig: {
          Type: "AWS::AutoScaling::LaunchConfiguration",
          Properties: {
            // Make the root disk 100 GB.
            BlockDeviceMappings: [{
              DeviceName: "/dev/xvda",
              Ebs: {
                DeleteOnTermination: "true",
                VolumeType: "gp2",
                VolumeSize: "100",
              },
            }],
            IamInstanceProfile: { Ref: "KubeNodeInstanceProfileName" },
            ImageId: aws_images.coreos_ami_id(aws_region),
            InstanceMonitoring: "true",
            InstanceType: "m4.2xlarge",
            KeyName: constants.InstanceKeyName,
            SecurityGroups: [{ Ref: "KubernetesNodeSecurityGroupId" }],
            UserData: std.base64($.kube_node_cloud_config(aws_region, aws_zone, peer_zones)),
          },
        },

        KubernetesNodeAutoscaler: {
          Type: "AWS::AutoScaling::AutoScalingGroup",
          DependsOn: ["KubernetesNodeLaunchConfig", "KubernetesMasterDNSRecord"],
          // TODO: This could be improved to wait for each node to fully come up before tearing down the next.
          UpdatePolicy: {
            AutoScalingRollingUpdate: {
              MinInstancesInService: "2",
              MaxBatchSize: "1",
              PauseTime: "PT1M",
            },
            AutoScalingScheduledAction: {
              IgnoreUnmodifiedGroupSizeProperties: "true",
            },
          },
          Properties: {
            HealthCheckGracePeriod: "60",
            HealthCheckType: "EC2",
            LaunchConfigurationName: { Ref: "KubernetesNodeLaunchConfig" },
            LoadBalancerNames: [
              { Ref: "KubernetesServicesELBTarget" },
            ],
            MaxSize: "12",
            MinSize: "0",
            DesiredCapacity: node_count,
            NotificationConfigurations:
              alert_utils.define_auto_scaler_change_notifications_with_default_sns_alert_topic_arn(),
            Tags: [
              { Key: "Name", Value: "kubernetes-node", PropagateAtLaunch: "true" },

              // The Kubernetes stack looks for this Tag in the EC2 metadata server as a way of isolating logically
              // separated clusters running in the same AZ.
              // See pkg/cloudprovider/providers/aws/aws.go#L50 in the kubernetes source for more details.
              { Key: "KubernetesCluster", Value: aws_zone, PropagateAtLaunch: "true" },
            ],
            VPCZoneIdentifier: [{ Ref: "PrivateSubnetId" }],

          },
        },

        // Setup AutoScaler alerts
        // CPU
        CPUAlarmHighAlert: alert_utils.define_auto_scaler_cpu_alert_with_default_sns_topic(stack_name,
                                                                                           "KubernetesNodeAutoscaler"),

        // Create an EBS volume to store long term metrics.
        MetricsStorageVolume: {
          Type: "AWS::EC2::Volume",
          DeletionPolicy: "Retain",  // Don't delete this if we tear down the stack.
          Properties: {
            AvailabilityZone: aws_zone,
            Size: "512",  // 512 GB to start with
            Tags: cf_utils.make_tags({ Name: aws_zone + "-metrics-data" }),
            VolumeType: "gp2",
          },
        },

        // Create an EBS volume to store logs for elasticsearch.
        LogsStorageVolume: {
          Type: "AWS::EC2::Volume",
          DeletionPolicy: "Retain",  // Don't delete this if we tear down the stack.
          Properties: {
            AvailabilityZone: aws_zone,
            Size: "1024",  // 1 TB to start with
            Tags: cf_utils.make_tags({ Name: aws_zone + "-logs-data" }),
            VolumeType: "gp2",
          },
        },
      },
    },

  etcd_endpoints(region, peer_zones)::
    std.join(",", etcd_cluster_template.etcd_endpoints(region, peer_zones)),

  absolute_kube_master_api_dns_alias(zone)::
    "%s.kube.%s" % [zone, dns_utils.absolute_public_ops_zone_name()],

  absolute_kube_services_dns_alias(zone)::
    "%s.kube-services.%s" % [zone, dns_utils.absolute_public_ops_zone_name()],

  relative_kube_master_internal_dns_alias(region, zone)::
    dns_utils.relative_instance_dns_name(region, "kube-master-" + zone),

  absolute_kube_master_internal_dns_alias(region, zone)::
    dns_utils.absolute_instance_dns_name(region, "kube-master-" + zone),

  kube_master_cloud_config(region, zone, peer_zones)::
    (importstr "kube_master_cloud_config.yaml") % {
      etcd_endpoints: $.etcd_endpoints(region, peer_zones),
      overlay_network_cidr: network_cidrs.overlay_cidr_block(region),
      cert_bucket: cert_storage_template.bucket_name(region),
      kube_version: kube_version,
      kube_dns_ip: network_cidrs.kube_internal_dns_ip(region),
      kube_apiserver_content: $.offset_multiline_string(6, $.kube_apiserver_yaml(region, zone, peer_zones)),
      kube_proxy_content: $.offset_multiline_string(6, $.kube_proxy_master_yaml()),
      kube_controller_manager_content: $.offset_multiline_string(6, $.kube_controller_manager_yaml(zone)),
      kube_scheduler_content: $.offset_multiline_string(6, $.kube_scheduler_yaml()),
      log_rotate_content: $.offset_multiline_string(6, $.log_rotate_conf()),
    },

  kube_node_cloud_config(region, zone, peer_zones)::
    (importstr "kube_node_cloud_config.yaml") % {
      etcd_endpoints: $.etcd_endpoints(region, peer_zones),
      cert_bucket: cert_storage_template.bucket_name(region),
      kube_version: kube_version,
      kube_dns_ip: network_cidrs.kube_internal_dns_ip(region),
      kube_master_address: $.relative_kube_master_internal_dns_alias(region, zone),
      kube_proxy_content: $.offset_multiline_string(6, $.kube_proxy_node_yaml(region, zone)),
      kube_config_content: $.offset_multiline_string(6, $.kube_config_node_yaml()),
      log_rotate_content: $.offset_multiline_string(6, $.log_rotate_conf()),
    },

  kube_apiserver_yaml(region, zone, peer_zones)::
    (importstr "kube_apiserver.yaml") % {
      cluster_prefix: zone,
      etcd_endpoints: $.etcd_endpoints(region, peer_zones),
      service_ip_range: network_cidrs.kube_services_cidr_block(region),
      kube_version: kube_version,
    },

  kube_proxy_master_yaml()::
    (importstr "kube_proxy_master.yaml") % {
      kube_version: kube_version,
    },

  kube_proxy_node_yaml(region, zone)::
    (importstr "kube_proxy_node.yaml") % {
      kube_version: kube_version,
      kube_master_address: $.relative_kube_master_internal_dns_alias(region, zone),
    },

  kube_controller_manager_yaml(zone)::
    (importstr "kube_controller_manager.yaml") % {
      cluster_prefix: zone,
      kube_version: kube_version,
    },

  kube_scheduler_yaml()::
    (importstr "kube_scheduler.yaml") % {
      kube_version: kube_version,
    },

  log_rotate_conf()::
    (importstr "log_rotate.conf"),

  kube_config_node_yaml()::
    (importstr "kube_config_node.yaml"),

  /**
   * Offsets the given multiline_string by indenting each line by offset spaces.
   */
  offset_multiline_string(offset, multiline_string)::
    local offset_string = std.join("", std.makeArray(offset, function(x) " "));
    local lines = std.split(multiline_string, "\n");
    local offset_lines = std.map(function(x) offset_string + x, lines);
    std.lines(offset_lines),
}
