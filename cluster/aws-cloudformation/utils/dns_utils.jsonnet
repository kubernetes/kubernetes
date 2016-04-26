/**
 * DNS utility methods
 */

local constants = import "constants.jsonnet";

{
  /**
   * Defines an 'A' record in the VPC-internal DNS fabric pointing to the instance's private IP.
   */
  define_internal_a_record(region, logical_instance_name, dns_name_prefix, comment):: {
    Type: "AWS::Route53::RecordSet",
    Properties: {
      HostedZoneName: $.absolute_hosted_zone_name(region),
      Comment: comment,
      Name: $.absolute_instance_dns_name(region, dns_name_prefix),
      Type: "A",
      TTL: "900",
      ResourceRecords: [
        { "Fn::GetAtt": [logical_instance_name, "PrivateIp"] },
      ],
    },
  },

  relative_instance_dns_name(region, name_prefix)::
    name_prefix + "." + $.relative_hosted_zone_name(region),

  absolute_instance_dns_name(region, name_prefix)::
    name_prefix + "." + $.absolute_hosted_zone_name(region),

  relative_hosted_zone_name(region)::
    region + constants.InternalDNSSuffix,

  absolute_hosted_zone_name(region)::
    $.relative_hosted_zone_name(region) + ".",

  relative_public_ops_zone_name()::
    constants.ExternalOpsDNSZoneName,

  absolute_public_ops_zone_name()::
    $.relative_public_zone_name() + ".",
}
