/*
 * Utilities for making it easier to generate CloudFormation conformant JSON.
 */

{
  /**
   * Base class for a CloudFormation-compatible JSON config.
   *
   * Includes the version, description, and common metadata fields used to indicate provenance and feed a push
   * tool.
   *
   * To use, import this file and derive a new object from this template.  As an example:

        local cf_utils = import "cf_utils.jsonnet";
        {
          cf_utils.cf_tempate(
              "my description",
              "cloud-formation-stack-name",
              "us-west-2",
              ["CAPABILITY_IAM"],
              [
                {"InputParam1": {"Stack": "StackName", "Output": "OutputName1"}},
                {"InputParam2": {"Stack": "StackName", "Output": "OutputName2"}}
              ]) {

            "Resources": {
              ...
            }
          }
        }
   */
  cf_template(description, stack_name, aws_region, caps, parameter_sources):: {
    AWSTemplateFormatVersion: "2010-09-09",
    Description: description,
    Metadata: {
      User: std.extVar("USER"),
      Hostname: std.extVar("HOSTNAME"),
      "Build Timestamp": std.extVar("TIMESTAMP"),
      StackName: stack_name,
      Region: aws_region,
      RequiredCaps: caps,
      ParamSources: parameter_sources,
    },
  },

  /**
   * Expands a key-value array into separated named key and value pairs; output matches expected CloudFormation tags
   * field common to many resource types.
   *
   * @param table a dictionary of key/value pairs; e.g. { "Name": "Foo", "Env": "Prod" }
   * @return an expanded list of dictionaries; e.g. { "Key": "Name", "Value": "Foo"}, { "Key": "Env", "Value": "Prod"}
   */
  make_tags(table)::
    [{ Key: k, Value: table[k] } for k in std.objectFields(table)],

  /**
   * Helper method which converts an AZ name into a string which can be included in a CloudFormation object logical
   * name.  For now, we just strip out the dashes.
   *
   * @param zone an aws zone name; e.g., us-west-2a
   * @return the zone name without dashes; e.g., uswest2a
   */
  logical_zone_name(zone)::
    std.join("", std.split(zone, "-")),

  /**
   * Given a zone name of "us-west-2a", returns the suffix letter 'a' which is the region's zone identifier.
   */
  zone_suffix(zone)::
    std.substr(zone, std.length(zone) - 1, 1),

  /**
   * Helper method which converts a bucket name name into a string which can be included in a CloudFormation
   * object logical name.  For now, we just strip out the dashes.
   *
   * @param bucket an aws bucket name; e.g., mycorp-backups
   * @return the zone name without dashes; e.g., mycorpbackups
   */
  logical_bucket_name(bucket)::
    std.join("", std.split(bucket, "-")),
}
