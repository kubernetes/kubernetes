/*
 * Regional AWS etcd cluster specification for the us-west-2 region.
 */

local ectd_cluster_template = import "../templates/etcd_cluster_template.jsonnet";
local aws_region = "us-west-2";
local aws_zones = ["us-west-2a", "us-west-2b", "us-west-2c"];

// Instantiate the template for this region.
ectd_cluster_template.regional_etcd_cluster_template(aws_region, aws_zones)
