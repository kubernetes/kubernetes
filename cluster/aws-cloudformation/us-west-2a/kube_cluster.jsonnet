/**
 * Kubernetes cluster specification for the us-west-2a availability zone.
 */

local kube_cluster_template = import "../templates/kube_cluster_template.jsonnet";
local aws_region = "us-west-2";
local aws_zone = "us-west-2a";
local peer_zones = ["us-west-2a", "us-west-2b", "us-west-2c"];

// Instantiate the template for this zone.
kube_cluster_template.kube_cluster_template(
  aws_region,
  aws_zone,
  peer_zones,
  "4")
