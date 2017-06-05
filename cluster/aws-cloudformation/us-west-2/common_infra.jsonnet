/*
 * Regional AWS infra cluster specification for the us-west-2 region.
 */

local common_infra_template = import "../templates/regional_common_infra_template.jsonnet";
local aws_region = "us-west-2";
local aws_zones = ["us-west-2a", "us-west-2b", "us-west-2c"];

// Instantiate the template for this region.
common_infra_template.regional_common_infra_environment_template(aws_region, aws_zones)
