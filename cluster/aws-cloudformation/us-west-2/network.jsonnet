/*
 * Regional AWS network specification for the us-west-2 region.
 */

local network_template = import "../templates/network_template.jsonnet";
local aws_region = "us-west-2";
local aws_zones = ["us-west-2a", "us-west-2b", "us-west-2c"];

// Instantiate the template for this region.
network_template.regional_network_definition(aws_region, aws_zones)
