/**
 * Regional certificate storage specification for the us-west-2 region.
 */

local cert_storage_template = import "../templates/cert_storage_template.jsonnet";
local aws_region = "us-west-2";

// Instantiate the template for this region.
cert_storage_template.regional_cert_storage_definition(aws_region)
