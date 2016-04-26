/*
 * Regional AWS notifications specification for the us-west-2 region.
 */

local notifications_template = import "../templates/notifications_template.jsonnet";
local aws_region = "us-west-2";

// Instantiate the template for this region.
notifications_template.regional_notification_template(aws_region)
