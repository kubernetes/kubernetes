/**
 * Template for a notifications configuration running in an AWS region.
 */

local alert_utils = import "../utils/alert_utils.jsonnet";
local cf_utils = import "../utils/cf_utils.jsonnet";
local constants = import "../utils/constants.jsonnet";

{
  // The template definition
  regional_notification_template(aws_region):: cf_utils.cf_template(
    aws_region + " notifications config",
    "notifications-" + aws_region,
    aws_region,
    [],
    []) {

    // Export AWS chosen values so other stacks can depend on these objects.
    Outputs: {
      AlertTopicArn: {
        Description: "ARN for the SNS alerts topic.",
        Value: { Ref: "AlertTopic" },
      },
    },

    Resources: {
      AlertTopic: {
        Type: "AWS::SNS::Topic",
        Properties: {
          DisplayName: "Production Alerts",
          TopicName: alert_utils.default_sns_alert_topic_name,
          Subscription: [
            {
              Endpoint: constants.AlertsEmail,
              Protocol: "email",
            },
          ],
        },
      },
    },
  },
}
