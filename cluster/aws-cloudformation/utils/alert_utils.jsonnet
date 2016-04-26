/**
 * Utility methods for setting up alert rules.
 */

local constants = import "constants.jsonnet";

{
  default_sns_alert_topic_name: "alerts",
  default_sns_alert_topic_arn: "arn:aws:sns:us-west-2:%s:%s" % [
    constants.AwsProjectId, $.default_sns_alert_topic_name],

  /**
   * Defines an alarm for an instance which fires if the average CPU utilization gets too high.
   */
  define_cpu_alert_with_default_sns_topic(stack_name, instance_id)::
    $.define_cpu_alert(stack_name, instance_id, $.default_sns_alert_topic_arn),

  define_cpu_alert(stack_name, instance_id, sns_alert_topic):: {
    Type: "AWS::CloudWatch::Alarm",
    Properties: {
      AlarmName: stack_name + "-" + instance_id + "-cpu-alarm",
      AlarmDescription: "CPU alarm instance " + instance_id,
      AlarmActions: [sns_alert_topic],
      MetricName: "CPUUtilization",
      Namespace: "AWS/EC2",
      Statistic: "Average",
      Period: "60",
      EvaluationPeriods: "3",
      Threshold: "80",
      ComparisonOperator: "GreaterThanThreshold",
      Dimensions: [{
        Name: "InstanceId",
        Value: { Ref: instance_id },
      }],
    },
  },

  /**
   * Defines an alarm for an autoscaler which triggers a scale-up to occur if CPU utilization gets really high.
   */
  define_auto_scaler_cpu_alert_with_default_sns_topic(stack_name, autoscaler_id)::
    $.define_auto_scaler_cpu_alert(stack_name, autoscaler_id, $.default_sns_alert_topic_arn),

  define_auto_scaler_cpu_alert(stack_name, autoscaler_id, sns_alert_topic):: {
    Type: "AWS::CloudWatch::Alarm",
    Properties: {
      AlarmDescription: "Scale-up if CPU > 90% for 10 minutes: " + autoscaler_id,
      AlarmName: stack_name + "-" + autoscaler_id + "-cpu-alarm",
      AlarmActions: [sns_alert_topic],
      MetricName: "CPUUtilization",
      Namespace: "AWS/EC2",
      Statistic: "Average",
      Period: "300",
      EvaluationPeriods: "2",
      Threshold: "90",
      ComparisonOperator: "GreaterThanThreshold",
      Dimensions: [
        {
          Name: "AutoScalingGroupName",
          Value: { Ref: autoscaler_id },
        },
      ],
    },
  },

  /**
   * Defines an alarm for an auto-scaler which sends a notification when auto-scaling occurs.
   */
  define_auto_scaler_change_notifications_with_default_sns_alert_topic_arn()::
    $.define_auto_scaler_change_notifications($.default_sns_alert_topic_arn),

  define_auto_scaler_change_notifications(sns_alert_topic):: [
    {
      TopicARN: sns_alert_topic,
      NotificationTypes: [
        "autoscaling:EC2_INSTANCE_LAUNCH",
        "autoscaling:EC2_INSTANCE_LAUNCH_ERROR",
        "autoscaling:EC2_INSTANCE_TERMINATE",
        "autoscaling:EC2_INSTANCE_TERMINATE_ERROR",
      ],
    },
  ],

  /**
   * Defines an alarm for an instance which fires when status checks fail.
   */
  define_instance_alert_with_default_sns_topic(stack_name, instance_id)::
    $.define_instance_alert(stack_name, instance_id, $.default_sns_alert_topic_arn),

  define_instance_alert(stack_name, instance_id, sns_alert_topic):: {
    Type: "AWS::CloudWatch::Alarm",
    Properties: {
      AlarmName: stack_name + "-" + instance_id + "-instance-health-alarm",
      AlarmDescription: "Instance health alert after 5 minutes: " + instance_id,
      AlarmActions: [sns_alert_topic],
      Namespace: "AWS/EC2",
      MetricName: "StatusCheckFailed_System",
      Statistic: "Minimum",
      Period: "60",
      EvaluationPeriods: "5",
      ComparisonOperator: "GreaterThanThreshold",
      Threshold: "0",
      Dimensions: [{
        Name: "InstanceId",
        Value: { Ref: instance_id },
      }],
    },
  },

  /**
   * Defines an alarm for an instance which triggers EC2 to auto-recover if the instance is down for more than
   * 15 minutes.
   */
  define_instance_recovery_action_with_default_sns_topic(stack_name, instance_id)::
    $.define_instance_recovery_action(stack_name, instance_id, $.default_sns_alert_topic_arn),

  define_instance_recovery_action(stack_name, instance_id, sns_alert_topic):: {
    Type: "AWS::CloudWatch::Alarm",
    Properties: {
      AlarmName: stack_name + "-" + instance_id + "-instance-health-recovery",
      AlarmDescription: "Instance recovery after 15 minutes: " + instance_id,
      AlarmActions: [{ "Fn::Join": ["", ["arn:aws:automate:", { Ref: "AWS::Region" }, ":ec2:recover"]] }],
      Namespace: "AWS/EC2",
      MetricName: "StatusCheckFailed_System",
      Statistic: "Minimum",
      Period: "60",
      EvaluationPeriods: "15",
      ComparisonOperator: "GreaterThanThreshold",
      Threshold: "0",
      Dimensions: [{
        Name: "InstanceId",
        Value: { Ref: instance_id },
      }],
    },
  },
}
