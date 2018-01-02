# language: en
@autoscaling @client
Feature: Auto Scaling

  Scenario: Making a request
    When I call the "DescribeScalingProcessTypes" API
    Then the value at "Processes" should be a list

  Scenario: Handing errors
    When I attempt to call the "CreateLaunchConfiguration" API with:
    | LaunchConfigurationName |              |
    | ImageId                 | ami-12345678 |
    | InstanceType            | m1.small     |
    Then I expect the response error code to be "InvalidParameter"
    And I expect the response error message to include:
    """
    LaunchConfigurationName
    """
