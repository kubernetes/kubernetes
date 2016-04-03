# language: en
@cloudwatchlogs @logs
Feature: Amazon CloudWatch Logs

  Scenario: Making a request
    When I call the "DescribeLogGroups" API
    Then the value at "logGroups" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetLogEvents" API with:
    | logGroupName  | fakegroup  |
    | logStreamName | fakestream |
    Then I expect the response error code to be "ResourceNotFoundException"
    And I expect the response error message to include:
    """
    The specified log group does not exist.
    """
