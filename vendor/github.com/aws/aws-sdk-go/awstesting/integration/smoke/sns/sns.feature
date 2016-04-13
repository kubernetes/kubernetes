# language: en
@sns @client
Feature: Amazon Simple Notification Service

  Scenario: Making a request
    When I call the "ListTopics" API
    Then the value at "Topics" should be a list

  Scenario: Handling errors
    When I attempt to call the "Publish" API with:
    | Message  | hello      |
    | TopicArn | fake_topic |
    Then I expect the response error code to be "InvalidParameter"
    And I expect the response error message to include:
    """
    Invalid parameter: TopicArn Reason: fake_topic does not start with arn
    """
