# language: en
@kinesis @client
Feature: AWS Kinesis

  Scenario: Making a request
    When I call the "ListStreams" API
    Then the value at "StreamNames" should be a list

  Scenario: Handling errors
    When I attempt to call the "DescribeStream" API with:
    | StreamName | bogus-stream-name |
    Then I expect the response error code to be "ResourceNotFoundException"
    And I expect the response error message to include:
    """
    Stream bogus-stream-name under account
    """
