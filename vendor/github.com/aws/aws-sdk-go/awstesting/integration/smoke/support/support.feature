# language: en
@support @client
Feature: AWS Support

  I want to use AWS Support

  Scenario: Making a request
    When I call the "DescribeServices" API
    Then the value at "services" should be a list

  Scenario: Handling errors
    When I attempt to call the "CreateCase" API with:
    | subject           | subject         |
    | communicationBody | communication   |
    | categoryCode      | category        |
    | serviceCode       | amazon-dynamodb |
    | severityCode      | low             |
    Then I expect the response error code to be "InvalidParameterValueException"
    And the error message should contain:
    """
    Invalid category code
    """
