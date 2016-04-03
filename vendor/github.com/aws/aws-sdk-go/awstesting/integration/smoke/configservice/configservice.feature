# language: en
@configservice @config @client
Feature: AWS Config

  Scenario: Making a request
    When I call the "DescribeConfigurationRecorders" API
    Then the value at "ConfigurationRecorders" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetResourceConfigHistory" API with:
    | resourceType | fake-type |
    | resourceId   | fake-id   |
    Then I expect the response error code to be "ValidationException"
    And I expect the response error message to include:
    """
    failed to satisfy constraint
    """
