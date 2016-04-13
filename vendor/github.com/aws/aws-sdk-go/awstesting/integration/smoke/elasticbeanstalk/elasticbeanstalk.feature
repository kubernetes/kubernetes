# language: en
@elasticbeanstalk @client
Feature: AWS Elastic Beanstalk

  Scenario: Making a request
    When I call the "ListAvailableSolutionStacks" API
    Then the value at "SolutionStacks" should be a list

  Scenario: Handling errors
    When I attempt to call the "DescribeEnvironmentResources" API with:
    | EnvironmentId | fake_environment |
    Then I expect the response error code to be "InvalidParameterValue"
    And I expect the response error message to include:
    """
    No Environment found for EnvironmentId = 'fake_environment'.
    """
