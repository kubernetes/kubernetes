# language: en
@workspaces @client
Feature: Amazon WorkSpaces

  I want to use Amazon WorkSpaces

  Scenario: Making a request
    When I call the "DescribeWorkspaces" API
    Then the value at "Workspaces" should be a list

  Scenario: Handling errors
    When I attempt to call the "DescribeWorkspaces" API with:
    | DirectoryId | fake-id |
    Then I expect the response error code to be "ValidationException"
    And I expect the response error message to include:
    """
    The Directory ID fake-id in the request is invalid.
    """
