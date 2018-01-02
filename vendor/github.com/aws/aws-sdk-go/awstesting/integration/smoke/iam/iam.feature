# language: en
@iam @client
Feature: AWS Identity and Access Management

  Scenario: Making a request
    When I call the "ListUsers" API
    Then the value at "Users" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetUser" API with:
    | UserName | fake_user |
    Then I expect the response error code to be "NoSuchEntity"
    And I expect the response error message to include:
    """
    The user with name fake_user cannot be found.
    """
