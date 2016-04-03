# language: en
@apigateway @client
Feature: Amazon API Gateway

  Scenario: Making a request
    When I call the "GetAccountRequest" API
    Then the request should be successful

  Scenario: Handing errors
    When I attempt to call the "GetRestApi" API with:
    | RestApiId | api123 |
    Then I expect the response error code to be "NotFoundException"
    And I expect the response error message to include:
    """
    Invalid REST API identifier specified
    """
