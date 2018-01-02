# language: en
@storagegateway @client
Feature: AWS Storage Gateway

  Scenario: Making a request
    When I call the "ListGateways" API
    Then the value at "Gateways" should be a list

  Scenario: Handling errors
    When I attempt to call the "ListVolumes" API with:
    | GatewayARN | fake_gateway |
    Then I expect the response error code to be "InvalidParameter"
    And I expect the response error message to include:
    """
    GatewayARN
    """
