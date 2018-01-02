# language: en
@devicefarm @client
Feature: AWS Device Farm

  Scenario: Making a request
    When I call the "ListDevices" API
    Then the value at "devices" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetDevice" API with:
    | arn | arn:aws:devicefarm:us-west-2::device:000000000000000000000000fake-arn |
    Then I expect the response error code to be "NotFoundException"
    And I expect the response error message to include:
    """
    No device was found for arn
    """
