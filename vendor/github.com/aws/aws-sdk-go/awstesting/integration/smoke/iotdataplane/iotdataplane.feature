# language: en
@iotdataplane @client
Feature: AWS IoT Data Plane

  Scenario: Handling errors
    When I attempt to call the "GetThingShadow" API with:
    | ThingName | "fake_thing" |
    Then I expect the response error code to be "ResourceNotFoundException"
    And I expect the response error message to include:
    """
    Not Found
    """
