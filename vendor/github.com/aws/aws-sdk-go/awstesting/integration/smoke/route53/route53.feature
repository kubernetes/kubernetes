# language: en
@route53 @client
Feature: Amazon Route 53

  Scenario: Making a request
    When I call the "ListHostedZones" API
    Then the value at "HostedZones" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetHostedZone" API with:
    | Id | fake-zone |
    Then I expect the response error code to be "NoSuchHostedZone"
    And I expect the response error message to include:
    """
    No hosted zone found with ID: fake-zone
    """
