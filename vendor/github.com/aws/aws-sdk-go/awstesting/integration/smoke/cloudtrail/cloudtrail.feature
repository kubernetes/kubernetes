# language: en
@cloudtrail @client
Feature: AWS CloudTrail

  Scenario: Making a request
    When I call the "DescribeTrails" API
    Then the request should be successful

  Scenario: Handling errors
    When I attempt to call the "DeleteTrail" API with:
    | Name | faketrail |
    Then I expect the response error code to be "TrailNotFoundException"
