# language: en
@cloudhsm @client
Feature: Amazon CloudHSM

  Scenario: Making a request
    When I call the "ListHapgs" API
    Then the value at "HapgList" should be a list

  Scenario: Handling errors
    When I attempt to call the "DescribeHapg" API with:
    | HapgArn | bogus-arn |
    Then I expect the response error code to be "ValidationException"
    And I expect the response error message to include:
    """
    Value 'bogus-arn' at 'hapgArn' failed to satisfy constraint
    """
