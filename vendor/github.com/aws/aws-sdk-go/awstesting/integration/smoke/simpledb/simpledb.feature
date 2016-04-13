# language: en
@simpledb @sdb
Feature: Amazon SimpleDB

  I want to use Amazon SimpleDB

  Scenario: Making a request
    When I call the "CreateDomain" API with:
    | DomainName  | sample-domain  |
    Then the request should be successful
    And I call the "ListDomains" API
    Then the value at "DomainNames" should be a list
    And I call the "DeleteDomain" API with:
    | DomainName  | sample-domain  |
    Then the request should be successful

  Scenario: Handling errors
    When I attempt to call the "CreateDomain" API with:
    | DomainName  |   |
    Then I expect the response error code to be "InvalidParameterValue"
    And I expect the response error message to include:
    """
    DomainName is invalid
    """
