# language: en
@route53domains @client
Feature: Amazon Route53 Domains

  Scenario: Making a request
    When I call the "ListDomains" API
    Then the value at "Domains" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetDomainDetail" API with:
    | DomainName | fake-domain-name |
    Then I expect the response error code to be "InvalidInput"
    And I expect the response error message to include:
    """
    domain name must contain more than 1 label
    """
