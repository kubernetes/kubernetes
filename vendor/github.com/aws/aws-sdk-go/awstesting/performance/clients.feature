# language: en
@performance @clients
Feature: Client Performance
  Background:
    Given I have loaded my SDK and its dependencies
    And I have a list of services
    And I take a snapshot of my resources

  Scenario: Creating and then cleaning up clients doesn't leak resources
    When I create and discard 100 clients for each service
    Then I should not have leaked any resources

  Scenario: Sending requests doesn't leak resources
    When I create a client for each service
    And I execute 100 command(s) on each client
    And I destroy all the clients
    Then I should not have leaked any resources
