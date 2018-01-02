# language: en
@elasticache @client
Feature: ElastiCache

  Scenario: Making a request
    When I call the "DescribeEvents" API
    Then the value at "Events" should be a list

  Scenario: Handling errors
    When I attempt to call the "DescribeCacheClusters" API with:
    | CacheClusterId | fake_cluster |
    Then I expect the response error code to be "InvalidParameterValue"
    And I expect the response error message to include:
    """
    The parameter CacheClusterIdentifier is not a valid identifier.
    """
