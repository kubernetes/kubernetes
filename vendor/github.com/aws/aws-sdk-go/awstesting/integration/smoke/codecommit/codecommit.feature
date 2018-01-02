# language: en
@codecommit @client
Feature: Amazon CodeCommit

  Scenario: Making a request
    When I call the "ListRepositories" API
    Then the value at "repositories" should be a list

  Scenario: Handling errors
    When I attempt to call the "ListBranches" API with:
    | repositoryName | fake-repo |
    Then I expect the response error code to be "RepositoryDoesNotExistException"
    And I expect the response error message to include:
    """
    fake-repo does not exist
    """
