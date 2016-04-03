# language: en
@codedeploy @client
Feature: Amazon CodeDeploy

  Scenario: Making a request
    When I call the "ListApplications" API
    Then the value at "applications" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetDeployment" API with:
      | deploymentId | d-USUAELQEX |
    Then I expect the response error code to be "DeploymentDoesNotExistException"
    And I expect the response error message to include:
    """
    The deployment d-USUAELQEX could not be found
    """
