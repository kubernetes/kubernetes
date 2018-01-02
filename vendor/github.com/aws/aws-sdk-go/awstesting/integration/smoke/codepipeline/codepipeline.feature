# language: en
@codepipeline @client
Feature: Amazon CodePipeline

  Scenario: Making a request
    When I call the "ListPipelines" API
    Then the value at "pipelines" should be a list

  Scenario: Handling errors
    When I attempt to call the "GetPipeline" API with:
    | name | fake-pipeline |
    Then I expect the response error code to be "PipelineNotFoundException"
    And I expect the response error message to include:
    """
    does not have a pipeline with name 'fake-pipeline'
    """
