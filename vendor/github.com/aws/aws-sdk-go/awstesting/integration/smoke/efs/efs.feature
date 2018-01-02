# language: en
@efs @elasticfilesystem @client
Feature: Amazon Elastic File System

  I want to use Amazon Elastic File System

  Scenario: Making a request
    When I call the "DescribeFileSystems" API
    Then the value at "FileSystems" should be a list

  Scenario: Handling errors
    When I attempt to call the "DeleteFileSystem" API with:
    | FileSystemId | fake-id |
    Then the error code should be "BadRequest"
