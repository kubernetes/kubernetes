# language: en
@performance @streaming
Feature: Streaming transfers consume a fixed amount of memory

  Scenario Outline: Streaming uploads are O(1) in memory usage
    Given I have a <bytes> byte file
    And I take a snapshot of my resources
    When I upload the file
    Then I should not have leaked any resources

    Examples:
    | bytes     |
    | 2097152   |
    | 209715200 |

  Scenario Outline: Streaming download are O(1) in memory usage
    Given I have a <bytes> byte file
    And I take a snapshot of my resources
    When I upload the file
    And then download the file
    Then I should not have leaked any resources

    Examples:
      | bytes     |
      | 2097152   |
      | 209715200 |
