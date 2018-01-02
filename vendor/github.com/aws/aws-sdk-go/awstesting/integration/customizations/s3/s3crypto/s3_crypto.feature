# language: en
@s3crypto @client
Feature: S3 Integration Crypto Tests

  Scenario: Uploading Go's SDK fixtures
    When I get all fixtures for "aes_gcm" from "aws-s3-shared-tests"
    Then I encrypt each fixture with "kms" "AWS_SDK_TEST_ALIAS" "us-west-2" and "aes_gcm"
    And upload "Go" data with folder "version_2"

  Scenario: Uploading Go's SDK fixtures
    When I get all fixtures for "aes_cbc" from "aws-s3-shared-tests"
    Then I encrypt each fixture with "kms" "AWS_SDK_TEST_ALIAS" "us-west-2" and "aes_cbc"
    And upload "Go" data with folder "version_2"

  Scenario: Get all plaintext fixtures for symmetric masterkey aes gcm 
    When I get all fixtures for "aes_gcm" from "aws-s3-shared-tests"
    Then I decrypt each fixture against "Go" "version_2"
    And I compare the decrypted ciphertext to the plaintext

  Scenario: Get all plaintext fixtures for symmetric masterkey aes cbc
    When I get all fixtures for "aes_cbc" from "aws-s3-shared-tests"
    Then I decrypt each fixture against "Go" "version_2"
    And I compare the decrypted ciphertext to the plaintext

  Scenario: Get all plaintext fixtures for symmetric masterkey aes gcm
    When I get all fixtures for "aes_gcm" from "aws-s3-shared-tests"
    Then I decrypt each fixture against "Java" "version_2"
    And I compare the decrypted ciphertext to the plaintext

  Scenario: Get all plaintext fixtures for symmetric masterkey aes cbc
    When I get all fixtures for "aes_cbc" from "aws-s3-shared-tests"
    Then I decrypt each fixture against "Java" "version_2"
    And I compare the decrypted ciphertext to the plaintext
