Feature: Use existing ginkgo framework
  As a test contributor
  I want to not throw away all our old tests
  In order to retain the value generated in them
  @ginko @Conformance @NodeConformance @k8s.io
  Scenario: Container Lifecycle Hook when create a pod with lifecycle hook should execute poststart exec hook properly
    Given existing test "[k8s.io] Container Lifecycle Hook when create a pod with lifecycle hook should execute poststart exec hook properly [NodeConformance] [Conformance]"
    When I run the test
    Then the existing test will pass
    And this is fine
  Scenario: Container Runtime blackbox test when running a container with a new image should be able to pull from private registry with secret
    Given existing test "[k8s.io] Container Runtime blackbox test when running a container with a new image should be able to pull from private registry with secret [NodeConformance]"
    When I run the test
    Then the existing test will pass
    And this is fine
