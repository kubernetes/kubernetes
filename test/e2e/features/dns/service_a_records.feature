Feature: DNS A records for Services and Pods
  As a Certified Kubernetes Service Provider
  I want to provide DNS for Services and Pods
  In order to locate services across the cluster
  Background:
    Given a DNS cluster
  Scenario: Pod in namespace quux can lookup service foo in namespace bar
    Given a namespace foo
      And a pod running http
      And with service named bar
      And a namespace quux
    When I create a pod A in namespace quux
    Then the pod A will be able to resolve foo.bar
  Scenario: A records for Services with a Cluster IP are assigned a DNS A record pointing to it
    Given a pod running http
    When I configure a service with a cluster IP pointing to that service
    Then my-svc.my-namespace.svc.cluster-domain.example resolves to the cluster IP
  Scenario: A records for Services without a Cluster IP are assigned a DNS A records for each pod the services runs on
    Given a pod running http
    When I configure a service without a cluster IP pointing to that pod
    Then my-svc.my-namespace.svc.cluster-domain.example resolves to the a list of pods
