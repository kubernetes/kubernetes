Feature: DNS SRV records for Services and Pods
  As a Certified Kubernetes Service Provider
  I want to provide DNS for Services and Pods
  In order to locate services across the cluster
  Background:
    Given a DNS cluster
  # SRV Records are created for named ports that are part of normal or Headless Services.
  # For each named port, the SRV record would have the form
  # _my-port-name._my-port-protocol.my-svc.my-namespace.svc.cluster-domain.example

  # For a regular service, this resolves to the port number and the domain name:
  # my-svc.my-namespace.svc.cluster-domain.example.
  Scenario: Regular Services with selectors are assigned Endpoints and DNS SRV records
  # For a headless service, this resolves to multiple answers,
  # one for each pod that is backing the service,
  # and contains the port number and the domain name of the pod of the form
  # auto-generated-name.my-svc.my-namespace.svc.cluster-domain.example
  Scenario: Headless Services with selectors are assigned Endpoints and DNS SRV records
    Given a pod running http
    When I configure a service with a cluster IP pointing to that service
    Then my-svc.my-namespace.svc.cluster-domain.example resolves to the cluster IP
  Scenario: Headless Services without selectors but configured with External name
  Scenario: Headless Services without selectors but with Endpoints sharing the same name 
    Given a pod running http
    When I configure a service without a cluster IP pointing to that pod
    Then my-svc.my-namespace.svc.cluster-domain.example resolves to the a list of pods
