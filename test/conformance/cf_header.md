# Kubernetes Conformance Test Suite -  {{.Version}}

## **Summary**
This document provides a summary of the tests included in the Kubernetes conformance test suite.
Each test lists a set of formal requirements that a platform that meets conformance requirements must adhere to.

The tests are a subset of the "e2e" tests that make up the Kubernetes testing infrastructure.
Each test is identified by the presence of the `[Conformance]` keyword in the ginkgo descriptive function calls.
The contents of this document are extracted from comments preceding those `[Conformance]` keywords
and those comments are expected to include a descriptive overview of what the test is validating using
RFC2119 keywords. This will provide a clear distinction between which bits of code in the tests are
there for the purposes of validating the platform rather than simply infrastructure logic used to setup, or
clean up the tests.

Example:
```
/*
  Release: v1.13
  Testname: Kubelet, log output, default
  Description: By default the stdout and stderr from the process being executed in a pod MUST be sent to the pod's logs.
*/
framework.ConformanceIt("should print the output to logs [NodeConformance]", func(ctx context.Context) {
```

would generate the following documentation for the test. Note that the "TestName" from the Documentation above will
be used to document the test which make it more human readable. The "Description" field will be used as the
documentation for that test.

### **Output:**
## [Kubelet, log output, default](https://github.com/kubernetes/kubernetes/tree/master/test/e2e/common/node/kubelet.go#L49)

- Added to conformance in release v1.13
- Defined in code as: [k8s.io] Kubelet when scheduling a busybox command in a pod should print the output to logs [NodeConformance] [Conformance]

By default the stdout and stderr from the process being executed in a pod MUST be sent to the pod's logs.

Notational Conventions when documenting the tests with the key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" are to be interpreted as described in [RFC 2119](https://tools.ietf.org/html/rfc2119).

Note: Please see the Summary at the end of this document to find the number of tests documented for conformance.

## **List of Tests**
