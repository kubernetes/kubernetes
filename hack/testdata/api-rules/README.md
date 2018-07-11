## Existing API Rule Violations

This folder contains the checked-in report file of known API rule violations.
The file violation\_exceptions.list is used by Make rule during OpenAPI spec generation to make
sure that no new API rule violation is introduced into our code base.

The report file [violation\_exceptions.list](./violation_exceptions.list) is in format of:

 * ***API rule violation: \<RULE\>,\<PACKAGE\>,\<TYPE\>,\<FIELD\>***

e.g.

 * ***API rule violation: names_match,k8s.io/api/core/v1,Event,ReportingController***

Make rule returns error when new generated violation report differs from this
checked-in violation report. If new API rule violation is detected, please fix
the API Go source file to pass the API rule check. **The entries in the checked-in
violation report should only be removed when existing API rule violation is
being fixed, but not added.**

For more information about the API rules being checked, please refer to
https://github.com/kubernetes/kube-openapi/tree/master/pkg/generators/rules
