# Existing API Rule Violations

This folder contains the checked-in report file of known API rule violations.
The file violation\_exceptions.list is used by Make rule during OpenAPI spec generation to make
sure that no new API rule violation is introduced into our code base.

## API Rule Violation Format

The report file [violation\_exceptions.list](./violation_exceptions.list) is in format of:

 * ***API rule violation: \<RULE\>,\<PACKAGE\>,\<TYPE\>,\<FIELD\>***

e.g.

 * ***API rule violation: names_match,k8s.io/api/core/v1,Event,ReportingController***

And the violation list is sorted alphabetically in each of the \<RULE\>, \<PACKAGE\>, \<TYPE\>, \<FIELD\> levels.

## How to resolve API Rule Check Failure

Make rule returns an error when the newly generated violation report differs from this
checked-in violation report.

Our goal is that exceptions should never be added to this list, only fixed and removed.
For new APIs, this is a hard requirement. For APIs that are e.g. being moved between
versions or groups without other changes, it is OK for your API reviewer to make an
exception.

If you're removing violations from the exception list, or if you have good
reasons to add new violations to this list, please update the file using:

 - `UPDATE_API_KNOWN_VIOLATIONS=true ./hack/update-codegen.sh`

It is up to API reviewers to review the list and make sure new APIs follow our API conventions.

**NOTE**: please don't hide changes to this file in a "generated changes" commit, treat it as
source code instead.

## API Rules Being Enforced

For more information about the API rules being checked, please refer to
https://github.com/kubernetes/kube-openapi/tree/master/pkg/generators/rules
