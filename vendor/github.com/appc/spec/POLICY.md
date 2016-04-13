# App Container Specification Governance and Contribution Policy

This document defines the current policies around governance and contributions for the App Container Specification (appc) project.
This is written from the perspective that there is a group of people who cooperatively support and manage the project (the _maintainers_, hereafter "we").
We will move towards an open governance model where multiple parties have commit access, roll-back rights, and can provide explicit support for features.

## Contributions

We use the following rules for accepting contributions to the specification repository, including the text of the specification itself and the associated schema and tooling.
In contrast to maintainers, _contributors_ are not actively managing the direction of the project, but contributing individual changes.
- We require all contributors to agree to the [DCO](https://github.com/appc/spec/blob/master/DCO)
- We accept well-written, clearly explained changes to the specification itself.
- We accept well-written, well-tested cleanup and refactoring changes of the schema code.
- We accept well-written, well-tested bug fixes to the schema code.
- We accept well-written, well-tested feature contributions to the schema tooling if a maintainer assumes support responsibilities, i.e., readily answers support questions and works on bugs. This includes feature contributions from external contributors. If there is no maintainer to support a feature, then we will deprecate and subsequently delete the feature - we will give three months' notice in such cases.
- We will not accept untested changes, except in very rare cases.
- We require a pre-commit code review from a maintainer for all changes. For changes submitted by maintainers, a review is required from at least one other maintainer.

## Major Schema Changes

Any changes that would require a new version of the schema/specification to be released must be approved by at least two of the current maintainers. A new version must be released any time a change is made that would break compatibility with the existing schema.

## Maintainers

Maintainers are the only contributors with merge privileges on the [specification repository](https://github.com/appc/spec). The group of maintainers is self-managing - new maintainers are added by a simple majority of votes from existing maintainers on the mailing list with zero no-votes within four business days. Maintainers may voluntarily step down with no voting required. Removal of existing maintainers is done through a supermajority of two-thirds of the votes of all maintainers. The voting period is 10 business days.

We expect that new contributors will submit a number of patches before they become maintainers.

The current set of maintainers is always recorded in the [MAINTAINERS](https://github.com/appc/spec/blob/master/MAINTAINERS) file.

The mailing list for the maintainers team is [appc-dev@googlegroups.com](https://groups.google.com/forum/#!forum/appc-dev)

## Governance

Changes to the rules in this document follow the same process as removing maintainers: a supermajority of two-thirds of the votes of all maintainers within a period of 10 business days.

## Credits

This document was inspired by and modelled on the [initial governance policy for Google's Bazel project](https://github.com/google/bazel/blob/efbcf00220a95c5ea1dfa7e3a5bff8311b52727d/site/governance.md)
