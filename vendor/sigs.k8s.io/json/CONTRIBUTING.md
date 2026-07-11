# Contributing Guidelines

Welcome to Kubernetes. We are excited about the prospect of you joining our [community](https://git.k8s.io/community)! The Kubernetes community abides by the CNCF [code of conduct](code-of-conduct.md). Here is an excerpt:

_As contributors and maintainers of this project, and in the interest of fostering an open and welcoming community, we pledge to respect all people who contribute through reporting issues, posting feature requests, updating documentation, submitting pull requests or patches, and other activities._

## Criteria for adding code here

This library adapts the stdlib `encoding/json` decoder to be compatible with 
Kubernetes JSON decoding, and is not expected to actively add new features.

It may be updated with changes from the stdlib `encoding/json` decoder.

Any code that is added must:
* Have full unit test and benchmark coverage
* Be backward compatible with the existing exposed go API
* Have zero external dependencies
* Preserve existing benchmark performance
* Preserve compatibility with existing decoding behavior of `UnmarshalCaseSensitivePreserveInts()` or `UnmarshalStrict()`
* Avoid use of `unsafe`

## Getting Started

We have full documentation on how to get started contributing here:

<!---
If your repo has certain guidelines for contribution, put them here ahead of the general k8s resources
-->

- [Contributor License Agreement](https://git.k8s.io/community/CLA.md) Kubernetes projects require that you sign a Contributor License Agreement (CLA) before we can accept your pull requests
- [Kubernetes Contributor Guide](https://git.k8s.io/community/contributors/guide) - Main contributor documentation, or you can just jump directly to the [contributing section](https://git.k8s.io/community/contributors/guide#contributing)
- [Contributor Cheat Sheet](https://git.k8s.io/community/contributors/guide/contributor-cheatsheet) - Common resources for existing developers

## Community, discussion, contribution, and support

You can reach the maintainers of this project via the 
[sig-api-machinery mailing list / channels](https://github.com/kubernetes/community/tree/master/sig-api-machinery#contact).

## Mentorship

- [Mentoring Initiatives](https://git.k8s.io/community/mentoring) - We have a diverse set of mentorship programs available that are always looking for volunteers!

