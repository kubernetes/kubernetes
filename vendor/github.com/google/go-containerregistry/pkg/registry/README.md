# `pkg/registry`

This package implements a Docker v2 registry and the OCI distribution specification.

It is designed to be used anywhere a low dependency container registry is needed, with an initial focus on tests.

Its goal is to be standards compliant and its strictness will increase over time.

This is currently a low flightmiles system. It's likely quite safe to use in tests; If you're using it in production, please let us know how and send us PRs for integration tests.

Before sending a PR, understand that the expectation of this package is that it remain free of extraneous dependencies.
This means that we expect `pkg/registry` to only have dependencies on Go's standard library, and other packages in `go-containerregistry`.

You may be asked to change your code to reduce dependencies, and your PR might be rejected if this is deemed impossible.
