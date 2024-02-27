[![GoDoc Widget]][GoDoc]  [![GoReport]][GoReportStatus]

[GoDoc]: https://godoc.org/k8s.io/gengo
[GoDoc Widget]: https://godoc.org/k8s.io/gengo?status.svg
[GoReport]: https://goreportcard.com/badge/github.com/kubernetes/gengo
[GoReportStatus]: https://goreportcard.com/report/github.com/kubernetes/gengo

# Gengo: a framework for building simple code generators

This repo is used by Kubernetes to build some codegen tooling.  It is not
intended to be general-purpose and makes some assumptions that may not hold
outside of Kubernetes.

In the past this repo was partially supported for external use (outside of the
Kubernetes project overall), but that is no longer true.  We may change the API
in incompatible ways, without warning.

If you are not building something that is part of Kubernetes, DO NOT DEPEND ON
THIS REPO.

## New usage within Kubernetes

Gengo is a very opinionated framework.  It is primarily aimed at generating Go
code derived from types defined in other Go code, but it is possible to use it
for other things (e.g. proto files).  Net new tools should consider using
`golang.org/x/tools/go/packages` directly.  Gengo can serve as an example of
how to do that.

If you still decide you want to use gengo, see the
[simple examples](./examples) in this repo or the more extensive tools in the
Kubernetes [code-generator](https://github.com/kubernetes/code-generator/)
repo.

## Overview

Gengo is used to build tools (generally a tool is a binary).  Each tool
describes some number of `Targets`. A target is a single output package, which
may be the same as the inputs (if the tool generates code alongside the inputs)
or different.  Each `Target` describes some number of `Generators`.  A
generator is responsible for emitting a single file into the target directory.

Gengo helps the tool to load and process input packages, e.g. extracting type
information and associating comments.  Each target will be offered every known
type, and can filter that down to the set of types it cares about.  Each
generator will be offered the result of the target's filtering, and can filter
the set of types further.  Finally, the generator will be called to emit code
for all of the remaining types.

The `tracer` example in this repo can be used to examine all of the hooks.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.
