# Contributing

Logr is open to pull-requests, provided they fit within the intended scope of
the project.  Specifically, this library aims to be VERY small and minimalist,
with no external dependencies.

## Compatibility

This project intends to follow [semantic versioning](http://semver.org) and
is very strict about compatibility.  Any proposed changes MUST follow those
rules.

## Performance

As a logging library, logr must be as light-weight as possible.  Any proposed
code change must include results of running the [benchmark](./benchmark)
before and after the change.
