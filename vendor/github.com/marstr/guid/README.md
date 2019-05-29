[![Build Status](https://travis-ci.org/marstr/guid.svg?branch=master)](https://travis-ci.org/marstr/guid)
[![GoDoc](https://godoc.org/github.com/marstr/guid?status.svg)](https://godoc.org/github.com/marstr/guid)
[![Go Report Card](https://goreportcard.com/badge/github.com/marstr/guid)](https://goreportcard.com/report/github.com/marstr/guid)

# Guid
Globally unique identifiers offer a quick means of generating non-colliding values across a distributed system. For this implemenation, [RFC 4122](http://ietf.org/rfc/rfc4122.txt) governs the desired behavior.

## What's in a name?
You have likely already noticed that RFC and some implementations refer to these structures as UUIDs (Universally Unique Identifiers), where as this project is annotated as  GUIDs (Globally Unique Identifiers). The name Guid was selected to make clear this project's ties to the [.NET struct Guid.](https://msdn.microsoft.com/en-us/library/system.guid(v=vs.110).aspx) The most obvious relationship is the desire to have the same format specifiers available in this library's Format and Parse methods as .NET would have in its ToString and Parse methods.

# Installation
- Ensure you have the [Go Programming Language](https://golang.org/) installed on your system.
- Run the command: `go get -u github.com/marstr/guid`

# Contribution
Contributions are welcome! Feel free to send Pull Requests. Continuous Integration will ensure that you have conformed to Go conventions. Please remember to add tests for your changes.

# Versioning
This library will adhere to the
[Semantic Versioning 2.0.0](http://semver.org/spec/v2.0.0.html) specification. It may be worth noting this should allow for tools like [glide](https://glide.readthedocs.io/en/latest/) to pull in this library with ease.

The Release Notes portion of this file will be updated to reflect the most recent major/minor updates, with the option to tag particular bug-fixes as well. Updates to the Release Notes for patches should be addative, where as major/minor updates should replace the previous version. If one desires to see the release notes for an older version, checkout that version of code and open this file.

# Release Notes 1.1.*

## v1.1.0
Adding support for JSON marshaling and unmarshaling.
