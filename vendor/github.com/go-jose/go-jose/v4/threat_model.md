# Threat model

## What this project does and where untrusted input enters
go-jose is a Go implementation of JOSE family of algorithms.

It is a library which will parse attacker-controlled data in network-reachable
services.

## Components that matter most / least
The primary attack surface of interest is parsing and validating JWS, JWE, JWK,
etc when this library is used to implement authentication and authorization.

## How to exercise it
Testing is primarily done by Golang unit tests, which can be run with the
tests.Container file's default CMD.

## How you rate severity
Any authentication bypass enabled by a go-jose issue is a high+ severity.

Denial-of-service attacks are of interest when the functions may be used in a protocol like OIDC, ACME, etc
which could have a server processing attacker-controlled data.

## Anything to leave alone
Any bug which only affects the jose-util CLI tool, but it can be used to exercise the library.

