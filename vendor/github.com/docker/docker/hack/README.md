## About

This directory contains a collection of scripts used to build and manage this
repository. If there are any issues regarding the intention of a particular
script (or even part of a certain script), please reach out to us.
It may help us either refine our current scripts, or add on new ones
that are appropriate for a given use case.

## DinD (dind.sh)

DinD is a wrapper script which allows Docker to be run inside a Docker
container. DinD requires the container to
be run with privileged mode enabled.

## Generate Authors (generate-authors.sh)

Generates AUTHORS; a file with all the names and corresponding emails of
individual contributors. AUTHORS can be found in the home directory of
this repository.

## Make

There are two make files, each with different extensions. Neither are supposed
to be called directly; only invoke `make`. Both scripts run inside a Docker
container.

### make.ps1

- The Windows native build script that uses PowerShell semantics; it is limited
unlike `hack\make.sh` since it does not provide support for the full set of
operations provided by the Linux counterpart, `make.sh`. However, `make.ps1`
does provide support for local Windows development and Windows to Windows CI.
More information is found within `make.ps1` by the author, @jhowardmsft

### make.sh

- Referenced via `make test` when running tests on a local machine,
or directly referenced when running tests inside a Docker development container.  
- When running on a local machine, `make test` to run all tests found in
`test`, `test-unit`, `test-integration-cli`, and `test-docker-py` on
your local machine. The default timeout is set in `make.sh` to 60 minutes
(`${TIMEOUT:=60m}`), since it currently takes up to an hour to run
all of the tests.
- When running inside a Docker development container, `hack/make.sh` does
not have a single target that runs all the tests. You need to provide a
single command line with multiple targets that performs the same thing.
An example referenced from [Run targets inside a development container](https://docs.docker.com/opensource/project/test-and-docs/#run-targets-inside-a-development-container): `root@5f8630b873fe:/go/src/github.com/moby/moby# hack/make.sh dynbinary binary cross test-unit test-integration-cli test-docker-py`
- For more information related to testing outside the scope of this README,
refer to
[Run tests and test documentation](https://docs.docker.com/opensource/project/test-and-docs/)

## Release (release.sh)

Releases any bundles built by `make` on a public AWS S3 bucket.
For information regarding configuration, please view `release.sh`.

## Vendor (vendor.sh)

A shell script that is a wrapper around Vndr. For information on how to use
this, please refer to [vndr's README](https://github.com/LK4D4/vndr/blob/master/README.md)
