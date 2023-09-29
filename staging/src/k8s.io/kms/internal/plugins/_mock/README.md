# Mock KMS Plugin

This is a mock KMS plugin for testing purposes. It implements the KMS plugin using PKCS#11 interface backed by [SoftHSM](https://www.opendnssec.org/softhsm/). It is intended to be used for testing only and not for production use.

The directory is named `_mock` so that it is ignored by the `go mod` tooling in the root directory.
