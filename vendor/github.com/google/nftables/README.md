[![Build Status](https://github.com/google/nftables/actions/workflows/push.yml/badge.svg)](https://github.com/google/nftables/actions/workflows/push.yml)
[![GoDoc](https://godoc.org/github.com/google/nftables?status.svg)](https://godoc.org/github.com/google/nftables)

**This is not the correct repository for issues with the Linux nftables
project!** This repository contains a third-party Go package to programmatically
interact with nftables. Find the official nftables website at
https://wiki.nftables.org/

This package manipulates Linux nftables (the iptables successor). It is
implemented in pure Go, i.e. does not wrap libnftnl.

This is not an official Google product.

## Breaking changes

This package is in very early stages, and only contains enough data types and
functions to install very basic nftables rules. It is likely that mistakes with
the data types/API will be identified as more functionality is added.

## Contributions

Contributions are very welcome!


