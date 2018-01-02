# rkt and the Trusted Platform Module

rkt supports *measuring* container state and configuration into the [Trusted Platform Module (TPM)][wiki-tpm] event log. Enable this functionality by building rkt with the [`--enable-tpm=yes` option to `./configure`][build-configure-tpm]. rkt accesses the TPM via the [`tpmd` executable available from the go-tspi project][go-tspi]. This `tpmd` is expected to listen on port 12041.

Events are logged to PCR 15, with event type `0x1000`. Each event contains the following data:

1. The hash of the container root filesystem
2. The hash of the contents of the container manifest data
3. The hash of the arguments passed to `stage1`

This provides a cryptographically verifiable audit log of the containers executed on a node, including the configuration of each.


[build-configure-tpm]: ../build-configure.md#security
[go-tspi]: https://github.com/coreos/go-tspi
[wiki-tpm]: https://en.wikipedia.org/wiki/Trusted_Platform_Module
