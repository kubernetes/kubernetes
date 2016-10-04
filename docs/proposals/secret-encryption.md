## Proposal and Motivation

Objects of type `Secret` are currently stored inside the etcd cluster in plain
text. Enabling encryption of these secrets would be a good first step into
making the cluster more secure with data at rest.

The implementation should minic the current Cloud Provider pattern in which
different encryption providers can be written by the community depending on
their infrastructure or desire for how to implement the encryption.

Currently the implementation will only encrypt Secrets across all namespaces,
but could be extended to encrypt all data.

## Design

Two flags will be added to the API server which will let you define the which
encryption provider to utilize as well as pass configuration information is
required for that specific encryption provider.
- encryption-provider
- encryption-config

### Encryption provider

An encryption provider implements the `EncryptionProvider` interface which
allows for multiple encryption schemes to be implemented. At run-time, the API
server will look at the flag passed in and utilize that encryption provider.

To start, an AES encryption scheme will be implemented utilizing the built-in
libraries in Golang. There is an interest at UPMC Enterprises to also utlize the
Transit backend (https://www.vaultproject.io/docs/secrets/transit/) but that
specific protocol is out of scope initially.  

## Future work

* Implement Vault integration as an encryption provider
* Enable encryption of all items in etcd
