# go-tspi - Go bindings and support code for libtspi and tpm communication

This is a library providing a set of bindings for communication between
code written in Go and libtspi, the library responsible for providing a TPM
control interface. It consists of the following components:

## tspi

The tspi bindings for Go. These are a low-level interface intended for use by
people writing new TPM-using applications in Go. Code using these bindings must
run on the same system as the TPM. For example:
```
// Create a new TSPI context
context, err := tspi.NewContext()
// Connect to the TPM daemon
context.connect()
// Obtain a handle to the TPM itself
tpm := context.GetTPM()
// Obtain the TPM event log
log, err := tpm.GetEventLog()
```
## attestation and verification

Helper functions for performing attestation-related tasks
```
// Retrieve the EK certificate
ekcert, err := attestation.GetEKCert(context)
// Verify that the EK certificate is signed by a TPM vendor
err = verification.VerifyEKCert(ekcert)
if err != nil {
   log.Fatal("Unable to verify EK certificate!")
}
```
## tpmd

Daemon for performing certain TPM operations at a higher level API or via a
network. Takes the listening port number as the only argument.

## tpmclient

Library for client applications communicating with tpmd. Avoids the need for
individual applications to care about TSPI context or resource lifecycles
themselves.
```
`// Connect to the TPM daemon on localhost port 12401
client := tpmclient.New("127.0.0.1:12401")
// Extend a PCR with some data
data := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
client.Extend(15, 0x1000, data, "Test extension")`
```