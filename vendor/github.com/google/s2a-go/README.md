# Secure Session Agent Client Libraries

The Secure Session Agent is a service that enables a workload to offload select
operations from the mTLS handshake and protects a workload's private key
material from exfiltration. Specifically, the workload asks the Secure Session
Agent for the TLS configuration to use during the handshake, to perform private
key operations, and to validate the peer certificate chain. The Secure Session
Agent's client libraries enable applications to communicate with the Secure
Session Agent during the TLS handshake, and to encrypt traffic to the peer
after the TLS handshake is complete.

This repository contains the source code for the Secure Session Agent's Go
client libraries, which allow gRPC-Go applications to use the Secure Session
Agent. This repository supports the Bazel and Golang build systems.

All code in this repository is experimental and subject to change. We do not
guarantee API stability at this time.
