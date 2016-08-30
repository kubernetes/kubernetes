# SCADA Client

This library provides a Golang client for the [HashiCorp SCADA service](http://scada.hashicorp.com).
SCADA stands for Supervisory Control And Data Acquisition, and as the name implies it allows
[Atlas](https://atlas.hashicorp.com) to provide control functions and request data from the tools that integrate.

The technical details about how SCADA works are fairly simple. Clients first open a connection to
the SCADA service at scada.hashicorp.com on port 7223. This connection is secured by TLS, allowing
clients to verify the identity of the servers and to encrypt all communications. Once connected, a
handshake is performed where a client provides it's Atlas API credentials so that Atlas can verify
the client identity. Once complete, clients keep the connection open in an idle state waiting for
commands to be received. Commands map to APIs exposed by the product, and are subject to any ACLs,
authentication or authorization mechanisms of the client.

This library is used in various HashiCorp products to integrate with the SCADA system.

## Environmental Variables

This library respects the following environment variables:

* ATLAS_TOKEN: The Atlas token to use for authentication
* SCADA_ENDPOINT: Overrides the default SCADA endpoint

