# libtrust

Libtrust is library for managing authentication and authorization using public key cryptography.

Authentication is handled using the identity attached to the public key.
Libtrust provides multiple methods to prove possession of the private key associated with an identity.
 - TLS x509 certificates
 - Signature verification
 - Key Challenge

Authorization and access control is managed through a distributed trust graph.
Trust servers are used as the authorities of the trust graph and allow caching portions of the graph for faster access.

## Copyright and license

Code and documentation copyright 2014 Docker, inc. Code released under the Apache 2.0 license.
Docs released under Creative commons.

