This demonstrates using Cloudflare's [cfssl](https://github.com/cloudflare/cfssl) to easily generate certificates for an etcd cluster.

Defaults generate an ECDSA-384 root and leaf certificates for `localhost`. etcd nodes will use the same certificates for both sides of mutual authentication, but won't require client certs for non-peer clients.

**Instructions**

1. Install git, go, and make
2. Run `make` to generate the certs
3. Run `goreman start`
