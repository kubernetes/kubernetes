// Package bundler provides an API for creating certificate bundles,
// which contain a trust chain of certificates. Generally, the bundles
// will also include the private key (but this is not strictly
// required). In this package, a bundle refers to a certificate with
// full trust chain -- all certificates in the chain in one file or
// buffer.
//
// The first step in creating a certificate bundle is to create a
// Bundler. A Bundler must be created from a pre-existing certificate
// authority bundle and an intermediate certificate bundle. Once the
// Bundler is initialised, bundles may be created using a variety of
// methods: from PEM- or DER-encoded files, directly from the relevant
// Go structures, or by starting with the certificate from a remote
// system. These functions return a Bundle value, which may be
// serialised to JSON.
package bundler
