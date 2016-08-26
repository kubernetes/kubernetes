# Go JOSE 

[![godoc](http://img.shields.io/badge/godoc-reference-blue.svg?style=flat)](https://godoc.org/gopkg.in/square/go-jose.v1) [![license](http://img.shields.io/badge/license-apache_2.0-red.svg?style=flat)](https://raw.githubusercontent.com/square/go-jose/master/LICENSE) [![build](https://travis-ci.org/square/go-jose.svg?branch=master)](https://travis-ci.org/square/go-jose) [![coverage](https://coveralls.io/repos/github/square/go-jose/badge.svg?branch=master)](https://coveralls.io/r/square/go-jose)

Package jose aims to provide an implementation of the Javascript Object Signing
and Encryption set of standards. For the moment, it mainly focuses on encryption
and signing based on the JSON Web Encryption and JSON Web Signature standards.

**Disclaimer**: This library contains encryption software that is subject to
the U.S. Export Administration Regulations. You may not export, re-export,
transfer or download this code or any part of it in violation of any United
States law, directive or regulation. In particular this software may not be
exported or re-exported in any form or on any media to Iran, North Sudan,
Syria, Cuba, or North Korea, or to denied persons or entities mentioned on any
US maintained blocked list.

## Overview

The implementation follows the
[JSON Web Encryption](http://dx.doi.org/10.17487/RFC7516)
standard (RFC 7516) and
[JSON Web Signature](http://dx.doi.org/10.17487/RFC7515)
standard (RFC 7515). Tables of supported algorithms are shown below.
The library supports both the compact and full serialization formats, and has
optional support for multiple recipients. It also comes with a small
command-line utility
([`jose-util`](https://github.com/square/go-jose/tree/master/jose-util))
for dealing with JOSE messages in a shell.

**Note**: We use a forked version of the `encoding/json` package from the Go
standard library which uses case-sensitive matching for member names (instead
of [case-insensitive matching](https://www.ietf.org/mail-archive/web/json/current/msg03763.html)).
This is to avoid differences in interpretation of messages between go-jose and
libraries in other languages. If you do not like this behavior, you can use the
`std_json` build tag to disable it (though we do not recommend doing so).

### Versions

We use [gopkg.in](https://gopkg.in) for versioning.

[Version 1](https://gopkg.in/square/go-jose.v1) is the current stable version:

    import "gopkg.in/square/go-jose.v1"

The interface for [go-jose.v1](https://gopkg.in/square/go-jose.v1) will remain
backwards compatible. We're currently sketching out ideas for a new version, to
clean up the interface a bit. If you have ideas or feature requests [please let
us know](https://github.com/square/go-jose/issues/64)!

### Supported algorithms

See below for a table of supported algorithms. Algorithm identifiers match
the names in the
[JSON Web Algorithms](http://dx.doi.org/10.17487/RFC7518)
standard where possible. The
[Godoc reference](https://godoc.org/github.com/square/go-jose#pkg-constants)
has a list of constants.

 Key encryption             | Algorithm identifier(s)
 :------------------------- | :------------------------------
 RSA-PKCS#1v1.5             | RSA1_5
 RSA-OAEP                   | RSA-OAEP, RSA-OAEP-256
 AES key wrap               | A128KW, A192KW, A256KW
 AES-GCM key wrap           | A128GCMKW, A192GCMKW, A256GCMKW
 ECDH-ES + AES key wrap     | ECDH-ES+A128KW, ECDH-ES+A192KW, ECDH-ES+A256KW
 ECDH-ES (direct)           | ECDH-ES<sup>1</sup>
 Direct encryption          | dir<sup>1</sup>

<sup>1. Not supported in multi-recipient mode</sup>

 Signing / MAC              | Algorithm identifier(s)
 :------------------------- | :------------------------------
 RSASSA-PKCS#1v1.5          | RS256, RS384, RS512
 RSASSA-PSS                 | PS256, PS384, PS512
 HMAC                       | HS256, HS384, HS512
 ECDSA                      | ES256, ES384, ES512

 Content encryption         | Algorithm identifier(s)
 :------------------------- | :------------------------------
 AES-CBC+HMAC               | A128CBC-HS256, A192CBC-HS384, A256CBC-HS512
 AES-GCM                    | A128GCM, A192GCM, A256GCM 

 Compression                | Algorithm identifiers(s)
 :------------------------- | -------------------------------
 DEFLATE (RFC 1951)         | DEF

### Supported key types

See below for a table of supported key types. These are understood by the
library, and can be passed to corresponding functions such as `NewEncrypter` or
`NewSigner`. Note that if you are creating a new encrypter or signer with a
JsonWebKey, the key id of the JsonWebKey (if present) will be added to any
resulting messages.

 Algorithm(s)               | Corresponding types
 :------------------------- | -------------------------------
 RSA                        | *[rsa.PublicKey](http://golang.org/pkg/crypto/rsa/#PublicKey), *[rsa.PrivateKey](http://golang.org/pkg/crypto/rsa/#PrivateKey), *[jose.JsonWebKey](https://godoc.org/github.com/square/go-jose#JsonWebKey)
 ECDH, ECDSA                | *[ecdsa.PublicKey](http://golang.org/pkg/crypto/ecdsa/#PublicKey), *[ecdsa.PrivateKey](http://golang.org/pkg/crypto/ecdsa/#PrivateKey), *[jose.JsonWebKey](https://godoc.org/github.com/square/go-jose#JsonWebKey)
 AES, HMAC                  | []byte, *[jose.JsonWebKey](https://godoc.org/github.com/square/go-jose#JsonWebKey)

## Examples

Encryption/decryption example using RSA:

```Go
// Generate a public/private key pair to use for this example. The library
// also provides two utility functions (LoadPublicKey and LoadPrivateKey)
// that can be used to load keys from PEM/DER-encoded data.
privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
if err != nil {
	panic(err)
}

// Instantiate an encrypter using RSA-OAEP with AES128-GCM. An error would
// indicate that the selected algorithm(s) are not currently supported.
publicKey := &privateKey.PublicKey
encrypter, err := NewEncrypter(RSA_OAEP, A128GCM, publicKey)
if err != nil {
	panic(err)
}

// Encrypt a sample plaintext. Calling the encrypter returns an encrypted
// JWE object, which can then be serialized for output afterwards. An error
// would indicate a problem in an underlying cryptographic primitive.
var plaintext = []byte("Lorem ipsum dolor sit amet")
object, err := encrypter.Encrypt(plaintext)
if err != nil {
	panic(err)
}

// Serialize the encrypted object using the full serialization format.
// Alternatively you can also use the compact format here by calling
// object.CompactSerialize() instead.
serialized := object.FullSerialize()

// Parse the serialized, encrypted JWE object. An error would indicate that
// the given input did not represent a valid message.
object, err = ParseEncrypted(serialized)
if err != nil {
	panic(err)
}

// Now we can decrypt and get back our original plaintext. An error here
// would indicate the the message failed to decrypt, e.g. because the auth
// tag was broken or the message was tampered with.
decrypted, err := object.Decrypt(privateKey)
if err != nil {
	panic(err)
}

fmt.Printf(string(decrypted))
// output: Lorem ipsum dolor sit amet
```

Signing/verification example using RSA:

```Go
// Generate a public/private key pair to use for this example. The library
// also provides two utility functions (LoadPublicKey and LoadPrivateKey)
// that can be used to load keys from PEM/DER-encoded data.
privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
if err != nil {
	panic(err)
}

// Instantiate a signer using RSASSA-PSS (SHA512) with the given private key.
signer, err := NewSigner(PS512, privateKey)
if err != nil {
	panic(err)
}

// Sign a sample payload. Calling the signer returns a protected JWS object,
// which can then be serialized for output afterwards. An error would
// indicate a problem in an underlying cryptographic primitive.
var payload = []byte("Lorem ipsum dolor sit amet")
object, err := signer.Sign(payload)
if err != nil {
	panic(err)
}

// Serialize the encrypted object using the full serialization format.
// Alternatively you can also use the compact format here by calling
// object.CompactSerialize() instead.
serialized := object.FullSerialize()

// Parse the serialized, protected JWS object. An error would indicate that
// the given input did not represent a valid message.
object, err = ParseSigned(serialized)
if err != nil {
	panic(err)
}

// Now we can verify the signature on the payload. An error here would
// indicate the the message failed to verify, e.g. because the signature was
// broken or the message was tampered with.
output, err := object.Verify(&privateKey.PublicKey)
if err != nil {
	panic(err)
}

fmt.Printf(string(output))
// output: Lorem ipsum dolor sit amet
```

More examples can be found in the [Godoc
reference](https://godoc.org/github.com/square/go-jose) for this package. The
[`jose-util`](https://github.com/square/go-jose/tree/master/jose-util)
subdirectory also contains a small command-line utility which might
be useful as an example.
