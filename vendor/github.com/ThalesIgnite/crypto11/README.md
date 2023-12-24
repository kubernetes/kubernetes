Crypto11
========

[![GoDoc](https://godoc.org/github.com/ThalesIgnite/crypto11?status.svg)](https://godoc.org/github.com/ThalesIgnite/crypto11)
[![Build Status](https://travis-ci.com/ThalesIgnite/crypto11.svg?branch=master)](https://travis-ci.com/ThalesIgnite/crypto11)

This is an implementation of the standard Golang crypto interfaces that
uses [PKCS#11](http://docs.oasis-open.org/pkcs11/pkcs11-base/v2.40/errata01/os/pkcs11-base-v2.40-errata01-os-complete.html) as a backend. The supported features are:

* Generation and retrieval of RSA, DSA and ECDSA keys.
* Importing and retrieval of x509 certificates
* PKCS#1 v1.5 signing.
* PKCS#1 PSS signing.
* PKCS#1 v1.5 decryption
* PKCS#1 OAEP decryption
* ECDSA signing.
* DSA signing.
* Random number generation.
* AES and DES3 encryption and decryption.
* HMAC support.

Signing is done through the
[crypto.Signer](https://golang.org/pkg/crypto/#Signer) interface and
decryption through
[crypto.Decrypter](https://golang.org/pkg/crypto/#Decrypter).

To verify signatures or encrypt messages, retrieve the public key and do it in software.

See [the documentation](https://godoc.org/github.com/ThalesIgnite/crypto11) for details of various limitations,
especially regarding symmetric crypto.


Installation
============

Since v1.0.0, crypto11 requires Go v1.11+. Install the library by running:

```bash
go get github.com/ThalesIgnite/crypto11
```

The crypto11 library needs to be configured with information about your PKCS#11 installation. This is either done programmatically
(see the `Config` struct in [the documentation](https://godoc.org/github.com/ThalesIgnite/crypto11)) or via a configuration
file. The configuration file is a JSON representation of the `Config` struct.

A minimal configuration file looks like this:

```json
{
  "Path" : "/usr/lib/softhsm/libsofthsm2.so",
  "TokenLabel": "token1",
  "Pin" : "password"
}
```

- `Path` points to the library from your PKCS#11 vendor.
- `TokenLabel` is the `CKA_LABEL` of the token you wish to use.
- `Pin` is the password for the `CKU_USER` user.

Testing Guidance
================

Disabling tests
---------------

To disable specific tests, set the environment variable `CRYPTO11_SKIP=<flags>` where `<flags>` is a comma-separated
list of the following options:

*  `CERTS` - disables certificate-related tests. Needed for AWS CloudHSM, which doesn't support certificates.
*  `OAEP_LABEL` - disables RSA OAEP encryption tests that use source data encoding parameter (also known as a 'label' 
in some crypto libraries). Needed for AWS CloudHSM.
*  `DSA` - disables DSA tests. Needed for AWS CloudHSM (and any other tokens not supporting DSA).

Testing with Thales Luna HSM
----------------------------




Testing with AWS CloudHSM
-------------------------

A minimal configuration file for CloudHSM will look like this:

```json
{
  "Path" : "/opt/cloudhsm/lib/libcloudhsm_pkcs11_standard.so",
  "TokenLabel": "cavium",
  "Pin" : "username:password",
  "UseGCMIVFromHSM" : true,
}
```

To run the test suite you must skip unsupported tests:

```
CRYPTO11_SKIP=CERTS,OAEP_LABEL,DSA go test -v
```

Be sure to take note of the supported mechanisms, key types and other idiosyncrasies described at
https://docs.aws.amazon.com/cloudhsm/latest/userguide/pkcs11-library.html. Here's a collection of things we
noticed when testing with the  v2.0.4 PKCS#11 library:

- 1024-bit RSA keys don't appear to be supported, despite what `C_GetMechanismInfo` tells you.
- The `CKM_RSA_PKCS_OAEP` mechanism doesn't support source data. I.e. when constructing a `CK_RSA_PKCS_OAEP_PARAMS`, 
one must set `pSourceData` to `NULL` and `ulSourceDataLen` to zero.
- CloudHSM will generate it's own IV for GCM mode. This is described in their documentation, see footnote 4 on
https://docs.aws.amazon.com/cloudhsm/latest/userguide/pkcs11-mechanisms.html.
- It appears that `CKA_ID` values must be unique, otherwise you get a `CKR_ATTRIBUTE_VALUE_INVALID` error.
- Very rapid session opening can trigger the following error:
  ```
  C_OpenSession failed with error CKR_ARGUMENTS_BAD : 0x00000007
  HSM error 8c: HSM Error: Already maximum number of sessions are issued
  ```

Testing with SoftHSM2
---------------------

To set up a slot:

    $ cat softhsm2.conf
    directories.tokendir = /home/rjk/go/src/github.com/ThalesIgnite/crypto11/tokens
    objectstore.backend = file
    log.level = INFO
    $ mkdir tokens
    $ export SOFTHSM2_CONF=`pwd`/softhsm2.conf
    $ softhsm2-util --init-token --slot 0 --label test
    === SO PIN (4-255 characters) ===
    Please enter SO PIN: ********
    Please reenter SO PIN: ********
    === User PIN (4-255 characters) ===
    Please enter user PIN: ********
    Please reenter user PIN: ********
    The token has been initialized.

The configuration looks like this:

    $ cat config
    {
      "Path" : "/usr/lib/softhsm/libsofthsm2.so",
      "TokenLabel": "test",
      "Pin" : "password"
    }

(At time of writing) OAEP is only partial and HMAC is unsupported, so expect test skips.

Testing with nCipher nShield
--------------------

In all cases, it's worth enabling nShield PKCS#11 log output:

    export CKNFAST_DEBUG=2

To protect keys with a 1/N operator cardset:

    $ cat config
    {
      "Path" : "/opt/nfast/toolkits/pkcs11/libcknfast.so",
      "TokenLabel": "rjk",
      "Pin" : "password"
    }

You can also identify the token by serial number, which in this case
means the first 16 hex digits of the operator cardset's token hash:

    $ cat config
    {
      "Path" : "/opt/nfast/toolkits/pkcs11/libcknfast.so",
      "TokenSerial": "1d42780caa22efd5",
      "Pin" : "password"
    }

A card from the cardset must be in the slot when you run `go test`.

To protect keys with the module only, use the 'accelerator' token:

    $ cat config
    {
      "Path" : "/opt/nfast/toolkits/pkcs11/libcknfast.so",
      "TokenLabel": "accelerator",
      "Pin" : "password"
    }

(At time of writing) GCM is not implemented, so expect test skips.

Limitations
===========

 * The [PKCS1v15DecryptOptions SessionKeyLen](https://golang.org/pkg/crypto/rsa/#PKCS1v15DecryptOptions) field
is not implemented and an error is returned if it is nonzero.
The reason for this is that it is not possible for crypto11 to guarantee the constant-time behavior in the specification.
See [issue #5](https://github.com/ThalesIgnite/crypto11/issues/5) for further discussion.
 * Symmetric crypto support via [cipher.Block](https://golang.org/pkg/crypto/cipher/#Block) is very slow.
You can use the `BlockModeCloser` API
(over 400 times as fast on my computer)
but you must call the Close()
interface (not found in [cipher.BlockMode](https://golang.org/pkg/crypto/cipher/#BlockMode)).
See [issue #6](https://github.com/ThalesIgnite/crypto11/issues/6) for further discussion.

Contributions
========

Contributions are gratefully received. Before beginning work on sizeable changes, please open an issue first to
discuss.

Here are some topics we'd like to cover:

* Full test instructions for additional PKCS#11 implementations.
