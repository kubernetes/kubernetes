# PKCS#11 [![Build Status](https://travis-ci.org/miekg/pkcs11.png?branch=master)](https://travis-ci.org/miekg/pkcs11) [![GoDoc](https://img.shields.io/badge/godoc-reference-blue.svg)](http://godoc.org/github.com/miekg/pkcs11)

This is a Go implementation of the PKCS#11 API. It wraps the library closely, but uses Go idiom were
it makes sense. It has been tested with SoftHSM.

## SoftHSM

 *  Make it use a custom configuration file `export SOFTHSM_CONF=$PWD/softhsm.conf`

 *  Then use `softhsm` to init it

    ~~~
    softhsm --init-token --slot 0 --label test --pin 1234
    ~~~

 *  Then use `libsofthsm.so` as the pkcs11 module:

    ~~~ go
    p := pkcs11.New("/usr/lib/softhsm/libsofthsm.so")
    ~~~

## Examples

A skeleton program would look somewhat like this (yes, pkcs#11 is verbose):

~~~ go
p := pkcs11.New("/usr/lib/softhsm/libsofthsm.so")
err := p.Initialize()
if err != nil {
    panic(err)
}

defer p.Destroy()
defer p.Finalize()

slots, err := p.GetSlotList(true)
if err != nil {
    panic(err)
}

session, err := p.OpenSession(slots[0], pkcs11.CKF_SERIAL_SESSION|pkcs11.CKF_RW_SESSION)
if err != nil {
    panic(err)
}
defer p.CloseSession(session)

err = p.Login(session, pkcs11.CKU_USER, "1234")
if err != nil {
    panic(err)
}
defer p.Logout(session)

p.DigestInit(session, []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_SHA_1, nil)})
hash, err := p.Digest(session, []byte("this is a string"))
if err != nil {
    panic(err)
}

for _, d := range hash {
        fmt.Printf("%x", d)
}
fmt.Println()
~~~

Further examples are included in the tests.

To expose PKCS#11 keys using the [crypto.Signer interface](https://golang.org/pkg/crypto/#Signer),
please see [github.com/thalesignite/crypto11](https://github.com/thalesignite/crypto11).
