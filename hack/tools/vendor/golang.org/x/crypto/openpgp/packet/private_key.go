// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto"
	"crypto/cipher"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/sha1"
	"io"
	"io/ioutil"
	"math/big"
	"strconv"
	"time"

	"golang.org/x/crypto/openpgp/elgamal"
	"golang.org/x/crypto/openpgp/errors"
	"golang.org/x/crypto/openpgp/s2k"
)

// PrivateKey represents a possibly encrypted private key. See RFC 4880,
// section 5.5.3.
type PrivateKey struct {
	PublicKey
	Encrypted     bool // if true then the private key is unavailable until Decrypt has been called.
	encryptedData []byte
	cipher        CipherFunction
	s2k           func(out, in []byte)
	PrivateKey    interface{} // An *{rsa|dsa|ecdsa}.PrivateKey or crypto.Signer/crypto.Decrypter (Decryptor RSA only).
	sha1Checksum  bool
	iv            []byte
}

func NewRSAPrivateKey(creationTime time.Time, priv *rsa.PrivateKey) *PrivateKey {
	pk := new(PrivateKey)
	pk.PublicKey = *NewRSAPublicKey(creationTime, &priv.PublicKey)
	pk.PrivateKey = priv
	return pk
}

func NewDSAPrivateKey(creationTime time.Time, priv *dsa.PrivateKey) *PrivateKey {
	pk := new(PrivateKey)
	pk.PublicKey = *NewDSAPublicKey(creationTime, &priv.PublicKey)
	pk.PrivateKey = priv
	return pk
}

func NewElGamalPrivateKey(creationTime time.Time, priv *elgamal.PrivateKey) *PrivateKey {
	pk := new(PrivateKey)
	pk.PublicKey = *NewElGamalPublicKey(creationTime, &priv.PublicKey)
	pk.PrivateKey = priv
	return pk
}

func NewECDSAPrivateKey(creationTime time.Time, priv *ecdsa.PrivateKey) *PrivateKey {
	pk := new(PrivateKey)
	pk.PublicKey = *NewECDSAPublicKey(creationTime, &priv.PublicKey)
	pk.PrivateKey = priv
	return pk
}

// NewSignerPrivateKey creates a PrivateKey from a crypto.Signer that
// implements RSA or ECDSA.
func NewSignerPrivateKey(creationTime time.Time, signer crypto.Signer) *PrivateKey {
	pk := new(PrivateKey)
	// In general, the public Keys should be used as pointers. We still
	// type-switch on the values, for backwards-compatibility.
	switch pubkey := signer.Public().(type) {
	case *rsa.PublicKey:
		pk.PublicKey = *NewRSAPublicKey(creationTime, pubkey)
	case rsa.PublicKey:
		pk.PublicKey = *NewRSAPublicKey(creationTime, &pubkey)
	case *ecdsa.PublicKey:
		pk.PublicKey = *NewECDSAPublicKey(creationTime, pubkey)
	case ecdsa.PublicKey:
		pk.PublicKey = *NewECDSAPublicKey(creationTime, &pubkey)
	default:
		panic("openpgp: unknown crypto.Signer type in NewSignerPrivateKey")
	}
	pk.PrivateKey = signer
	return pk
}

func (pk *PrivateKey) parse(r io.Reader) (err error) {
	err = (&pk.PublicKey).parse(r)
	if err != nil {
		return
	}
	var buf [1]byte
	_, err = readFull(r, buf[:])
	if err != nil {
		return
	}

	s2kType := buf[0]

	switch s2kType {
	case 0:
		pk.s2k = nil
		pk.Encrypted = false
	case 254, 255:
		_, err = readFull(r, buf[:])
		if err != nil {
			return
		}
		pk.cipher = CipherFunction(buf[0])
		pk.Encrypted = true
		pk.s2k, err = s2k.Parse(r)
		if err != nil {
			return
		}
		if s2kType == 254 {
			pk.sha1Checksum = true
		}
	default:
		return errors.UnsupportedError("deprecated s2k function in private key")
	}

	if pk.Encrypted {
		blockSize := pk.cipher.blockSize()
		if blockSize == 0 {
			return errors.UnsupportedError("unsupported cipher in private key: " + strconv.Itoa(int(pk.cipher)))
		}
		pk.iv = make([]byte, blockSize)
		_, err = readFull(r, pk.iv)
		if err != nil {
			return
		}
	}

	pk.encryptedData, err = ioutil.ReadAll(r)
	if err != nil {
		return
	}

	if !pk.Encrypted {
		return pk.parsePrivateKey(pk.encryptedData)
	}

	return
}

func mod64kHash(d []byte) uint16 {
	var h uint16
	for _, b := range d {
		h += uint16(b)
	}
	return h
}

func (pk *PrivateKey) Serialize(w io.Writer) (err error) {
	// TODO(agl): support encrypted private keys
	buf := bytes.NewBuffer(nil)
	err = pk.PublicKey.serializeWithoutHeaders(buf)
	if err != nil {
		return
	}
	buf.WriteByte(0 /* no encryption */)

	privateKeyBuf := bytes.NewBuffer(nil)

	switch priv := pk.PrivateKey.(type) {
	case *rsa.PrivateKey:
		err = serializeRSAPrivateKey(privateKeyBuf, priv)
	case *dsa.PrivateKey:
		err = serializeDSAPrivateKey(privateKeyBuf, priv)
	case *elgamal.PrivateKey:
		err = serializeElGamalPrivateKey(privateKeyBuf, priv)
	case *ecdsa.PrivateKey:
		err = serializeECDSAPrivateKey(privateKeyBuf, priv)
	default:
		err = errors.InvalidArgumentError("unknown private key type")
	}
	if err != nil {
		return
	}

	ptype := packetTypePrivateKey
	contents := buf.Bytes()
	privateKeyBytes := privateKeyBuf.Bytes()
	if pk.IsSubkey {
		ptype = packetTypePrivateSubkey
	}
	err = serializeHeader(w, ptype, len(contents)+len(privateKeyBytes)+2)
	if err != nil {
		return
	}
	_, err = w.Write(contents)
	if err != nil {
		return
	}
	_, err = w.Write(privateKeyBytes)
	if err != nil {
		return
	}

	checksum := mod64kHash(privateKeyBytes)
	var checksumBytes [2]byte
	checksumBytes[0] = byte(checksum >> 8)
	checksumBytes[1] = byte(checksum)
	_, err = w.Write(checksumBytes[:])

	return
}

func serializeRSAPrivateKey(w io.Writer, priv *rsa.PrivateKey) error {
	err := writeBig(w, priv.D)
	if err != nil {
		return err
	}
	err = writeBig(w, priv.Primes[1])
	if err != nil {
		return err
	}
	err = writeBig(w, priv.Primes[0])
	if err != nil {
		return err
	}
	return writeBig(w, priv.Precomputed.Qinv)
}

func serializeDSAPrivateKey(w io.Writer, priv *dsa.PrivateKey) error {
	return writeBig(w, priv.X)
}

func serializeElGamalPrivateKey(w io.Writer, priv *elgamal.PrivateKey) error {
	return writeBig(w, priv.X)
}

func serializeECDSAPrivateKey(w io.Writer, priv *ecdsa.PrivateKey) error {
	return writeBig(w, priv.D)
}

// Decrypt decrypts an encrypted private key using a passphrase.
func (pk *PrivateKey) Decrypt(passphrase []byte) error {
	if !pk.Encrypted {
		return nil
	}

	key := make([]byte, pk.cipher.KeySize())
	pk.s2k(key, passphrase)
	block := pk.cipher.new(key)
	cfb := cipher.NewCFBDecrypter(block, pk.iv)

	data := make([]byte, len(pk.encryptedData))
	cfb.XORKeyStream(data, pk.encryptedData)

	if pk.sha1Checksum {
		if len(data) < sha1.Size {
			return errors.StructuralError("truncated private key data")
		}
		h := sha1.New()
		h.Write(data[:len(data)-sha1.Size])
		sum := h.Sum(nil)
		if !bytes.Equal(sum, data[len(data)-sha1.Size:]) {
			return errors.StructuralError("private key checksum failure")
		}
		data = data[:len(data)-sha1.Size]
	} else {
		if len(data) < 2 {
			return errors.StructuralError("truncated private key data")
		}
		var sum uint16
		for i := 0; i < len(data)-2; i++ {
			sum += uint16(data[i])
		}
		if data[len(data)-2] != uint8(sum>>8) ||
			data[len(data)-1] != uint8(sum) {
			return errors.StructuralError("private key checksum failure")
		}
		data = data[:len(data)-2]
	}

	return pk.parsePrivateKey(data)
}

func (pk *PrivateKey) parsePrivateKey(data []byte) (err error) {
	switch pk.PublicKey.PubKeyAlgo {
	case PubKeyAlgoRSA, PubKeyAlgoRSASignOnly, PubKeyAlgoRSAEncryptOnly:
		return pk.parseRSAPrivateKey(data)
	case PubKeyAlgoDSA:
		return pk.parseDSAPrivateKey(data)
	case PubKeyAlgoElGamal:
		return pk.parseElGamalPrivateKey(data)
	case PubKeyAlgoECDSA:
		return pk.parseECDSAPrivateKey(data)
	}
	panic("impossible")
}

func (pk *PrivateKey) parseRSAPrivateKey(data []byte) (err error) {
	rsaPub := pk.PublicKey.PublicKey.(*rsa.PublicKey)
	rsaPriv := new(rsa.PrivateKey)
	rsaPriv.PublicKey = *rsaPub

	buf := bytes.NewBuffer(data)
	d, _, err := readMPI(buf)
	if err != nil {
		return
	}
	p, _, err := readMPI(buf)
	if err != nil {
		return
	}
	q, _, err := readMPI(buf)
	if err != nil {
		return
	}

	rsaPriv.D = new(big.Int).SetBytes(d)
	rsaPriv.Primes = make([]*big.Int, 2)
	rsaPriv.Primes[0] = new(big.Int).SetBytes(p)
	rsaPriv.Primes[1] = new(big.Int).SetBytes(q)
	if err := rsaPriv.Validate(); err != nil {
		return err
	}
	rsaPriv.Precompute()
	pk.PrivateKey = rsaPriv
	pk.Encrypted = false
	pk.encryptedData = nil

	return nil
}

func (pk *PrivateKey) parseDSAPrivateKey(data []byte) (err error) {
	dsaPub := pk.PublicKey.PublicKey.(*dsa.PublicKey)
	dsaPriv := new(dsa.PrivateKey)
	dsaPriv.PublicKey = *dsaPub

	buf := bytes.NewBuffer(data)
	x, _, err := readMPI(buf)
	if err != nil {
		return
	}

	dsaPriv.X = new(big.Int).SetBytes(x)
	pk.PrivateKey = dsaPriv
	pk.Encrypted = false
	pk.encryptedData = nil

	return nil
}

func (pk *PrivateKey) parseElGamalPrivateKey(data []byte) (err error) {
	pub := pk.PublicKey.PublicKey.(*elgamal.PublicKey)
	priv := new(elgamal.PrivateKey)
	priv.PublicKey = *pub

	buf := bytes.NewBuffer(data)
	x, _, err := readMPI(buf)
	if err != nil {
		return
	}

	priv.X = new(big.Int).SetBytes(x)
	pk.PrivateKey = priv
	pk.Encrypted = false
	pk.encryptedData = nil

	return nil
}

func (pk *PrivateKey) parseECDSAPrivateKey(data []byte) (err error) {
	ecdsaPub := pk.PublicKey.PublicKey.(*ecdsa.PublicKey)

	buf := bytes.NewBuffer(data)
	d, _, err := readMPI(buf)
	if err != nil {
		return
	}

	pk.PrivateKey = &ecdsa.PrivateKey{
		PublicKey: *ecdsaPub,
		D:         new(big.Int).SetBytes(d),
	}
	pk.Encrypted = false
	pk.encryptedData = nil

	return nil
}
