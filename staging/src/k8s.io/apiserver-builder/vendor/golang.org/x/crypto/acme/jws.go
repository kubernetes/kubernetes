// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package acme

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	_ "crypto/sha512" // need for EC keys
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/big"
)

// jwsEncodeJSON signs claimset using provided key and a nonce.
// The result is serialized in JSON format.
// See https://tools.ietf.org/html/rfc7515#section-7.
func jwsEncodeJSON(claimset interface{}, key crypto.Signer, nonce string) ([]byte, error) {
	jwk, err := jwkEncode(key.Public())
	if err != nil {
		return nil, err
	}
	alg, sha := jwsHasher(key)
	if alg == "" || !sha.Available() {
		return nil, ErrUnsupportedKey
	}
	phead := fmt.Sprintf(`{"alg":%q,"jwk":%s,"nonce":%q}`, alg, jwk, nonce)
	phead = base64.RawURLEncoding.EncodeToString([]byte(phead))
	cs, err := json.Marshal(claimset)
	if err != nil {
		return nil, err
	}
	payload := base64.RawURLEncoding.EncodeToString(cs)
	hash := sha.New()
	hash.Write([]byte(phead + "." + payload))
	sig, err := jwsSign(key, sha, hash.Sum(nil))
	if err != nil {
		return nil, err
	}

	enc := struct {
		Protected string `json:"protected"`
		Payload   string `json:"payload"`
		Sig       string `json:"signature"`
	}{
		Protected: phead,
		Payload:   payload,
		Sig:       base64.RawURLEncoding.EncodeToString(sig),
	}
	return json.Marshal(&enc)
}

// jwkEncode encodes public part of an RSA or ECDSA key into a JWK.
// The result is also suitable for creating a JWK thumbprint.
// https://tools.ietf.org/html/rfc7517
func jwkEncode(pub crypto.PublicKey) (string, error) {
	switch pub := pub.(type) {
	case *rsa.PublicKey:
		// https://tools.ietf.org/html/rfc7518#section-6.3.1
		n := pub.N
		e := big.NewInt(int64(pub.E))
		// Field order is important.
		// See https://tools.ietf.org/html/rfc7638#section-3.3 for details.
		return fmt.Sprintf(`{"e":"%s","kty":"RSA","n":"%s"}`,
			base64.RawURLEncoding.EncodeToString(e.Bytes()),
			base64.RawURLEncoding.EncodeToString(n.Bytes()),
		), nil
	case *ecdsa.PublicKey:
		// https://tools.ietf.org/html/rfc7518#section-6.2.1
		p := pub.Curve.Params()
		n := p.BitSize / 8
		if p.BitSize%8 != 0 {
			n++
		}
		x := pub.X.Bytes()
		if n > len(x) {
			x = append(make([]byte, n-len(x)), x...)
		}
		y := pub.Y.Bytes()
		if n > len(y) {
			y = append(make([]byte, n-len(y)), y...)
		}
		// Field order is important.
		// See https://tools.ietf.org/html/rfc7638#section-3.3 for details.
		return fmt.Sprintf(`{"crv":"%s","kty":"EC","x":"%s","y":"%s"}`,
			p.Name,
			base64.RawURLEncoding.EncodeToString(x),
			base64.RawURLEncoding.EncodeToString(y),
		), nil
	}
	return "", ErrUnsupportedKey
}

// jwsSign signs the digest using the given key.
// It returns ErrUnsupportedKey if the key type is unknown.
// The hash is used only for RSA keys.
func jwsSign(key crypto.Signer, hash crypto.Hash, digest []byte) ([]byte, error) {
	switch key := key.(type) {
	case *rsa.PrivateKey:
		return key.Sign(rand.Reader, digest, hash)
	case *ecdsa.PrivateKey:
		r, s, err := ecdsa.Sign(rand.Reader, key, digest)
		if err != nil {
			return nil, err
		}
		rb, sb := r.Bytes(), s.Bytes()
		size := key.Params().BitSize / 8
		if size%8 > 0 {
			size++
		}
		sig := make([]byte, size*2)
		copy(sig[size-len(rb):], rb)
		copy(sig[size*2-len(sb):], sb)
		return sig, nil
	}
	return nil, ErrUnsupportedKey
}

// jwsHasher indicates suitable JWS algorithm name and a hash function
// to use for signing a digest with the provided key.
// It returns ("", 0) if the key is not supported.
func jwsHasher(key crypto.Signer) (string, crypto.Hash) {
	switch key := key.(type) {
	case *rsa.PrivateKey:
		return "RS256", crypto.SHA256
	case *ecdsa.PrivateKey:
		switch key.Params().Name {
		case "P-256":
			return "ES256", crypto.SHA256
		case "P-384":
			return "ES384", crypto.SHA384
		case "P-512":
			return "ES512", crypto.SHA512
		}
	}
	return "", 0
}

// JWKThumbprint creates a JWK thumbprint out of pub
// as specified in https://tools.ietf.org/html/rfc7638.
func JWKThumbprint(pub crypto.PublicKey) (string, error) {
	jwk, err := jwkEncode(pub)
	if err != nil {
		return "", err
	}
	b := sha256.Sum256([]byte(jwk))
	return base64.RawURLEncoding.EncodeToString(b[:]), nil
}
