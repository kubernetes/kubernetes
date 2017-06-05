// +build ignore

// This file is used to generate keys for tests.

package main

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"text/template"

	jose "gopkg.in/square/go-jose.v2"
)

type key struct {
	name string
	new  func() (crypto.Signer, error)
}

var keys = []key{
	{
		"ECDSA_256", func() (crypto.Signer, error) {
			return ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		},
	},
	{
		"ECDSA_384", func() (crypto.Signer, error) {
			return ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		},
	},
	{
		"ECDSA_521", func() (crypto.Signer, error) {
			return ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
		},
	},
	{
		"RSA_1024", func() (crypto.Signer, error) {
			return rsa.GenerateKey(rand.Reader, 1024)
		},
	},
	{
		"RSA_2048", func() (crypto.Signer, error) {
			return rsa.GenerateKey(rand.Reader, 2048)
		},
	},
	{
		"RSA_4096", func() (crypto.Signer, error) {
			return rsa.GenerateKey(rand.Reader, 4096)
		},
	},
}

func newJWK(k key, prefix, ident string) (privBytes, pubBytes []byte, err error) {
	priv, err := k.new()
	if err != nil {
		return nil, nil, fmt.Errorf("generate %s: %v", k.name, err)
	}
	pub := priv.Public()

	privKey := &jose.JSONWebKey{Key: priv}
	thumbprint, err := privKey.Thumbprint(crypto.SHA256)
	if err != nil {
		return nil, nil, fmt.Errorf("computing thumbprint: %v", err)
	}

	keyID := hex.EncodeToString(thumbprint)
	privKey.KeyID = keyID
	pubKey := &jose.JSONWebKey{Key: pub, KeyID: keyID}

	privBytes, err = json.MarshalIndent(privKey, prefix, ident)
	if err != nil {
		return
	}
	pubBytes, err = json.MarshalIndent(pubKey, prefix, ident)
	return
}

type keyData struct {
	Name string
	Priv string
	Pub  string
}

var tmpl = template.Must(template.New("").Parse(`// +build !golint

// This file contains statically created JWKs for tests created by gen.go

package oidc

import (
	"encoding/json"

	jose "gopkg.in/square/go-jose.v2"
)

func mustLoadJWK(s string) jose.JSONWebKey {
	var jwk jose.JSONWebKey
	if err := json.Unmarshal([]byte(s), &jwk); err != nil {
		panic(err)
	}
	return jwk
}

var (
{{- range $i, $key := .Keys }}
	testKey{{ $key.Name }} = mustLoadJWK(` + "`" + `{{ $key.Pub }}` + "`" + `)
	testKey{{ $key.Name }}_Priv = mustLoadJWK(` + "`" + `{{ $key.Priv }}` + "`" + `)
{{ end -}}
)
`))

func main() {
	var tmplData struct {
		Keys []keyData
	}
	for _, k := range keys {
		for i := 0; i < 4; i++ {
			log.Printf("generating %s", k.name)
			priv, pub, err := newJWK(k, "\t", "\t")
			if err != nil {
				log.Fatal(err)
			}
			name := fmt.Sprintf("%s_%d", k.name, i)

			tmplData.Keys = append(tmplData.Keys, keyData{
				Name: name,
				Priv: string(priv),
				Pub:  string(pub),
			})
		}
	}

	buff := new(bytes.Buffer)
	if err := tmpl.Execute(buff, tmplData); err != nil {
		log.Fatalf("excuting template: %v", err)
	}

	if err := ioutil.WriteFile("jose_test.go", buff.Bytes(), 0644); err != nil {
		log.Fatal(err)
	}
}
