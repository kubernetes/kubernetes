/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package serviceaccount_test

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"encoding/pem"
	"fmt"

	"google.golang.org/grpc"
	jose "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/cryptosigner"

	externalsigner "k8s.io/kubernetes/pkg/serviceaccount/externalsigner/v1alpha1"
)

func rsaPubKeyBytes(pk *rsa.PrivateKey) ([]byte, error) {
	if pk == nil {
		return nil, fmt.Errorf("No private key!")
	}
	return pubKeyPKIX(&pk.PublicKey)
}

func pubKeyPKIX(kU interface{}) ([]byte, error) {
	der, err := x509.MarshalPKIXPublicKey(kU)
	if err != nil {
		return nil, fmt.Errorf("Could not marshal public key (%T): %v", kU, err)
	}
	block := pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: der,
	}
	return pem.EncodeToMemory(&block), nil
}

var _ externalsigner.KeyServiceClient = &mockKeyServiceClient{}

func newMockKeyServiceClient(privatekey interface{}) *mockKeyServiceClient {
	return &mockKeyServiceClient{
		key: privatekey,
	}
}

type mockKeyServiceClient struct {
	key interface{}
}

func (c *mockKeyServiceClient) SignPayload(ctx context.Context, in *externalsigner.SignPayloadRequest, opts ...grpc.CallOption) (*externalsigner.SignPayloadResponse, error) {
	cSigner, ok := c.key.(crypto.Signer)
	if !ok {
		return nil, fmt.Errorf("private key type %T must be a valid crypto.Signer", c.key)
	}
	signer := cryptosigner.Opaque(cSigner)
	signedData, err := signer.SignPayload(in.Payload, jose.SignatureAlgorithm(in.Algorithm))
	if err != nil {
		return nil, err
	}
	return &externalsigner.SignPayloadResponse{
		Content: signedData,
	}, nil
}

func (c *mockKeyServiceClient) ListPublicKeys(ctx context.Context, in *externalsigner.ListPublicKeysRequest, opts ...grpc.CallOption) (*externalsigner.ListPublicKeysResponse, error) {
	var alg jose.SignatureAlgorithm
	var kU []byte
	var err error
	switch pk := c.key.(type) {
	case *rsa.PrivateKey:
		alg = jose.RS256
		kU, err = pubKeyPKIX(&pk.PublicKey)
		if err != nil {
			return nil, err
		}
	case *ecdsa.PrivateKey:
		kU, err = pubKeyPKIX(&pk.PublicKey)
		if err != nil {
			return nil, err
		}
		switch pk.Curve {
		case elliptic.P256():
			alg = jose.ES256
		case elliptic.P384():
			alg = jose.ES384
		case elliptic.P521():
			alg = jose.ES512
		default:
			return nil, fmt.Errorf("unknown private key curve, must be 256, 384, or 521")
		}
	default:
		return nil, fmt.Errorf("unknown private key type %T, must be *rsa.PrivateKey, *ecdsa.PrivateKey", pk)
	}

	// for simplicity purposes, KeyID is the sha1sum(PKIX(pubkey))
	kid := fmt.Sprintf("%x", sha1.Sum(kU))

	keys := []*externalsigner.PublicKey{
		{
			PublicKey: kU,
			KeyId:     kid,
			Algorithm: string(alg),
		},
	}
	return &externalsigner.ListPublicKeysResponse{
		ActiveKeyId: kid,
		PublicKeys:  keys,
	}, nil

}
