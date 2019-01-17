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

package serviceaccount

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	jose "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/jwt"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/klog"
	externalsigner "k8s.io/kubernetes/pkg/serviceaccount/externalsigner/v1alpha1"
)

// ExternalJWTTokenGenerator returns a TokenGenerator that generates signed JWT Tokens using a remote signing service
func ExternalJWTTokenGenerator(iss string, socketPath string) (TokenGenerator, error) {
	// TODO: @micahhausler conditionally add unix:// prefix
	conn, err := grpc.Dial(fmt.Sprintf("unix://%s", socketPath), grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	generator := &ExternalTokenGenerator{
		Iss:    iss,
		Client: externalsigner.NewKeyServiceClient(conn),
	}
	return generator, nil
}

type ExternalTokenGenerator struct {
	Iss    string
	Client externalsigner.KeyServiceClient
}

func (g *ExternalTokenGenerator) GenerateToken(claims *jwt.Claims, privateClaims interface{}) (string, error) {
	signer := NewRemoteOpaqueSigner(g.Client)
	generator, err := JWTTokenGenerator(g.Iss, signer.Public().KeyID, signer)
	if err != nil {
		return "", err
	}
	return generator.GenerateToken(claims, privateClaims)
}

// NewRemoteOpaqueSigner returns an jose.OpaqueSigner that communicates over the client
func NewRemoteOpaqueSigner(client externalsigner.KeyServiceClient) *RemoteOpaqueSigner {
	return &RemoteOpaqueSigner{Client: client}
}

type RemoteOpaqueSigner struct {
	Client externalsigner.KeyServiceClient
}

// check that OpaqueSigner conforms to the interface
var _ jose.OpaqueSigner = &RemoteOpaqueSigner{}

func (s *RemoteOpaqueSigner) Public() *jose.JSONWebKey {
	resp, err := s.Client.ListPublicKeys(context.Background(), &externalsigner.ListPublicKeysRequest{})
	if err != nil {
		klog.Errorf("Error getting public keys: %v", err)
		return nil
	}
	var currentPublicKey *externalsigner.PublicKey
	for _, key := range resp.PublicKeys {
		if resp.ActiveKeyId == key.KeyId {
			currentPublicKey = key
			break
		}
	}
	if currentPublicKey == nil {
		klog.Errorf("Current key_id %s not found in list", resp.ActiveKeyId)
		return nil
	}
	response := &jose.JSONWebKey{
		KeyID:     currentPublicKey.KeyId,
		Algorithm: currentPublicKey.Algorithm,
		Use:       "sig",
	}
	keys, err := keyutil.ParsePublicKeysPEM(currentPublicKey.PublicKey)
	if err != nil {
		klog.Errorf("Error getting public key: %v", err)
		return nil
	}
	if len(keys) == 0 {
		klog.Error("No public key returned")
		return nil
	}
	response.Key = keys[0]
	response.Certificates, err = certutil.ParseCertsPEM(currentPublicKey.Certificates)
	if err != nil && err != certutil.ErrNoCerts {
		klog.Errorf("Error parsing x509 certificate: %v", err)
		return nil
	}
	return response
}

func (s *RemoteOpaqueSigner) Algs() []jose.SignatureAlgorithm {
	resp, err := s.Client.ListPublicKeys(context.Background(), &externalsigner.ListPublicKeysRequest{})
	if err != nil {
		klog.Errorf("Error getting public keys: %v", err)
		return nil
	}
	algos := map[string]bool{}
	for _, key := range resp.PublicKeys {
		algos[key.Algorithm] = true
	}
	response := []jose.SignatureAlgorithm{}
	for alg := range algos {
		response = append(response, jose.SignatureAlgorithm(alg))
	}
	return response
}

func (s *RemoteOpaqueSigner) SignPayload(payload []byte, alg jose.SignatureAlgorithm) ([]byte, error) {
	resp, err := s.Client.SignPayload(context.Background(), &externalsigner.SignPayloadRequest{
		Payload:   payload,
		Algorithm: string(alg),
	})
	if err != nil {
		return nil, err
	}
	return resp.Content, nil
}

type ExternalTokenAuthenticator struct {
	Client       externalsigner.KeyServiceClient
	Iss          string
	Validator    Validator
	ImplicitAuds authenticator.Audiences
}

func (a *ExternalTokenAuthenticator) AuthenticateToken(ctx context.Context, tokenData string) (*authenticator.Response, bool, error) {
	keyResp, err := a.Client.ListPublicKeys(ctx, &externalsigner.ListPublicKeysRequest{})
	if err != nil {
		return nil, false, err
	}
	var keyData []byte
	for _, pubKey := range keyResp.PublicKeys {
		keyData = append(keyData, pubKey.PublicKey...)
		keyData = append(keyData, '\n')
	}
	keys, err := keyutil.ParsePublicKeysPEM(keyData)
	if err != nil {
		return nil, false, err
	}
	return JWTTokenAuthenticator(
		a.Iss,
		keys,
		a.ImplicitAuds,
		a.Validator,
	).AuthenticateToken(
		ctx, tokenData,
	)
}

// ExternalJWTTokenAuthenticator authenticates JWT tokens signed externally
func ExternalJWTTokenAuthenticator(socketPath string, iss string, implicitAuds authenticator.Audiences, validator Validator) (authenticator.Token, error) {
	conn, err := grpc.Dial(fmt.Sprintf("unix://%s", socketPath), grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return &ExternalTokenAuthenticator{
		Client:       externalsigner.NewKeyServiceClient(conn),
		Iss:          iss,
		Validator:    validator,
		ImplicitAuds: implicitAuds,
	}, nil
}
