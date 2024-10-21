/*
Copyright 2024 The Kubernetes Authors.

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

package plugin

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"gopkg.in/square/go-jose.v2/jwt"
	ejspb "k8s.io/externaljwt/apis/v1alpha1"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

type VerificationKeys struct {
	Keys          []serviceaccount.PublicKey
	DataTimestamp time.Time
}

type Metadata struct {
	MaxTokenExpirationSeconds int
}

// New sets up all dependencies for the plugin and cache, in
// order to minimize changes to upstream files.
func New(issuer, socketPath string) (*Plugin, *KeyCache, error) {
	conn, err := grpc.Dial(
		socketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("while dialing go/gke-external-token-signing grpc socket: %w", err)
	}

	plugin := NewPlugin(issuer, conn)

	initialFillCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cache := NewKeyCache(plugin)
	if err := cache.InitialFill(initialFillCtx); err != nil {
		return nil, nil, fmt.Errorf("while initially filling go/gke-external-token-signing key cache: %w", err)
	}
	go cache.StartPeriodicSync(context.Background())

	return plugin, cache, nil
}

// enables pluging in an external jwt signer.
type Plugin struct {
	iss    string
	client ejspb.ExternalJWTSignerClient
}

func NewPlugin(iss string, conn *grpc.ClientConn) *Plugin {
	client := ejspb.NewExternalJWTSignerClient(conn)
	return &Plugin{
		iss:    iss,
		client: client,
	}
}

// GenerateToken creates a service account token with the provided claims by
// calling out to the external signer binary.
func (s *Plugin) GenerateToken(claims *jwt.Claims, privateClaims interface{}) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	claims.Issuer = s.iss

	mergedClaims, err := mergeClaims(claims, privateClaims)
	if err != nil {
		return "", fmt.Errorf("while merging claims: %w", err)
	}

	payload := base64.RawURLEncoding.EncodeToString(mergedClaims)

	request := &ejspb.SignJWTRequest{
		Claims: payload,
	}

	response, err := s.client.Sign(ctx, request)
	if err != nil {
		return "", fmt.Errorf("while signing jwt: %w", err)
	}

	if err := validateJWTHeader(response); err != nil {
		return "", fmt.Errorf("while validating header: %w", err)
	}

	return response.GetHeader() + "." + payload + "." + response.GetSignature(), nil
}

// GetTokenVerificationKeys returns a map of supported external keyIDs to keys
// the keys are PKIX-serialized.
func (s *Plugin) GetTokenVerificationKeys(ctx context.Context) (*VerificationKeys, int, error) {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	req := &ejspb.FetchKeysRequest{}
	resp, err := s.client.FetchKeys(ctx, req)
	if err != nil {
		return nil, 0, fmt.Errorf("while getting externally supported jwt signing keys: %w", err)
	}

	keys := []serviceaccount.PublicKey{}
	for _, protoKey := range resp.GetKeys() {
		parsedPublicKey, err := x509.ParsePKIXPublicKey(protoKey.GetKey())
		if err != nil {
			return nil, 0, fmt.Errorf("while parsing external public keys: %w", err)
		}

		keys = append(keys, serviceaccount.PublicKey{
			KeyID:     protoKey.GetKeyId(),
			PublicKey: parsedPublicKey,
		})
	}

	vk := &VerificationKeys{
		Keys:          keys,
		DataTimestamp: resp.GetDataTimestamp().AsTime(),
	}

	return vk, int(resp.RefreshHintSeconds), nil
}

// GetServiceMetadata returns metadata associated with externalJWTSigner
// It Includes details like max token lifetime supported by externalJWTSigner, etc.
func (s *Plugin) GetServiceMetadata(ctx context.Context) (*Metadata, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := &ejspb.MetadataRequest{}
	resp, err := s.client.Metadata(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("while getting metadata from external jwt signer: %w", err)
	}

	return &Metadata{
		MaxTokenExpirationSeconds: int(resp.MaxTokenExpirationSeconds),
	}, nil
}

func mergeClaims(claims *jwt.Claims, privateClaims interface{}) ([]byte, error) {
	privateBytes, err := json.Marshal(privateClaims)
	if err != nil {
		return nil, fmt.Errorf("while marshalling private claims: %w", err)
	}

	var unifiedMap map[string]any
	if err := json.Unmarshal(privateBytes, &unifiedMap); err != nil {
		return nil, fmt.Errorf("while unmarshalling private claims: %w", err)
	}

	unifiedMap["iss"] = claims.Issuer
	unifiedMap["sub"] = claims.Subject
	unifiedMap["aud"] = claims.Audience
	unifiedMap["exp"] = claims.Expiry
	unifiedMap["nbf"] = claims.NotBefore
	unifiedMap["iat"] = claims.IssuedAt
	if claims.ID != "" {
		unifiedMap["jti"] = claims.ID
	}

	unifiedBytes, err := json.Marshal(unifiedMap)
	if err != nil {
		return nil, fmt.Errorf("while marshalling unified claims: %w", err)
	}

	return unifiedBytes, nil
}

func validateJWTHeader(response *ejspb.SignJWTResponse) error {
	jsonBytes, err := base64.RawURLEncoding.DecodeString(string(response.GetHeader()))
	if err != nil {
		return fmt.Errorf("while unwrapping header: %w", err)
	}

	decoder := json.NewDecoder(bytes.NewBuffer(jsonBytes))
	decoder.DisallowUnknownFields()

	header := &struct {
		Algorithm string `json:"alg,omitempty"`
		KeyID     string `json:"kid,omitempty"`
		Type      string `json:"typ,omitempty"`
	}{}

	if err := decoder.Decode(header); err != nil {
		return fmt.Errorf("while parsing header JSON: %w", err)
	}

	if header.Type != "JWT" {
		return fmt.Errorf("bad type")
	}
	if header.KeyID == "" {
		return fmt.Errorf("key id missing")
	}
	switch header.Algorithm {
	case "RS256", "ES256", "ES384", "ES512":
		// OK
	default:
		return fmt.Errorf("bad signing algorithm %q", header.Algorithm)
	}

	return nil
}
