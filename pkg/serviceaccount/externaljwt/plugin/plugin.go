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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	jose "gopkg.in/go-jose/go-jose.v2"
	"gopkg.in/go-jose/go-jose.v2/jwt"

	externaljwtv1 "k8s.io/externaljwt/apis/v1"
	"k8s.io/kubernetes/pkg/serviceaccount"
	externaljwtmetrics "k8s.io/kubernetes/pkg/serviceaccount/externaljwt/metrics"
)

func init() {
	externaljwtmetrics.RegisterMetrics()
}

type VerificationKeys struct {
	Keys            []serviceaccount.PublicKey
	DataTimestamp   time.Time
	NextRefreshHint time.Time
}

// New calls external signer to fill out supported keys.
// It also starts a periodic sync of external keys.
// In order for the key cache and external signing to work correctly, pass a context that will live as
// long as the dependent process; is used to maintain the lifetime of the connection to external signer.
func New(ctx context.Context, issuer, socketPath string, keySyncTimeout time.Duration, allowSigningWithNonOIDCKeys bool) (*Plugin, *keyCache, error) {
	conn, err := grpc.Dial(
		socketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithAuthority("localhost"),
		grpc.WithDefaultCallOptions(grpc.WaitForReady(true)),
		grpc.WithContextDialer(func(ctx context.Context, path string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}),
		grpc.WithChainUnaryInterceptor(externaljwtmetrics.OuboundRequestMetricsInterceptor),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("while dialing grpc socket at %q: %w", socketPath, err)
	}

	plugin := newPlugin(issuer, conn, allowSigningWithNonOIDCKeys)

	initialFillCtx, cancel := context.WithTimeout(ctx, keySyncTimeout)
	defer cancel()

	if err := plugin.keyCache.initialFill(initialFillCtx); err != nil {
		return nil, nil, fmt.Errorf("while initially filling key cache: %w", err)
	}
	go plugin.keyCache.scheduleSync(ctx, keySyncTimeout)

	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	return plugin, plugin.keyCache, nil
}

// enables plugging in an external jwt signer.
type Plugin struct {
	iss                         string
	client                      externaljwtv1.ExternalJWTSignerClient
	keyCache                    *keyCache
	allowSigningWithNonOIDCKeys bool
}

// newPlugin constructs an implementation of external JWT signer plugin.
func newPlugin(iss string, conn *grpc.ClientConn, allowSigningWithNonOIDCKeys bool) *Plugin {
	client := externaljwtv1.NewExternalJWTSignerClient(conn)
	plugin := &Plugin{
		iss:                         iss,
		client:                      client,
		allowSigningWithNonOIDCKeys: allowSigningWithNonOIDCKeys,
		keyCache:                    newKeyCache(client),
	}
	return plugin
}

// GenerateToken creates a service account token with the provided claims by
// calling out to the external signer binary.
func (p *Plugin) GenerateToken(ctx context.Context, claims *jwt.Claims, privateClaims interface{}) (string, error) {
	jwt, err := p.signAndAssembleJWT(ctx, claims, privateClaims)
	externaljwtmetrics.RecordTokenGenAttempt(err)
	return jwt, err
}

func (p *Plugin) signAndAssembleJWT(ctx context.Context, claims *jwt.Claims, privateClaims interface{}) (string, error) {
	payload, err := mergeClaims(p.iss, claims, privateClaims)
	if err != nil {
		return "", fmt.Errorf("while merging claims: %w", err)
	}

	payloadBase64 := base64.RawURLEncoding.EncodeToString(payload)

	request := &externaljwtv1.SignJWTRequest{
		Claims: payloadBase64,
	}

	response, err := p.client.Sign(ctx, request)
	if err != nil {
		return "", fmt.Errorf("while signing jwt: %w", err)
	}

	if err := p.validateJWTHeader(ctx, response); err != nil {
		return "", fmt.Errorf("while validating header: %w", err)
	}

	if len(response.Signature) == 0 {
		return "", fmt.Errorf("empty signature returned")
	}

	return response.Header + "." + payloadBase64 + "." + response.Signature, nil
}

// GetServiceMetadata returns metadata associated with externalJWTSigner
// It Includes details like max token lifetime supported by externalJWTSigner, etc.
func (p *Plugin) GetServiceMetadata(ctx context.Context) (*externaljwtv1.MetadataResponse, error) {
	req := &externaljwtv1.MetadataRequest{}
	return p.client.Metadata(ctx, req)
}

func (p *Plugin) validateJWTHeader(ctx context.Context, response *externaljwtv1.SignJWTResponse) error {
	jsonBytes, err := base64.RawURLEncoding.DecodeString(response.Header)
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
	if len(header.KeyID) == 0 {
		return fmt.Errorf("key id missing")
	}
	if len(header.KeyID) > 1024 {
		return fmt.Errorf("key id longer than 1 kb")
	}
	switch header.Algorithm {
	// IMPORTANT: If this function is updated to support additional algorithms,
	// JWTTokenGenerator, signerFromRSAPrivateKey, signerFromECDSAPrivateKey in
	// kubernetes/pkg/serviceaccount/jwt.go must also be updated to support the same Algorithms.
	case "RS256", "ES256", "ES384", "ES512":
		// OK
	default:
		return fmt.Errorf("bad signing algorithm %q", header.Algorithm)
	}

	if !p.allowSigningWithNonOIDCKeys {
		publicKeys := p.keyCache.GetPublicKeys(ctx, header.KeyID)
		for _, key := range publicKeys {
			// Such keys shall only be used for validating formerly issued tokens.
			if key.ExcludeFromOIDCDiscovery {
				return fmt.Errorf("key used for signing JWT (kid: %s) is excluded from OIDC discovery docs", header.KeyID)
			}
		}
	}

	return nil
}

func mergeClaims(iss string, claims *jwt.Claims, privateClaims interface{}) ([]byte, error) {
	var out []byte
	signer := payloadGrabber(func(payload []byte) { out = payload })
	_, err := serviceaccount.GenerateToken(signer, iss, claims, privateClaims)
	if len(out) == 0 {
		return nil, fmt.Errorf("failed to marshal: %w", err)
	}
	return out, nil // error is safe to ignore as long as we have the payload bytes
}

var _ jose.Signer = payloadGrabber(nil)

type payloadGrabber func(payload []byte)

func (p payloadGrabber) Sign(payload []byte) (*jose.JSONWebSignature, error) {
	p(payload)
	return nil, jose.ErrUnprotectedNonce // return some error to stop after we have the payload
}

func (p payloadGrabber) Options() jose.SignerOptions { return jose.SignerOptions{} }
