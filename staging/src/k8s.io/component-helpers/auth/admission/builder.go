/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/coreos/go-oidc/v3/oidc"
	"github.com/go-jose/go-jose/v4/jwt"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/rest"
)

const (
	localJWKSPath                    = "/openid/v1/jwks"
	projectedServiceAccountNamespace = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)

// Option configures [WithTokenVerification].
type Option func(*builder)

type builder struct {
	// remoteSet prevents incomplete remote options from selecting local mode.
	remoteSet        bool
	issuer           string
	audience         string
	remoteHTTPClient *http.Client

	serviceAudienceSet bool
	serviceName        string
	servicePath        string
	servicePort        uint16
}

func newBuilder() *builder {
	return &builder{}
}

// WithIssuer uses the issuer URL for OIDC discovery and as the expected
// token issuer. issuer and audience are required. client is used for discovery
// and signing-key requests and must not carry credentials: the discovered
// signing-key endpoint may be on another host. A nil client uses the default
// HTTP transport.
func WithIssuer(issuer, audience string, client *http.Client) Option {
	return func(b *builder) {
		b.remoteSet = true
		b.issuer = issuer
		b.audience = audience
		b.remoteHTTPClient = client
	}
}

// WithServiceAudience sets the expected audience of a Service-backed in-cluster
// webhook to https://<name>.<namespace>.svc:<port><path>, reading <namespace>
// from the Pod's projected ServiceAccount namespace file. Supply the registered
// Service's name, path (starting with "/", or "/" if omitted in the
// registration), and non-zero port (443 if omitted in the registration). Use
// [WithIssuer] for URL-backed registrations; it cannot be combined with
// WithServiceAudience. Never derive the audience from a request or token.
func WithServiceAudience(name, path string, port uint16) Option {
	return func(b *builder) {
		b.serviceAudienceSet = true
		b.serviceName = name
		b.servicePath = path
		b.servicePort = port
	}
}

func newVerifier(ctx context.Context, opts ...Option) (*verifier, error) {
	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}
	return b.buildVerifier(ctx)
}

func (b *builder) buildVerifier(ctx context.Context) (*verifier, error) {
	if b.remoteSet && b.serviceAudienceSet {
		return nil, errors.New("admission: WithServiceAudience cannot be combined with WithIssuer")
	}
	if b.remoteSet {
		if b.issuer == "" || b.audience == "" {
			return nil, errors.New("admission: remote verification requires both an issuer and an audience")
		}
		return b.remoteVerifier(ctx)
	}
	if !b.serviceAudienceSet {
		return nil, errors.New("admission: in-cluster verification requires WithServiceAudience")
	}
	return b.localVerifier(ctx)
}

func (b *builder) localVerifier(ctx context.Context) (*verifier, error) {
	audience, err := buildInClusterServiceAudience(b.serviceName, b.servicePath, b.servicePort, projectedServiceAccountNamespace)
	if err != nil {
		return nil, err
	}
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("admission: loading in-cluster configuration: %w", err)
	}
	return newLocalVerifier(ctx, config, audience)
}

// buildInClusterServiceAudience validates the Service name, path, port, and
// projected Pod namespace file and formats the expected Service URL audience.
func buildInClusterServiceAudience(name, path string, port uint16, namespaceFile string) (string, error) {
	var errs []error
	if len(validation.IsDNS1123Label(name)) != 0 {
		errs = append(errs, errors.New("Service name must be a valid DNS-1123 label"))
	}
	if port == 0 {
		errs = append(errs, errors.New("Service port must be non-zero"))
	}
	if path == "" || !strings.HasPrefix(path, "/") {
		errs = append(errs, errors.New("Service path must be non-empty and start with \"/\""))
	} else {
		u, err := url.Parse(path)
		if err != nil || u.Scheme != "" || u.Opaque != "" || u.Host != "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || strings.Contains(path, "#") || !strings.HasPrefix(u.EscapedPath(), "/") {
			errs = append(errs, errors.New("Service path must be a relative URL path without query or fragment"))
		}
	}
	if len(errs) != 0 {
		return "", fmt.Errorf("admission: invalid WithServiceAudience: %w", errors.Join(errs...))
	}
	data, err := os.ReadFile(namespaceFile)
	if err != nil {
		return "", fmt.Errorf("admission: reading projected Pod namespace for WithServiceAudience: %w", err)
	}
	podNamespace := strings.TrimSpace(string(data))
	if len(validation.IsDNS1123Label(podNamespace)) != 0 {
		return "", errors.New("admission: projected Pod namespace for WithServiceAudience is empty or invalid")
	}
	u := url.URL{
		Scheme: "https",
		Host:   fmt.Sprintf("%s.%s.svc:%d", name, podNamespace, port),
		Path:   path,
	}
	return u.String(), nil
}

func (b *builder) remoteVerifier(ctx context.Context) (*verifier, error) {
	issuerURL, err := url.Parse(b.issuer)
	if err != nil || issuerURL.Scheme != "https" || issuerURL.Host == "" || issuerURL.User != nil || issuerURL.RawQuery != "" || issuerURL.Fragment != "" {
		return nil, errors.New("admission: remote issuer must be an HTTPS URL without credentials, query, or fragment")
	}
	client := b.remoteHTTPClient
	if client == nil {
		var err error
		client, err = rest.HTTPClientFor(&rest.Config{})
		if err != nil {
			return nil, fmt.Errorf("admission: creating default HTTP client: %w", err)
		}
	}

	ctx = oidc.ClientContext(ctx, client)
	provider, err := oidc.NewProvider(ctx, b.issuer)
	if err != nil {
		return nil, fmt.Errorf("admission: discovering issuer: %w", err)
	}
	var metadata struct {
		JWKSURI string `json:"jwks_uri"`
	}
	if err := provider.Claims(&metadata); err != nil {
		return nil, fmt.Errorf("admission: reading issuer metadata: %w", err)
	}
	jwksURL, err := url.Parse(metadata.JWKSURI)
	if err != nil || jwksURL.Scheme != "https" || jwksURL.Host == "" || jwksURL.User != nil || jwksURL.Fragment != "" {
		return nil, errors.New("admission: signing-key endpoint must be an HTTPS URL without credentials or fragment")
	}
	return &verifier{verifier: provider.VerifierContext(ctx, &oidc.Config{
		ClientID:             b.audience,
		SupportedSigningAlgs: supportedSigningAlgs,
	})}, nil
}

func extractLocalIssuer(tokenFile string) (string, error) {
	if tokenFile == "" {
		return "", errors.New("admission: in-cluster configuration requires BearerTokenFile for rotatable ServiceAccount credentials")
	}
	tokenData, err := os.ReadFile(tokenFile)
	if err != nil {
		return "", fmt.Errorf("admission: reading projected ServiceAccount token file: %w", err)
	}
	token, err := jwt.ParseSigned(strings.TrimSpace(string(tokenData)), allowedSigningAlgs)
	if err != nil {
		return "", fmt.Errorf("admission: parsing projected ServiceAccount token: %w", err)
	}
	var claims jwt.Claims
	if err := token.UnsafeClaimsWithoutVerification(&claims); err != nil {
		return "", fmt.Errorf("admission: reading projected ServiceAccount issuer: %w", err)
	}
	return claims.Issuer, nil
}

func newLocalVerifier(ctx context.Context, config *rest.Config, audience string) (*verifier, error) {
	issuer, err := extractLocalIssuer(config.BearerTokenFile)
	if err != nil {
		return nil, err
	}
	client, err := rest.HTTPClientFor(config)
	if err != nil {
		return nil, fmt.Errorf("admission: creating in-cluster HTTP client: %w", err)
	}
	jwksURL := strings.TrimRight(config.Host, "/") + localJWKSPath

	verifierConfig := &oidc.Config{
		ClientID:             audience,
		SupportedSigningAlgs: supportedSigningAlgs,
	}
	return &verifier{
		verifier: oidc.NewVerifier(issuer, oidc.NewRemoteKeySet(oidc.ClientContext(ctx, client), jwksURL), verifierConfig),
	}, nil
}
