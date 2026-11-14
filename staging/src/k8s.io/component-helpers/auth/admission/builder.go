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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/coreos/go-oidc/v3/oidc"
	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
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

	// serviceAudienceSet records whether WithServiceAudience was supplied so
	// buildVerifier can reject missing or conflicting audience options.
	serviceAudienceSet bool
	serviceName        string
	servicePath        string
	servicePort        uint16
	// namespaceFile is the projected ServiceAccount namespace file path,
	// defaulted by newBuilder to projectedServiceAccountNamespace.
	namespaceFile string

	mismatchProviderSet bool
	mismatchProvider    WebhookSubresourceGroupAllowedMismatch
}

func newBuilder() *builder {
	return &builder{namespaceFile: projectedServiceAccountNamespace}
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
// webhook to https://<serviceName>.<namespace>.svc:<servicePort><servicePath>,
// reading <namespace> from the Pod's projected ServiceAccount namespace file.
// Supply the registered Service's serviceName, servicePath (starting with "/",
// or "/" if omitted in the registration), and non-zero servicePort (443 if
// omitted in the registration; this is Service spec.ports[].port /
// clientConfig.service.port, not the container's targetPort). Use [WithIssuer]
// for URL-backed registrations; it cannot be combined with WithServiceAudience.
// Never derive the audience from a request or token.
func WithServiceAudience(serviceName, servicePath string, servicePort uint16) Option {
	return func(b *builder) {
		b.serviceAudienceSet = true
		b.serviceName = serviceName
		b.servicePath = servicePath
		b.servicePort = servicePort
	}
}

// WithSubresourceGroupMismatches overrides the allowed cross-group subresource
// object Kind map using provider.Build(ctx) at startup. When omitted,
// [WithServiceAudience] queries cluster discovery once at startup via a cached
// discovery client (seeded with [AllowedSubresourceGroupMismatchList]) and
// [WithIssuer] uses [AllowedSubresourceGroupMismatchList].
func WithSubresourceGroupMismatches(provider WebhookSubresourceGroupAllowedMismatch) Option {
	return func(b *builder) {
		b.mismatchProviderSet = true
		b.mismatchProvider = provider
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
	if b.mismatchProviderSet && b.mismatchProvider == nil {
		return nil, errors.New("admission: WithSubresourceGroupMismatches requires a non-nil provider")
	}
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
	audience, err := b.buildInClusterServiceAudience()
	if err != nil {
		return nil, err
	}
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("admission: loading in-cluster configuration: %w", err)
	}
	return b.localVerifierWithConfig(ctx, config, audience)
}

func (b *builder) localVerifierWithConfig(ctx context.Context, config *rest.Config, audience string) (*verifier, error) {
	v, err := newLocalVerifier(ctx, config, audience)
	if err != nil {
		return nil, err
	}
	if b.mismatchProviderSet {
		expectedMismatches, err := b.mismatchProvider.Build(ctx)
		if err != nil {
			klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").Error(err, "Failed to build subresource group mismatch map")
			return nil, fmt.Errorf("admission: building subresource group mismatches: %w", err)
		}
		v.expectedSubresourceMismatches = expectedMismatches
		return v, nil
	}
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("admission: creating in-cluster discovery client: %w", err)
	}
	cachedDiscovery := memory.NewMemCacheClient(discoveryClient)
	expectedMismatches, err := DiscoverSubresourceGroupMismatches(ctx, cachedDiscovery)
	if err != nil {
		klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").Error(err, "Failed to discover in-cluster subresource group mismatches on startup")
		return nil, err
	}
	v.expectedSubresourceMismatches = expectedMismatches
	return v, nil
}

// buildInClusterServiceAudience validates the Service name, path, port, and
// projected Pod namespace file and formats the expected Service URL audience.
func (b *builder) buildInClusterServiceAudience() (string, error) {
	var errs []error
	if len(validation.IsDNS1123Label(b.serviceName)) != 0 {
		errs = append(errs, errors.New("service name must be a valid DNS-1123 label"))
	}
	if b.servicePort == 0 {
		errs = append(errs, errors.New("service port must be non-zero"))
	}
	if b.servicePath == "" || !strings.HasPrefix(b.servicePath, "/") {
		errs = append(errs, errors.New("service path must be non-empty and start with \"/\""))
	} else {
		u, err := url.Parse(b.servicePath)
		if err != nil || u.Scheme != "" || u.Opaque != "" || u.Host != "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || strings.Contains(b.servicePath, "#") || !strings.HasPrefix(u.EscapedPath(), "/") {
			errs = append(errs, errors.New("service path must be a relative URL path without query or fragment"))
		}
	}
	if len(errs) != 0 {
		return "", fmt.Errorf("admission: invalid WithServiceAudience: %w", errors.Join(errs...))
	}
	data, err := os.ReadFile(b.namespaceFile)
	if err != nil {
		return "", fmt.Errorf("admission: reading projected Pod namespace for WithServiceAudience: %w", err)
	}
	podNamespace := strings.TrimSpace(string(data))
	if len(validation.IsDNS1123Label(podNamespace)) != 0 {
		return "", errors.New("admission: projected Pod namespace for WithServiceAudience is empty or invalid")
	}
	u := url.URL{
		Scheme: "https",
		Host:   fmt.Sprintf("%s.%s.svc:%d", b.serviceName, podNamespace, b.servicePort),
		Path:   b.servicePath,
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
	expectedMismatches := AllowedSubresourceGroupMismatchList()
	if b.mismatchProviderSet {
		expectedMismatches, err = b.mismatchProvider.Build(ctx)
		if err != nil {
			klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").Error(err, "Failed to build subresource group mismatch map")
			return nil, fmt.Errorf("admission: building subresource group mismatches: %w", err)
		}
	}
	return &verifier{
		verifier: provider.VerifierContext(ctx, &oidc.Config{
			ClientID:             b.audience,
			SupportedSigningAlgs: supportedSigningAlgs,
		}),
		expectedSubresourceMismatches: expectedMismatches,
	}, nil
}

func extractLocalIssuer(tokenFile string) (string, error) {
	if tokenFile == "" {
		return "", errors.New("admission: in-cluster configuration requires BearerTokenFile for rotatable ServiceAccount credentials")
	}
	tokenData, err := os.ReadFile(tokenFile)
	if err != nil {
		return "", fmt.Errorf("admission: reading projected ServiceAccount token file: %w", err)
	}
	token, err := jose.ParseSignedCompact(strings.TrimSpace(string(tokenData)), allowedSigningAlgs)
	if err != nil {
		return "", fmt.Errorf("admission: parsing projected ServiceAccount token: %w", err)
	}
	var claims jwt.Claims
	if err := json.Unmarshal(token.UnsafePayloadWithoutVerification(), &claims); err != nil {
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
		verifier:                      oidc.NewVerifier(issuer, oidc.NewRemoteKeySet(oidc.ClientContext(ctx, client), jwksURL), verifierConfig),
		expectedSubresourceMismatches: AllowedSubresourceGroupMismatchList(),
	}, nil
}
