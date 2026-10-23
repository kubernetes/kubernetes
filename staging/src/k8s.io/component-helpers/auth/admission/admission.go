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

// Package admission verifies API-server authentication tokens for admission webhooks.
package admission // import "k8s.io/component-helpers/auth/admission"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/auth/admission/internal/verify"
	"k8s.io/klog/v2"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(admissionv1.AddToScheme(scheme))
}

// Match controller-runtime's admission request limit:
// https://github.com/kubernetes-sigs/controller-runtime/blob/2e312096f4555f330939cfd5e30a0ae2436829c0/pkg/webhook/admission/http.go#L37-L55
const defaultMaxBodyBytes int64 = 7 << 20

const (
	verificationTimeout = 10 * time.Second
	localJWKSPath       = "/openid/v1/jwks"
	genericDenyMessage  = "webhook token verification failed"
)

var (
	errVerificationFailed = errors.New(genericDenyMessage)
	errInvalidReview      = errors.New("request body is not a decodable AdmissionReview")
	errBodyTooLarge       = errors.New("request body exceeds the configured limit")

	// Keep in sync with pkg/serviceaccount; staging cannot import k8s.io/kubernetes.
	// https://github.com/kubernetes/kubernetes/blob/9b965ac43f05d304180c7b8b9611264530788948/pkg/serviceaccount/jwt.go#L340-L351
	serviceAccountSigningAlgs = []jose.SignatureAlgorithm{
		jose.ES256, jose.ES384, jose.ES512, jose.RS256,
	}
)

// AdmissionHandler decides an authenticated admission request.
type AdmissionHandler func(ctx context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse

// Option configures [WithTokenVerification] or [NewVerifier].
type Option func(*builder)

type builder struct {
	// remoteSet prevents incomplete remote options from selecting local mode.
	remoteSet  bool
	issuer     string
	audience   string
	httpClient *http.Client

	explicitAudience bool
}

func newBuilder() *builder {
	return &builder{}
}

// WithRemoteIssuer uses the issuer URL for OIDC discovery and as the expected
// token issuer. issuer and audience are required. client is used for discovery
// and signing-key requests and must not carry credentials: the discovered
// signing-key endpoint may be on another host. A nil client uses the default
// HTTP transport.
func WithRemoteIssuer(issuer, audience string, client *http.Client) Option {
	return func(b *builder) {
		b.remoteSet = true
		b.issuer = issuer
		b.audience = audience
		b.httpClient = client
	}
}

// WithAudience sets the in-cluster webhook's expected audience. Use its configured
// URL verbatim, or https://<name>.<namespace>.svc:<port><path> for a Service
// reference, with default port 443 and path "/". It cannot be combined with
// [WithRemoteIssuer]. The audience must come from trusted webhook configuration.
func WithAudience(audience string) Option {
	return func(b *builder) {
		b.explicitAudience = true
		b.audience = audience
	}
}

// Handler authenticates and serves v1 AdmissionReviews.
type Handler struct {
	verifier *Verifier
	next     AdmissionHandler
	maxBody  int64
}

// WithTokenVerification authenticates requests before decoding their bodies and
// calling next. Register admissionReviewVersions as ["v1"]. Use [WithAudience]
// in-cluster or [WithRemoteIssuer] for another issuer. Failed authentication
// returns HTTP 401; authenticated admission-policy denials return an AdmissionReview
// with Allowed:false. Use failurePolicy: Fail when authentication or transport
// failures must block admission. A nil next returns an error.
func WithTokenVerification(ctx context.Context, next AdmissionHandler, opts ...Option) (*Handler, error) {
	if next == nil {
		return nil, errors.New("admission: next AdmissionHandler must not be nil")
	}
	v, err := NewVerifier(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return newHandler(v, next, defaultMaxBodyBytes), nil
}

func (b *builder) buildVerifier(ctx context.Context) (*Verifier, error) {
	if b.remoteSet && b.explicitAudience {
		return nil, errors.New("admission: WithAudience cannot be combined with WithRemoteIssuer")
	}
	if b.remoteSet {
		if b.issuer == "" || b.audience == "" {
			return nil, errors.New("admission: remote verification requires both an issuer and an audience")
		}
		return b.remoteVerifier(ctx)
	}
	if !b.explicitAudience || b.audience == "" {
		return nil, errors.New("admission: in-cluster verification requires WithAudience set to the webhook's configured URL or Service audience")
	}
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("admission: loading in-cluster configuration: %w", err)
	}
	return newLocalVerifier(ctx, config, b.audience)
}

func (b *builder) remoteVerifier(ctx context.Context) (*Verifier, error) {
	issuerURL, err := url.Parse(b.issuer)
	if err != nil || issuerURL.Scheme != "https" || issuerURL.Host == "" || issuerURL.User != nil || issuerURL.RawQuery != "" || issuerURL.Fragment != "" {
		return nil, errors.New("admission: remote issuer must be an HTTPS URL without credentials, query, or fragment")
	}
	client := b.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	clientCopy := *client
	// RemoteKeySet fetches outlive the verification request's context.
	if clientCopy.Timeout <= 0 || clientCopy.Timeout > verificationTimeout {
		clientCopy.Timeout = verificationTimeout
	}
	previousRedirectPolicy := clientCopy.CheckRedirect
	clientCopy.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != "https" {
			return errors.New("admission: signing-key and discovery redirects must use HTTPS")
		}
		if previousRedirectPolicy != nil {
			return previousRedirectPolicy(req, via)
		}
		if len(via) >= 10 {
			return errors.New("admission: stopped after 10 redirects")
		}
		return nil
	}
	client = &clientCopy
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
	v := &Verifier{
		verifier: provider.VerifierContext(ctx, &oidc.Config{ClientID: b.audience}),
		client:   client,
		jwksURL:  metadata.JWKSURI,
	}
	if err := v.healthCheck(ctx); err != nil {
		return nil, err
	}
	return v, nil
}

func newLocalVerifier(ctx context.Context, config *rest.Config, audience string) (*Verifier, error) {
	token, err := jwt.ParseSigned(strings.TrimSpace(config.BearerToken), serviceAccountSigningAlgs)
	if err != nil {
		return nil, fmt.Errorf("admission: parsing projected ServiceAccount token: %w", err)
	}
	var claims jwt.Claims
	if err := token.UnsafeClaimsWithoutVerification(&claims); err != nil {
		return nil, fmt.Errorf("admission: reading projected ServiceAccount issuer: %w", err)
	}
	if claims.Issuer == "" {
		return nil, errors.New("admission: projected ServiceAccount token has no issuer")
	}
	config = rest.CopyConfig(config)
	if config.Timeout <= 0 || config.Timeout > verificationTimeout {
		config.Timeout = verificationTimeout
	}
	client, err := rest.HTTPClientFor(config)
	if err != nil {
		return nil, fmt.Errorf("admission: creating in-cluster HTTP client: %w", err)
	}
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return errors.New("admission: local signing-key requests must not redirect")
	}
	jwksURL := strings.TrimRight(config.Host, "/") + localJWKSPath
	verifierConfig := &oidc.Config{ClientID: audience}
	for _, algorithm := range serviceAccountSigningAlgs {
		verifierConfig.SupportedSigningAlgs = append(verifierConfig.SupportedSigningAlgs, string(algorithm))
	}
	v := &Verifier{
		verifier: oidc.NewVerifier(claims.Issuer, oidc.NewRemoteKeySet(oidc.ClientContext(ctx, client), jwksURL), verifierConfig),
		client:   client,
		jwksURL:  jwksURL,
	}
	if err := v.healthCheck(ctx); err != nil {
		return nil, err
	}
	return v, nil
}

func newHandler(v *Verifier, next AdmissionHandler, maxBody int64) *Handler {
	return &Handler{verifier: v, next: next, maxBody: maxBody}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	token, ok := BearerToken(r)
	if !ok {
		denyPreDecode(ctx, w, "no bearer token presented")
		return
	}
	groups, err := h.verifier.authenticateToken(ctx, token)
	if err != nil {
		http.Error(w, genericDenyMessage, http.StatusUnauthorized)
		return
	}
	review, err := h.decodeReview(r)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			http.Error(w, errBodyTooLarge.Error(), http.StatusRequestEntityTooLarge)
		} else {
			denyPreDecode(ctx, w, err.Error())
		}
		return
	}
	req := review.Request
	if err := h.verifier.checkDecodedRequest(ctx, req, groups); err != nil {
		writeResponse(w, req, deniedResponse())
		return
	}
	klog.FromContext(ctx).V(4).Info("Admission webhook token verified")
	writeResponse(w, req, h.next(ctx, req))
}

func denyPreDecode(ctx context.Context, w http.ResponseWriter, reason string) {
	klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied", "reason", reason)
	http.Error(w, genericDenyMessage, http.StatusUnauthorized)
}

func deniedResponse() *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: false,
		Result: &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: genericDenyMessage,
			Code:    http.StatusUnauthorized,
		},
	}
}

func writeResponse(w http.ResponseWriter, req *admissionv1.AdmissionRequest, resp *admissionv1.AdmissionResponse) {
	if resp == nil {
		resp = deniedResponse()
	}
	response := *resp
	response.UID = req.UID
	out := &admissionv1.AdmissionReview{Response: &response}
	data, err := runtime.Encode(codecs.LegacyCodec(admissionv1.SchemeGroupVersion), out)
	if err != nil {
		http.Error(w, "failed to encode admission response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

func (h *Handler) decodeReview(r *http.Request) (*admissionv1.AdmissionReview, error) {
	if r.Body == nil {
		return nil, errInvalidReview
	}
	buf, err := io.ReadAll(io.LimitReader(r.Body, h.maxBody+1))
	_ = r.Body.Close()
	if err != nil {
		return nil, errInvalidReview
	}
	if int64(len(buf)) > h.maxBody {
		return nil, errBodyTooLarge
	}
	obj, _, err := codecs.UniversalDeserializer().Decode(buf, nil, &admissionv1.AdmissionReview{})
	if err != nil {
		return nil, errInvalidReview
	}
	review, ok := obj.(*admissionv1.AdmissionReview)
	if !ok || review.Request == nil {
		return nil, errInvalidReview
	}
	return review, nil
}

// checkObjectGroup binds the objects to the admitted Kind, allowing only known
// subresources to carry an object outside the attested resource group.
func checkObjectGroup(req *admissionv1.AdmissionRequest) error {
	if req == nil {
		return errors.New("admission request is nil")
	}
	group, version, kind := req.Resource.Group, "", ""
	kindSpecified := req.Kind.Group != "" || req.Kind.Version != "" || req.Kind.Kind != ""
	if req.SubResource != "" && !kindSpecified {
		return errors.New("subresource admission request has no kind")
	}
	if kindSpecified {
		if req.Kind.Version == "" || req.Kind.Kind == "" {
			return errors.New("admission request kind is incomplete")
		}
		if req.Kind.Group != req.Resource.Group {
			switch {
			case req.SubResource == "scale" && req.Kind.Group == "autoscaling" && req.Kind.Version == "v1" && req.Kind.Kind == "Scale":
			case req.Resource.Group == "" && req.Resource.Resource == "pods" && req.SubResource == "eviction" && req.Kind.Group == "policy" && req.Kind.Version == "v1" && req.Kind.Kind == "Eviction":
			case req.Resource.Group == "" && req.Resource.Resource == "serviceaccounts" && req.SubResource == "token" && req.Kind.Group == "authentication.k8s.io" && req.Kind.Version == "v1" && req.Kind.Kind == "TokenRequest":
			default:
				return fmt.Errorf("object kind group %q differs from authorized resource group %q", req.Kind.Group, req.Resource.Group)
			}
		}
		group, version, kind = req.Kind.Group, req.Kind.Version, req.Kind.Kind
	}
	if err := checkExtensionObjectGroup("object", req.Object, group, version, kind); err != nil {
		return err
	}
	return checkExtensionObjectGroup("oldObject", req.OldObject, group, version, kind)
}

func checkExtensionObjectGroup(name string, extension runtime.RawExtension, group, version, kind string) error {
	raw := extension.Raw
	if len(raw) == 0 && extension.Object != nil {
		var err error
		raw, err = json.Marshal(extension.Object)
		if err != nil {
			return fmt.Errorf("%s is undecodable: %w", name, err)
		}
	}
	return checkRawObjectGroup(name, raw, group, version, kind)
}

func checkRawObjectGroup(name string, raw []byte, group, version, kind string) error {
	if len(raw) == 0 {
		return nil
	}
	var tm metav1.TypeMeta
	if err := json.Unmarshal(raw, &tm); err != nil {
		return fmt.Errorf("%s is undecodable: %w", name, err)
	}
	if tm.APIVersion == "" || tm.Kind == "" {
		return fmt.Errorf("%s has no apiVersion or kind", name)
	}
	gv, err := schema.ParseGroupVersion(tm.APIVersion)
	if err != nil {
		return fmt.Errorf("%s apiVersion %q is invalid: %w", name, tm.APIVersion, err)
	}
	if gv.Version == "" {
		return fmt.Errorf("%s apiVersion %q is missing a version", name, tm.APIVersion)
	}
	if gv.Group != group {
		return fmt.Errorf("%s group %q differs from admitted kind group %q", name, gv.Group, group)
	}
	if (version != "" && gv.Version != version) || (kind != "" && tm.Kind != kind) {
		return fmt.Errorf("%s GVK %s %q differs from the admitted kind", name, gv.String(), tm.Kind)
	}
	return nil
}

// Verifier authenticates tokens and authorizes already-decoded admission requests.
type Verifier struct {
	verifier *oidc.IDTokenVerifier
	client   *http.Client
	jwksURL  string
}

// NewVerifier constructs a verifier using [WithAudience] for local issuer/keys or
// [WithRemoteIssuer] for remote discovery. Configuration and endpoint availability
// are checked during construction; signing keys are loaded lazily by coreos.
func NewVerifier(ctx context.Context, opts ...Option) (*Verifier, error) {
	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}
	return b.buildVerifier(ctx)
}

// Verify authenticates token and checks the request's API group and object kinds.
// Any denial, including a nil request, returns the same generic error.
func (v *Verifier) Verify(ctx context.Context, req *admissionv1.AdmissionRequest, token string) error {
	if req == nil {
		return errVerificationFailed
	}
	groups, err := v.authenticateToken(ctx, token)
	if err != nil {
		return err
	}
	return v.checkDecodedRequest(ctx, req, groups)
}

func (v *Verifier) authenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	if ctx.Err() != nil {
		return nil, errVerificationFailed
	}
	token, err := v.verifier.Verify(ctx, rawToken)
	if err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "token authentication failed", "detail", err.Error())
		return nil, errVerificationFailed
	}
	groups, err := verify.TokenGroups(token)
	if err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "invalid Kubernetes token claims", "detail", err.Error())
		return nil, errVerificationFailed
	}
	return groups, nil
}

func (v *Verifier) checkDecodedRequest(ctx context.Context, req *admissionv1.AdmissionRequest, groups []string) error {
	if ctx.Err() != nil || req == nil {
		return errVerificationFailed
	}
	if !verify.AllowsAPIGroup(groups, req.Resource.Group) {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "token admissionReviewAPIGroups does not authorize the review's API group")
		return errVerificationFailed
	}
	if err := checkObjectGroup(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "admitted object does not match AdmissionReview kind", "detail", err.Error())
		return errVerificationFailed
	}
	return nil
}

// HealthCheck checks signing-key endpoint availability, not key usability or
// freshness. It neither populates nor invalidates the verifier's key cache.
func (v *Verifier) HealthCheck() error {
	return v.healthCheck(context.Background())
}

func (v *Verifier) healthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.jwksURL, nil)
	if err != nil {
		return fmt.Errorf("admission: creating signing-key availability request: %w", err)
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return fmt.Errorf("admission: signing-key endpoint is unavailable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("admission: signing-key endpoint returned %s", resp.Status)
	}
	return nil
}

// BearerToken extracts a bearer using the apiserver's header parsing rules.
// Callers that decode AdmissionReviews themselves can pass it to [Verifier.Verify].
func BearerToken(r *http.Request) (token string, ok bool) {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return "", false
	}
	parts := strings.SplitN(auth, " ", 3)
	if len(parts) < 2 || strings.ToLower(parts[0]) != "bearer" {
		return "", false
	}
	token = parts[1]
	if token == "" {
		return "", false
	}
	return token, true
}

// HealthCheck checks signing-key endpoint availability, not key usability or freshness.
func (h *Handler) HealthCheck() error {
	return h.verifier.HealthCheck()
}
