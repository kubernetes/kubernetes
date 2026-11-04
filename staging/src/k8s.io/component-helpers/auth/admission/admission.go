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
package admission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/coreos/go-oidc/v3/oidc"
	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(admissionv1.AddToScheme(scheme))
}

// Match API server's 3MB limit
// staging/src/k8s.io/apiserver/pkg/server/config.go#L485-L494
const defaultMaxBodyBytes int64 = int64(3 * 1024 * 1024)

const (
	localJWKSPath                    = "/openid/v1/jwks"
	projectedServiceAccountNamespace = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	genericDenyMessage               = "webhook token verification failed"
)

var (
	errVerificationFailed = errors.New(genericDenyMessage)
	errInvalidReview      = errors.New("request body is not a decodable AdmissionReview")
	errBodyTooLarge       = errors.New("request body exceeds the configured limit")

	// Keep in sync with allowedSigningAlgs in k8s.io/apiserver's OIDC JWT authenticator;
	// component-helpers cannot import k8s.io/apiserver.
	// https://github.com/kubernetes/kubernetes/blob/51072aa8e1954f42274dd7b76213f58f7ad7c87d/staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/oidc.go#L257-L268
	allowedSigningAlgs = []jose.SignatureAlgorithm{
		jose.RS256,
		jose.RS384,
		jose.RS512,
		jose.ES256,
		jose.ES384,
		jose.ES512,
		jose.PS256,
		jose.PS384,
		jose.PS512,
	}
	supportedSigningAlgs = []string{
		oidc.RS256,
		oidc.RS384,
		oidc.RS512,
		oidc.ES256,
		oidc.ES384,
		oidc.ES512,
		oidc.PS256,
		oidc.PS384,
		oidc.PS512,
	}
)

// AdmissionHandler decides an authenticated admission request.
type AdmissionHandler func(ctx context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse

// Option configures [WithTokenVerification].
type Option func(*builder)

type builder struct {
	// remoteSet prevents incomplete remote options from selecting local mode.
	remoteSet  bool
	issuer     string
	audience   string
	httpClient *http.Client

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
		b.httpClient = client
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

// Handler authenticates and serves v1 AdmissionReviews.
type Handler struct {
	verifier *verifier
	next     AdmissionHandler
	maxBody  int64
}

// WithTokenVerification authenticates requests before decoding their bodies and
// calling next. Register admissionReviewVersions as ["v1"]. Use
// [WithServiceAudience] in-cluster or [WithIssuer] for explicit issuer
// discovery. Failed authentication returns HTTP 401; authenticated
// admission-policy denials return an AdmissionReview with Allowed:false. Use
// failurePolicy: Fail when authentication or transport failures must block
// admission. A nil next returns an error.
func WithTokenVerification(ctx context.Context, next AdmissionHandler, opts ...Option) (*Handler, error) {
	if next == nil {
		return nil, errors.New("admission: next AdmissionHandler must not be nil")
	}
	v, err := newVerifier(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return newHandler(v, next, defaultMaxBodyBytes), nil
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
	client := b.httpClient
	if client == nil {
		client = http.DefaultClient
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

func newLocalVerifier(ctx context.Context, config *rest.Config, audience string) (*verifier, error) {
	// Require BearerTokenFile rather than static in-memory BearerToken so local
	// verification always uses rotatable projected ServiceAccount credentials.
	if config.BearerTokenFile == "" {
		return nil, errors.New("admission: in-cluster configuration requires BearerTokenFile for rotatable ServiceAccount credentials")
	}
	tokenData, err := os.ReadFile(config.BearerTokenFile)
	if err != nil {
		return nil, fmt.Errorf("admission: reading projected ServiceAccount token file: %w", err)
	}
	// Bootstrap the expected issuer from our projected credential, never the incoming token.
	token, err := jwt.ParseSigned(strings.TrimSpace(string(tokenData)), allowedSigningAlgs)
	if err != nil {
		return nil, fmt.Errorf("admission: parsing projected ServiceAccount token: %w", err)
	}
	var claims jwt.Claims
	// NOTE: why do we call this? Because we trust our environment. We trust environment
	// variables and file system.  We have to.  No process can run in an untrusted environment reasonably.
	if err := token.UnsafeClaimsWithoutVerification(&claims); err != nil {
		return nil, fmt.Errorf("admission: reading projected ServiceAccount issuer: %w", err)
	}
	config = rest.CopyConfig(config)
	config.BearerToken = ""
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
		verifier: oidc.NewVerifier(claims.Issuer, oidc.NewRemoteKeySet(oidc.ClientContext(ctx, client), jwksURL), verifierConfig),
	}, nil
}

func newHandler(v *verifier, next AdmissionHandler, maxBody int64) *Handler {
	return &Handler{verifier: v, next: next, maxBody: maxBody}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	token, ok := bearerToken(r)
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

func checkRequestObjectKinds(req *admissionv1.AdmissionRequest) error {
	if req == nil {
		return errors.New("admission request is nil")
	}
	if kindOrResourceIncomplete(req) {
		return errors.New("admission request kind or resource is incomplete")
	}
	if req.Kind.Group != req.Resource.Group && !allowedKindResourceGroupMismatch(req) {
		return fmt.Errorf("object kind group %q differs from authorized resource group %q", req.Kind.Group, req.Resource.Group)
	}
	admittedKind := schema.GroupVersionKind(req.Kind)
	if err := checkObjectMatchesAdmittedKind("object", req.Object, admittedKind); err != nil {
		return err
	}
	return checkObjectMatchesAdmittedKind("oldObject", req.OldObject, admittedKind)
}

func kindOrResourceIncomplete(req *admissionv1.AdmissionRequest) bool {
	return req.Kind.Version == "" || req.Kind.Kind == "" || req.Resource.Version == "" || req.Resource.Resource == ""
}

// Tokens attest the parent resource group; these subresources have cross-group kinds.
func allowedKindResourceGroupMismatch(req *admissionv1.AdmissionRequest) bool {
	switch req.Kind {
	case metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}:
		return req.SubResource == "scale"
	case metav1.GroupVersionKind{Group: "policy", Version: "v1", Kind: "Eviction"}:
		return req.Resource == metav1.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "pods",
		} && req.SubResource == "eviction"
	case metav1.GroupVersionKind{Group: "authentication.k8s.io", Version: "v1", Kind: "TokenRequest"}:
		return req.Resource == metav1.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "serviceaccounts",
		} && req.SubResource == "token"
	default:
		return false
	}
}

// checkObjectMatchesAdmittedKind validates both RawExtension representations against the admitted GVK.
func checkObjectMatchesAdmittedKind(name string, extension runtime.RawExtension, admittedKind schema.GroupVersionKind) error {
	if err := checkRawObjectMatchesAdmittedKind(name, extension.Raw, admittedKind); err != nil {
		return err
	}
	return checkRuntimeObjectMatchesAdmittedKind(name, extension.Object, admittedKind)
}

// checkRuntimeObjectMatchesAdmittedKind reads in-memory GVK without marshaling object contents.
func checkRuntimeObjectMatchesAdmittedKind(name string, object runtime.Object, admittedKind schema.GroupVersionKind) error {
	if object == nil {
		return nil
	}
	objectKind := object.GetObjectKind()
	if objectKind == nil {
		return fmt.Errorf("%s has no kind", name)
	}
	return checkGVKMatchesAdmittedKind(name, objectKind.GroupVersionKind(), admittedKind)
}

// checkRawObjectMatchesAdmittedKind reads JSON TypeMeta; empty data is absent.
func checkRawObjectMatchesAdmittedKind(name string, raw []byte, admittedKind schema.GroupVersionKind) error {
	if len(raw) == 0 {
		return nil
	}
	var tm metav1.TypeMeta
	if err := json.Unmarshal(raw, &tm); err != nil {
		return fmt.Errorf("%s is undecodable: %w", name, err)
	}
	gv, err := schema.ParseGroupVersion(tm.APIVersion)
	if err != nil {
		return fmt.Errorf("%s apiVersion %q is invalid: %w", name, tm.APIVersion, err)
	}
	return checkGVKMatchesAdmittedKind(name, gv.WithKind(tm.Kind), admittedKind)
}

func checkGVKMatchesAdmittedKind(name string, gvk schema.GroupVersionKind, admittedKind schema.GroupVersionKind) error {
	if gvk.Version == "" || gvk.Kind == "" {
		return fmt.Errorf("%s has no apiVersion or kind", name)
	}
	if gvk != admittedKind {
		return fmt.Errorf("%s GVK %s differs from the admitted kind", name, gvk.String())
	}
	return nil
}

// verifier is unexported so callers must enter through [WithTokenVerification],
// which authenticates the Authorization header before reading or decoding the
// HTTP request body.
type verifier struct {
	verifier *oidc.IDTokenVerifier
}

func newVerifier(ctx context.Context, opts ...Option) (*verifier, error) {
	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}
	return b.buildVerifier(ctx)
}

func (v *verifier) verify(ctx context.Context, req *admissionv1.AdmissionRequest, token string) error {
	if req == nil {
		return errVerificationFailed
	}
	groups, err := v.authenticateToken(ctx, token)
	if err != nil {
		return err
	}
	return v.checkDecodedRequest(ctx, req, groups)
}

func (v *verifier) authenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	if ctx.Err() != nil {
		return nil, errVerificationFailed
	}
	// JWTs use compact serialization; the JOSE parser also accepts JSON JWS.
	if strings.HasPrefix(strings.TrimSpace(rawToken), "{") {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "token is not compact JWT")
		return nil, errVerificationFailed
	}
	token, err := v.verifier.Verify(ctx, rawToken)
	if err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "token authentication failed", "detail", err.Error())
		return nil, errVerificationFailed
	}
	groups, err := tokenGroups(token)
	if err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "invalid Kubernetes token claims", "detail", err.Error())
		return nil, errVerificationFailed
	}
	return groups, nil
}

func (v *verifier) checkDecodedRequest(ctx context.Context, req *admissionv1.AdmissionRequest, groups []string) error {
	if ctx.Err() != nil || req == nil {
		return errVerificationFailed
	}
	if !allowsAPIGroup(groups, req.Resource.Group) {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", "token admissionReviewAPIGroups does not authorize the review's API group")
		return errVerificationFailed
	}
	if err := checkRequestObjectKinds(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Admission request rejected", "reason", "invalid admission request",
			"requestUID", req.UID, "operation", req.Operation, "resource", schema.GroupVersionResource(req.Resource).String(),
			"detail", err.Error())
		return errVerificationFailed
	}
	return nil
}

func bearerToken(r *http.Request) (token string, ok bool) {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return "", false
	}
	parts := strings.SplitN(auth, " ", 3)
	if len(parts) < 2 || !strings.EqualFold(parts[0], "bearer") {
		return "", false
	}
	token = parts[1]
	if token == "" {
		return "", false
	}
	return token, true
}
