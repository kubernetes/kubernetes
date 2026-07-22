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

// Package admissionhttp adapts the core token verifier to a plain net/http
// admission webhook. WithTokenVerification builds the verifier and returns an
// http.Handler that decodes the incoming AdmissionReview once, enforces KEP-6060
// token verification, and — only on success — hands the decoded AdmissionRequest
// to a downstream handler.
//
// Enforcement is unconditional: a verification failure never reaches the
// downstream handler. Whether a webhook adopts verification at all is a
// deployment-time concern that defaults off, not a runtime knob here.
//
// WithTokenVerification selects an out-of-cluster or in-cluster verifier by
// option presence: with no option it is the zero-config in-cluster default, and
// WithRemoteConfig opts into out-of-cluster verification. See its doc for the two
// modes and the fail-closed/not-ready behavior until an in-cluster audience binds.
//
// Callers that have already decoded the review (for example controller-runtime)
// should build a verifier with oidc.NewRemoteVerifier and use
// [VerifyAdmissionRequest] directly. Because WithTokenVerification constructs the
// verifier, this package imports verify/oidc and therefore its JOSE/JWT
// dependency tree; a consumer that only needs [VerifyAdmissionRequest] and
// [BearerToken] still pulls that tree transitively.
package admissionhttp // import "k8s.io/webhookauth/verify/admissionhttp"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/oidc"
)

// scheme and codecs decode incoming AdmissionReview bodies through the codec
// factory (TypeMeta-aware) rather than a raw json.Unmarshal.
var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(admissionv1.AddToScheme(scheme))
}

// defaultMaxBodyBytes bounds the request body read before decoding, guarding
// against unbounded reads while staying generous for real AdmissionReviews.
const defaultMaxBodyBytes int64 = 3 << 20 // 3 MiB

// genericDenyMessage is the fixed body/message returned for every denial. It
// carries no claim values, so a caller cannot tell which check failed
// (anti-enumeration).
const genericDenyMessage = "webhook token verification failed"

// Adapter-level denial reasons. Like the core verifier's reasons they are
// surfaced only to the observation hook / logs, never to the HTTP client, and
// carry no claim values.
const (
	reasonNoBearerToken   = "no bearer token presented"
	reasonUndecodableBody = "request body is not a decodable AdmissionReview"
	reasonBodyTooLarge    = "request body exceeds the configured limit"
)

// AdmissionHandler performs the admission decision for an already-verified,
// already-decoded request, mirroring the shape of controller-runtime's
// Handle(ctx, Request) Response. The adapter echoes the request UID onto the
// response and wraps it in an AdmissionReview, so the downstream never touches
// the raw body.
type AdmissionHandler func(ctx context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse

// Option configures WithTokenVerification. Options mutate an internal builder
// BEFORE the verifier is constructed, so an option can influence verifier
// construction (mode selection, remote issuer/audience) as well as handler
// behavior (body limit).
type Option func(*builder)

// builder accumulates option state before WithTokenVerification constructs the
// verifier and assembles the Handler. It holds both verifier inputs (remote
// issuer/audience/client and the unexported in-cluster endpoint overrides) and
// handler config (maxBody).
//
// remoteSet is the single bright-line mode selector: only WithRemoteConfig sets
// it, so remote mode is chosen by the PRESENCE of WithRemoteConfig and never by a
// stray HTTPClient. This keeps the fail-closed contract unambiguous — an injected
// CA can never half-select remote mode.
type builder struct {
	// remote verifier inputs; remoteSet true selects remote mode.
	remoteSet  bool
	issuer     string
	audience   string
	httpClient *http.Client

	// in-cluster endpoint overrides, set only via withInClusterEndpoint (unexported;
	// exposed to tests through export_test.go). They do NOT set remoteSet, so the
	// handler stays in-cluster. inClusterURL=="" uses oidc.InClusterAPIServerURL;
	// inClusterClient==nil uses the projected service-account CA client built inside
	// oidc.NewLocalKeySetVerifier — so the default in-cluster path is unchanged.
	inClusterURL    string
	inClusterClient *http.Client

	// handler config.
	maxBody int64
}

// newBuilder returns a builder seeded with the handler defaults, ready for
// options to mutate.
func newBuilder() *builder {
	return &builder{maxBody: defaultMaxBodyBytes}
}

// WithMaxBodyBytes overrides the limit applied when reading the request body. A
// non-positive value is ignored and the default is retained.
func WithMaxBodyBytes(n int64) Option {
	return func(b *builder) {
		if n > 0 {
			b.maxBody = n
		}
	}
}

// WithRemoteConfig selects out-of-cluster verification against cfg.Issuer and
// cfg.Audience (both required; the token must be issued by Issuer and carry
// Audience). Its PRESENCE selects remote mode: passing it commits the handler to
// remote, and an incomplete cfg (missing Issuer or Audience) is a construction
// error from WithTokenVerification, never a silent fallback to in-cluster. Issuer
// and Audience travel together in one struct so the jointly-required invariant is
// preserved structurally (an empty audience must never mean "accept any
// audience"). cfg.HTTPClient, when non-nil, is used for OIDC discovery and key
// refreshes (for example to trust a private CA).
func WithRemoteConfig(cfg RemoteConfig) Option {
	return func(b *builder) {
		b.remoteSet = true
		b.issuer = cfg.Issuer
		b.audience = cfg.Audience
		b.httpClient = cfg.HTTPClient
	}
}

// withInClusterEndpoint redirects the in-cluster verifier at apiServerURL and
// client instead of oidc.InClusterAPIServerURL and the projected service-account
// CA client. It stays IN-CLUSTER mode (it deliberately does NOT set remoteSet). It
// is unexported and exposed only through export_test.go, so the real
// WithTokenVerification entrypoint — not just the NewHandlerForTest seam — can be
// driven offline against a throwaway apiserver, closing the in-cluster e2e gap.
func withInClusterEndpoint(apiServerURL string, client *http.Client) Option {
	return func(b *builder) {
		b.inClusterURL = apiServerURL
		b.inClusterClient = client
	}
}

// AudienceResolver derives the expected token audience from the first admission
// request. It is used internally by the in-cluster handler, whose audience is not
// known at startup (see [InClusterAudienceResolver]). A non-nil error denies the
// request fail-closed and leaves the verifier not-ready.
type AudienceResolver func(r *http.Request) (audience string, err error)

// Handler is the http.Handler returned by WithTokenVerification. Beyond serving
// admission requests it exposes [Handler.HealthCheck] for readiness wiring.
type Handler struct {
	verifier *verify.Verifier
	next     AdmissionHandler
	maxBody  int64

	// resolve, when non-nil, derives the expected audience from the first request
	// and binds it (the in-cluster deferred-audience path); nil means the
	// verifier's audience is already bound.
	resolve AudienceResolver
	// bound is the lock-free steady-state fast path: once the audience has been
	// bound successfully it flips false→true and every later request observes it
	// without taking bindMu. bindMu serializes bind attempts while still unbound.
	//
	// The bind is committed (bound set true) ONLY on a successful resolve+bind; a
	// failed attempt leaves bound false so the NEXT request retries under the lock.
	// This matters because the resolver's inputs are not fixed for the pod's
	// lifetime — the request Host is per-request and attacker-influenceable — so
	// caching a failure would let a single poisoned or misdirected request wedge
	// the handler deny-all forever (a fail-closed crash-loop). A genuinely
	// permanent misconfiguration (for example a missing Service port env var)
	// still fails identically on every retry and surfaces via HealthCheck (an
	// unhealthy pod is restarted).
	bound  atomic.Bool
	bindMu sync.Mutex
}

// RemoteConfig configures an out-of-cluster verifier for WithTokenVerification.
// Issuer and Audience are required; the token must be issued by Issuer and
// carry Audience. HTTPClient, when non-nil, is used for OIDC discovery and key
// refreshes (for example to trust a private CA); a nil HTTPClient uses the
// default transport.
type RemoteConfig struct {
	Issuer     string
	Audience   string
	HTTPClient *http.Client
}

// WithTokenVerification builds a verifier and returns an http.Handler that checks
// the bearer token before reading the body, decodes the AdmissionReview once,
// verifies the token, and — only on success — invokes next with the decoded
// AdmissionRequest and writes next's AdmissionResponse.
//
// The verifier is selected by option presence — the safe path is the zero-config
// default:
//   - no remote option builds the in-cluster verifier (zero-config, secure by
//     default; see oidc.NewLocalKeySetVerifier): the issuer is discovered over the
//     in-cluster network and the audience is derived from the first request via
//     [InClusterAudienceResolver]. Until the audience binds, requests are denied
//     fail-closed and [Handler.HealthCheck] reports not-ready.
//   - WithRemoteConfig builds an out-of-cluster verifier bound to the config's
//     Issuer and Audience (see oidc.NewRemoteVerifier). Its presence commits the
//     handler to remote mode; a present-but-incomplete config (missing issuer or
//     audience) is a construction error, never a silent fallback to in-cluster and
//     never a verifier built with an empty, audience-skipping audience.
//
// A failure before decoding (missing token, undecodable or over-limit body) is a
// bare 401. A verification failure after decoding is a denied AdmissionResponse
// (HTTP 200, Allowed:false, Result.Code 401): an explicit deny that, unlike a
// non-2xx status, failurePolicy: Ignore cannot turn into an allow.
//
// next is required and performs the real admission logic; a nil next panics so a
// misconfiguration fails fast instead of becoming a no-op auth gate. ctx governs
// the verifier's discovery and background key refreshes, so pass a
// process-lifetime context. It returns an error if the verifier cannot be built.
func WithTokenVerification(ctx context.Context, next AdmissionHandler, opts ...Option) (*Handler, error) {
	if next == nil {
		panic("admissionhttp: next AdmissionHandler must not be nil")
	}

	b := newBuilder()
	for _, opt := range opts {
		opt(b)
	}

	v, resolve, err := b.buildVerifier(ctx)
	if err != nil {
		return nil, err
	}
	return newHandler(v, next, resolve, b.maxBody), nil
}

// buildVerifier constructs the verifier the Handler will use and the audience
// resolver it needs (nil for remote, the in-cluster resolver otherwise). It is
// the SINGLE build point where the fail-closed mode contract is enforced.
func (b *builder) buildVerifier(ctx context.Context) (*verify.Verifier, AudienceResolver, error) {
	if b.remoteSet {
		// Remote mode is committed: issuer AND audience are jointly required and
		// validated HERE, before constructing the verifier. A partial config is a
		// hard error — never a silent fallback to in-cluster, and never a verifier
		// built with an empty (audience-skipping) audience. The message carries no
		// claim values.
		if b.issuer == "" || b.audience == "" {
			return nil, nil, errors.New("admissionhttp: remote verification requires both an issuer and an audience")
		}
		v, err := oidc.NewRemoteVerifier(ctx, b.issuer, b.audience, oidc.WithHTTPClient(b.httpClient))
		if err != nil {
			return nil, nil, err
		}
		// resolve stays nil: the audience is fixed at construction.
		return v, nil, nil
	}

	// No remote option: in-cluster (zero-config, secure by default). The verifier
	// discovers its issuer over the in-cluster network (trusting the projected
	// service account CA) and derives its audience from the first request via
	// [InClusterAudienceResolver]. A construction error (missing SA CA bundle,
	// failed discovery) propagates unchanged — a not-securely-buildable in-cluster
	// handler is never returned. url/client default to the real in-cluster address
	// and SA-CA client; the unexported withInClusterEndpoint override only
	// redirects them for offline tests.
	url := b.inClusterURL
	if url == "" {
		url = oidc.InClusterAPIServerURL
	}
	v, err := oidc.NewLocalKeySetVerifier(ctx, url, oidc.WithHTTPClient(b.inClusterClient))
	if err != nil {
		return nil, nil, err
	}
	return v, InClusterAudienceResolver(), nil
}

// newHandler assembles a Handler around an already-built verifier. resolve is nil
// for the out-of-cluster path (audience fixed at construction) and non-nil for
// the in-cluster path (audience derived from the first request). maxBody is the
// resolved request-body read limit. It is the shared seam the exported
// constructor and tests build through.
func newHandler(v *verify.Verifier, next AdmissionHandler, resolve AudienceResolver, maxBody int64) *Handler {
	return &Handler{
		verifier: v,
		next:     next,
		maxBody:  maxBody,
		resolve:  resolve,
	}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Check the bearer token first: it is far cheaper than reading and decoding
	// the body, and rejects an unauthenticated caller before any body work. No
	// UID exists yet, so a pre-decode failure is a bare 401.
	token, ok := BearerToken(r)
	if !ok {
		denyPreDecode(ctx, w, reasonNoBearerToken)
		return
	}

	// Decode the AdmissionReview once via the scheme's serializer.
	review, reason, ok := h.decodeReview(r)
	if !ok {
		denyPreDecode(ctx, w, reason)
		return
	}
	req := review.Request

	// Bind the expected audience from the first request if the verifier needs it
	// (in-cluster deferred path). A failure is fail-closed: deny with an explicit
	// denied response and stay not-ready, so a scheduling race surfaces as a
	// restart rather than a silent accept.
	if err := h.ensureAudience(r); err != nil {
		klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied",
			"reason", "expected audience could not be derived", "detail", err.Error())
		writeResponse(w, req, deniedResponse())
		return
	}

	// decodeReview guarantees req is non-nil, so the resource API group is
	// well-defined here.
	if err := h.verifier.Verify(ctx, token, req.Resource.Group); err != nil {
		// Verify has already logged the specific reason. Fail closed with an
		// explicit denied AdmissionResponse (HTTP 200, Allowed:false).
		writeResponse(w, req, deniedResponse())
		return
	}

	// Defense in depth: the token authorized req.Resource.Group, so require the
	// admitted object's own group to match — an authorized-group envelope must not
	// carry a different-group payload.
	if err := checkObjectGroup(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied",
			"reason", "admitted object group does not match the resource group", "detail", err.Error())
		writeResponse(w, req, deniedResponse())
		return
	}

	// next is required, so it is safe to call. The verified outcome is logged at
	// high verbosity for operators; the token is never logged.
	klog.FromContext(ctx).V(4).Info("Admission webhook token verified")
	writeResponse(w, req, h.next(ctx, req))
}

// denyPreDecode answers a failure that occurred before a review was decoded (so
// there is no UID to echo): it logs the non-sensitive reason and writes a bare
// generic 401. reason never contains claim values.
func denyPreDecode(ctx context.Context, w http.ResponseWriter, reason string) {
	klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied", "reason", reason)
	http.Error(w, genericDenyMessage, http.StatusUnauthorized)
}

// deniedResponse builds the fail-closed denial: Allowed:false with a uniform
// generic message and a 401 Result.Code and no claim values (anti-enumeration).
// Wrapped in an AdmissionReview it is an EXPLICIT deny — which, unlike a non-2xx
// HTTP status, failurePolicy: Ignore cannot override.
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

// writeResponse echoes the request UID onto resp, wraps it in a v1
// AdmissionReview, and writes it as HTTP 200 JSON — the admission response wire
// format. A nil resp (a misbehaving downstream) fails closed as a denial.
func writeResponse(w http.ResponseWriter, req *admissionv1.AdmissionRequest, resp *admissionv1.AdmissionResponse) {
	if resp == nil {
		resp = deniedResponse()
	}
	resp.UID = req.UID
	out := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			APIVersion: admissionv1.SchemeGroupVersion.String(),
			Kind:       "AdmissionReview",
		},
		Response: resp,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// decodeReview reads the body once (bounded by maxBody) and decodes it into an
// AdmissionReview via the scheme's serializer. ok is false, with a non-sensitive
// log reason, for an over-limit body or anything that is not a decodable
// AdmissionReview carrying a Request — so the handler fails closed.
func (h *Handler) decodeReview(r *http.Request) (review *admissionv1.AdmissionReview, reason string, ok bool) {
	if r.Body == nil {
		return nil, reasonUndecodableBody, false
	}
	// Read one byte past the limit so an over-limit body is rejected rather than
	// silently decoding truncated bytes.
	buf, err := io.ReadAll(io.LimitReader(r.Body, h.maxBody+1))
	_ = r.Body.Close()
	if err != nil {
		return nil, reasonUndecodableBody, false
	}
	if int64(len(buf)) > h.maxBody {
		return nil, reasonBodyTooLarge, false
	}
	obj, _, err := codecs.UniversalDeserializer().Decode(buf, nil, &admissionv1.AdmissionReview{})
	if err != nil {
		return nil, reasonUndecodableBody, false
	}
	ar, ok := obj.(*admissionv1.AdmissionReview)
	if !ok || ar.Request == nil {
		return nil, reasonUndecodableBody, false
	}
	return ar, "", true
}

// checkObjectGroup enforces that the admitted object's own API group matches the
// request's Resource.Group (the group the token was authorized for), so an
// authorized-group envelope cannot carry a different-group payload.
//
// It checks BOTH req.Object (the incoming object on CREATE/UPDATE) AND
// req.OldObject (the prior object on UPDATE/DELETE) rather than only the first
// non-empty one: on an UPDATE both are populated, and checking only one would let
// a mismatched group slip through in the unchecked slot. An empty slot is skipped
// (for example CONNECT populates neither), a request with no embedded object at
// all passes, and an undecodable or mismatched object in either slot fails
// closed.
func checkObjectGroup(req *admissionv1.AdmissionRequest) error {
	for _, obj := range []struct {
		name string
		raw  []byte
	}{
		{"object", req.Object.Raw},
		{"oldObject", req.OldObject.Raw},
	} {
		if len(obj.raw) == 0 {
			continue
		}
		var tm metav1.TypeMeta
		if err := json.Unmarshal(obj.raw, &tm); err != nil {
			return fmt.Errorf("%s is undecodable: %w", obj.name, err)
		}
		gv, err := schema.ParseGroupVersion(tm.APIVersion)
		if err != nil {
			return fmt.Errorf("%s apiVersion %q is invalid: %w", obj.name, tm.APIVersion, err)
		}
		if gv.Group != req.Resource.Group {
			return fmt.Errorf("%s group %q != resource group %q", obj.name, gv.Group, req.Resource.Group)
		}
	}
	return nil
}

// VerifyAdmissionRequest verifies token against the KEP-6060 contract for an
// already-decoded AdmissionRequest — the entry point for consumers (for example
// controller-runtime) that decode the review themselves.
//
// It returns nil on success or a generic error satisfying
// errors.Is(err, verify.ErrVerificationFailed) on any failure (including a nil
// request). The reason is logged; do not branch on the error.
func VerifyAdmissionRequest(ctx context.Context, v *verify.Verifier, req *admissionv1.AdmissionRequest, token string) error {
	if req == nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", reasonUndecodableBody)
		return verify.ErrVerificationFailed
	}
	if err := v.Verify(ctx, token, req.Resource.Group); err != nil {
		return err
	}
	// Defense in depth: the admitted object's own group must match the authorized
	// resource group.
	if err := checkObjectGroup(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied",
			"reason", "admitted object group does not match the resource group", "detail", err.Error())
		return verify.ErrVerificationFailed
	}
	return nil
}

// BearerToken extracts the token from an "Authorization: Bearer <token>" header,
// mirroring k8s.io/apiserver/pkg/authentication/request/bearertoken so this
// adapter does not drift from the apiserver's parsing. ok is false for a missing
// header, a non-Bearer scheme, or an empty token. It is exported so a caller that
// decodes the review itself can obtain the token for [VerifyAdmissionRequest].
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

// ensureAudience binds the expected audience from the first request when the
// handler was given a resolver (the in-cluster deferred path). Once the audience
// is bound it is a lock-free read; while still unbound it serializes bind
// attempts under bindMu and commits ONLY on a successful resolve+bind. A failed
// attempt is NOT cached: the next request retries (see the bound/bindMu field
// comment), so a transient failure or a poisoned first request cannot
// permanently wedge the handler, while a permanent misconfiguration keeps
// failing every attempt and surfaces through HealthCheck.
//
// With no resolver (the out-of-cluster path, audience already bound at
// construction) it does not bind, but it still verifies the backing verifier is
// ready via HealthCheck and fails closed if it is not, so a not-ready verifier
// can never silently serve.
func (h *Handler) ensureAudience(r *http.Request) error {
	if h.resolve == nil {
		if err := h.verifier.HealthCheck(); err != nil {
			return fmt.Errorf("verifier is not ready: %w", err)
		}
		return nil
	}
	// Fast path: once bound, every request observes it without taking the lock.
	if h.bound.Load() {
		return nil
	}
	// Slow path: still unbound. Serialize bind attempts so concurrent first
	// requests do not resolve/bind in parallel.
	h.bindMu.Lock()
	defer h.bindMu.Unlock()
	// Re-check under the lock: another goroutine may have bound while we waited.
	if h.bound.Load() {
		return nil
	}
	audience, err := h.resolve(r)
	if err != nil {
		// Fail closed WITHOUT committing: bound stays false so a later request
		// retries rather than caching this (possibly attacker-induced) failure.
		return err
	}
	if err := h.verifier.BindAudience(audience); err != nil {
		return err
	}
	// TODO(kep-6060, possible GA hardening): discuss committing the bind only
	// after a token in THIS same request verifies against the resolved audience,
	// so trust-critical state is gated on a valid token rather than a merely
	// resolvable (and potentially unauthenticated) request. Deferred for Alpha;
	// the <NAME>_SERVICE_PORT env var is the interim trust anchor. This may not
	// be the right direction, but it should be raised for discussion.
	// Commit only on success: from here the fast path serves lock-free.
	h.bound.Store(true)
	return nil
}

// HealthCheck reports whether the handler is ready to verify tokens by returning
// the backing verifier's readiness. For an in-cluster deferred verifier it is
// non-nil until the audience has been derived from a request. Wire it into a
// controller-runtime health/readiness check so a webhook that can never derive
// its audience (for example a scheduling race that never yields the Service env
// var) is restarted rather than silently denying every request.
func (h *Handler) HealthCheck() error {
	return h.verifier.HealthCheck()
}

// InClusterAudienceResolver returns an [AudienceResolver] that derives the
// expected token audience for an in-cluster, Service-backed admission webhook
// from the first request, with no static configuration:
//
//   - host ← request.Host (the DNS name the apiserver dialed, e.g.
//     <name>.<ns>.svc); its first two labels are the Service name and namespace,
//   - port ← the kubelet-injected "<NAME>_SERVICE_PORT" env var for that Service
//     (see k8s.io/kubernetes/pkg/kubelet/envvars), which also acts as a trust
//     anchor: only a real Service in this pod's namespace has one, so a spoofed
//     Host is rejected rather than trusted,
//   - path ← request.URL.Path.
//
// The result mirrors kube-apiserver's validateWebhookAudience. It fails (denying
// the request) if any component is missing; because the process environment is
// fixed, a missing port env var keeps failing until the pod is rescheduled,
// surfaced through [Handler.HealthCheck].
func InClusterAudienceResolver() AudienceResolver {
	return func(r *http.Request) (string, error) {
		host := r.Host
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		if host == "" {
			return "", errors.New("request Host is empty")
		}
		labels := strings.Split(host, ".")
		if len(labels) < 2 || labels[0] == "" || labels[1] == "" {
			return "", fmt.Errorf("request Host %q is not a <name>.<namespace>.svc service name", r.Host)
		}
		name, namespace := labels[0], labels[1]
		portStr, ok := os.LookupEnv(serviceEnvPrefix(name) + "_SERVICE_PORT")
		if !ok || portStr == "" {
			return "", fmt.Errorf("no service port env var for %q: not a Service in this namespace, or a scheduling race", name)
		}
		port, err := strconv.Atoi(portStr)
		if err != nil {
			return "", fmt.Errorf("service port env var for %q is not numeric: %w", name, err)
		}
		return verify.AudienceForService(name, namespace, int32(port), r.URL.Path), nil
	}
}

// serviceEnvPrefix mirrors the kubelet's makeEnvVariableName
// (k8s.io/kubernetes/pkg/kubelet/envvars): uppercase, dashes to underscores.
func serviceEnvPrefix(name string) string {
	return strings.ToUpper(strings.ReplaceAll(name, "-", "_"))
}
