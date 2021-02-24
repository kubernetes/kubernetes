package openshiftkubeapiserver

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	"github.com/openshift/library-go/pkg/apiserver/httprequest"
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/util/flowcontrol"
	patchfilters "k8s.io/kubernetes/openshift-kube-apiserver/filters"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest"
)

// TODO switch back to taking a kubeapiserver config.  For now make it obviously safe for 3.11
func BuildHandlerChain(consolePublicURL string, oauthMetadataFile string, genericConfig *genericapiserver.Config) (func(apiHandler http.Handler, kc *genericapiserver.Config) http.Handler, map[string]genericapiserver.PostStartHookFunc, error) {
	// load the oauthmetadata when we can return an error
	oAuthMetadata := []byte{}
	if len(oauthMetadataFile) > 0 {
		var err error
		oAuthMetadata, err = loadOAuthMetadataFile(oauthMetadataFile)
		if err != nil {
			return nil, nil, err
		}
	}
	deprecatedAPIRequestController := deprecatedapirequest.NewController(genericConfig.LoopbackClientConfig)
	return func(apiHandler http.Handler, genericConfig *genericapiserver.Config) http.Handler {
			// well-known comes after the normal handling chain. This shows where to connect for oauth information
			handler := withOAuthInfo(apiHandler, oAuthMetadata)

			// we rate limit watches after building the regular handler chain so we have the context information
			handler = withWatchRateLimit(handler)

			// after normal chain, so that user is in context
			handler = patchfilters.WithDeprecatedApiRequestLogging(handler, deprecatedAPIRequestController)

			// this is the normal kube handler chain
			handler = genericapiserver.DefaultBuildHandlerChain(handler, genericConfig)

			// these handlers are all before the normal kube chain
			handler = translateLegacyScopeImpersonation(handler)

			// redirects from / and /console to consolePublicURL if you're using a browser
			handler = withConsoleRedirect(handler, consolePublicURL)

			return handler
		},
		map[string]genericapiserver.PostStartHookFunc{
			"openshift.io-deprecated-api-requests-filter": deprecatedapirequest.NewPostStartHookFunc(deprecatedAPIRequestController),
		},
		nil
}

// If we know the location of the asset server, redirect to it when / is requested
// and the Accept header supports text/html
func withOAuthInfo(handler http.Handler, oAuthMetadata []byte) http.Handler {
	if len(oAuthMetadata) == 0 {
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != oauthMetadataEndpoint {
			// Dispatch to the next handler
			handler.ServeHTTP(w, req)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(oAuthMetadata)
	})
}

// If we know the location of the asset server, redirect to it when / is requested
// and the Accept header supports text/html
func withConsoleRedirect(handler http.Handler, consolePublicURL string) http.Handler {
	if len(consolePublicURL) == 0 {
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if strings.HasPrefix(req.URL.Path, "/console") ||
			(req.URL.Path == "/" && httprequest.PrefersHTML(req)) {
			http.Redirect(w, req, consolePublicURL, http.StatusFound)
			return
		}
		// Dispatch to the next handler
		handler.ServeHTTP(w, req)
	})
}

// legacyImpersonateUserScopeHeader is the header name older servers were using
// just for scopes, so we need to translate it from clients that may still be
// using it.
const legacyImpersonateUserScopeHeader = "Impersonate-User-Scope"

// translateLegacyScopeImpersonation is a filter that will translates user scope impersonation for openshift into the equivalent kube headers.
func translateLegacyScopeImpersonation(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		for _, scope := range req.Header[legacyImpersonateUserScopeHeader] {
			req.Header[authenticationv1.ImpersonateUserExtraHeaderPrefix+authorizationv1.ScopesKey] =
				append(req.Header[authenticationv1.ImpersonateUserExtraHeaderPrefix+authorizationv1.ScopesKey], scope)
		}

		handler.ServeHTTP(w, req)
	})
}

type authorizerAttributesFunc func(ctx context.Context) (authorizer.Attributes, error)

type watchRateLimit struct {
	delegate http.Handler

	earlyRateLimiter  flowcontrol.RateLimiter
	middleRateLimiter flowcontrol.RateLimiter

	authorizerAttributesFn authorizerAttributesFunc
	clock                  clock.Clock
	earlyEndTime           time.Time
	middleEndTime          time.Time
}

// ServeHTTP rate limits the establishment of watches to keep kube-apiservers from crashing.
// Rate limiting watches effectively creates an upper bound on secret and configmap mounts of 10*QPS because of a
// ten minute watch timeout and kubelets use watches to get the content for the mount.
// We will break the rate limiting into three timespans
// 1. first ten minutes: this is the most restrictive 10000 total mounted secrets and configmaps, 1000 per minute.
//    We're trying to break up the slug of kubelet traffic and
//    we want to be sure that operators can make progress during this time if we need to recover a cluster in
//    a bad state.
// 2. second ten minutes: this is less restrictive  20000 total mounted secrets and configmaps, 2000 per minute.
//    This lets us start to ramp up during a relative steady state.
// 3. no limit.  We have this to handle cases of large clusters with more than 20000 mounted secrets and configmaps.
//    I honestly don't know how common this is, but I don't want to break on it.
// Recall that we observed more than the the 30,000 per minute observed during some disruptive events on a cluster.
// In addition, we special case watches in the platform operator namespaces, cluster scope, and kube-system.
// We have not observed large numbers of these and they are required in order to make progress when trying to correct
// some kinds of cluster failures.
func (h watchRateLimit) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var effectiveRateLimiter flowcontrol.RateLimiter
	now := h.clock.Now()
	switch {
	case now.After(h.middleEndTime):
		// we are past our rate limiting time
		h.delegate.ServeHTTP(w, req)
		return
	case now.After(h.earlyEndTime):
		effectiveRateLimiter = h.middleRateLimiter
	default:
		effectiveRateLimiter = h.earlyRateLimiter
	}

	ctx := req.Context()

	attributes, err := h.authorizerAttributesFn(ctx)
	if err != nil {
		// if we cannot get attributes, don't fail the request
		h.delegate.ServeHTTP(w, req)
		return
	}
	if attributes.GetUser() == nil {
		// if we cannot get user, don't fail the request
		h.delegate.ServeHTTP(w, req)
		return
	}
	for _, group := range attributes.GetUser().GetGroups() {
		if group == user.SystemPrivilegedGroup {
			// system:masters always have the power!
			h.delegate.ServeHTTP(w, req)
			return
		}
	}

	if attributes.GetVerb() != "watch" {
		// only throttle watch establishment
		h.delegate.ServeHTTP(w, req)
		return
	}
	namespace := attributes.GetNamespace()
	switch {
	case len(namespace) == 0:
		// don't rate limit cluster scoped watches because operators that need to make progress may use those
		// if we have to restrict further we can
		h.delegate.ServeHTTP(w, req)
		return

	case strings.HasPrefix("kube-", namespace):
		// don't rate limit kube- because some operators use and we use this for delegated authn
		h.delegate.ServeHTTP(w, req)
		return

	case strings.HasPrefix("openshift-", namespace):
		// don't rate limit openshift- because we need operators to make progress, so we need openshift- mounts
		// to succeed in order to progress
		h.delegate.ServeHTTP(w, req)
		return

	}

	if !effectiveRateLimiter.TryAccept() {
		// add a metric for us to observe
		if requestInfo, ok := request.RequestInfoFrom(ctx); ok {
			metrics.RecordRequestTermination(req, requestInfo, "apiserver-watch", http.StatusTooManyRequests)
		}

		ae := request.AuditEventFrom(ctx)
		audit.LogAnnotation(ae, "apiserver.openshift.io/watch-rate-limit", "rate-limited")
		retryAfter := rand.Intn(15) + 5 // evenly weighted from 5-20 second wait
		// Return a 429 status indicating "Too Many Requests", but make sure its recognizeable
		w.Header().Set("Retry-After", fmt.Sprintf("%d", retryAfter))
		http.Error(w, "Too many WATCH requests, please try again later.", http.StatusTooManyRequests)
		return
	}
	h.delegate.ServeHTTP(w, req)
}

func newWatchRateLimit(handler http.Handler, theClock clock.Clock) watchRateLimit {
	startTime := theClock.Now()

	return watchRateLimit{
		delegate:               handler,
		earlyRateLimiter:       flowcontrol.NewTokenBucketRateLimiterWithClock(16.6, 100, theClock),
		middleRateLimiter:      flowcontrol.NewTokenBucketRateLimiterWithClock(33.3, 100, theClock),
		authorizerAttributesFn: filters.GetAuthorizerAttributes,
		clock:                  theClock,
		earlyEndTime:           startTime.Add(10 * time.Minute),
		middleEndTime:          startTime.Add(20 * time.Minute),
	}
}

func withWatchRateLimit(handler http.Handler) http.Handler {
	return newWatchRateLimit(handler, clock.RealClock{})
}
