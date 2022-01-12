package openshiftkubeapiserver

import (
	"net/http"
	"strings"

	authenticationv1 "k8s.io/api/authentication/v1"
	genericapiserver "k8s.io/apiserver/pkg/server"
	coreinformers "k8s.io/client-go/informers/core/v1"
	patchfilters "k8s.io/kubernetes/openshift-kube-apiserver/filters"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/apirequestcount"

	authorizationv1 "github.com/openshift/api/authorization/v1"
)

const (
	openShiftConfigManagedNamespaceName = "openshift-config-managed"
	consolePublicConfigMapName          = "console-public"
)

// TODO switch back to taking a kubeapiserver config.  For now make it obviously safe for 3.11
func BuildHandlerChain(oauthMetadataFile string, cmInformer coreinformers.ConfigMapInformer, requestLogger apirequestcount.APIRequestLogger) (func(apiHandler http.Handler, kc *genericapiserver.Config) http.Handler, error) {
	// load the oauthmetadata when we can return an error
	oAuthMetadata := []byte{}
	if len(oauthMetadataFile) > 0 {
		var err error
		oAuthMetadata, err = loadOAuthMetadataFile(oauthMetadataFile)
		if err != nil {
			return nil, err
		}
	}

	return func(apiHandler http.Handler, genericConfig *genericapiserver.Config) http.Handler {
			// well-known comes after the normal handling chain. This shows where to connect for oauth information
			handler := withOAuthInfo(apiHandler, oAuthMetadata)

			// after normal chain, so that user is in context
			handler = patchfilters.WithAPIRequestCountLogging(handler, requestLogger)

			// this is the normal kube handler chain
			handler = genericapiserver.DefaultBuildHandlerChain(handler, genericConfig)

			// these handlers are all before the normal kube chain
			handler = translateLegacyScopeImpersonation(handler)

			// redirects from / and /console to consolePublicURL if you're using a browser
			handler = withConsoleRedirect(handler, cmInformer)

			return handler
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
func withConsoleRedirect(handler http.Handler, cmInformer coreinformers.ConfigMapInformer) http.Handler {
	cmLister := cmInformer.Lister()
	informer := cmInformer.Informer()
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !strings.HasPrefix(req.URL.Path, "/console") {
			// Dispatch to the next handler
			handler.ServeHTTP(w, req)
			return
		}

		consoleUrl := ""
		if informer.HasSynced() {
			consolePublicConfig, err := cmLister.ConfigMaps(openShiftConfigManagedNamespaceName).Get(consolePublicConfigMapName)
			if err == nil {
				consoleUrl = consolePublicConfig.Data["consoleURL"]
			}
		}
		if len(consoleUrl) > 0 {
			http.Redirect(w, req, consoleUrl, http.StatusFound)
			return
		}
		http.Error(w, "redirection failed: console URL not found", http.StatusInternalServerError)
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
