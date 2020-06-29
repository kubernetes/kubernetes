/*
Copyright 2016 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	endpointmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/egressselector"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
	apiregistrationv1api "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	apiregistrationv1apihelper "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper"
)

const (
	aggregatorComponent string = "aggregator"

	aggregatedDiscoveryTimeout = 5 * time.Second
)

// proxyHandler provides a http.Handler which will proxy traffic to locations
// specified by items implementing Redirector.
type proxyHandler struct {
	// localDelegate is used to satisfy local APIServices
	localDelegate http.Handler

	// proxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	proxyClientCert []byte
	proxyClientKey  []byte
	proxyTransport  *http.Transport

	// Endpoints based routing to map from cluster IP to routable IP
	serviceResolver ServiceResolver

	handlingInfo atomic.Value

	// egressSelector selects the proper egress dialer to communicate with the custom apiserver
	// overwrites proxyTransport dialer if not nil
	egressSelector *egressselector.EgressSelector
}

type proxyHandlingInfo struct {
	// local indicates that this APIService is locally satisfied
	local bool

	// name is the name of the APIService
	name string
	// restConfig holds the information for building a roundtripper
	restConfig *restclient.Config
	// transportBuildingError is an error produced while building the transport.  If this
	// is non-nil, it will be reported to clients.
	transportBuildingError error
	// proxyRoundTripper is the re-useable portion of the transport.  It does not vary with any request.
	proxyRoundTripper http.RoundTripper
	// serviceName is the name of the service this handler proxies to
	serviceName string
	// namespace is the namespace the service lives in
	serviceNamespace string
	// serviceAvailable indicates this APIService is available or not
	serviceAvailable bool
	// servicePort is the port of the service this handler proxies to
	servicePort int32
}

func proxyError(w http.ResponseWriter, req *http.Request, error string, code int) {
	http.Error(w, error, code)

	ctx := req.Context()
	info, ok := genericapirequest.RequestInfoFrom(ctx)
	if !ok {
		klog.Warning("no RequestInfo found in the context")
		return
	}
	// TODO: record long-running request differently? The long-running check func does not necessarily match the one of the aggregated apiserver
	endpointmetrics.RecordRequestTermination(req, info, aggregatorComponent, code)
}

func (r *proxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	value := r.handlingInfo.Load()
	if value == nil {
		r.localDelegate.ServeHTTP(w, req)
		return
	}
	handlingInfo := value.(proxyHandlingInfo)
	if handlingInfo.local {
		if r.localDelegate == nil {
			http.Error(w, "", http.StatusNotFound)
			return
		}
		r.localDelegate.ServeHTTP(w, req)
		return
	}

	if !handlingInfo.serviceAvailable {
		proxyError(w, req, "service unavailable", http.StatusServiceUnavailable)
		return
	}

	if handlingInfo.transportBuildingError != nil {
		proxyError(w, req, handlingInfo.transportBuildingError.Error(), http.StatusInternalServerError)
		return
	}

	user, ok := genericapirequest.UserFrom(req.Context())
	if !ok {
		proxyError(w, req, "missing user", http.StatusInternalServerError)
		return
	}

	r.serveHTTPWithRetry(w, req, handlingInfo, user)
}

// serveHTTPWithRetry serves the request with a retry mechanism for increasing high availability of the aggregator.
// it takes into account the number of available endpoints, used HTTP method and type of an error (if any)
func (r *proxyHandler) serveHTTPWithRetry(w http.ResponseWriter, req *http.Request, handlingInfo proxyHandlingInfo, user user.Info) {
	w = newResponseWriterInterceptor(w)
	errRsp := newHijackErrorResponder(&responder{w: w}, req)
	retryDecorator, endpointsCount, err := newRetryDecoratorForResolver(w, errRsp, handlingInfo, r.serviceResolver, 3)
	if err != nil {
		proxyError(w, req, "service unavailable", http.StatusServiceUnavailable)
		return
	}

	visitedURLs := []*url.URL{}
	for {
		visitedURL := serveHTTP(w, utilnet.CloneRequest(req), handlingInfo, errRsp, user, r.serviceResolverWrapper(visitedURLs, endpointsCount))
		if visitedURL != nil {
			visitedURLs = append(visitedURLs, visitedURL)
		}

		if !retryDecorator.RetryIfNeeded() {
			break
		}
	}

	// if an error is not retriable and we haven't sent a response to the client return StatusServiceUnavailable
	if w.(responseWriterInterceptor).StatusCode() == 0 && !w.(responseWriterInterceptor).WasHijacked() {
		proxyError(w, req, "service unavailable", http.StatusServiceUnavailable)
	}
}

func serveHTTP(w http.ResponseWriter, req *http.Request, handlingInfo proxyHandlingInfo, errResponder proxy.ErrorResponder, user user.Info, serviceResolverFn func(namespace, name string, port int32) (*url.URL, error)) *url.URL {
	// write a new location based on the existing request pointed at the target service
	location := &url.URL{}
	location.Scheme = "https"
	rloc, err := serviceResolverFn(handlingInfo.serviceNamespace, handlingInfo.serviceName, handlingInfo.servicePort)
	if err != nil {
		klog.Errorf("error resolving %s/%s: %v", handlingInfo.serviceNamespace, handlingInfo.serviceName, err)
		proxyError(w, req, "service unavailable", http.StatusServiceUnavailable)
		return nil
	}
	location.Host = rloc.Host
	location.Path = req.URL.Path
	location.RawQuery = req.URL.Query().Encode()

	newReq, cancelFn := newRequestForProxy(location, req)
	defer cancelFn()

	if handlingInfo.proxyRoundTripper == nil {
		proxyError(w, req, "", http.StatusNotFound)
		return nil
	}

	// we need to wrap the roundtripper in another roundtripper which will apply the front proxy headers
	proxyRoundTripper, upgrade, err := maybeWrapForConnectionUpgrades(handlingInfo.restConfig, handlingInfo.proxyRoundTripper, req)
	if err != nil {
		proxyError(w, req, err.Error(), http.StatusInternalServerError)
		return nil
	}
	proxyRoundTripper = transport.NewAuthProxyRoundTripper(user.GetName(), user.GetGroups(), user.GetExtra(), proxyRoundTripper)

	// if we are upgrading, then the upgrade path tries to use this request with the TLS config we provide, but it does
	// NOT use the roundtripper.  Its a direct call that bypasses the round tripper.  This means that we have to
	// attach the "correct" user headers to the request ahead of time.  After the initial upgrade, we'll be back
	// at the roundtripper flow, so we only have to muck with this request, but we do have to do it.
	if upgrade {
		transport.SetAuthProxyHeaders(newReq, user.GetName(), user.GetGroups(), user.GetExtra())
	}

	handler := proxy.NewUpgradeAwareHandler(location, proxyRoundTripper, true, upgrade, errResponder)
	handler.ServeHTTP(w, newReq)
	return rloc
}

// newRequestForProxy returns a shallow copy of the original request with a context that may include a timeout for discovery requests
func newRequestForProxy(location *url.URL, req *http.Request) (*http.Request, context.CancelFunc) {
	newCtx := req.Context()
	cancelFn := func() {}

	if requestInfo, ok := genericapirequest.RequestInfoFrom(req.Context()); ok {
		// trim leading and trailing slashes. Then "/apis/group/version" requests are for discovery, so if we have exactly three
		// segments that we are going to proxy, we have a discovery request.
		if !requestInfo.IsResourceRequest && len(strings.Split(strings.Trim(requestInfo.Path, "/"), "/")) == 3 {
			// discovery requests are used by kubectl and others to determine which resources a server has.  This is a cheap call that
			// should be fast for every aggregated apiserver.  Latency for aggregation is expected to be low (as for all extensions)
			// so forcing a short timeout here helps responsiveness of all clients.
			newCtx, cancelFn = context.WithTimeout(newCtx, aggregatedDiscoveryTimeout)
		}
	}

	// WithContext creates a shallow clone of the request with the same context.
	newReq := req.WithContext(newCtx)
	newReq.Header = utilnet.CloneHeader(req.Header)
	newReq.URL = location
	newReq.Host = location.Host

	return newReq, cancelFn
}

// maybeWrapForConnectionUpgrades wraps the roundtripper for upgrades.  The bool indicates if it was wrapped
func maybeWrapForConnectionUpgrades(restConfig *restclient.Config, rt http.RoundTripper, req *http.Request) (http.RoundTripper, bool, error) {
	if !httpstream.IsUpgradeRequest(req) {
		return rt, false, nil
	}

	tlsConfig, err := restclient.TLSConfigFor(restConfig)
	if err != nil {
		return nil, true, err
	}
	followRedirects := utilfeature.DefaultFeatureGate.Enabled(genericfeatures.StreamingProxyRedirects)
	requireSameHostRedirects := utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ValidateProxyRedirects)
	upgradeRoundTripper := spdy.NewRoundTripper(tlsConfig, followRedirects, requireSameHostRedirects)
	wrappedRT, err := restclient.HTTPWrappersForConfig(restConfig, upgradeRoundTripper)
	if err != nil {
		return nil, true, err
	}

	return wrappedRT, true, nil
}

// newRetryDecoratorForResolver returns a new retry decorator object. It reads the total number of endpoints for the given service from the resolver and use it as the number of reties.
// In case only a single endpoint will be used it will retry defaultRetryCount.
func newRetryDecoratorForResolver(w http.ResponseWriter, errRsp *retriableHijackErrorResponder, handlingInfo proxyHandlingInfo, serviceResolver ServiceResolver, defaultRetryCount int) (*retryDecorator, int, error) {
	var allEPs int

	if extendedServiceResolver, ok := serviceResolver.(RetryServiceResolver); ok {
		var err error
		allEPs, err = extendedServiceResolver.EndpointCount(handlingInfo.serviceNamespace, handlingInfo.serviceName, handlingInfo.servicePort)
		if err != nil {
			return nil, 0, fmt.Errorf("error resolving endpoints count %s/%s: %v", handlingInfo.serviceNamespace, handlingInfo.serviceName, err)
		}

		if allEPs == 0 {
			return nil, 0, fmt.Errorf("an empty set returned while resolving endpoints count for %s/%s", handlingInfo.serviceNamespace, handlingInfo.serviceName)
		}
	} else {
		// assume a single endpoint if the provided serviceResolver doesn't support the extended interface
		allEPs = 1
	}

	if allEPs == 1 {
		return newRetryDecorator(w.(responseWriterInterceptor), errRsp, true, defaultRetryCount), allEPs, nil
	}

	return newRetryDecorator(w.(responseWriterInterceptor), errRsp, false, allEPs), allEPs, nil
}

// serviceResolverWrapper wraps the serviceResolver for encapsulating possible implementations.
// It is used by serveHTTP method to pick up a backend server for serving a request.
func (r *proxyHandler) serviceResolverWrapper(visitedEPs []*url.URL, totalEndpointsCount int) func(namespace, name string, port int32) (*url.URL, error) {
	serviceResolverWrapper := func(namespace, name string, port int32) (*url.URL, error) {
		// in case of a single host scenario always use the default service resolver to resolve to the same instance
		if totalEndpointsCount == 1 {
			return r.serviceResolver.ResolveEndpoint(namespace, name, port)
		}
		if extendedServiceResolver, ok := r.serviceResolver.(RetryServiceResolver); ok {
			return extendedServiceResolver.ResolveEndpointWithVisited(namespace, name, port, visitedEPs)
		}
		return r.serviceResolver.ResolveEndpoint(namespace, name, port)
	}
	return serviceResolverWrapper
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	w http.ResponseWriter
}

// TODO this should properly handle content type negotiation
// if the caller asked for protobuf and you write JSON bad things happen.
func (r *responder) Object(statusCode int, obj runtime.Object) {
	responsewriters.WriteRawJSON(statusCode, obj, r.w)
}

func (r *responder) Error(_ http.ResponseWriter, _ *http.Request, err error) {
	http.Error(r.w, err.Error(), http.StatusServiceUnavailable)
}

// these methods provide locked access to fields

func (r *proxyHandler) updateAPIService(apiService *apiregistrationv1api.APIService) {
	if apiService.Spec.Service == nil {
		r.handlingInfo.Store(proxyHandlingInfo{local: true})
		return
	}

	newInfo := proxyHandlingInfo{
		name: apiService.Name,
		restConfig: &restclient.Config{
			TLSClientConfig: restclient.TLSClientConfig{
				Insecure:   apiService.Spec.InsecureSkipTLSVerify,
				ServerName: apiService.Spec.Service.Name + "." + apiService.Spec.Service.Namespace + ".svc",
				CertData:   r.proxyClientCert,
				KeyData:    r.proxyClientKey,
				CAData:     apiService.Spec.CABundle,
			},
		},
		serviceName:      apiService.Spec.Service.Name,
		serviceNamespace: apiService.Spec.Service.Namespace,
		servicePort:      *apiService.Spec.Service.Port,
		serviceAvailable: apiregistrationv1apihelper.IsAPIServiceConditionTrue(apiService, apiregistrationv1api.Available),
	}
	if r.egressSelector != nil {
		networkContext := egressselector.Cluster.AsNetworkContext()
		var egressDialer utilnet.DialFunc
		egressDialer, err := r.egressSelector.Lookup(networkContext)
		if err != nil {
			klog.Warning(err.Error())
		} else {
			newInfo.restConfig.Dial = egressDialer
		}
	} else if r.proxyTransport != nil && r.proxyTransport.DialContext != nil {
		newInfo.restConfig.Dial = r.proxyTransport.DialContext
	}
	newInfo.proxyRoundTripper, newInfo.transportBuildingError = restclient.TransportFor(newInfo.restConfig)
	if newInfo.transportBuildingError != nil {
		klog.Warning(newInfo.transportBuildingError.Error())
	}
	r.handlingInfo.Store(newInfo)
}
