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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericrest "k8s.io/apiserver/pkg/registry/generic/rest"
	"k8s.io/apiserver/pkg/server"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"

	"crypto/rand"
	"github.com/golang/glog"
	apiregistrationapi "k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"math/big"
)

// proxyHandler provides a http.Handler which will proxy traffic to locations
// specified by items implementing Redirector.
type proxyHandler struct {
	contextMapper genericapirequest.RequestContextMapper

	// localDelegate is used to satisfy local APIServices
	localDelegate http.Handler

	// lookup map to allow proxy handlers to make sideways calls to other proxy handlers
	proxyHandlers map[string]*proxyHandler

	// proxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	proxyClientCert []byte
	proxyClientKey  []byte

	handlingInfo atomic.Value
}

type proxyHandlingInfo struct {
	// local indicates that this APIService is locally satisfied
	local bool

	// restConfig holds the information for building a roundtripper
	restConfig *restclient.Config
	// transportBuildingError is an error produced while building the transport.  If this
	// is non-nil, it will be reported to clients.
	transportBuildingError error
	// proxyRoundTripper is the re-useable portion of the transport.  It does not vary with any request.
	proxyRoundTripper http.RoundTripper
	// destinationHost is the hostname of the backing API server
	destinationHost string
	// serviceName is the name of the service this handler proxies to
	serviceName string
	// namespace is the namespace the service lives in
	namespace string
}

func (r *proxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	value := r.handlingInfo.Load()
	if value == nil {
		r.localDelegate.ServeHTTP(w, req)
		return
	}
	handlingInfo := value.(proxyHandlingInfo)
	if handlingInfo.local {
		r.localDelegate.ServeHTTP(w, req)
		return
	}

	if handlingInfo.transportBuildingError != nil {
		http.Error(w, handlingInfo.transportBuildingError.Error(), http.StatusInternalServerError)
		return
	}
	proxyRoundTripper := handlingInfo.proxyRoundTripper
	if proxyRoundTripper == nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}

	ctx, ok := r.contextMapper.Get(req)
	if !ok {
		http.Error(w, "missing context", http.StatusInternalServerError)
		return
	}
	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		http.Error(w, "missing user", http.StatusInternalServerError)
		return
	}

	// write a new location based on the existing request pointed at the target service
	location := &url.URL{}
	location.Scheme = "https"
	// Try to lookup the service and find the pods which implement this service.
	location.Host, ok = r.getPodIpForService(handlingInfo)
	if !ok {
		glog.V(2).Info("Aggregation unable to get pod ip for service.")
		location.Host = handlingInfo.destinationHost
	}
	location.Path = req.URL.Path
	location.RawQuery = req.URL.Query().Encode()

	// make a new request object with the updated location and the body we already have
	newReq, err := http.NewRequest(req.Method, location.String(), req.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	mergeHeader(newReq.Header, req.Header)
	newReq.ContentLength = req.ContentLength
	// Copy the TransferEncoding is for future-proofing. Currently Go only supports "chunked" and
	// it can determine the TransferEncoding based on ContentLength and the Body.
	newReq.TransferEncoding = req.TransferEncoding

	upgrade := false
	// we need to wrap the roundtripper in another roundtripper which will apply the front proxy headers
	proxyRoundTripper, upgrade, err = maybeWrapForConnectionUpgrades(handlingInfo.restConfig, proxyRoundTripper, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	proxyRoundTripper = transport.NewAuthProxyRoundTripper(user.GetName(), user.GetGroups(), user.GetExtra(), proxyRoundTripper)

	// if we are upgrading, then the upgrade path tries to use this request with the TLS config we provide, but it does
	// NOT use the roundtripper.  Its a direct call that bypasses the round tripper.  This means that we have to
	// attach the "correct" user headers to the request ahead of time.  After the initial upgrade, we'll be back
	// at the roundtripper flow, so we only have to muck with this request, but we do have to do it.
	if upgrade {
		transport.SetAuthProxyHeaders(newReq, user.GetName(), user.GetGroups(), user.GetExtra())
	}

	handler := genericrest.NewUpgradeAwareProxyHandler(location, proxyRoundTripper, true, upgrade, &responder{w: w})
	handler.ServeHTTP(w, newReq)
}

// Look up and pick a pod ip for the requested service.
func (r *proxyHandler) getPodIpForService(handlingInfo proxyHandlingInfo) (string, bool) {
	if handlingInfo.local || handlingInfo.serviceName == "endpoints" {
		glog.V(9).Infof("Aggregation refused local pod ip lookup for %s.", handlingInfo.serviceName)
		return "", false
	}
	if handlingInfo.namespace == "" {
		glog.V(2).Info("Aggregation missing namespace")
		return "", false
	}
	if handlingInfo.serviceName == "" {
		glog.V(2).Infof("Aggregation missing service name")
		return "", false
	}
	endpointsProxy, ok := r.proxyHandlers[legacyAPIServiceName]
	if !ok {
		glog.V(2).Infof("Aggregation missing legacy API proxy handler")
		return "", false
	}
	endpointLoc := &url.URL{}
	endpointLoc.Scheme = "https"
	// Setting current host - proxy will "fix" it.
	// endpointLoc.Host = handlingInfo.destinationHost
	endpointLoc.Path = "/api/v1/namespaces/" + handlingInfo.namespace + "/endpoints/" + handlingInfo.serviceName
	endpointReq, err := http.NewRequest("GET", endpointLoc.String(), nil)
	if err != nil {
		glog.V(2).Infof("Aggregation failed to create endpoints request %v.", err)
		return "", false
	}
	endpointReq.Proto = "HTTP/2.0"
	endpointReq.ProtoMajor = 2
	endpointReq.ProtoMinor = 0
	endpointReq.Header.Add("Accept", "application/vnd.kubernetes.protobuf")
	infoFactory := &genericapirequest.RequestInfoFactory{
		APIPrefixes: sets.NewString(strings.Trim(server.APIGroupPrefix, "/"),
			strings.Trim(server.DefaultLegacyAPIPrefix, "/")),
		GrouplessAPIPrefixes: sets.NewString(strings.Trim(server.DefaultLegacyAPIPrefix, "/")),
	}
	info, err := infoFactory.NewRequestInfo(endpointReq)
	if err != nil {
		glog.V(2).Infof("Aggregation failed to create endpoints request info %v.", err)
		return "", false
	}
	endpointCtx := genericapirequest.NewContext()
	endpointCtx = genericapirequest.WithNamespace(endpointCtx, handlingInfo.namespace)
	endpointCtx = genericapirequest.WithRequestInfo(endpointCtx, info)
	endpointReq = endpointReq.WithContext(endpointCtx)

	contextHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		err := r.contextMapper.Update(req, endpointCtx)
		if err != nil {
			glog.V(2).Infof("Aggregation could not update endpoints context, %v.", err)
		}
		endpointsProxy.ServeHTTP(w, req)
	})

	handler := genericapirequest.WithRequestContext(contextHandler, r.contextMapper)
	httpWriter := httptest.NewRecorder()
	handler.ServeHTTP(httpWriter, endpointReq)
	var endpoints v1.Endpoints
	d := api.Codecs.UniversalDeserializer()
	_, _, err = d.Decode(httpWriter.Body.Bytes(), &schema.GroupVersionKind{Kind: "Endpoints", Version: "v1"}, &endpoints)
	if err != nil {
		glog.V(2).Infof("Aggregation failed to decode endpoints %v.", err)
		return "", false
	}
	var ips []string
	for _, subset := range endpoints.Subsets {
		for _, address := range subset.Addresses {
			ips = append(ips, address.IP)
		}
	}
	if len(ips) > 0 {
		index, err := rand.Int(rand.Reader, big.NewInt(int64(len(ips))))
		if err != nil {
			glog.V(2).Infof("Aggregation failed to generate random number %v.", err)
			return ips[0], true
		}
		return ips[index.Int64()], true
	}
	glog.V(2).Infof("Aggregation found no ips for %v", handlingInfo.serviceName)
	return "", false
}

// maybeWrapForConnectionUpgrades wraps the roundtripper for upgrades.  The bool indicates if it was wrapped
func maybeWrapForConnectionUpgrades(restConfig *restclient.Config, rt http.RoundTripper, req *http.Request) (http.RoundTripper, bool, error) {
	connectionHeader := req.Header.Get("Connection")
	if len(connectionHeader) == 0 {
		return rt, false, nil
	}

	tlsConfig, err := restclient.TLSConfigFor(restConfig)
	if err != nil {
		return nil, true, err
	}
	upgradeRoundTripper := spdy.NewRoundTripper(tlsConfig)
	wrappedRT, err := restclient.HTTPWrappersForConfig(restConfig, upgradeRoundTripper)
	if err != nil {
		return nil, true, err
	}

	return wrappedRT, true, nil
}

func mergeHeader(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
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

func (r *responder) Error(err error) {
	http.Error(r.w, err.Error(), http.StatusInternalServerError)
}

// these methods provide locked access to fields

func (r *proxyHandler) updateAPIService(apiService *apiregistrationapi.APIService, destinationHost string) {
	if apiService.Spec.Service == nil {
		r.handlingInfo.Store(proxyHandlingInfo{local: true})
		return
	}

	newInfo := proxyHandlingInfo{
		destinationHost: destinationHost,
		restConfig: &restclient.Config{
			TLSClientConfig: restclient.TLSClientConfig{
				Insecure:   apiService.Spec.InsecureSkipTLSVerify,
				ServerName: apiService.Spec.Service.Name + "." + apiService.Spec.Service.Namespace + ".svc",
				CertData:   r.proxyClientCert,
				KeyData:    r.proxyClientKey,
				CAData:     apiService.Spec.CABundle,
			},
		},
		serviceName: apiService.Spec.Service.Name,
		namespace:   apiService.Spec.Service.Namespace,
	}
	newInfo.proxyRoundTripper, newInfo.transportBuildingError = restclient.TransportFor(newInfo.restConfig)
	r.handlingInfo.Store(newInfo)
}
