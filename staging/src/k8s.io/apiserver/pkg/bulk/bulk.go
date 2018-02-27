/*
Copyright 2017 The Kubernetes Authors.

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

package bulk

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/util/wsstream"
	restclient "k8s.io/client-go/rest"
)

// LocalAPIGroupInfo contains services to serve local group through bulk api.
type LocalAPIGroupInfo struct {

	// GroupVersion is uniqute group identifier
	GroupVersion schema.GroupVersion
	Preferred    bool

	Storage    map[string]rest.Storage
	Mapper     meta.RESTMapper
	Linker     runtime.SelfLinker
	Serializer runtime.NegotiatedSerializer

	Authorizer                 authorizer.Authorizer
	AuthroizationCachingPeriod time.Duration
}

// A ServiceResolver knows how to get an API endpoint URL given a service.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string) (*url.URL, error)
}

// ProxiedAPIGroupInfo contains settings to enable bulk forwarding for desired group.
type ProxiedAPIGroupInfo struct {

	// GroupVersion is uniqute group identifier
	GroupVersion schema.GroupVersion
	Preferred    bool

	// ServiceName is the name of the service this handler proxies to
	ServiceName string

	// ServiceNamespace is the namespace the service lives in
	ServiceNamespace string

	// If present, the Dial method will be used for dialing out to delegate apiservers.
	ProxyTransport *http.Transport

	// ProxyClientCert/Key are the client cert used to identify this proxy.
	// Backing APIServices use this to confirm the proxy's identity
	InsecureSkipTLSVerify bool
	ProxyClientCert       []byte
	ProxyClientKey        []byte
	CABundle              []byte

	// Endpoints based routing to map from cluster IP to routable IP
	ServiceResolver ServiceResolver

	tlsConfig         *tls.Config
	dialContext       func(ctx context.Context, network, addr string) (net.Conn, error)
	transportBuildErr error
}

// APIManagerFactory constructs instances of APIManager
type APIManagerFactory struct {
	Root                 string
	NegotiatedSerializer runtime.NegotiatedSerializer
	ContextMapper        request.RequestContextMapper
	Delegate             *APIManager
	WSTimeout            time.Duration
}

// APIManager installs web handlers for Bulk API.
type APIManager struct {

	// FIXME: support multiple group versions at the same time
	GroupVersion schema.GroupVersion

	// Root is path prefix for installed endpoints
	Root string

	// Available api groups.
	apiGroups map[schema.GroupVersion]*registeredAPIGroup

	// Map api group -> preferred version.
	preferredVersion map[string]string

	negotiatedSerializer runtime.NegotiatedSerializer
	mapper               request.RequestContextMapper
	wsTimeout            time.Duration
}

// registeredAPIGroup is either LocalAPIGroupInfo or ProxiedAPIGroupInfo
type registeredAPIGroup struct {
	Local   *LocalAPIGroupInfo
	Proxied *ProxiedAPIGroupInfo
}

// HTTP handler for bulk-watch, stateless.
type watchHTTPHandler struct {
	*APIManager
}

// New constructs new instance of *APIManager
func (f APIManagerFactory) New() *APIManager {
	glog.V(7).Infof("Construct new bulk.APIManager from %v", f)
	// TODO: merge negotiatedSerializer & ContextMapper from .Delegate

	// Merge API groups from delegate
	preferredVersion := make(map[string]string)
	groups := make(map[schema.GroupVersion]*registeredAPIGroup)
	if f.Delegate != nil {
		for k, v := range f.Delegate.apiGroups {
			glog.V(8).Infof("Reuse %v from delegated bulk.APIManager", k)
			groups[k] = v
		}
		for k, v := range f.Delegate.preferredVersion {
			preferredVersion[k] = v
		}
	}

	return &APIManager{
		// FIXME: Don't hardcode version
		GroupVersion:         schema.GroupVersion{Version: "v1alpha1", Group: bulkapi.GroupName},
		Root:                 f.Root,
		negotiatedSerializer: f.NegotiatedSerializer,
		mapper:               f.ContextMapper,
		wsTimeout:            f.WSTimeout,
		preferredVersion:     preferredVersion,
		apiGroups:            groups,
	}
}

// Install adds the handlers to the given mux.
func (m *APIManager) Install(c *mux.PathRecorderMux) {
	c.HandleFunc(m.Root, watchHTTPHandler{m}.ServeHTTP)
}

// UnregisterGroup unrgisters group from bulk manager.
func (m *APIManager) UnregisterGroup(gv schema.GroupVersion) (found bool) {
	glog.V(7).Infof("Unregister %v at bulk.APIManager", gv)
	if _, found := m.apiGroups[gv]; !found {
		return false
	}
	pversion := m.preferredVersion[gv.Group]
	if pversion == gv.Version {
		delete(m.preferredVersion, gv.Group)
	}
	delete(m.apiGroups, gv)
	return
}

// RegisterLocalGroup enables Bulk API for provided group.
func (m *APIManager) RegisterLocalGroup(agv LocalAPIGroupInfo) error {
	return m.registerAPIGroupCommon(agv.GroupVersion, registeredAPIGroup{Local: &agv}, agv.Preferred)
}

// RegisterProxiedGroup forward Bulk API request to other apiserver.
func (m *APIManager) RegisterProxiedGroup(agv ProxiedAPIGroupInfo) error {
	agv.prepareProxiedAPIGroup()
	return m.registerAPIGroupCommon(agv.GroupVersion, registeredAPIGroup{Proxied: &agv}, agv.Preferred)
}

// RegisterAPIGroup enables Bulk API for provided group.
func (m *APIManager) registerAPIGroupCommon(gv schema.GroupVersion, agv registeredAPIGroup, preferredVersion bool) error {
	if _, found := m.apiGroups[gv]; found {
		return fmt.Errorf("group %v already registered", agv)
	}
	if _, found := m.preferredVersion[gv.Group]; preferredVersion && found {
		return fmt.Errorf("group %v already has preferred version", agv)
	}
	glog.V(7).Infof("Register %v at bulk.APIManager", gv)
	m.apiGroups[gv] = &agv
	if preferredVersion {
		m.preferredVersion[gv.Group] = gv.Version
	}
	return nil
}

// ServeHTTP serves 'bulk-api' endpoint
func (h watchHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m := h.APIManager
	ctx, ok := m.mapper.Get(req)
	if !ok {
		responsewriters.InternalError(w, req, errors.New("no context found for request"))
		return
	}

	// negotiate for the stream serializerInfo - TODO: should we create new fn?
	si, err := negotiation.NegotiateOutputStreamSerializer(req, m.negotiatedSerializer)
	if err != nil {
		responsewriters.ErrorNegotiated(ctx, err, m.negotiatedSerializer, m.GroupVersion, w, req)
		return
	}

	// TODO: use correct mediaType here
	// - Respect 'Accept' header?
	// - Return err NewNotAcceptableError([]string{"**;stream=bulk.k8s.io"})
	mediaType := si.MediaType
	if mediaType != runtime.ContentTypeJSON {
		mediaType += ";stream=bulk.k8s.io"
	}
	w = httplog.Unlogged(w)
	w.Header().Set("Content-Type", mediaType)

	bc := newBulkConnectionImpl(m, req, ctx, si)
	defer bc.Close()

	if wsstream.IsWebSocketRequest(req) {
		err = handleWSBulkRequest(ctx, bc, w, req)
	} else {
		err = handleHTTPBulkRequest(ctx, bc, w, req)
	}
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
}

func (p *ProxiedAPIGroupInfo) prepareProxiedAPIGroup() {
	restConfig := &restclient.Config{
		TLSClientConfig: restclient.TLSClientConfig{
			Insecure:   p.InsecureSkipTLSVerify,
			ServerName: p.ServiceName + "." + p.ServiceNamespace + ".svc",
			CertData:   p.ProxyClientCert,
			KeyData:    p.ProxyClientKey,
			CAData:     p.CABundle,
		}}

	var err error
	p.tlsConfig, err = restclient.TLSConfigFor(restConfig)
	if err != nil {
		p.transportBuildErr = err
		return
	}

	// TODO(anjensan): We should reuse as much as possible from 'aggregator' here
	if p.ProxyTransport != nil {
		p.dialContext = p.ProxyTransport.DialContext
	} else {
		p.dialContext = (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext
	}
}
