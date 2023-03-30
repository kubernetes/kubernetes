/*
Copyright 2023 The Kubernetes Authors.

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

package unknownversionproxy

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"time"

	v1 "k8s.io/api/coordination/v1"

	apiv1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storageversion"
	apiserverproxyutil "k8s.io/apiserver/pkg/util/proxy"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
)

const (
	UvipReroutedHeader = "UVIP-rerouted"
)

// Interface defines how the Unknown Version Proxy filter interacts with the underlying system.
type Interface interface {
	Handle(handler http.Handler, localAPIServerId string, s runtime.NegotiatedSerializer) http.Handler
	WaitForCacheSync(stopCh <-chan struct{}) error
}

// New creates a new instance to implement API server proxy
func New(
	informerFactory kubeinformers.SharedInformerFactory,
	svm storageversion.Manager,
	proxyClientCertFile string,
	proxyClientKeyFile string,
	peerCAFile string,
	peerBindAddress string,

) Interface {
	return NewUVIPHandler(UVIPConfig{
		Name:                "UVIPHandler",
		InformerFactory:     informerFactory,
		Svm:                 svm,
		ProxyClientCertFile: proxyClientCertFile,
		ProxyClientKeyFile:  proxyClientKeyFile,
		PeerCAFile:          peerCAFile,
		PeerBindAddress:     peerBindAddress,
	})
}

// UVIPConfig carries the parameters to an implementation that is testable
type UVIPConfig struct {
	// Name of the handler
	Name string

	// InformerFactory to use in building the handler
	InformerFactory     kubeinformers.SharedInformerFactory
	Svm                 storageversion.Manager
	ProxyClientCertFile string
	ProxyClientKeyFile  string
	PeerCAFile          string
	PeerBindAddress     string
}

func (h *uvipHandler) WaitForCacheSync(stopCh <-chan struct{}) error {

	ok := cache.WaitForNamedCacheSync("unknown-version-proxy", stopCh, h.svi.HasSynced, h.leasei.HasSynced, h.svm.Completed)
	if !ok {
		return fmt.Errorf("error while waiting for initial cache sync")
	}
	klog.V(3).Infof("setting finishedSync to true")
	finishedSync.Store(true)
	return nil
}

func (h *uvipHandler) Handle(handler http.Handler, localAPIServerId string, s runtime.NegotiatedSerializer) http.Handler {
	if h.svLister == nil || h.leaseLister == nil {
		klog.Warningf("API server interoperability proxy support not found, skipping")
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {

		requestInfo, ok := apirequest.RequestInfoFrom(req.Context())

		if !ok {
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}

		// Allow non-resource requests
		if !requestInfo.IsResourceRequest {
			klog.V(3).Infof(fmt.Sprintf("Not a resource request skipping proxying"))
			handler.ServeHTTP(w, req)
			return
		}

		if req.Header.Get(UvipReroutedHeader) == "true" {
			klog.V(3).Infof(fmt.Sprintf("Already rerouted once, skipping proxying"))
			handler.ServeHTTP(w, req)
			return
		}

		if !h.HasFinishedSync() {
			handler.ServeHTTP(w, req)
			return
		}

		gvr := schema.GroupVersionResource{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion, Resource: requestInfo.Resource}
		if requestInfo.APIGroup == "" {
			gvr.Group = "core"
		}
		serviceableByResp, err := h.findServiceableByServers(gvr, localAPIServerId)
		if err != nil {
			// this means that resource is an aggregated API or a CR since it wasn't found in SV informer cache, pass as it is
			klog.Warningf(fmt.Sprintf("no StorageVersion/APIServerID found for the GVR: %v skipping proxying", gvr))
			handler.ServeHTTP(w, req)
			return
		}
		// found the gvr locally, pass handler as it is
		if serviceableByResp.locallyServiceable {
			klog.V(4).Infof("resource can be served locally, skipping proxying")
			handler.ServeHTTP(w, req)
			return
		}

		if len(serviceableByResp.serviceableBy) == 0 {
			// this means that no apiservers were found that could serve the requested resource, pass as it is
			klog.V(3).Infof(fmt.Sprintf("no StorageVersion/APIServerID found for the GVR: %v skipping proxying", gvr))
			handler.ServeHTTP(w, req)
			return
		}

		// randomly select an APIServer
		rand := rand.Intn(len(serviceableByResp.serviceableBy))
		apiserverId := serviceableByResp.serviceableBy[rand]

		gv := schema.GroupVersion{Group: gvr.Group, Version: gvr.Version}

		// fetch APIServerIdentity Lease object for this apiserver
		lease, err := h.leaseLister.Leases(metav1.NamespaceSystem).Get(apiserverId)

		if err != nil {
			klog.ErrorS(err, "uvip: Error getting apiserver lease")
			responsewriters.ErrorNegotiated(apierrors.NewServiceUnavailable(fmt.Sprintf("Error retrieving lease for destination API server for requested resource: %v,", gv)), s, gv, w, req)
			return
		}

		// check if lease is expired, which means that the apiserver that registered this resource has shutdown, serve 503
		if isLeaseExpired(lease) {
			responsewriters.ErrorNegotiated(apierrors.NewServiceUnavailable(fmt.Sprintf("Expired lease for API server for the requested GVR: %v", gv)), s, gv, w, req)
			return
		}

		// finally proxy
		peerIp := lease.Labels[apiv1.LabelPeerBindIp]
		peerPort := lease.Labels[apiv1.LabelPeerBindPort]

		err = proxyRequestToDestinationAPIServer(req, w, peerIp, peerPort, h.proxyClientCertFile, h.proxyClientKeyFile, h.peerCAFile)
		if err != nil {
			klog.ErrorS(err, "error proxying request for the requested GVR")
			responsewriters.ErrorNegotiated(apierrors.NewServiceUnavailable(fmt.Sprintf("Error proxying request for the requested GVR: %v, err: %v", gvr, err)), s, gv, w, req)
		}

	})
}

func proxyRequestToDestinationAPIServer(req *http.Request, w http.ResponseWriter, peerIp string, peerPort string, proxyClientCertFile string,
	proxyClientKeyFile string,
	rootCAFile string) error {
	user, ok := apirequest.UserFrom(req.Context())
	if !ok {
		return fmt.Errorf("failed to get user info from request")
	}

	// write a new location based on the existing request pointed at the target service
	location := &url.URL{}
	location.Scheme = "https"
	location.Host = fmt.Sprintf("%s:%s", peerIp, peerPort)
	location.Path = req.URL.Path
	location.RawQuery = req.URL.Query().Encode()

	newReq, cancelFn := apiserverproxyutil.NewRequestForProxy(location, req)
	newReq.Header.Add(UvipReroutedHeader, "true")
	//newReq, cancelFn := newRequestForProxy(location, req)
	defer cancelFn()

	// create transport
	clientConfig := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure:   false,
			CertFile:   proxyClientCertFile,
			KeyFile:    proxyClientKeyFile,
			CAFile:     rootCAFile,
			ServerName: "kubernetes.default.svc",
		},
	}

	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()
	proxyRoundTripper, transportBuildingError := transport.NewWithContext(ctx.Done(), clientConfig)
	if transportBuildingError != nil {
		klog.Warning(transportBuildingError.Error())
		return transportBuildingError
	}

	proxyRoundTripper = transport.NewAuthProxyRoundTripper(user.GetName(), user.GetGroups(), nil, proxyRoundTripper)

	handler := proxy.NewUpgradeAwareHandler(location, proxyRoundTripper, true, false, &responder{w: w})

	handler.ServeHTTP(w, newReq)
	return nil
}

func isLeaseExpired(lease *v1.Lease) bool {
	currentTime := time.Now()
	// Leases created by the apiserver lease controller should have non-nil renew time
	// and lease duration set. Leases without these fields set are invalid and should
	// be GC'ed.
	return lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime)
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

func (r *responder) Error(w http.ResponseWriter, req *http.Request, err error) {
	klog.Errorf("Error while proxying request to destination apiserver: %v", err)
	http.Error(w, err.Error(), http.StatusServiceUnavailable)
}
