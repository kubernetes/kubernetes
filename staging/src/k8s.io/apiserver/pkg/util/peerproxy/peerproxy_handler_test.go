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

package peerproxy

import (
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"net/http/httptest"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/reconcilers"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storageversion"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/peerproxy/metrics"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/transport"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const (
	requestTimeout = 30 * time.Second
	localServerId  = "local-apiserver"
	remoteServerId = "remote-apiserver"
)

type FakeSVMapData struct {
	gvr      schema.GroupVersionResource
	serverId string
}

type reconciler struct {
	do       bool
	publicIP string
	serverId string
}

func TestPeerProxy(t *testing.T) {
	testCases := []struct {
		desc                 string
		svdata               FakeSVMapData
		informerFinishedSync bool
		requestPath          string
		peerproxiedHeader    string
		expectedStatus       int
		metrics              []string
		want                 string
		reconcilerConfig     reconciler
	}{
		{
			desc:           "allow non resource requests",
			requestPath:    "/foo/bar/baz",
			expectedStatus: http.StatusOK,
		},
		{
			desc:              "allow if already proxied once",
			requestPath:       "/api/bar/baz",
			expectedStatus:    http.StatusOK,
			peerproxiedHeader: "true",
		},
		{
			desc:                 "allow if unsynced informers",
			requestPath:          "/api/bar/baz",
			expectedStatus:       http.StatusOK,
			informerFinishedSync: false,
		},
		{
			desc:                 "allow if no storage version found",
			requestPath:          "/api/bar/baz",
			expectedStatus:       http.StatusOK,
			informerFinishedSync: true,
		},
		{
			// since if no server id is found, we pass request to next handler
			//, and our last handler in local chain is an http ok handler
			desc:                 "200 if no serverid found",
			requestPath:          "/api/bar/baz",
			expectedStatus:       http.StatusOK,
			informerFinishedSync: true,
			svdata: FakeSVMapData{
				gvr: schema.GroupVersionResource{
					Group:    "core",
					Version:  "bar",
					Resource: "baz"},
				serverId: ""},
		},
		{
			desc:                 "503 if no endpoint fetched from lease",
			requestPath:          "/api/foo/bar",
			expectedStatus:       http.StatusServiceUnavailable,
			informerFinishedSync: true,
			svdata: FakeSVMapData{
				gvr: schema.GroupVersionResource{
					Group:    "core",
					Version:  "foo",
					Resource: "bar"},
				serverId: remoteServerId},
		},
		{
			desc:                 "200 if locally serviceable",
			requestPath:          "/api/foo/bar",
			expectedStatus:       http.StatusOK,
			informerFinishedSync: true,
			svdata: FakeSVMapData{
				gvr: schema.GroupVersionResource{
					Group:    "core",
					Version:  "foo",
					Resource: "bar"},
				serverId: localServerId},
		},
		{
			desc:                 "503 unreachable peer bind address",
			requestPath:          "/api/foo/bar",
			expectedStatus:       http.StatusServiceUnavailable,
			informerFinishedSync: true,
			svdata: FakeSVMapData{
				gvr: schema.GroupVersionResource{
					Group:    "core",
					Version:  "foo",
					Resource: "bar"},
				serverId: remoteServerId},
			reconcilerConfig: reconciler{
				do:       true,
				publicIP: "1.2.3.4",
				serverId: remoteServerId,
			},
			metrics: []string{
				"apiserver_rerouted_request_total",
			},
			want: `
			# HELP apiserver_rerouted_request_total [ALPHA] Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it
			# TYPE apiserver_rerouted_request_total counter
			apiserver_rerouted_request_total{code="503"} 1
			`,
		},
		{
			desc:                 "503 unreachable peer public address",
			requestPath:          "/api/foo/bar",
			expectedStatus:       http.StatusServiceUnavailable,
			informerFinishedSync: true,
			svdata: FakeSVMapData{
				gvr: schema.GroupVersionResource{
					Group:    "core",
					Version:  "foo",
					Resource: "bar"},
				serverId: remoteServerId},
			reconcilerConfig: reconciler{
				do:       true,
				publicIP: "1.2.3.4",
				serverId: remoteServerId,
			},
			metrics: []string{
				"apiserver_rerouted_request_total",
			},
			want: `
			# HELP apiserver_rerouted_request_total [ALPHA] Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it
			# TYPE apiserver_rerouted_request_total counter
			apiserver_rerouted_request_total{code="503"} 2
			`,
		},
	}

	metrics.Register()
	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			lastHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte("OK"))
			})
			reconciler := newFakePeerEndpointReconciler(t)
			handler := newHandlerChain(t, lastHandler, reconciler, tt.informerFinishedSync, tt.svdata)
			server, requestGetter := createHTTP2ServerWithClient(handler, requestTimeout*2)
			defer server.Close()

			if tt.reconcilerConfig.do {
				// need to enable feature flags first
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)

				reconciler.UpdateLease(tt.reconcilerConfig.serverId,
					tt.reconcilerConfig.publicIP,
					[]corev1.EndpointPort{{Name: "foo",
						Port: 8080, Protocol: "TCP"}})
			}

			req, err := http.NewRequest(http.MethodGet, server.URL+tt.requestPath, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			req.Header.Set(PeerProxiedHeader, tt.peerproxiedHeader)

			resp, _ := requestGetter(req)

			// compare response
			assert.Equal(t, tt.expectedStatus, resp.StatusCode)

			// compare metric
			if tt.want != "" {
				if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
					t.Fatal(err)
				}
			}
		})
	}

}

func newFakePeerEndpointReconciler(t *testing.T) reconcilers.PeerEndpointLeaseReconciler {
	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	t.Cleanup(func() { server.Terminate(t) })
	scheme := runtime.NewScheme()
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	//utilruntime.Must(core.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(scheme.SetVersionPriority(corev1.SchemeGroupVersion))
	codecs := serializer.NewCodecFactory(scheme)
	sc.Codec = apitesting.TestStorageCodec(codecs, corev1.SchemeGroupVersion)
	config := *sc.ForResource(schema.GroupResource{Resource: "endpoints"})
	baseKey := "/" + uuid.New().String() + "/peer-testleases/"
	leaseTime := 1 * time.Minute
	reconciler, err := reconcilers.NewPeerEndpointLeaseReconciler(&config, baseKey, leaseTime)
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	return reconciler
}

func newHandlerChain(t *testing.T, handler http.Handler, reconciler reconcilers.PeerEndpointLeaseReconciler, informerFinishedSync bool, svdata FakeSVMapData) http.Handler {
	// Add peerproxy handler
	s := serializer.NewCodecFactory(runtime.NewScheme()).WithoutConversion()
	peerProxyHandler, err := newFakePeerProxyHandler(informerFinishedSync, reconciler, svdata, localServerId, s)
	if err != nil {
		t.Fatalf("Error creating peer proxy handler: %v", err)
	}
	peerProxyHandler.finishedSync.Store(informerFinishedSync)
	handler = peerProxyHandler.WrapHandler(handler)

	// Add user info
	handler = withFakeUser(handler)

	// Add requestInfo handler
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	handler = apifilters.WithRequestInfo(handler, requestInfoFactory)
	return handler
}

func newFakePeerProxyHandler(informerFinishedSync bool, reconciler reconcilers.PeerEndpointLeaseReconciler, svdata FakeSVMapData, id string, s runtime.NegotiatedSerializer) (*peerProxyHandler, error) {
	clientset := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	clientConfig := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure: false,
		}}
	proxyRoundTripper, err := transport.New(clientConfig)
	if err != nil {
		return nil, err
	}
	ppI := NewPeerProxyHandler(informerFactory, storageversion.NewDefaultManager(), proxyRoundTripper, id, reconciler, s)
	if testDataExists(svdata.gvr) {
		ppI.addToStorageVersionMap(svdata.gvr, svdata.serverId)
	}
	return ppI, nil
}

func (h *peerProxyHandler) addToStorageVersionMap(gvr schema.GroupVersionResource, serverId string) {
	apiserversi, _ := h.svMap.LoadOrStore(gvr, &sync.Map{})
	apiservers := apiserversi.(*sync.Map)
	if serverId != "" {
		apiservers.Store(serverId, true)
	}
}

func testDataExists(gvr schema.GroupVersionResource) bool {
	return gvr.Group != "" && gvr.Version != "" && gvr.Resource != ""
}

func withFakeUser(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(apirequest.WithUser(r.Context(), &user.DefaultInfo{
			Groups: r.Header["Groups"],
		}))
		handler.ServeHTTP(w, r)
	})
}

// returns a started http2 server, with a client function to send request to the server.
func createHTTP2ServerWithClient(handler http.Handler, clientTimeout time.Duration) (*httptest.Server, func(req *http.Request) (*http.Response, error)) {
	server := httptest.NewUnstartedServer(handler)
	server.EnableHTTP2 = true
	server.StartTLS()
	cli := server.Client()
	cli.Timeout = clientTimeout
	return server, func(req *http.Request) (*http.Response, error) {
		return cli.Do(req)
	}
}
