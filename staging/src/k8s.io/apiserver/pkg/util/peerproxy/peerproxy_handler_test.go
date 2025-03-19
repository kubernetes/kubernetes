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
	"testing"
	"time"

	"net/http/httptest"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/reconcilers"
	"k8s.io/apiserver/pkg/util/peerproxy/metrics"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const (
	requestTimeout  = 30 * time.Second
	localServerID   = "local-apiserver"
	remoteServerID1 = "remote-apiserver-1"
	remoteServerID2 = "remote-apiserver-2"
)

type server struct {
	publicIP string
	serverID string
}

type reconciler struct {
	do      bool
	servers []server
}

func TestPeerProxy(t *testing.T) {
	testCases := []struct {
		desc                 string
		informerFinishedSync bool
		requestPath          string
		peerproxiedHeader    string
		reconcilerConfig     reconciler
		localCache           map[schema.GroupVersionResource]bool
		peerCache            map[string]map[schema.GroupVersionResource]bool
		wantStatus           int
		wantMetricsData      string
	}{
		{
			desc:        "allow non resource requests",
			requestPath: "/foo/bar/baz",
			wantStatus:  http.StatusOK,
		},
		{
			desc:              "allow if already proxied once",
			requestPath:       "/api/bar/baz",
			peerproxiedHeader: "true",
			wantStatus:        http.StatusOK,
		},
		{
			desc:                 "allow if unsynced informers",
			requestPath:          "/api/bar/baz",
			informerFinishedSync: false,
			wantStatus:           http.StatusOK,
		},
		{
			desc:        "Serve locally if serviceable",
			requestPath: "/api/foo/bar",
			localCache: map[schema.GroupVersionResource]bool{
				{Group: "core", Version: "foo", Resource: "bar"}: true,
			},
			wantStatus: http.StatusOK,
		},
		{
			desc:                 "200 if no appropriate peers found, serve locally",
			requestPath:          "/api/foo/bar",
			informerFinishedSync: true,
			wantStatus:           http.StatusOK,
		},
		{
			desc:                 "503 if no endpoint fetched from lease",
			requestPath:          "/api/foo/bar",
			informerFinishedSync: true,
			peerCache: map[string]map[schema.GroupVersionResource]bool{
				remoteServerID1: {
					{Group: "core", Version: "foo", Resource: "bar"}: true,
				},
			},
			wantStatus: http.StatusServiceUnavailable,
		},
		{
			desc:                 "503 unreachable peer bind address",
			requestPath:          "/api/foo/bar",
			informerFinishedSync: true,
			peerCache: map[string]map[schema.GroupVersionResource]bool{
				remoteServerID1: {
					{Group: "core", Version: "foo", Resource: "bar"}: true,
				},
			},
			reconcilerConfig: reconciler{
				do: true,
				servers: []server{
					{
						publicIP: "1.2.3.4",
						serverID: remoteServerID1,
					},
				},
			},
			wantStatus: http.StatusServiceUnavailable,
			wantMetricsData: `
				# HELP apiserver_rerouted_request_total [ALPHA] Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it
				# TYPE apiserver_rerouted_request_total counter
				apiserver_rerouted_request_total{code="503"} 1
				`,
		},
		{
			desc:                 "503 if one apiserver's endpoint lease wasnt found but another valid (unreachable) apiserver was found",
			requestPath:          "/api/foo/bar",
			informerFinishedSync: true,
			peerCache: map[string]map[schema.GroupVersionResource]bool{
				remoteServerID1: {
					{Group: "core", Version: "foo", Resource: "bar"}: true,
				},
				remoteServerID2: {
					{Group: "core", Version: "foo", Resource: "bar"}: true,
				},
			},
			reconcilerConfig: reconciler{
				do: true,
				servers: []server{
					{
						publicIP: "1.2.3.4",
						serverID: remoteServerID1,
					},
				},
			},
			wantStatus: http.StatusServiceUnavailable,
			wantMetricsData: `
				# HELP apiserver_rerouted_request_total [ALPHA] Total number of requests that were proxied to a peer kube apiserver because the local apiserver was not capable of serving it
				# TYPE apiserver_rerouted_request_total counter
				apiserver_rerouted_request_total{code="503"} 1
				`,
		},
	}

	metrics.Register()
	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer metrics.Reset()
			lastHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte("OK"))
			})
			serverIDs := []string{localServerID}
			for peerID := range tt.peerCache {
				serverIDs = append(serverIDs, peerID)
			}
			fakeReconciler := newFakePeerEndpointReconciler(t)
			handler := newHandlerChain(t, tt.informerFinishedSync, lastHandler, fakeReconciler, tt.localCache, tt.peerCache)
			server, requestGetter := createHTTP2ServerWithClient(handler, requestTimeout*2)
			defer server.Close()

			if tt.reconcilerConfig.do {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)

				for _, s := range tt.reconcilerConfig.servers {
					err := fakeReconciler.UpdateLease(s.serverID,
						s.publicIP,
						[]corev1.EndpointPort{{Name: "foo",
							Port: 8080, Protocol: "TCP"}})
					if err != nil {
						t.Errorf("Failed to update lease for server %s", s.serverID)
					}
				}
			}

			req, err := http.NewRequest(http.MethodGet, server.URL+tt.requestPath, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			req.Header.Set(PeerProxiedHeader, tt.peerproxiedHeader)

			resp, _ := requestGetter(req)

			// compare response
			assert.Equal(t, tt.wantStatus, resp.StatusCode)

			// compare metric
			if tt.wantMetricsData != "" {
				if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.wantMetricsData), []string{"apiserver_rerouted_request_total"}...); err != nil {
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

func newHandlerChain(t *testing.T, informerFinishedSync bool, handler http.Handler,
	reconciler reconcilers.PeerEndpointLeaseReconciler,
	localCache map[schema.GroupVersionResource]bool, peerCache map[string]map[schema.GroupVersionResource]bool) http.Handler {
	// Add peerproxy handler
	s := serializer.NewCodecFactory(runtime.NewScheme()).WithoutConversion()
	peerProxyHandler, err := newFakePeerProxyHandler(informerFinishedSync, reconciler, localServerID, s, localCache, peerCache)
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

func newFakePeerProxyHandler(informerFinishedSync bool,
	reconciler reconcilers.PeerEndpointLeaseReconciler, id string, s runtime.NegotiatedSerializer,
	localCache map[schema.GroupVersionResource]bool, peerCache map[string]map[schema.GroupVersionResource]bool) (*peerProxyHandler, error) {
	clientset := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(clientset, 0)
	leaseInformer := informerFactory.Coordination().V1().Leases()
	clientConfig := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure: false,
		}}
	loopbackClientConfig := &rest.Config{
		Host: "localhost:1010",
	}
	ppH, err := NewPeerProxyHandler(id, "identity=testserver", leaseInformer, reconciler, s, loopbackClientConfig, clientConfig)
	if err != nil {
		return nil, err
	}
	ppH.localDiscoveryInfoCache.Store(localCache)
	ppH.peerDiscoveryInfoCache.Store(peerCache)

	ppH.finishedSync.Store(informerFinishedSync)
	return ppH, nil
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
