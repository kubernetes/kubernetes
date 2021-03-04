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

package apiserver

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"k8s.io/utils/pointer"

	"github.com/davecgh/go-spew/spew"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/fake"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
)

const (
	testServicePort     = 1234
	testServicePortName = "testPort"
)

func newEndpoints(namespace, name string) *v1.Endpoints {
	return &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
	}
}

func newEndpointsWithAddress(namespace, name string, port int32, portName string) *v1.Endpoints {
	return &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Subsets: []v1.EndpointSubset{
			{
				Addresses: []v1.EndpointAddress{
					{
						IP: "val",
					},
				},
				Ports: []v1.EndpointPort{
					{
						Name: portName,
						Port: port,
					},
				},
			},
		},
	}
}

func newService(namespace, name string, port int32, portName string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{
				{Port: port, Name: portName},
			},
		},
	}
}

func newLocalAPIService(name string) *apiregistration.APIService {
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func newRemoteAPIService(name string) *apiregistration.APIService {
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: apiregistration.APIServiceSpec{
			Group:   strings.SplitN(name, ".", 2)[0],
			Version: strings.SplitN(name, ".", 2)[1],
			Service: &apiregistration.ServiceReference{
				Namespace: "foo",
				Name:      "bar",
				Port:      pointer.Int32Ptr(testServicePort),
			},
		},
	}
}

func setupAPIServices(apiServices []*apiregistration.APIService) (*AvailableConditionController, *fake.Clientset) {
	fakeClient := fake.NewSimpleClientset()
	apiServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	serviceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	endpointsIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	for _, o := range apiServices {
		apiServiceIndexer.Add(o)
	}

	c := AvailableConditionController{
		apiServiceClient: fakeClient.ApiregistrationV1(),
		apiServiceLister: listers.NewAPIServiceLister(apiServiceIndexer),
		serviceLister:    v1listers.NewServiceLister(serviceIndexer),
		endpointsLister:  v1listers.NewEndpointsLister(endpointsIndexer),
		serviceResolver:  &fakeServiceResolver{url: testServer.URL},
		queue: workqueue.NewNamedRateLimitingQueue(
			// We want a fairly tight requeue time.  The controller listens to the API, but because it relies on the routability of the
			// service network, it is possible for an external, non-watchable factor to affect availability.  This keeps
			// the maximum disruption time to a minimum, but it does prevent hot loops.
			workqueue.NewItemExponentialFailureRateLimiter(5*time.Millisecond, 30*time.Second),
			"AvailableConditionController"),
		tlsCache: &tlsTransportCache{transports: make(map[tlsCacheKey]http.RoundTripper)},
		metrics:  newAvailabilityMetrics(),
	}
	for _, svc := range apiServices {
		c.addAPIService(svc)
	}
	return &c, fakeClient
}

func BenchmarkBuildCache(b *testing.B) {
	apiServiceName := "remote.group"
	// model 1 APIService pointing at a given service, and 30 pointing at local group/versions
	apiServices := []*apiregistration.APIService{newRemoteAPIService(apiServiceName)}
	for i := 0; i < 30; i++ {
		apiServices = append(apiServices, newLocalAPIService(fmt.Sprintf("local.group%d", i)))
	}
	// model one service backing an API service, and 100 unrelated services
	services := []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)}
	for i := 0; i < 100; i++ {
		services = append(services, newService("foo", fmt.Sprintf("bar%d", i), testServicePort, testServicePortName))
	}
	c, _ := setupAPIServices(apiServices)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 1; n <= b.N; n++ {
		for _, svc := range services {
			c.addService(svc)
		}
		for _, svc := range services {
			c.updateService(svc, svc)
		}
		for _, svc := range services {
			c.deleteService(svc)
		}
	}
}

func TestBuildCache(t *testing.T) {
	tests := []struct {
		name string

		apiServiceName string
		apiServices    []*apiregistration.APIService
		services       []*v1.Service
		endpoints      []*v1.Endpoints

		expectedAvailability apiregistration.APIServiceCondition
	}{
		{
			name:           "api service",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c, fakeClient := setupAPIServices(tc.apiServices)
			for _, svc := range tc.services {
				c.addService(svc)
			}

			c.sync(tc.apiServiceName)

			// ought to have one action writing status
			if e, a := 1, len(fakeClient.Actions()); e != a {
				t.Fatalf("%v expected %v, got %v", tc.name, e, fakeClient.Actions())
			}
		})
	}
}

func TestTLSCache(t *testing.T) {
	apiServices := []*apiregistration.APIService{newRemoteAPIService("remote.group")}
	services := []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)}
	c, _ := setupAPIServices(apiServices)
	// TLS configs with customized dialers are uncacheable by the client-go
	// TLS transport cache. The local cache will be used.
	c.dialContext = (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}).DialContext
	for _, svc := range services {
		c.addService(svc)
	}
	tests := []struct {
		name                       string
		proxyCurrentCertKeyContent certKeyFunc
		expectedCacheSize          int
	}{
		{
			name:              "nil certKeyFunc",
			expectedCacheSize: 1,
		},
		{
			name:                       "empty certKeyFunc",
			proxyCurrentCertKeyContent: func() ([]byte, []byte) { return emptyCert(), emptyCert() },
			// the tlsCacheKey is the same, reuse existing transport
			expectedCacheSize: 1,
		},
		{
			name:                       "different certKeyFunc",
			proxyCurrentCertKeyContent: testCertKeyFunc,
			// the tlsCacheKey is different, create a new transport
			expectedCacheSize: 2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c.proxyCurrentCertKeyContent = tc.proxyCurrentCertKeyContent
			for _, apiService := range apiServices {
				c.sync(apiService.Name)
			}
			if len(c.tlsCache.transports) != tc.expectedCacheSize {
				t.Fatalf("%v cache size expected %v, got %v", tc.name, tc.expectedCacheSize, len(c.tlsCache.transports))
			}
		})
	}
}

func TestSync(t *testing.T) {
	tests := []struct {
		name string

		apiServiceName     string
		apiServices        []*apiregistration.APIService
		services           []*v1.Service
		endpoints          []*v1.Endpoints
		forceDiscoveryFail bool

		expectedAvailability apiregistration.APIServiceCondition
	}{
		{
			name:           "local",
			apiServiceName: "local.group",
			apiServices:    []*apiregistration.APIService{newLocalAPIService("local.group")},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionTrue,
				Reason:  "Local",
				Message: "Local APIServices are always available",
			},
		},
		{
			name:           "no service",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "not-bar", testServicePort, testServicePortName)},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "ServiceNotFound",
				Message: `service/bar in "foo" is not present`,
			},
		},
		{
			name:           "service on bad port",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services: []*v1.Service{{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeClusterIP,
					Ports: []v1.ServicePort{
						{Port: 6443},
					},
				},
			}},
			endpoints: []*v1.Endpoints{newEndpointsWithAddress("foo", "bar", testServicePort, testServicePortName)},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "ServicePortError",
				Message: fmt.Sprintf(`service/bar in "foo" is not listening on port %d`, testServicePort),
			},
		},
		{
			name:           "no endpoints",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "EndpointsNotFound",
				Message: `cannot find endpoints for service/bar in "foo"`,
			},
		},
		{
			name:           "missing endpoints",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
			endpoints:      []*v1.Endpoints{newEndpoints("foo", "bar")},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "MissingEndpoints",
				Message: `endpoints for service/bar in "foo" have no addresses with port name "testPort"`,
			},
		},
		{
			name:           "wrong endpoint port name",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
			endpoints:      []*v1.Endpoints{newEndpointsWithAddress("foo", "bar", testServicePort, "wrongName")},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "MissingEndpoints",
				Message: fmt.Sprintf(`endpoints for service/bar in "foo" have no addresses with port name "%s"`, testServicePortName),
			},
		},
		{
			name:           "remote",
			apiServiceName: "remote.group",
			apiServices:    []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:       []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
			endpoints:      []*v1.Endpoints{newEndpointsWithAddress("foo", "bar", testServicePort, testServicePortName)},
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionTrue,
				Reason:  "Passed",
				Message: `all checks passed`,
			},
		},
		{
			name:               "remote-bad-return",
			apiServiceName:     "remote.group",
			apiServices:        []*apiregistration.APIService{newRemoteAPIService("remote.group")},
			services:           []*v1.Service{newService("foo", "bar", testServicePort, testServicePortName)},
			endpoints:          []*v1.Endpoints{newEndpointsWithAddress("foo", "bar", testServicePort, testServicePortName)},
			forceDiscoveryFail: true,
			expectedAvailability: apiregistration.APIServiceCondition{
				Type:    apiregistration.Available,
				Status:  apiregistration.ConditionFalse,
				Reason:  "FailedDiscoveryCheck",
				Message: `failing or missing response from`,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset()
			apiServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			serviceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			endpointsIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, obj := range tc.apiServices {
				apiServiceIndexer.Add(obj)
			}
			for _, obj := range tc.services {
				serviceIndexer.Add(obj)
			}
			for _, obj := range tc.endpoints {
				endpointsIndexer.Add(obj)
			}

			testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !tc.forceDiscoveryFail {
					w.WriteHeader(http.StatusOK)
				}
				w.WriteHeader(http.StatusForbidden)
			}))
			defer testServer.Close()

			c := AvailableConditionController{
				apiServiceClient:           fakeClient.ApiregistrationV1(),
				apiServiceLister:           listers.NewAPIServiceLister(apiServiceIndexer),
				serviceLister:              v1listers.NewServiceLister(serviceIndexer),
				endpointsLister:            v1listers.NewEndpointsLister(endpointsIndexer),
				serviceResolver:            &fakeServiceResolver{url: testServer.URL},
				proxyCurrentCertKeyContent: func() ([]byte, []byte) { return emptyCert(), emptyCert() },
				tlsCache:                   &tlsTransportCache{transports: make(map[tlsCacheKey]http.RoundTripper)},
				metrics:                    newAvailabilityMetrics(),
			}
			c.sync(tc.apiServiceName)

			// ought to have one action writing status
			if e, a := 1, len(fakeClient.Actions()); e != a {
				t.Fatalf("%v expected %v, got %v", tc.name, e, fakeClient.Actions())
			}

			action, ok := fakeClient.Actions()[0].(clienttesting.UpdateAction)
			if !ok {
				t.Fatalf("%v got %v", tc.name, ok)
			}

			if e, a := 1, len(action.GetObject().(*apiregistration.APIService).Status.Conditions); e != a {
				t.Fatalf("%v expected %v, got %v", tc.name, e, action.GetObject())
			}
			condition := action.GetObject().(*apiregistration.APIService).Status.Conditions[0]
			if e, a := tc.expectedAvailability.Type, condition.Type; e != a {
				t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
			}
			if e, a := tc.expectedAvailability.Status, condition.Status; e != a {
				t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
			}
			if e, a := tc.expectedAvailability.Reason, condition.Reason; e != a {
				t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
			}
			if e, a := tc.expectedAvailability.Message, condition.Message; !strings.HasPrefix(a, e) {
				t.Errorf("%v expected %v, got %#v", tc.name, e, condition)
			}
			if condition.LastTransitionTime.IsZero() {
				t.Error("expected lastTransitionTime to be non-zero")
			}
		})
	}
}

type fakeServiceResolver struct {
	url string
}

func (f *fakeServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return url.Parse(f.url)
}

func TestUpdateAPIServiceStatus(t *testing.T) {
	foo := &apiregistration.APIService{Status: apiregistration.APIServiceStatus{Conditions: []apiregistration.APIServiceCondition{{Type: "foo"}}}}
	bar := &apiregistration.APIService{Status: apiregistration.APIServiceStatus{Conditions: []apiregistration.APIServiceCondition{{Type: "bar"}}}}

	fakeClient := fake.NewSimpleClientset()
	c := AvailableConditionController{
		apiServiceClient: fakeClient.ApiregistrationV1().(apiregistrationclient.APIServicesGetter),
		metrics:          newAvailabilityMetrics(),
	}

	c.updateAPIServiceStatus(foo, foo)
	if e, a := 0, len(fakeClient.Actions()); e != a {
		t.Error(spew.Sdump(fakeClient.Actions()))
	}

	fakeClient.ClearActions()
	c.updateAPIServiceStatus(foo, bar)
	if e, a := 1, len(fakeClient.Actions()); e != a {
		t.Error(spew.Sdump(fakeClient.Actions()))
	}
}

func emptyCert() []byte {
	return []byte{}
}

func testCertKeyFunc() ([]byte, []byte) {
	return []byte(`-----BEGIN CERTIFICATE-----
MIICBDCCAW2gAwIBAgIJAPgVBh+4xbGoMA0GCSqGSIb3DQEBCwUAMBsxGTAXBgNV
BAMMEHdlYmhvb2tfdGVzdHNfY2EwIBcNMTcwNzI4MjMxNTI4WhgPMjI5MTA1MTMy
MzE1MjhaMB8xHTAbBgNVBAMMFHdlYmhvb2tfdGVzdHNfY2xpZW50MIGfMA0GCSqG
SIb3DQEBAQUAA4GNADCBiQKBgQDkGXXSm6Yun5o3Jlmx45rItcQ2pmnoDk4eZfl0
rmPa674s2pfYo3KywkXQ1Fp3BC8GUgzPLSfJ8xXya9Lg1Wo8sHrDln0iRg5HXxGu
uFNhRBvj2S0sIff0ZG/IatB9I6WXVOUYuQj6+A0CdULNj1vBqH9+7uWbLZ6lrD4b
a44x/wIDAQABo0owSDAJBgNVHRMEAjAAMAsGA1UdDwQEAwIF4DAdBgNVHSUEFjAU
BggrBgEFBQcDAgYIKwYBBQUHAwEwDwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0B
AQsFAAOBgQCpN27uh/LjUVCaBK7Noko25iih/JSSoWzlvc8CaipvSPofNWyGx3Vu
OdcSwNGYX/pp4ZoAzFij/Y5u0vKTVLkWXATeTMVmlPvhmpYjj9gPkCSY6j/SiKlY
kGy0xr+0M5UQkMBcfIh9oAp9um1fZHVWAJAGP/ikZgkcUey0LmBn8w==
-----END CERTIFICATE-----`), []byte(`-----BEGIN RSA PRIVATE KEY-----
MIICWwIBAAKBgQDkGXXSm6Yun5o3Jlmx45rItcQ2pmnoDk4eZfl0rmPa674s2pfY
o3KywkXQ1Fp3BC8GUgzPLSfJ8xXya9Lg1Wo8sHrDln0iRg5HXxGuuFNhRBvj2S0s
Iff0ZG/IatB9I6WXVOUYuQj6+A0CdULNj1vBqH9+7uWbLZ6lrD4ba44x/wIDAQAB
AoGAZbWwowvCq1GBq4vPPRI3h739Uz0bRl1ymf1woYXNguXRtCB4yyH+2BTmmrrF
6AIWkePuUEdbUaKyK5nGu3iOWM+/i6NP3kopQANtbAYJ2ray3kwvFlhqyn1bxX4n
gl/Cbdw1If4zrDrB66y8mYDsjzK7n/gFaDNcY4GArjvOXKkCQQD9Lgv+WD73y4RP
yS+cRarlEeLLWVsX/pg2oEBLM50jsdUnrLSW071MjBgP37oOXzqynF9SoDbP2Y5C
x+aGux9LAkEA5qPlQPv0cv8Wc3qTI+LixZ/86PPHKWnOnwaHm3b9vQjZAkuVQg3n
Wgg9YDmPM87t3UFH7ZbDihUreUxwr9ZjnQJAZ9Z95shMsxbOYmbSVxafu6m1Sc+R
M+sghK7/D5jQpzYlhUspGf8n0YBX0hLhXUmjamQGGH5LXL4Owcb4/mM6twJAEVio
SF/qva9jv+GrKVrKFXT374lOJFY53Qn/rvifEtWUhLCslCA5kzLlctRBafMZPrfH
Mh5RrJP1BhVysDbenQJASGcc+DiF7rB6K++ZGyC11E2AP29DcZ0pgPESSV7npOGg
+NqPRZNVCSZOiVmNuejZqmwKhZNGZnBFx1Y+ChAAgw==
-----END RSA PRIVATE KEY-----`)
}
