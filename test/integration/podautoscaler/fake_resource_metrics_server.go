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

package podautoscaler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	"k8s.io/client-go/ktesting"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregator "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	testdiscovery "k8s.io/kubernetes/test/integration/apiserver/discovery"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
)

// fakeResourceMetricsServer serves empty PodMetrics lists through the API
// aggregation layer and tracks requests by API version. Empty responses are
// sufficient because the test verifies version selection, not replica calculation.
type fakeResourceMetricsServer struct {
	discovery   discoveryendpoint.ResourceManager
	service     testdiscovery.FakeService
	apiServices aggregator.Interface

	registered map[schema.GroupVersion]struct{}

	v1Requests      atomic.Int64
	v1beta1Requests atomic.Int64
}

func newFakeResourceMetricsServer(tCtx ktesting.TContext) *fakeResourceMetricsServer {
	server := &fakeResourceMetricsServer{
		discovery:   discoveryendpoint.NewResourceManager("apis"),
		apiServices: aggregator.NewForConfigOrDie(tCtx.RESTConfig()),
		registered:  map[schema.GroupVersion]struct{}{},
	}

	service := testdiscovery.NewFakeService("resource-metrics-server", tCtx.Client(), http.HandlerFunc(server.serveHTTP))
	server.service = service

	go func() {
		if err := service.Run(tCtx); err != nil {
			tCtx.Errorf("fake resource metrics server failed: %v", err)
		}
	}()

	if err := service.WaitForReady(tCtx); err != nil {
		tCtx.Fatalf("fake resource metrics server did not become ready: %v", err)
	}

	tCtx.CleanupCtx(func(cleanupCtx ktesting.TContext) {
		for gv := range server.registered {
			err := server.apiServices.ApiregistrationV1().APIServices().Delete(cleanupCtx, apiServiceName(gv), metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				cleanupCtx.Errorf("failed to delete APIService %q: %v", apiServiceName(gv), err)
			}
		}
	})

	return server
}

func (s *fakeResourceMetricsServer) serveHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/apis":
		s.discovery.ServeHTTP(w, r)
	case r.URL.Path == "/apis/"+metricsv1.SchemeGroupVersion.String():
		writeJSON(w, metav1.APIResourceList{
			GroupVersion: metricsv1.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{
					Name:       "pods",
					Kind:       "PodMetrics",
					Namespaced: true,
					Verbs:      metav1.Verbs{"get", "list"},
				},
			},
		})
	case r.URL.Path == "/apis/"+metricsv1beta1.SchemeGroupVersion.String():
		writeJSON(w, metav1.APIResourceList{
			GroupVersion: metricsv1beta1.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{
					Name:       "pods",
					Kind:       "PodMetrics",
					Namespaced: true,
					Verbs:      metav1.Verbs{"get", "list"},
				},
			},
		})
	case isPodMetricsListRequest(r.URL.Path, metricsv1.SchemeGroupVersion):
		s.v1Requests.Add(1)
		writeJSON(w, metricsv1.PodMetricsList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: metricsv1.SchemeGroupVersion.String(),
				Kind:       "PodMetricsList",
			},
		})
	case isPodMetricsListRequest(r.URL.Path, metricsv1beta1.SchemeGroupVersion):
		s.v1beta1Requests.Add(1)
		writeJSON(w, metricsv1beta1.PodMetricsList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: metricsv1beta1.SchemeGroupVersion.String(),
				Kind:       "PodMetricsList",
			},
		})

	default:
		http.NotFound(w, r)
	}
}

func (s *fakeResourceMetricsServer) replaceVersions(tCtx ktesting.TContext, versions ...schema.GroupVersion) {
	tCtx.Helper()

	desired := make(map[schema.GroupVersion]struct{}, len(versions))
	discoveredVersions := make([]apidiscoveryv2.APIVersionDiscovery, 0, len(versions))

	for _, gv := range versions {
		if gv.Group != metricsv1.GroupName {
			tCtx.Fatalf("unexpected resource metrics API group %q", gv.Group)
		}
		if _, duplicate := desired[gv]; duplicate {
			continue
		}

		desired[gv] = struct{}{}
		discoveredVersions = append(
			discoveredVersions,
			apidiscoveryv2.APIVersionDiscovery{
				Version: gv.Version,
				Resources: []apidiscoveryv2.APIResourceDiscovery{
					{
						Resource: "pods",
						ResponseKind: &metav1.GroupVersionKind{
							Group:   gv.Group,
							Version: gv.Version,
							Kind:    "PodMetrics",
						},
						Scope: apidiscoveryv2.ScopeNamespace,
						Verbs: []string{"get", "list"},
					},
				},
				Freshness: apidiscoveryv2.DiscoveryFreshnessCurrent,
			},
		)
	}

	s.discovery.SetGroups([]apidiscoveryv2.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: metricsv1.GroupName,
			},
			Versions: discoveredVersions,
		},
	})

	for gv := range desired {
		if _, alreadyRegistered := s.registered[gv]; alreadyRegistered {
			continue
		}

		if err := s.registerAPIService(tCtx, gv); err != nil {
			tCtx.Fatalf("failed to register APIService %q: %v", apiServiceName(gv), err)
		}
		s.registered[gv] = struct{}{}
	}
	for gv := range s.registered {
		if _, keep := desired[gv]; keep {
			continue
		}

		err := s.apiServices.ApiregistrationV1().APIServices().Delete(tCtx, apiServiceName(gv), metav1.DeleteOptions{})
		if err != nil {
			tCtx.Fatalf("failed to delete APIService %q: %v", apiServiceName(gv), err)
		}
		delete(s.registered, gv)
	}
}

func (s *fakeResourceMetricsServer) registerAPIService(ctx context.Context, gv schema.GroupVersion) error {
	port := s.service.Port()
	if port == nil {
		return fmt.Errorf("fake resource metrics service is not ready")
	}

	// Give v1beta1 a higher APIService priority to verify that the resource
	// metrics client prefers v1 based on its own version ordering.
	versionPriority := int32(100)
	if gv == metricsv1beta1.SchemeGroupVersion {
		versionPriority = 200
	}

	apiService := &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: apiServiceName(gv),
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:                 gv.Group,
			Version:               gv.Version,
			GroupPriorityMinimum:  100,
			VersionPriority:       versionPriority,
			InsecureSkipTLSVerify: true,
			Service: &apiregistrationv1.ServiceReference{
				Namespace: "default",
				Name:      s.service.Name(),
				Port:      port,
			},
		},
	}

	_, err := s.apiServices.ApiregistrationV1().APIServices().Create(ctx, apiService, metav1.CreateOptions{})

	return err
}

func isPodMetricsListRequest(path string, gv schema.GroupVersion) bool {
	prefix := "/apis/" + gv.String() + "/namespaces/"
	return strings.HasPrefix(path, prefix) && strings.HasSuffix(path, "/pods")
}

func writeJSON(w http.ResponseWriter, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

func apiServiceName(gv schema.GroupVersion) string {
	return gv.Version + "." + gv.Group
}
