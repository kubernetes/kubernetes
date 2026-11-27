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

package cache

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
)

func TestReflectorFullListPreservesZeroLimit(t *testing.T) {
	for _, clientType := range []string{"typed", "dynamic", "metadata", "filtered list watch"} {
		t.Run(clientType, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pods := []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", ResourceVersion: "10"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "default", ResourceVersion: "10"}},
			}
			var requests []url.Values
			var requestsMu sync.Mutex
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				query := req.URL.Query()
				requestsMu.Lock()
				requests = append(requests, query)
				requestsMu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				if query.Get("continue") != "" {
					w.WriteHeader(http.StatusGone)
					_ = json.NewEncoder(w).Encode(&metav1.Status{
						TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Status"},
						Status:   metav1.StatusFailure,
						Reason:   metav1.StatusReasonExpired,
						Code:     http.StatusGone,
						Message:  "continuation expired",
					})
					return
				}
				result := &v1.PodList{
					TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "PodList"},
					ListMeta: metav1.ListMeta{ResourceVersion: "12"},
					Items:    pods,
				}
				if query.Get("limit") != "" && query.Get("limit") != "0" {
					result.ResourceVersion = "11"
					result.Continue = "next"
					result.Items = pods[:1]
				}
				_ = json.NewEncoder(w).Encode(result)
			}))
			defer server.Close()

			config := &rest.Config{Host: server.URL}
			resource := schema.GroupVersionResource{Version: "v1", Resource: "pods"}
			modify := func(options *metav1.ListOptions) {
				options.Limit = 1
				options.LabelSelector = "app=example"
				options.FieldSelector = "status.phase=Running"
			}
			var listFn ListWithContextFunc
			var lw *ListWatch
			switch clientType {
			case "typed", "filtered list watch":
				client, err := corev1client.NewForConfig(config)
				require.NoError(t, err)
				if clientType == "filtered list watch" {
					lw = NewFilteredListWatchFromClient(client.RESTClient(), "pods", "default", modify)
				} else {
					listFn = func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
						return client.Pods("default").List(ctx, options)
					}
				}
			case "dynamic":
				client, err := dynamic.NewForConfig(config)
				require.NoError(t, err)
				listFn = func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
					return client.Resource(resource).Namespace("default").List(ctx, options)
				}
			case "metadata":
				client, err := metadata.NewForConfig(config)
				require.NoError(t, err)
				listFn = func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
					return client.Resource(resource).Namespace("default").List(ctx, options)
				}
			}
			if lw == nil {
				lw = &ListWatch{
					ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
						modify(&options)
						return listFn(ctx, options)
					},
				}
			}
			store := NewStore(MetaNamespaceKeyFunc)
			for i := range pods {
				require.NoError(t, store.Add(&pods[i]))
			}
			r := NewReflectorWithOptions(lw, nil, store, ReflectorOptions{Logger: &logger})
			r.WatchListPageSize = 2
			r.setLastSyncResourceVersion("10")

			require.NoError(t, r.list(ctx))
			requestsMu.Lock()
			assert.Equal(t, []url.Values{
				{"limit": {"1"}, "resourceVersion": {"10"}, "labelSelector": {"app=example"}, "fieldSelector": {"status.phase=Running"}},
				{"limit": {"1"}, "continue": {"next"}, "labelSelector": {"app=example"}, "fieldSelector": {"status.phase=Running"}},
				{"resourceVersion": {"10"}, "labelSelector": {"app=example"}, "fieldSelector": {"status.phase=Running"}},
			}, requests)
			requestsMu.Unlock()
			assert.ElementsMatch(t, []string{"default/pod-1", "default/pod-2"}, store.ListKeys())
			assert.Equal(t, "12", r.LastSyncResourceVersion())

			// Ordinary zero-limit requests must still allow a callback to set a page size.
			result, err := lw.ListWithContext(ctx, metav1.ListOptions{})
			require.NoError(t, err)
			listMeta, err := meta.ListAccessor(result)
			require.NoError(t, err)
			assert.Equal(t, "next", listMeta.GetContinue())
			requestsMu.Lock()
			assert.Equal(t, "1", requests[len(requests)-1].Get("limit"))
			requestsMu.Unlock()
		})
	}
}
