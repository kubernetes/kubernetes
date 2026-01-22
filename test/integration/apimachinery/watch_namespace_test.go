/*
Copyright 2024 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestWatchNamespaceEvent(t *testing.T) {
	timeout := 30 * time.Second

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()
	clientSet, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	newWatcher := func(ctx context.Context, ns *corev1.Namespace, resourceVersion string) (watch.Interface, error) {
		return clientSet.CoreV1().Namespaces().Watch(ctx, metav1.ListOptions{
			FieldSelector:   fields.OneTermEqualSelector("metadata.name", ns.Name).String(),
			ResourceVersion: resourceVersion,
		})
	}

	newLegacyNSWatcher := func(ctx context.Context, ns *corev1.Namespace, resourceVersion string) (watch.Interface, error) {
		if resourceVersion == "" {
			return clientSet.CoreV1().RESTClient().Get().
				RequestURI(fmt.Sprintf("/api/v1/watch/namespaces/%s", ns.Name)).Watch(ctx)
		} else {
			return clientSet.CoreV1().RESTClient().Get().
				RequestURI(fmt.Sprintf("/api/v1/watch/namespaces/%s?resourceVersion=%s", ns.Name, resourceVersion)).Watch(ctx)
		}
	}

	newTestNamespace := func(name string) *corev1.Namespace {
		return &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		}
	}

	tt := []struct {
		name            string
		namespace       *corev1.Namespace
		resourceVersion string
		getWatcher      func(ctx context.Context, ns *corev1.Namespace, resourceVersion string) (watch.Interface, error)
	}{
		{
			name:       "watch namespace",
			namespace:  newTestNamespace("watch-namespace"),
			getWatcher: newWatcher,
		},
		{
			name:            "watch namespace with resource version",
			namespace:       newTestNamespace("watch-ns-with-rv"),
			getWatcher:      newWatcher,
			resourceVersion: "0",
		},
		{
			name:       "legacy watch namespace api",
			namespace:  newTestNamespace("legacy-watch-ns"),
			getWatcher: newLegacyNSWatcher,
		},
		{
			name:            "legacy watch namespace api with resource version",
			namespace:       newTestNamespace("legacy-watch-ns-with-rv"),
			getWatcher:      newLegacyNSWatcher,
			resourceVersion: "0",
		},
	}

	t.Run("group", func(t *testing.T) {
		for _, tc := range tt {
			tc := tc // we need to copy it for parallel runs
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				startTest := time.Now()
				ctx, cancel := context.WithTimeout(context.Background(), timeout)
				defer cancel()

				ns, err := clientSet.CoreV1().Namespaces().Create(ctx, tc.namespace, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create testing namespace %s: %v", tc.namespace.Name, err)
				} else {
					defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
				}

				watcher, err := tc.getWatcher(ctx, ns, tc.resourceVersion)
				if err != nil {
					t.Fatalf("Failed to create watcher: %v", err)
				}
				defer watcher.Stop()

				t.Logf("start watching namespace %s event", tc.namespace.Name)
				generateAndWatchNamespaceEvent(ctx, t, clientSet, ns, watcher)
				t.Logf("Watch duration: %v", time.Since(startTest))
			})
		}
	})
}

func generateAndWatchNamespaceEvent(ctx context.Context, t *testing.T, clientSet *kubernetes.Clientset, ns *corev1.Namespace, watcher watch.Interface) {
	timeout := 10 * time.Second
	nsName := ns.Name

	t.Logf("Expectiong to watch an add event")
	_, ok := waitForEvent(watcher, watch.Added, ns, timeout)
	if !ok {
		t.Fatalf("Failed to watch add event")
	}

	if ns.Annotations == nil {
		ns.Annotations = map[string]string{"e2e-watch-namespace": "update-v1"}
	} else {
		ns.Annotations["e2e-watch-namespace"] = "update-v1"
	}

	ns, err := clientSet.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update namespace %s: %v", nsName, err)
	}

	t.Logf("Expectiong to watch an update event")
	_, ok = waitForEvent(watcher, watch.Modified, ns, timeout)
	if !ok {
		t.Fatalf("Failed to watch update event")
	}

	ns.Annotations["e2e-watch-namespace"] = "update-v2"
	ns, err = clientSet.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to delete namespace %s: %v", nsName, err)
	}

	t.Logf("Expectiong to watch an update event")
	_, ok = waitForEvent(watcher, watch.Modified, ns, timeout)
	if !ok {
		t.Fatalf("Failed to watch update event")
	}
}

func waitForEvent(w watch.Interface, expectType watch.EventType, expectObject runtime.Object, duration time.Duration) (watch.Event, bool) {
	stopTimer := time.NewTimer(duration)
	defer stopTimer.Stop()
	for {
		select {
		case actual, ok := <-w.ResultChan():
			if !ok {
				return watch.Event{}, false
			}
			if expectType == actual.Type && (expectObject == nil || apiequality.Semantic.DeepEqual(expectObject, actual.Object)) {
				return actual, true
			}

		case <-stopTimer.C:
			expected := watch.Event{
				Type:   expectType,
				Object: expectObject,
			}
			return expected, false
		}
	}
}
