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

package pods

import (
	"encoding/json"
	"testing"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestDisablePodSubresources(t *testing.T) {
	// Ensure pods/resize would normally be registered so we verify the flag suppresses gated subresources too.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)

	flags := append(
		framework.DefaultTestServerFlags(),
		"--disable-pod-subresources=exec,log,ephemeralcontainers,resize,eviction,proxy",
	)
	server, err := kubeapiservertesting.StartTestServer(t, &kubeapiservertesting.TestServerInstanceOptions{}, flags, framework.SharedEtcd())
	if err != nil {
		t.Fatalf("failed to start kube-apiserver with --disable-pod-subresources: %v", err)
	}
	defer server.TearDownFn()

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create clientset: %v", err)
	}
	ctx := t.Context()

	t.Run("discovery", func(t *testing.T) {
		v1Resources, err := client.Discovery().ServerResourcesForGroupVersion("v1")
		if err != nil {
			t.Fatalf("failed to fetch v1 discovery: %v", err)
		}
		discovered := sets.New[string]()
		for _, r := range v1Resources.APIResources {
			discovered.Insert(r.Name)
		}
		for _, name := range []string{"pods", "pods/status", "pods/binding", "pods/attach", "pods/portforward"} {
			if !discovered.Has(name) {
				t.Errorf("expected resource %q to be present in v1 discovery", name)
			}
		}
		for _, name := range []string{"pods/exec", "pods/log", "pods/ephemeralcontainers", "pods/resize", "pods/eviction", "pods/proxy"} {
			if discovered.Has(name) {
				t.Errorf("expected disabled subresource %q to be absent from v1 discovery", name)
			}
		}
	})

	t.Run("openapi", func(t *testing.T) {
		raw, err := client.Discovery().RESTClient().Get().
			AbsPath("/openapi/v3/api/v1").
			SetHeader("Accept", "application/json").
			DoRaw(ctx)
		if err != nil {
			t.Fatalf("failed to fetch /openapi/v3/api/v1: %v", err)
		}
		var doc struct {
			Paths map[string]json.RawMessage `json:"paths"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatalf("failed to unmarshal /openapi/v3/api/v1 JSON: %v", err)
		}
		paths := sets.KeySet(doc.Paths)

		wantPresent := []string{
			"/api/v1/namespaces/{namespace}/pods",
			"/api/v1/namespaces/{namespace}/pods/{name}",
			"/api/v1/namespaces/{namespace}/pods/{name}/status",
			"/api/v1/namespaces/{namespace}/pods/{name}/binding",
			"/api/v1/namespaces/{namespace}/pods/{name}/attach",
			"/api/v1/namespaces/{namespace}/pods/{name}/portforward",
		}
		wantAbsent := []string{
			"/api/v1/namespaces/{namespace}/pods/{name}/exec",
			"/api/v1/namespaces/{namespace}/pods/{name}/log",
			"/api/v1/namespaces/{namespace}/pods/{name}/ephemeralcontainers",
			"/api/v1/namespaces/{namespace}/pods/{name}/resize",
			"/api/v1/namespaces/{namespace}/pods/{name}/eviction",
			"/api/v1/namespaces/{namespace}/pods/{name}/proxy",
			"/api/v1/namespaces/{namespace}/pods/{name}/proxy/{path}",
		}
		for _, p := range wantPresent {
			if !paths.Has(p) {
				t.Errorf("expected path %q to be present in OpenAPI v3", p)
			}
		}
		for _, p := range wantAbsent {
			if paths.Has(p) {
				t.Errorf("expected disabled subresource path %q to be absent from OpenAPI v3", p)
			}
		}
	})

	t.Run("runtime", func(t *testing.T) {
		ns := framework.CreateNamespaceOrDie(client, "disable-pod-subresources", t)
		defer framework.DeleteNamespaceOrDie(client, ns, t)

		pod, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "c1", Image: "pause"}},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create test pod: %v", err)
		}

		// Check enabled streaming subresources before binding: the routed handler rejects an
		// unscheduled pod with 400 BadRequest without needing a Node object in etcd.
		for _, subresource := range []string{"attach", "portforward"} {
			err := client.CoreV1().RESTClient().Post().
				AbsPath("/api/v1/namespaces", ns.Name, "pods", pod.Name, subresource).
				Param("stdout", "true").
				Do(ctx).Error()
			if !apierrors.IsBadRequest(err) {
				t.Errorf("%s: expected 400 BadRequest from routed handler on unbound pod, got: %v", subresource, err)
			}
		}

		if err := client.CoreV1().Pods(ns.Name).Bind(ctx, &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.Name, Namespace: ns.Name},
			Target:     v1.ObjectReference{Kind: "Node", Name: "test-node"},
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("expected pods/binding to succeed, got: %v", err)
		}

		pod, err = client.CoreV1().Pods(ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get bound pod: %v", err)
		}
		if pod.Spec.NodeName != "test-node" {
			t.Fatalf("expected pod Spec.NodeName=\"test-node\" after binding, got %q", pod.Spec.NodeName)
		}

		// Populate a valid unicast PodIP so pods' Redirector would succeed if it ever intercepted
		// unregistered subresource paths such as /proxy.
		pod.Status.Phase = v1.PodRunning
		pod.Status.PodIP = "10.1.2.3"
		pod.Status.PodIPs = []v1.PodIP{{IP: "10.1.2.3"}}
		pod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("expected pods/status update to succeed, got: %v", err)
		}
		if pod.Status.Phase != v1.PodRunning || pod.Status.PodIP != "10.1.2.3" {
			t.Fatalf("expected updated pod status Running/10.1.2.3, got phase=%q podIP=%q", pod.Status.Phase, pod.Status.PodIP)
		}

		disabledChecks := []struct {
			name string
			call func() error
		}{
			{
				name: "ephemeralcontainers",
				call: func() error {
					_, err := client.CoreV1().Pods(ns.Name).UpdateEphemeralContainers(ctx, pod.Name, pod, metav1.UpdateOptions{})
					return err
				},
			},
			{
				name: "resize",
				call: func() error {
					_, err := client.CoreV1().Pods(ns.Name).UpdateResize(ctx, pod.Name, pod, metav1.UpdateOptions{})
					return err
				},
			},
			{
				name: "eviction",
				call: func() error {
					return client.CoreV1().Pods(ns.Name).EvictV1(ctx, &policyv1.Eviction{
						ObjectMeta: metav1.ObjectMeta{Name: pod.Name, Namespace: ns.Name},
					})
				},
			},
			{
				name: "log",
				call: func() error {
					return client.CoreV1().Pods(ns.Name).GetLogs(pod.Name, &v1.PodLogOptions{}).Do(ctx).Error()
				},
			},
			{
				name: "exec",
				call: func() error {
					return client.CoreV1().RESTClient().Post().
						AbsPath("/api/v1/namespaces", ns.Name, "pods", pod.Name, "exec").
						Do(ctx).Error()
				},
			},
			{
				name: "proxy",
				call: func() error {
					return client.CoreV1().RESTClient().Get().
						AbsPath("/api/v1/namespaces", ns.Name, "pods", pod.Name, "proxy").
						Do(ctx).Error()
				},
			},
			{
				name: "proxy subpath",
				call: func() error {
					return client.CoreV1().RESTClient().Get().
						AbsPath("/api/v1/namespaces", ns.Name, "pods", pod.Name, "proxy", "healthz").
						Do(ctx).Error()
				},
			},
		}

		for _, tc := range disabledChecks {
			if err := tc.call(); !apierrors.IsNotFound(err) {
				t.Errorf("%s: expected 404 NotFound for disabled subresource, got: %v", tc.name, err)
			}
		}
	})
}
