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

package apiclient

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/version"
	clientset "k8s.io/client-go/kubernetes"
	clienttesting "k8s.io/client-go/testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

func TestDryRunRoundTripperActions(t *testing.T) {
	ctx := context.Background()
	ns := metav1.NamespaceSystem
	configmaps := corev1.SchemeGroupVersion.WithResource("configmaps")
	secrets := corev1.SchemeGroupVersion.WithResource("secrets")
	nodes := corev1.SchemeGroupVersion.WithResource("nodes")
	pods := corev1.SchemeGroupVersion.WithResource("pods")
	deployments := appsv1.SchemeGroupVersion.WithResource("deployments")
	clusterroles := rbacv1.SchemeGroupVersion.WithResource("clusterroles")

	// Request bodies are decoded with the clientset scheme, so the object seen by the
	// reactors carries the TypeMeta that the caller left empty.
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Data:       map[string]string{"a": "b"},
	}
	decodedConfigMap := configMap.DeepCopy()
	decodedConfigMap.TypeMeta = metav1.TypeMeta{APIVersion: "v1", Kind: "ConfigMap"}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	decodedNode := node.DeepCopy()
	decodedNode.TypeMeta = metav1.TypeMeta{APIVersion: "v1", Kind: "Node"}
	patch := []byte(`{"metadata":{"labels":{"a":"b"}}}`)

	tests := []struct {
		name    string
		apiCall func(c clientset.Interface) error
		want    clienttesting.Action
	}{
		{
			name: "get namespaced",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().ConfigMaps(ns).Get(ctx, "foo", metav1.GetOptions{})
				return err
			},
			want: clienttesting.NewGetAction(configmaps, ns, "foo"),
		},
		{
			name: "get cluster-scoped",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().Nodes().Get(ctx, "node-1", metav1.GetOptions{})
				return err
			},
			want: clienttesting.NewRootGetAction(nodes, "node-1"),
		},
		{
			name: "get in a non-core group",
			apiCall: func(c clientset.Interface) error {
				_, err := c.AppsV1().Deployments(ns).Get(ctx, "coredns", metav1.GetOptions{})
				return err
			},
			want: clienttesting.NewGetAction(deployments, ns, "coredns"),
		},
		{
			name: "list namespaced with selectors",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
					LabelSelector: "component=etcd,tier=control-plane",
					FieldSelector: "spec.nodeName=node-1",
				})
				return err
			},
			want: clienttesting.NewListActionWithOptions(pods, corev1.SchemeGroupVersion.WithKind("Pod"), ns, metav1.ListOptions{
				LabelSelector: "component=etcd,tier=control-plane",
				FieldSelector: "spec.nodeName=node-1",
			}),
		},
		{
			name: "list cluster-scoped",
			apiCall: func(c clientset.Interface) error {
				_, err := c.RbacV1().ClusterRoles().List(ctx, metav1.ListOptions{})
				return err
			},
			want: clienttesting.NewRootListActionWithOptions(clusterroles, rbacv1.SchemeGroupVersion.WithKind("ClusterRole"), metav1.ListOptions{}),
		},
		{
			name: "create",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().ConfigMaps(ns).Create(ctx, configMap, metav1.CreateOptions{})
				return err
			},
			want: clienttesting.NewCreateAction(configmaps, ns, decodedConfigMap),
		},
		{
			name: "update",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().ConfigMaps(ns).Update(ctx, configMap, metav1.UpdateOptions{})
				return err
			},
			want: clienttesting.NewUpdateAction(configmaps, ns, decodedConfigMap),
		},
		{
			name: "update subresource",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
				return err
			},
			want: clienttesting.NewRootUpdateSubresourceAction(nodes, "status", decodedNode),
		},
		{
			name: "patch",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().Nodes().Patch(ctx, "node-1", types.StrategicMergePatchType, patch, metav1.PatchOptions{})
				return err
			},
			want: clienttesting.NewRootPatchAction(nodes, "node-1", types.StrategicMergePatchType, patch),
		},
		{
			name: "delete",
			apiCall: func(c clientset.Interface) error {
				return c.CoreV1().Secrets(ns).Delete(ctx, "foo", metav1.DeleteOptions{})
			},
			want: clienttesting.NewDeleteAction(secrets, ns, "foo"),
		},
	}

	// Selectors hold unexported state; compare their canonical string form instead.
	selectorsByString := cmp.Options{
		cmp.Comparer(func(a, b labels.Selector) bool { return a.String() == b.String() }),
		cmp.Comparer(func(a, b fields.Selector) bool { return a.String() == b.String() }),
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got clienttesting.Action
			d := NewDryRun().WithDefaultMarshalFunction().WithWriter(io.Discard)
			d.PrependReactor(&clienttesting.SimpleReactor{
				Verb:     "*",
				Resource: "*",
				Reaction: func(action clienttesting.Action) (bool, runtime.Object, error) {
					got = action
					return true, nil, nil
				},
			})
			if err := tc.apiCall(d.FakeClient()); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got, selectorsByString); diff != "" {
				t.Errorf("action differs (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestDryRunRoundTripperListFilter(t *testing.T) {
	pod := func(name string) corev1.Pod {
		return corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: metav1.NamespaceSystem, Labels: map[string]string{"component": name}}}
	}
	d := NewDryRun().WithDefaultMarshalFunction().WithWriter(io.Discard)
	d.PrependReactor(&clienttesting.SimpleReactor{
		Verb:     "list",
		Resource: "pods",
		Reaction: func(clienttesting.Action) (bool, runtime.Object, error) {
			return true, &corev1.PodList{Items: []corev1.Pod{pod("kube-apiserver"), pod("etcd")}}, nil
		},
	})
	for _, tc := range []struct{ selector, want string }{
		{"", "kube-apiserver,etcd"},
		{"component=kube-apiserver", "kube-apiserver"},
		{"component=none", ""},
	} {
		list, err := d.FakeClient().CoreV1().Pods(metav1.NamespaceSystem).List(context.Background(), metav1.ListOptions{LabelSelector: tc.selector})
		if err != nil {
			t.Fatalf("%q: unexpected error: %v", tc.selector, err)
		}
		var names []string
		for _, p := range list.Items {
			names = append(names, p.Name)
		}
		if got := strings.Join(names, ","); got != tc.want {
			t.Errorf("%q: got %q, want %q", tc.selector, got, tc.want)
		}
	}
}

func TestDryRunRoundTripperServerVersion(t *testing.T) {
	want := &version.Info{Major: "1", Minor: "33", GitVersion: "v1.33.0"}
	d := NewDryRun().WithServerVersion(want)
	got, err := d.FakeClient().Discovery().ServerVersion()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("server version differs (-want,+got):\n%s", diff)
	}
}

func TestDryRunRoundTripperErrors(t *testing.T) {
	ctx := context.Background()
	ns := metav1.NamespaceSystem
	tests := []struct {
		name        string
		reactionErr error
		apiCall     func(c clientset.Interface) error
		wantCode    int32
		wantReason  metav1.StatusReason
	}{
		{
			name:        "API status error keeps its code",
			reactionErr: apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, "foo"),
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().ConfigMaps(ns).Get(ctx, "foo", metav1.GetOptions{})
				return err
			},
			wantCode:   http.StatusNotFound,
			wantReason: metav1.StatusReasonNotFound,
		},
		{
			name:        "plain error becomes an internal error",
			reactionErr: errors.New("boom"),
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().ConfigMaps(ns).Get(ctx, "foo", metav1.GetOptions{})
				return err
			},
			wantCode:   http.StatusInternalServerError,
			wantReason: metav1.StatusReasonInternalError,
		},
		{
			name: "watch is not served",
			apiCall: func(c clientset.Interface) error {
				_, err := c.CoreV1().Pods(ns).Watch(ctx, metav1.ListOptions{})
				return err
			},
			wantCode:   http.StatusMethodNotAllowed,
			wantReason: metav1.StatusReasonMethodNotAllowed,
		},
		{
			name: "non-resource paths are not served",
			apiCall: func(c clientset.Interface) error {
				return c.Discovery().RESTClient().Get().AbsPath("/healthz").Do(ctx).Error()
			},
			wantCode:   http.StatusNotFound,
			wantReason: metav1.StatusReasonNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewDryRun().WithDefaultMarshalFunction().WithWriter(io.Discard)
			if tc.reactionErr != nil {
				d.PrependReactor(&clienttesting.SimpleReactor{
					Verb:     "*",
					Resource: "*",
					Reaction: func(clienttesting.Action) (bool, runtime.Object, error) {
						return true, nil, tc.reactionErr
					},
				})
			}
			err := tc.apiCall(d.FakeClient())
			var apiStatus apierrors.APIStatus
			if !errors.As(err, &apiStatus) {
				t.Fatalf("expected an API status error, got: %v", err)
			}
			if status := apiStatus.Status(); status.Code != tc.wantCode || status.Reason != tc.wantReason {
				t.Errorf("expected code %d and reason %q, got code %d and reason %q: %v",
					tc.wantCode, tc.wantReason, status.Code, status.Reason, err)
			}
		})
	}
}
