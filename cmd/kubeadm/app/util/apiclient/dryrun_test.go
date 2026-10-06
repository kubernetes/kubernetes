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

package apiclient

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/version"
	clienttesting "k8s.io/client-go/testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

func TestNewDryRunWithKubeConfigFile(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "dryrun-test")
	if err != nil {
		t.Errorf("Unable to create temporary directory: %v", err)
	}
	defer func() {
		_ = os.RemoveAll(tmpDir)
	}()

	kubeconfig := kubeconfigphase.CreateWithToken(
		"some-server:6443",
		"cluster-foo",
		"user-bar",
		[]byte("fake-ca-cert"),
		"fake-token",
	)
	path := filepath.Join(tmpDir, "some-file")
	if err := kubeconfigphase.WriteToDisk(path, kubeconfig); err != nil {
		t.Fatal(err)
	}

	d := NewDryRun()
	if err := d.WithKubeConfigFile(path); err != nil {
		t.Fatal(err)
	}
	if d.FakeClient() == nil {
		t.Fatal("expected fakeClient to be non-nil")
	}
	if d.Client() == nil {
		t.Fatal("expected client to be non-nil")
	}
	if d.DynamicClient() == nil {
		t.Fatal("expected dynamicClient to be non-nil")
	}
}

func TestPrependAppendReactor(t *testing.T) {
	foo := &clienttesting.SimpleReactor{Verb: "foo"}
	bar := &clienttesting.SimpleReactor{Verb: "bar"}
	baz := &clienttesting.SimpleReactor{Verb: "baz"}
	qux := &clienttesting.SimpleReactor{Verb: "qux"}

	d := NewDryRun()
	lenBefore := len(d.fake.ReactionChain)
	d.PrependReactor(foo).PrependReactor(bar).
		AppendReactor(baz).AppendReactor(qux)

	// [ log, bar, foo, get, list, baz, qux, default ]
	//         1    2               5    6
	expectedIdx := map[string]int{
		foo.Verb: 2,
		bar.Verb: 1,
		baz.Verb: 5,
		qux.Verb: 6,
	}
	expectedLen := lenBefore + len(expectedIdx)

	if len(d.fake.ReactionChain) != expectedLen {
		t.Fatalf("expected len of reactor chain: %d, got: %d",
			expectedLen, len(d.fake.ReactionChain))
	}

	for actual, r := range d.fake.ReactionChain {
		s := r.(*clienttesting.SimpleReactor)
		expected, exists := expectedIdx[s.Verb]
		if exists {
			delete(expectedIdx, s.Verb)
			if actual != expected {
				t.Errorf("expected idx for verb %q: %d, got %d", s.Verb, expected, actual)
			}
		}
	}

	if len(expectedIdx) != 0 {
		t.Fatalf("expected len of exists map to be 0 after iteration, got: %d", len(expectedIdx))
	}
}

func TestReactors(t *testing.T) {
	type apiCallCase struct {
		name          string
		namespace     string
		expectedError bool
	}
	ctx := context.Background()
	tests := []struct {
		name         string
		setup        func(d *DryRun)
		apiCall      func(d *DryRun, namespace, name string) error
		apiCallCases []apiCallCase
	}{
		{
			name: "HealthCheckJobReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.HealthCheckJobReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().BatchV1().Jobs(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if diff := cmp.Diff(getJob(name, namespace), obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "upgrade-health-check",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "PatchNodeReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.PatchNodeReactor()))
			},
			apiCall: func(d *DryRun, _, name string) error {
				obj, err := d.FakeClient().CoreV1().Nodes().Patch(ctx, name, "", []byte{}, metav1.PatchOptions{})
				if err != nil {
					return err
				}
				if diff := cmp.Diff(getNode(name), obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "some-node",
					expectedError: false,
				},
			},
		},
		{
			name: "GetNodeReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetNodeReactor()))
			},
			apiCall: func(d *DryRun, _, name string) error {
				obj, err := d.FakeClient().CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if diff := cmp.Diff(getNode(name), obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "some-node",
					expectedError: false,
				},
			},
		},
		{
			name: "GetClusterInfoReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetClusterInfoReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getClusterInfoConfigMap()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "cluster-info",
					namespace:     metav1.NamespacePublic,
					expectedError: false,
				},
			},
		},
		{
			name: "GetKubeadmConfigReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetKubeadmConfigReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getKubeadmConfigMap()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "kubeadm-config",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "GetKubeletConfigReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetKubeletConfigReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getKubeletConfigMap()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "kubelet-config",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "GetKubeProxyConfigReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetKubeProxyConfigReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getKubeProxyConfigMap()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "kube-proxy",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "GetCoreDNSConfigReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetCoreDNSConfigReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getCoreDNSConfigMap()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "coredns",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "DeleteBootstrapTokenReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.DeleteBootstrapTokenReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				err := d.FakeClient().CoreV1().Secrets(namespace).Delete(ctx, name, metav1.DeleteOptions{})
				if err != nil {
					return err
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "bootstrap-token-foo",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "GetKubeadmCertsReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.GetKubeadmCertsReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				expectedObj := getKubeadmCertsSecret()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					name:          "foo",
					namespace:     "bar",
					expectedError: true,
				},
				{
					name:          "kubeadm-certs",
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "ListPodsReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.ListPodsReactor("foo")))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return err
				}
				expectedObj := getPodList("foo")
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					namespace:     "bar",
					expectedError: true,
				},
				{
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
		{
			name: "ListDeploymentsReactor",
			setup: func(d *DryRun) {
				d.PrependReactor((d.ListDeploymentsReactor()))
			},
			apiCall: func(d *DryRun, namespace, name string) error {
				obj, err := d.FakeClient().AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					return err
				}
				expectedObj := getDeploymentList()
				if diff := cmp.Diff(expectedObj, obj); diff != "" {
					return errors.Errorf("object differs (-want,+got):\n%s", diff)
				}
				return nil
			},
			apiCallCases: []apiCallCase{
				{
					namespace:     "bar",
					expectedError: true,
				},
				{
					namespace:     metav1.NamespaceSystem,
					expectedError: false,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewDryRun().WithDefaultMarshalFunction().WithWriter(io.Discard)
			tc.setup(d)
			for _, ac := range tc.apiCallCases {
				if err := tc.apiCall(d, ac.namespace, ac.name); (err != nil) != ac.expectedError {
					t.Errorf("expected error: %v, got: %v, error: %v", ac.expectedError, err != nil, err)
				}
			}
		})
	}
}

func TestDecodeUnstructuredIntoAPIObject(t *testing.T) {
	tests := []struct {
		name          string
		action        clienttesting.Action
		unstructured  runtime.Unstructured
		expectedObj   runtime.Object
		expectedError bool
	}{
		{
			name: "valid: ConfigMap is decoded",
			action: clienttesting.NewGetAction(
				schema.GroupVersionResource{
					Group:    "",
					Version:  "v1",
					Resource: "configmaps",
				},
				metav1.NamespaceSystem,
				"kubeadm-config",
			),
			unstructured: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "ConfigMap",
					"metadata": map[string]interface{}{
						"namespace": "foo",
						"name":      "bar",
					},
				},
			},
			expectedObj: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "bar",
				},
			},
			expectedError: false,
		},
		{
			name: "invalid: unknown GVR cannot be decoded",
			action: clienttesting.NewGetAction(
				schema.GroupVersionResource{
					Group:    "foo",
					Version:  "bar",
					Resource: "baz",
				},
				"some-ns",
				"baz01",
			),
			unstructured: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "foo/bar",
					"kind":       "baz",
					"metadata": map[string]interface{}{
						"namespace": "some-ns",
						"name":      "baz01",
					},
				},
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewDryRun().WithDefaultMarshalFunction().WithWriter(io.Discard)
			obj, err := d.decodeUnstructuredIntoAPIObject(tc.action, tc.unstructured)
			if (err != nil) != tc.expectedError {
				t.Errorf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if diff := cmp.Diff(tc.expectedObj, obj); diff != "" {
				t.Errorf("object differs (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestFakeClientRoundTrip(t *testing.T) {
	ctx := context.Background()
	d := NewDryRun().WithWriter(io.Discard).WithDefaultMarshalFunction().
		WithServerVersion(&version.Info{GitVersion: "v1.99.0"})
	client := d.FakeClient()
	cms := client.CoreV1().ConfigMaps(metav1.NamespaceSystem)

	tests := []struct {
		name    string
		call    func() error
		wantErr func(error) bool
	}{
		{
			name: "server version",
			call: func() error {
				v, err := client.Discovery().ServerVersion()
				if err != nil {
					return err
				}
				if v.GitVersion != "v1.99.0" {
					return errors.Errorf("expected server version v1.99.0, got %s", v.GitVersion)
				}
				return nil
			},
		},
		{
			name: "create",
			call: func() error {
				_, err := cms.Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
					Name: "foo", Labels: map[string]string{"app": "foo"},
				}}, metav1.CreateOptions{})
				return err
			},
		},
		{
			name: "create existing",
			call: func() error {
				_, err := cms.Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}, metav1.CreateOptions{})
				return err
			},
			wantErr: apierrors.IsAlreadyExists,
		},
		{
			name: "update",
			call: func() error {
				_, err := cms.Update(ctx, &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "foo", Labels: map[string]string{"app": "foo"}},
					Data:       map[string]string{"k": "v"},
				}, metav1.UpdateOptions{})
				return err
			},
		},
		{
			name: "patch",
			call: func() error {
				_, err := cms.Patch(ctx, "foo", types.MergePatchType, []byte(`{"data":{"k2":"v2"}}`), metav1.PatchOptions{})
				return err
			},
		},
		{
			name: "get",
			call: func() error {
				cm, err := cms.Get(ctx, "foo", metav1.GetOptions{})
				if err != nil {
					return err
				}
				want := map[string]string{"k": "v", "k2": "v2"}
				if diff := cmp.Diff(want, cm.Data); diff != "" {
					return errors.Errorf("data differs (-want,+got):\n%s", diff)
				}
				return nil
			},
		},
		{
			name: "list with label selector",
			call: func() error {
				if _, err := cms.Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}, metav1.CreateOptions{}); err != nil {
					return err
				}
				list, err := cms.List(ctx, metav1.ListOptions{LabelSelector: "app=foo"})
				if err != nil {
					return err
				}
				if len(list.Items) != 1 || list.Items[0].Name != "foo" {
					return errors.Errorf("expected only configmap foo, got %v", list.Items)
				}
				return nil
			},
		},
		{
			name: "invalid label selector",
			call: func() error {
				_, err := cms.List(ctx, metav1.ListOptions{LabelSelector: "app in ("})
				return err
			},
			wantErr: apierrors.IsBadRequest,
		},
		{
			name: "delete",
			call: func() error {
				return cms.Delete(ctx, "foo", metav1.DeleteOptions{})
			},
		},
		{
			name: "get deleted",
			call: func() error {
				_, err := cms.Get(ctx, "foo", metav1.GetOptions{})
				return err
			},
			wantErr: apierrors.IsNotFound,
		},
		{
			name: "watch is not supported",
			call: func() error {
				_, err := cms.Watch(ctx, metav1.ListOptions{})
				return err
			},
			wantErr: apierrors.IsMethodNotSupported,
		},
	}
	// The cases share one store and run in order.
	for _, tc := range tests {
		err := tc.call()
		switch {
		case tc.wantErr == nil && err != nil:
			t.Fatalf("%s: unexpected error: %v", tc.name, err)
		case tc.wantErr != nil && !tc.wantErr(err):
			t.Fatalf("%s: unexpected error: %v", tc.name, err)
		}
	}
}
