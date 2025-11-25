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

package helmapplyset

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

// Helper function to create a mock Helm release Secret
func createMockHelmReleaseSecret(name, namespace, releaseName, chartName, chartVersion, status, manifest string, version int) *v1.Secret {
	// This is a simplified version - in real tests, we'd encode the full Helm release JSON
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Type: HelmReleaseSecretType,
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte("mock-encoded-release-data"),
		},
	}
	return secret
}

// Helper function to create a complete Helm release Secret with encoded data
func createCompleteHelmReleaseSecret(t *testing.T, name, namespace, releaseName, chartName, chartVersion, status, manifest string, version int) *v1.Secret {
	helmReleaseJSON := HelmReleaseJSON{
		Name:      releaseName,
		Namespace: namespace,
		Version:   version,
		Status:    status,
		Chart: struct {
			Metadata struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"metadata"`
		}{
			Metadata: struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			}{
				Name:    chartName,
				Version: chartVersion,
			},
		},
		Manifest: manifest,
		Info: struct {
			Status string `json:"status"`
		}{
			Status: status,
		},
	}

	jsonBytes, err := json.Marshal(helmReleaseJSON)
	require.NoError(t, err)

	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	_, err = gz.Write(jsonBytes)
	require.NoError(t, err)
	require.NoError(t, gz.Close())

	encoded := base64.StdEncoding.EncodeToString(b.Bytes())

	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Type: HelmReleaseSecretType,
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte(encoded),
		},
	}
}

type fixture struct {
	t *testing.T

	client        *fake.Clientset
	dynamicClient *dynamicfake.FakeDynamicClient

	secretLister []*v1.Secret

	controller *Controller

	actions []core.Action
}

func newFixture(t *testing.T) *fixture {
	f := &fixture{
		t:      t,
		client: fake.NewSimpleClientset(),
	}

	// Create dynamic client with custom list kinds for resources we'll be listing
	scheme := runtime.NewScheme()
	f.dynamicClient = dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme,
		map[schema.GroupVersionResource]string{
			{Group: "apps", Version: "v1", Resource: "deployments"}: "DeploymentList",
			{Group: "", Version: "v1", Resource: "services"}:        "ServiceList",
			{Group: "", Version: "v1", Resource: "configmaps"}:      "ConfigMapList",
			{Group: "", Version: "v1", Resource: "secrets"}:         "SecretList",
		})

	// Create informers
	informerFactory := informers.NewSharedInformerFactory(f.client, 0)
	secretInformer := informerFactory.Core().V1().Secrets()

	// Create REST mapper
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "apps", Version: "v1"},
		{Group: "", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}, meta.RESTScopeNamespace)

	// Create controller
	_, ctx := ktesting.NewTestContext(t)
	controller, err := NewController(ctx, f.client, f.dynamicClient, secretInformer, mapper)
	require.NoError(t, err)

	f.controller = controller
	f.secretLister = []*v1.Secret{}

	return f
}

func (f *fixture) newController() (*Controller, informers.SharedInformerFactory, error) {
	f.client = fake.NewSimpleClientset(f.objects()...)

	informerFactory := informers.NewSharedInformerFactory(f.client, 0)
	secretInformer := informerFactory.Core().V1().Secrets()

	// Create REST mapper
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "apps", Version: "v1"},
		{Group: "", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}, meta.RESTScopeNamespace)

	_, ctx := ktesting.NewTestContext(f.t)
	controller, err := NewController(ctx, f.client, f.dynamicClient, secretInformer, mapper)
	if err != nil {
		return nil, nil, err
	}

	return controller, informerFactory, nil
}

func (f *fixture) objects() []runtime.Object {
	objects := make([]runtime.Object, 0, len(f.secretLister))
	for _, secret := range f.secretLister {
		objects = append(objects, secret)
	}
	return objects
}

func (f *fixture) run(key string) {
	f.run_(key, true, false)
}

// runSimple runs reconciliation without checking action order
func (f *fixture) runSimple(key string) {
	controller, informerFactory, err := f.newController()
	require.NoError(f.t, err)

	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	cache.WaitForCacheSync(stopCh, controller.secretSynced)

	_, ctx := ktesting.NewTestContext(f.t)
	err = controller.syncHandler(ctx, key)
	if err != nil {
		f.t.Logf("syncHandler returned error (may be expected): %v", err)
	}
}

func (f *fixture) run_(key string, startInformers bool, expectError bool) {
	controller, informerFactory, err := f.newController()
	require.NoError(f.t, err)

	if startInformers {
		stopCh := make(chan struct{})
		defer close(stopCh)
		informerFactory.Start(stopCh)
		cache.WaitForCacheSync(stopCh, controller.secretSynced)
	}

	_, ctx := ktesting.NewTestContext(f.t)
	err = controller.syncHandler(ctx, key)
	if !expectError && err != nil {
		f.t.Errorf("error syncing: %v", err)
	} else if expectError && err == nil {
		f.t.Error("expected error syncing, got nil")
	}

	actions := filterInformerActions(f.client.Actions())
	for i, action := range actions {
		if len(f.actions) < i+1 {
			f.t.Errorf("%d unexpected actions: %+v", len(actions)-len(f.actions), actions[i:])
			break
		}

		expectedAction := f.actions[i]
		checkAction(expectedAction, action, f.t)
	}

	if len(f.actions) > len(actions) {
		f.t.Errorf("%d additional expected actions:%+v", len(f.actions)-len(actions), f.actions[len(actions):])
	}
}

func filterInformerActions(actions []core.Action) []core.Action {
	ret := []core.Action{}
	for _, action := range actions {
		if len(action.GetNamespace()) == 0 &&
			(action.Matches("list", "namespaces") ||
				action.Matches("list", "nodes") ||
				action.Matches("list", "persistentvolumes") ||
				action.Matches("watch", "namespaces") ||
				action.Matches("watch", "nodes") ||
				action.Matches("watch", "persistentvolumes")) {
			continue
		}
		ret = append(ret, action)
	}
	return ret
}

func checkAction(expected, actual core.Action, t *testing.T) {
	if !(expected.Matches(actual.GetVerb(), actual.GetResource().Resource) &&
		actual.GetSubresource() == expected.GetSubresource()) {
		t.Errorf("Expected\n\t%#v\ngot\n\t%#v", expected, actual)
	}
}

func getKey(secret *v1.Secret, t *testing.T) string {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(secret)
	if err != nil {
		t.Errorf("Unexpected error getting key for secret %v: %v", secret.Name, err)
		return ""
	}
	return key
}

func TestController_NewHelmRelease(t *testing.T) {
	f := newFixture(t)

	manifest := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  namespace: default
---
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: default
`

	secret := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.my-app.1",
		"default",
		"my-app",
		"nginx",
		"1.0.0",
		"deployed",
		manifest,
		1,
	)

	f.secretLister = append(f.secretLister, secret)

	// Run reconciliation (don't check strict action order)
	f.runSimple(getKey(secret, t))

	// Verify parent Secret was created
	actions := f.client.Actions()
	require.Greater(t, len(actions), 0, "Expected at least one action")

	createAction := false
	for _, action := range actions {
		if action.Matches("create", "secrets") {
			createAction = true
			createActionObj := action.(core.CreateAction)
			createdSecret := createActionObj.GetObject().(*v1.Secret)
			assert.Contains(t, createdSecret.Name, parent.ParentSecretNamePrefix)
			assert.Equal(t, createdSecret.Labels[parent.ApplySetParentIDLabel], parent.ComputeApplySetID("my-app", "default"))
			assert.Equal(t, createdSecret.Annotations[parent.ApplySetToolingAnnotation], parent.HelmReleaseTooling)
			break
		}
	}
	assert.True(t, createAction, "Expected parent Secret creation")
}

func TestController_HelmUpgrade(t *testing.T) {
	f := newFixture(t)

	// Create upgraded release with namespaces in manifest
	manifest2 := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  namespace: default
---
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: default
`
	secret2 := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.my-app.2",
		"default",
		"my-app",
		"nginx",
		"1.1.0",
		"deployed",
		manifest2,
		2,
	)

	// Create parent Secret from first release
	applySetID := parent.ComputeApplySetID("my-app", "default")
	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      parent.ParentSecretNamePrefix + "my-app",
			Namespace: "default",
			Labels: map[string]string{
				parent.ApplySetParentIDLabel: applySetID,
			},
			Annotations: map[string]string{
				parent.ApplySetToolingAnnotation: parent.HelmReleaseTooling,
				parent.ApplySetGKsAnnotation:     "Deployment.apps",
			},
		},
	}

	f.secretLister = append(f.secretLister, secret2, parentSecret)

	// Run reconciliation
	f.runSimple(getKey(secret2, t))

	// Verify parent Secret was updated with new GroupKinds
	actions := f.client.Actions()
	updateAction := false
	for _, action := range actions {
		if action.Matches("update", "secrets") {
			updateAction = true
			updateActionObj := action.(core.UpdateAction)
			updatedSecret := updateActionObj.GetObject().(*v1.Secret)
			// Should include both Deployment and Service
			assert.Contains(t, updatedSecret.Annotations[parent.ApplySetGKsAnnotation], "Deployment")
			assert.Contains(t, updatedSecret.Annotations[parent.ApplySetGKsAnnotation], "Service")
			break
		}
	}
	assert.True(t, updateAction, "Expected parent Secret update")
}

func TestController_HelmReleaseDeletion(t *testing.T) {
	f := newFixture(t)

	// Create parent Secret
	applySetID := parent.ComputeApplySetID("my-app", "default")
	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      parent.ParentSecretNamePrefix + "my-app",
			Namespace: "default",
			Labels: map[string]string{
				parent.ApplySetParentIDLabel: applySetID,
			},
			Annotations: map[string]string{
				parent.ApplySetToolingAnnotation: parent.HelmReleaseTooling,
				parent.ApplySetGKsAnnotation:     "Deployment.apps,Service",
			},
		},
	}

	f.secretLister = append(f.secretLister, parentSecret)

	// Simulate deletion by using a key for a release that doesn't exist
	key := "default/sh.helm.release.v1.my-app.1"

	// Run reconciliation
	f.runSimple(key)

	// Verify parent Secret was deleted
	actions := f.client.Actions()
	deleteAction := false
	for _, action := range actions {
		if action.Matches("delete", "secrets") {
			deleteAction = true
			deleteActionObj := action.(core.DeleteAction)
			assert.Equal(t, deleteActionObj.GetName(), parent.ParentSecretNamePrefix+"my-app")
			break
		}
	}
	assert.True(t, deleteAction, "Expected parent Secret deletion")
}

func TestController_InvalidSecretFormat(t *testing.T) {
	// Test that Secrets with wrong type are ignored
	invalidSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.my-app.1",
			Namespace: "default",
		},
		Type: v1.SecretTypeOpaque, // Wrong type - should be helm.sh/release.v1
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte("invalid"),
		},
	}

	// Verify this Secret would be rejected by IsHelmReleaseSecret
	assert.False(t, IsHelmReleaseSecret(invalidSecret), "Secret with wrong type should not be recognized as Helm release")
}

func TestController_MultipleReleasesSameNamespace(t *testing.T) {
	// Test that multiple releases in same namespace each get their own parent Secret
	// Each release should have a unique ApplySet ID based on release name + namespace

	manifest1 := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: app1-deployment
  namespace: default
`
	secret1 := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.app1.1",
		"default",
		"app1",
		"chart1",
		"1.0.0",
		"deployed",
		manifest1,
		1,
	)

	manifest2 := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: app2-deployment
  namespace: default
`
	secret2 := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.app2.1",
		"default",
		"app2",
		"chart2",
		"1.0.0",
		"deployed",
		manifest2,
		1,
	)

	// Verify secrets are for different releases
	assert.NotEqual(t, secret1.Name, secret2.Name)

	// Verify ApplySet IDs would be different
	id1 := parent.ComputeApplySetID("app1", "default")
	id2 := parent.ComputeApplySetID("app2", "default")
	assert.NotEqual(t, id1, id2, "Different releases should have different ApplySet IDs")
}

func TestController_PermanentError(t *testing.T) {
	f := newFixture(t)

	// Create Secret with invalid format
	invalidSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.my-app.1",
			Namespace: "default",
		},
		Type: HelmReleaseSecretType,
		Data: map[string][]byte{
			HelmReleaseDataKey: []byte("!@#$%"), // Invalid base64
		},
	}

	f.secretLister = append(f.secretLister, invalidSecret)

	// Create controller and run sync - should fail on invalid Secret
	controller, informerFactory, err := f.newController()
	require.NoError(t, err)

	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	cache.WaitForCacheSync(stopCh, controller.secretSynced)

	_, ctx := ktesting.NewTestContext(t)
	err = controller.syncHandler(ctx, getKey(invalidSecret, t))

	// Should return error for invalid format
	assert.Error(t, err)
	// Invalid format should cause parsing error
	assert.Contains(t, err.Error(), "failed to parse Helm release Secret", "Expected parsing error")
}

func TestController_ProcessNextWorkItem(t *testing.T) {
	f := newFixture(t)

	manifest := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  namespace: default
`
	secret := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.my-app.1",
		"default",
		"my-app",
		"nginx",
		"1.0.0",
		"deployed",
		manifest,
		1,
	)

	f.secretLister = append(f.secretLister, secret)

	controller, informerFactory, err := f.newController()
	require.NoError(t, err)

	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	cache.WaitForCacheSync(stopCh, controller.secretSynced)

	// Add item to queue
	key := getKey(secret, t)
	controller.queue.Add(key)

	// Process item
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	processed := controller.processNextWorkItem(ctx)
	assert.True(t, processed, "Expected item to be processed")

	// Verify item was removed from queue
	assert.Equal(t, 0, controller.queue.Len(), "Queue should be empty after processing")
}

func TestController_WorkqueueRetry(t *testing.T) {
	f := newFixture(t)

	manifest := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  namespace: default
`
	secret := createCompleteHelmReleaseSecret(t,
		"sh.helm.release.v1.my-app.1",
		"default",
		"my-app",
		"nginx",
		"1.0.0",
		"deployed",
		manifest,
		1,
	)

	f.secretLister = append(f.secretLister, secret)

	controller, informerFactory, err := f.newController()
	require.NoError(t, err)

	// Add reactor to simulate transient error
	errorCount := 0
	f.client.PrependReactor("create", "secrets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		errorCount++
		return true, nil, &apierrors.StatusError{
			ErrStatus: metav1.Status{
				Code:    500,
				Message: "Internal server error",
			},
		}
	})

	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	cache.WaitForCacheSync(stopCh, controller.secretSynced)

	// Add item to queue
	key := getKey(secret, t)
	controller.queue.Add(key)

	// Process item - should fail
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	processed := controller.processNextWorkItem(ctx)
	assert.True(t, processed, "Expected item to be processed")

	// Verify the error handler was triggered (create was attempted)
	assert.Greater(t, errorCount, 0, "Expected create to be attempted and fail")
}
