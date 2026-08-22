/*
Copyright 2026 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

const (
	testCAConfigMapNamespace = "kube-system"
	testCAConfigMapName      = "extension-apiserver-authentication"
	testCAConfigMapKey       = "requestheader-client-ca-file"
)

type countingListener struct {
	count int
}

func (l *countingListener) Enqueue() {
	l.count++
}

func testCAConfigMap(caBundle string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testCAConfigMapName,
			Namespace: testCAConfigMapNamespace,
		},
		Data: map[string]string{testCAConfigMapKey: caBundle},
	}
}

// setConfigMaps points the controller at a lister holding exactly the given configmaps,
// so passing none models the configmap having been deleted.
func setConfigMaps(c *ConfigMapCAController, configMaps ...*corev1.ConfigMap) error {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, cm := range configMaps {
		if err := indexer.Add(cm); err != nil {
			return err
		}
	}
	c.configmapLister = corev1listers.NewConfigMapLister(indexer)
	return nil
}

func newTestConfigMapCAController(t *testing.T) (*ConfigMapCAController, *countingListener) {
	t.Helper()

	c := &ConfigMapCAController{
		name:               "test::" + testCAConfigMapNamespace + "::" + testCAConfigMapName + "::" + testCAConfigMapKey,
		configmapNamespace: testCAConfigMapNamespace,
		configmapName:      testCAConfigMapName,
		configmapKey:       testCAConfigMapKey,
	}
	listener := &countingListener{}
	c.AddListener(listener)

	if err := setConfigMaps(c, testCAConfigMap(string(serverCert))); err != nil {
		t.Fatal(err)
	}
	if err := c.loadCABundle(); err != nil {
		t.Fatal(err)
	}
	if _, ok := c.VerifyOptions(); !ok {
		t.Fatal("expected verify options to be available after the initial load")
	}
	if listener.count != 1 {
		t.Fatalf("expected 1 notification for the initial load, got %d", listener.count)
	}

	return c, listener
}

func TestConfigMapCAControllerClearsCAOnConfigMapDeletion(t *testing.T) {
	c, listener := newTestConfigMapCAController(t)

	if err := setConfigMaps(c); err != nil {
		t.Fatal(err)
	}
	if err := c.loadCABundle(); err != nil {
		t.Fatalf("loadCABundle should swallow the not found error, got %v", err)
	}

	if _, ok := c.VerifyOptions(); ok {
		t.Error("expected verify options to be unavailable once the configmap is deleted")
	}
	if content := c.CurrentCABundleContent(); len(content) != 0 {
		t.Errorf("expected no ca bundle content once the configmap is deleted, got %d bytes", len(content))
	}
	// the serving certificate controller has to rebuild its client CA pool
	if listener.count != 2 {
		t.Errorf("expected the clear to notify listeners, got %d notifications", listener.count)
	}
}

func TestConfigMapCAControllerColdStartDoesNotClear(t *testing.T) {
	// RunOnce reads the lister before Run starts the informer, so the very first read always
	// reports NotFound even for a healthy configmap. That must not be recorded as a deletion.
	c := &ConfigMapCAController{
		name:               "test-cold-start",
		configmapNamespace: testCAConfigMapNamespace,
		configmapName:      testCAConfigMapName,
		configmapKey:       testCAConfigMapKey,
	}
	listener := &countingListener{}
	c.AddListener(listener)
	if err := setConfigMaps(c); err != nil {
		t.Fatal(err)
	}

	if err := c.loadCABundle(); err != nil {
		t.Fatal(err)
	}

	if listener.count != 0 {
		t.Errorf("expected no notification when nothing was ever loaded, got %d", listener.count)
	}
	if c.caBundle.Load() != nil {
		t.Error("expected no ca bundle to be stored when nothing was ever loaded")
	}
	if _, ok := c.VerifyOptions(); ok {
		t.Error("expected verify options to be unavailable before anything is loaded")
	}
}

func TestConfigMapCAControllerDoesNotRenotifyWhileCleared(t *testing.T) {
	c, listener := newTestConfigMapCAController(t)

	if err := setConfigMaps(c); err != nil {
		t.Fatal(err)
	}
	// the controller resyncs on a timer, so a deleted configmap is observed over and over
	for i := 0; i < 3; i++ {
		if err := c.loadCABundle(); err != nil {
			t.Fatal(err)
		}
	}

	if listener.count != 2 {
		t.Errorf("expected exactly one notification for the clear, got %d total notifications", listener.count)
	}
}

func TestConfigMapCAControllerKeepsCAWhenContentIsMissing(t *testing.T) {
	c, _ := newTestConfigMapCAController(t)

	// an empty key is not a deletion, so the last good bundle has to survive
	if err := setConfigMaps(c, testCAConfigMap("")); err != nil {
		t.Fatal(err)
	}
	if err := c.loadCABundle(); err == nil {
		t.Fatal("expected loadCABundle to report the missing ca bundle content")
	}

	if _, ok := c.VerifyOptions(); !ok {
		t.Error("expected the previous verify options to survive a configmap with no content")
	}
	if content := c.CurrentCABundleContent(); len(content) == 0 {
		t.Error("expected the previous ca bundle content to survive a configmap with no content")
	}
}

func TestConfigMapCAControllerReloadsAfterRecreation(t *testing.T) {
	c, listener := newTestConfigMapCAController(t)

	if err := setConfigMaps(c); err != nil {
		t.Fatal(err)
	}
	if err := c.loadCABundle(); err != nil {
		t.Fatal(err)
	}

	if err := setConfigMaps(c, testCAConfigMap(string(serverCert))); err != nil {
		t.Fatal(err)
	}
	if err := c.loadCABundle(); err != nil {
		t.Fatal(err)
	}

	if _, ok := c.VerifyOptions(); !ok {
		t.Error("expected verify options to be available again after the configmap is recreated")
	}
	if listener.count != 3 {
		t.Errorf("expected the reload to notify listeners, got %d notifications", listener.count)
	}
}
