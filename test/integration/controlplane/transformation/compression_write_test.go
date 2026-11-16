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

package transformation

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/integration/framework"
)

// configMapsGVR is where the write-path tests store their objects.
var configMapsGVR = gvr("", "v1", "configmaps")

// storedForm is how a raw etcd value is laid out. It's a string so a failed
// check says what it found.
type storedForm string

const (
	formUnframed     storedForm = "unframed protobuf"
	formUnframedJSON storedForm = "unframed JSON"
	formDeflate      storedForm = "deflate frame"
	formStoredFrame  storedForm = "stored frame"
)

func formOf(raw []byte) storedForm {
	switch {
	case len(raw) >= frameHeaderSize && bytes.HasPrefix(raw, []byte(frameMagic)) && raw[4] == algDeflate:
		return formDeflate
	case len(raw) >= frameHeaderSize && bytes.HasPrefix(raw, []byte(frameMagic)) && raw[4] == algStored:
		return formStoredFrame
	case bytes.HasPrefix(raw, []byte(protobufPrefix)):
		return formUnframed
	case bytes.HasPrefix(raw, []byte("{")):
		// Custom resources are stored as JSON.
		return formUnframedJSON
	}
	return storedForm(fmt.Sprintf("unrecognized value %q", raw[:min(len(raw), 16)]))
}

// compressionConfig returns a StorageCompressionConfiguration file made of
// entries from compressionEntry.
func compressionConfig(entries ...string) string {
	return "apiVersion: apiserver.config.k8s.io/v1alpha1\nkind: StorageCompressionConfiguration\ncompressionResources:\n" + strings.Join(entries, "")
}

// compressionEntry returns one compressionResources entry. An empty minSize
// leaves the field out, so the 1Ki default applies.
func compressionEntry(resource, algorithm, minSize string) string {
	entry := fmt.Sprintf("- resources: [%q]\n  algorithm: %s\n", resource, algorithm)
	if minSize != "" {
		entry += "  minSize: " + minSize + "\n"
	}
	return entry
}

// compressionFlags writes config to a file and returns the flags that load it.
// A file that compresses anything needs a min compatibility version of 1.38 or
// later to start.
func compressionFlags(t *testing.T, config, minCompatibilityVersion string) []string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "storage-compression.yaml")
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	return []string{"--storage-compression-config=" + path, "--min-compatibility-version=" + minCompatibilityVersion}
}

// compressibleText returns n bytes of repeated text, which DEFLATE squeezes
// almost all of the way out.
func compressibleText(n int) string {
	const unit = "compress me "
	return strings.Repeat(unit, n/len(unit)+1)[:n]
}

// compressibleConfigMap holds 2400 bytes of repeated text, well above the
// default 1Ki floor.
func compressibleConfigMap(name string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Data:       map[string]string{"data": compressibleText(2400)},
	}
}

// incompressibleConfigMap holds 4KiB of random bytes, which DEFLATE can't
// shrink. The seed is fixed so a failure reproduces.
func incompressibleConfigMap(name string) *corev1.ConfigMap {
	blob := make([]byte, 4096)
	_, _ = rand.NewChaCha8([32]byte{}).Read(blob) // ChaCha8's Read never fails.
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		BinaryData: map[string][]byte{"blob": blob},
	}
}

func (e *transformTest) mustCreateConfigMap(t *testing.T, cm *corev1.ConfigMap) {
	t.Helper()
	if _, err := e.restClient.CoreV1().ConfigMaps(testNamespace).Create(testContext(t), cm, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create configmap %s: %v", cm.Name, err)
	}
}

// wantForm reads the raw etcd value of object name in testNamespace and checks
// that it's stored as want.
func (e *transformTest) wantForm(t *testing.T, gvr schema.GroupVersionResource, name string, want storedForm) {
	t.Helper()
	if got := formOf(e.rawValue(t, gvr.GroupResource(), testNamespace, name)); got != want {
		t.Errorf("%s %s is stored as %s, want %s", gvr.Resource, name, got, want)
	}
}

// wantConfigMapContent reads want back through the apiserver and checks that it
// still holds what the client sent.
func (e *transformTest) wantConfigMapContent(t *testing.T, want *corev1.ConfigMap) {
	t.Helper()
	got, err := e.restClient.CoreV1().ConfigMaps(testNamespace).Get(testContext(t), want.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to read configmap %s: %v", want.Name, err)
	}
	if !reflect.DeepEqual(got.Data, want.Data) || !reflect.DeepEqual(got.BinaryData, want.BinaryData) {
		t.Errorf("configmap %s read back with different data", want.Name)
	}
}

// noOpUpdate puts object name in testNamespace back unchanged and reports
// whether its resource version moved. The store skips writing an unchanged
// object unless its read reported the stored value stale.
func (e *transformTest) noOpUpdate(t *testing.T, gvr schema.GroupVersionResource, name string) bool {
	t.Helper()
	ctx := testContext(t)
	objects := dynamic.NewForConfigOrDie(e.kubeAPIServer.ClientConfig).Resource(gvr).Namespace(testNamespace)
	obj, err := objects.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to read %s %s: %v", gvr.Resource, name, err)
	}
	updated, err := objects.Update(ctx, obj, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to update %s %s: %v", gvr.Resource, name, err)
	}
	return updated.GetResourceVersion() != obj.GetResourceVersion()
}

// wantRewriteOnce makes two no-op updates of object name. The first has to
// rewrite it as want and the second has to leave it alone, since a migration
// that doesn't converge rewrites the value on every update.
func (e *transformTest) wantRewriteOnce(t *testing.T, gvr schema.GroupVersionResource, name string, want storedForm) {
	t.Helper()
	if !e.noOpUpdate(t, gvr, name) {
		t.Errorf("the first no-op update of %s %s kept its resource version, want a rewrite", gvr.Resource, name)
	}
	e.wantForm(t, gvr, name, want)
	if e.noOpUpdate(t, gvr, name) {
		t.Errorf("the second no-op update of %s %s changed its resource version again", gvr.Resource, name)
	}
}

// TestStorageCompressionWritePath stores configmaps under a policy that
// compresses them from the default 1Ki up, then checks each raw etcd value and
// reads the object back through the apiserver.
func TestStorageCompressionWritePath(t *testing.T) {
	config := compressionConfig(compressionEntry("configmaps", "Deflate", ""))
	tests := []struct {
		name      string
		configMap *corev1.ConfigMap
		want      storedForm
	}{
		{
			name:      "compressible value above the floor",
			configMap: compressibleConfigMap("compressible"),
			want:      formDeflate,
		},
		{
			name:      "incompressible value above the floor",
			configMap: incompressibleConfigMap("incompressible"),
			want:      formStoredFrame,
		},
		{
			name:      "value below the floor",
			configMap: &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "small"}, Data: map[string]string{configMapKey: configMapVal}},
			want:      formUnframed,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			test, err := newTransformTest(t, transformTestConfig{extraFlags: compressionFlags(t, config, "1.38")})
			if err != nil {
				t.Fatalf("failed to start kube-apiserver: %v", err)
			}
			defer test.cleanUp()
			test.mustCreateConfigMap(t, tc.configMap)
			raw := test.rawValue(t, configMapsGVR.GroupResource(), testNamespace, tc.configMap.Name)
			if got := formOf(raw); got != tc.want {
				t.Fatalf("stored as %s, want %s", got, tc.want)
			}
			switch tc.want {
			case formDeflate:
				// A frame smaller than the data it holds shows DEFLATE ran, not just the framing.
				if size := len(tc.configMap.Data["data"]); len(raw) >= size {
					t.Errorf("the frame is %d bytes, want fewer than the %d bytes of data in it", len(raw), size)
				}
			case formStoredFrame:
				// A stored frame's body is the serialized object, unchanged.
				if !bytes.HasPrefix(raw[frameHeaderSize:], []byte(protobufPrefix)) {
					t.Errorf("the stored frame's body starts %q, want the protobuf prefix", raw[frameHeaderSize:min(len(raw), frameHeaderSize+8)])
				}
			}
			test.wantConfigMapContent(t, tc.configMap)
		})
	}
}

// TestStorageCompressionPolicyChanges runs four apiservers in turn against the
// same etcd data: no configuration file, configmaps compressed, the file
// removed, and configmaps compressed again. A read never needs the policy that
// wrote the value, and a value changes form only when something rewrites it.
func TestStorageCompressionPolicyChanges(t *testing.T) {
	storageConfig := framework.SharedEtcd()
	compress := compressionConfig(compressionEntry("configmaps", "Deflate", ""))
	legacy := compressibleConfigMap("legacy")
	fresh := compressibleConfigMap("fresh")
	random := incompressibleConfigMap("random")
	all := []*corev1.ConfigMap{legacy, fresh, random}

	start := func(t *testing.T, flags []string) *transformTest {
		t.Helper()
		test, err := newTransformTest(t, transformTestConfig{storageConfig: storageConfig, extraFlags: flags})
		if err != nil {
			t.Fatalf("failed to start kube-apiserver: %v", err)
		}
		return test
	}

	// Each run checks what the one before it stored, so stop at the first failure.
	if !t.Run("no configuration file", func(t *testing.T) {
		test := start(t, nil)
		defer test.cleanUp()
		// A policy would frame legacy, so plain protobuf here means the write path is off.
		test.mustCreateConfigMap(t, legacy)
		test.wantForm(t, configMapsGVR, legacy.Name, formUnframed)
	}) {
		return
	}

	if !t.Run("compression enabled", func(t *testing.T) {
		test := start(t, compressionFlags(t, compress, "1.38"))
		defer test.cleanUp()
		// Turning the policy on doesn't touch legacy until something writes it.
		test.wantConfigMapContent(t, legacy)
		test.wantForm(t, configMapsGVR, legacy.Name, formUnframed)
		test.mustCreateConfigMap(t, fresh)
		test.mustCreateConfigMap(t, random)
		test.wantForm(t, configMapsGVR, fresh.Name, formDeflate)
		test.wantForm(t, configMapsGVR, random.Name, formStoredFrame)
		test.wantRewriteOnce(t, configMapsGVR, legacy.Name, formDeflate)
		test.wantConfigMapContent(t, legacy)
	}) {
		return
	}

	if !t.Run("configuration file removed", func(t *testing.T) {
		test := start(t, nil)
		defer test.cleanUp()
		// Frames still read with no policy at all, and reading them rewrites nothing.
		for _, cm := range all {
			test.wantConfigMapContent(t, cm)
		}
		test.wantForm(t, configMapsGVR, legacy.Name, formDeflate)
		test.wantForm(t, configMapsGVR, fresh.Name, formDeflate)
		test.wantForm(t, configMapsGVR, random.Name, formStoredFrame)
		test.wantRewriteOnce(t, configMapsGVR, fresh.Name, formUnframed)
		test.wantRewriteOnce(t, configMapsGVR, random.Name, formUnframed)
		for _, cm := range all {
			test.wantConfigMapContent(t, cm)
		}
		// Nothing has written legacy since the policy went away, so it's still a frame.
		test.wantForm(t, configMapsGVR, legacy.Name, formDeflate)
	}) {
		return
	}

	t.Run("compression enabled again", func(t *testing.T) {
		test := start(t, compressionFlags(t, compress, "1.38"))
		defer test.cleanUp()
		// legacy is already stored the way the policy writes it, so there's nothing to rewrite.
		test.wantForm(t, configMapsGVR, legacy.Name, formDeflate)
		if test.noOpUpdate(t, configMapsGVR, legacy.Name) {
			t.Errorf("a no-op update of configmap %s changed its resource version, but it was already a frame", legacy.Name)
		}
		test.wantForm(t, configMapsGVR, fresh.Name, formUnframed)
		test.wantRewriteOnce(t, configMapsGVR, fresh.Name, formDeflate)
		// DEFLATE can't shrink random, so it settles in a stored frame rather than
		// being rewritten on every update.
		test.wantForm(t, configMapsGVR, random.Name, formUnframed)
		test.wantRewriteOnce(t, configMapsGVR, random.Name, formStoredFrame)
		for _, cm := range all {
			test.wantConfigMapContent(t, cm)
		}
	})
}
