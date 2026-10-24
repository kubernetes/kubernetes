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

package cacher

import (
	"context"
	"encoding/json"
	"fmt"
	goruntime "runtime"
	"runtime/debug"
	"testing"

	"go.etcd.io/etcd/server/v3/embed"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/utils/clock"
)

// sbomComponent and newSBOMReport build custom resources shaped like
// aquasecurity.github.io/v1alpha1 SbomReport: a long array of small objects
// with many short keys, which is the worst case for the decoded
// *unstructured.Unstructured form kept by the watch cache.
func sbomComponent(i int) map[string]interface{} {
	purl := fmt.Sprintf("pkg:deb/debian/libpkg-%06d@2.36-9+deb12u7?arch=amd64&distro=debian-12", i)
	return map[string]interface{}{
		"bom-ref": purl,
		"type":    "library",
		"name":    fmt.Sprintf("libpkg-%06d", i),
		"version": "2.36-9+deb12u7",
		"purl":    purl,
		"scope":   "required",
		"supplier": map[string]interface{}{
			"name": "Debian Security Team",
			"url":  []interface{}{"https://www.debian.org/security/"},
		},
		"licenses": []interface{}{
			map[string]interface{}{"license": map[string]interface{}{"name": "GPL-2.0-only"}},
			map[string]interface{}{"license": map[string]interface{}{"name": "LGPL-2.1-or-later"}},
		},
		"hashes": []interface{}{
			map[string]interface{}{"alg": "SHA-256", "content": fmt.Sprintf("%064x", i)},
			map[string]interface{}{"alg": "SHA-1", "content": fmt.Sprintf("%040x", i)},
		},
		"properties": []interface{}{
			map[string]interface{}{"name": "aquasecurity:trivy:PkgID", "value": fmt.Sprintf("libpkg-%06d@2.36-9", i)},
			map[string]interface{}{"name": "aquasecurity:trivy:PkgType", "value": "debian"},
			map[string]interface{}{"name": "aquasecurity:trivy:SrcName", "value": fmt.Sprintf("glibc-%06d", i)},
			map[string]interface{}{"name": "aquasecurity:trivy:SrcVersion", "value": "2.36-9"},
			map[string]interface{}{"name": "aquasecurity:trivy:LayerDigest", "value": fmt.Sprintf("sha256:%064x", i*7)},
		},
	}
}

func newSBOMReport(namespace, name string, components int) *unstructured.Unstructured {
	comps := make([]interface{}, 0, components)
	for i := 0; i < components; i++ {
		comps = append(comps, sbomComponent(i))
	}
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "aquasecurity.github.io/v1alpha1",
		"kind":       "SbomReport",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
			"labels": map[string]interface{}{
				"trivy-operator.resource.kind": "ReplicaSet",
				"trivy-operator.resource.name": name,
			},
		},
		"report": map[string]interface{}{
			"scanner": map[string]interface{}{"name": "Trivy", "vendor": "Aqua Security", "version": "0.58.1"},
			"components": map[string]interface{}{
				"bomFormat":   "CycloneDX",
				"specVersion": "1.5",
				"components":  comps,
			},
		},
	}}
}

var sbomGVK = schema.GroupVersionKind{Group: "aquasecurity.github.io", Version: "v1alpha1", Kind: "SbomReport"}

func newSBOMEtcdTestStorage(t *testing.T, resourcePrefix string) storage.Interface {
	server := &etcd3testing.EtcdTestServer{V3Client: testserver.RunEtcd(t, func(cfg *embed.Config) {
		cfg.QuotaBackendBytes = 8 << 30
	})}
	t.Cleanup(func() { server.Terminate(t) })
	versioner := storage.APIObjectVersioner{}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	codec := unstructured.UnstructuredJSONScheme
	s, err := etcd3.New(
		server.V3Client,
		compactor,
		codec,
		newSBOMReportObject,
		newSBOMReportList,
		etcd3testing.PathPrefix(),
		resourcePrefix,
		schema.GroupResource{Group: sbomGVK.Group, Resource: "sbomreports"},
		identity.NewEncryptCheckTransformer(),
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(codec, versioner),
		versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func newSBOMReportObject() runtime.Object {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(sbomGVK)
	return u
}

func newSBOMReportList() runtime.Object {
	u := &unstructured.UnstructuredList{}
	u.SetGroupVersionKind(sbomGVK.GroupVersion().WithKind(sbomGVK.Kind + "List"))
	return u
}

// seedSBOMReports stores SbomReports and returns their total size as JSON.
func seedSBOMReports(ctx context.Context, t *testing.T, s storage.Interface, resourcePrefix string, objects, components int) uint64 {
	t.Helper()
	var wireBytes uint64
	for i := 0; i < objects; i++ {
		obj := newSBOMReport("trivy-system", fmt.Sprintf("replicaset-app-%04d", i), components)
		raw, err := json.Marshal(obj.Object)
		if err != nil {
			t.Fatal(err)
		}
		wireBytes += uint64(len(raw))
		key := fmt.Sprintf("%strivy-system/%s", resourcePrefix, obj.GetName())
		if err := s.Create(ctx, key, obj, &unstructured.Unstructured{}, 0); err != nil {
			t.Fatalf("create %s: %v", key, err)
		}
	}
	return wireBytes
}

func sbomCacherConfig(s storage.Interface, resourcePrefix string, budget int64) Config {
	return Config{
		Storage:             s,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       schema.GroupResource{Group: sbomGVK.Group, Resource: "sbomreports"},
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      resourcePrefix,
		KeyFunc: func(obj runtime.Object) (string, error) {
			return storage.NamespaceKeyFunc(resourcePrefix, obj)
		},
		GetAttrsFunc: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			u, ok := obj.(*unstructured.Unstructured)
			if !ok {
				return nil, nil, fmt.Errorf("not unstructured: %T", obj)
			}
			return labels.Set(u.GetLabels()), fields.Set{"metadata.name": u.GetName(), "metadata.namespace": u.GetNamespace()}, nil
		},
		NewFunc:     newSBOMReportObject,
		NewListFunc: newSBOMReportList,
		Codec:       unstructured.UnstructuredJSONScheme,
		Clock:       clock.RealClock{},

		MaxAverageObjectSizeBytes: budget,
	}
}

// liveHeap returns the live heap in bytes, after a full garbage collection.
func liveHeap() uint64 {
	// Collect twice: objects held in sync.Pools survive the first collection
	// in the pools' victim caches.
	goruntime.GC()
	debug.FreeOSMemory()
	var m goruntime.MemStats
	goruntime.ReadMemStats(&m)
	return m.HeapAlloc
}

func heapGrowth(before uint64) uint64 {
	if after := liveHeap(); after > before {
		return after - before
	}
	return 0
}

func mib(b uint64) string { return fmt.Sprintf("%.1f MiB", float64(b)/(1<<20)) }

// TestObjectSizeBudgetMemory measures the heap retained by the watch cache of a
// resource with large custom resources, without and with an object size
// budget. It demonstrates both the problem -- a populated watch cache costs
// several times the objects' serialized size, with nothing watching -- and
// that the budget avoids it, whether the resource is over budget from the
// start or grows past it later.
func TestObjectSizeBudgetMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates hundreds of MiB")
	}
	const (
		objects    = 100
		components = 500 // ~550 KiB of JSON per object
		prefix     = "/sbomreports/"
		budget     = 100 * 1000
	)
	ctx := context.Background()

	t.Run("without budget", func(t *testing.T) {
		s := newSBOMEtcdTestStorage(t, prefix)
		wire := seedSBOMReports(ctx, t, s, prefix, objects, components)

		before := liveHeap()
		c, err := NewCacherFromConfig(sbomCacherConfig(s, prefix, 0))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.Stop)
		if err := c.Wait(ctx); err != nil {
			t.Fatal(err)
		}
		heap := heapGrowth(before)
		goruntime.KeepAlive(c)

		ratio := float64(heap) / float64(wire)
		t.Logf("%d objects, %s of JSON -> watch cache retains %s (%.2fx)", objects, mib(wire), mib(heap), ratio)
		if ratio < 3 {
			t.Errorf("expected the populated watch cache to retain more than 3x the serialized size, got %.2fx", ratio)
		}
	})

	t.Run("over budget before populating", func(t *testing.T) {
		s := newSBOMEtcdTestStorage(t, prefix)
		wire := seedSBOMReports(ctx, t, s, prefix, objects, components)

		before := liveHeap()
		c, err := NewCacherFromConfig(sbomCacherConfig(s, prefix, budget))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.Stop)
		waitForBypass(ctx, t, c)
		c.stopWg.Wait()
		heap := heapGrowth(before)
		goruntime.KeepAlive(c)

		ratio := float64(heap) / float64(wire)
		t.Logf("%d objects, %s of JSON -> bypassed watch cache retains %s (%.2fx)", objects, mib(wire), mib(heap), ratio)
		if ratio > 0.1 {
			t.Errorf("expected a bypassed watch cache to retain less than 0.1x the serialized size, got %.2fx", ratio)
		}
	})

	t.Run("over budget after populating", func(t *testing.T) {
		s := newSBOMEtcdTestStorage(t, prefix)
		wire := seedSBOMReports(ctx, t, s, prefix, objects, components)

		before := liveHeap()
		// Populate the watch cache first, without a budget, so that there is
		// no monitor racing the measurement; then apply the budget and run
		// the check the monitor would have run.
		c, err := NewCacherFromConfig(sbomCacherConfig(s, prefix, 0))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.Stop)
		if err := c.Wait(ctx); err != nil {
			t.Fatal(err)
		}
		populated := heapGrowth(before)

		c.maxAverageObjectSize = budget
		if !c.checkObjectSize() {
			t.Fatal("expected the check to take the resource out of the watch cache")
		}
		released := heapGrowth(before)
		goruntime.KeepAlive(c)

		t.Logf("%d objects, %s of JSON -> watch cache retains %s (%.2fx) populated, %s (%.2fx) after bypass",
			objects, mib(wire), mib(populated), float64(populated)/float64(wire), mib(released), float64(released)/float64(wire))
		if ratio := float64(populated) / float64(wire); ratio < 3 {
			t.Errorf("expected the populated watch cache to retain more than 3x the serialized size, got %.2fx", ratio)
		}
		if ratio := float64(released) / float64(wire); ratio > 0.25 {
			t.Errorf("expected the watch cache to retain less than 0.25x the serialized size after bypass, got %.2fx", ratio)
		}
	})
}
