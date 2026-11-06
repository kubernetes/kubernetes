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

package factory

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"slices"
	"strings"
	"testing"
	"time"

	"go.etcd.io/etcd/client/v3/kubernetes"
	noopoteltrace "go.opentelemetry.io/otel/trace/noop"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

var (
	podsGR             = schema.GroupResource{Resource: "pods"}
	compressAll        = compression.Rule{Algorithm: compression.AlgorithmDeflate, MinSize: 1}
	compressionMetrics = []string{
		"apiserver_storage_compression_operations_total",
		"apiserver_storage_compression_duration_seconds",
		"apiserver_storage_compression_format_errors_total",
		"apiserver_storage_compression_unsupported_algorithm_total",
		"apiserver_storage_compression_rewrites_needed_total",
		"apiserver_storage_compression_inflation_wait_seconds",
	}
)

// newPodStorage builds a pods store the way kube-apiserver does. encryption is
// the transformer an EncryptionConfiguration would give pods, or nil when
// encryption at rest isn't configured.
func newPodStorage(t *testing.T, encryption value.Transformer) (storage.Interface, *kubernetes.Client) {
	t.Helper()
	client := testserver.RunEtcd(t)
	cfg := storagebackend.Config{
		Type: storagebackend.StorageTypeETCD3,
		Transport: storagebackend.TransportConfig{
			ServerList:     client.Endpoints(),
			TracerProvider: noopoteltrace.NewTracerProvider(),
		},
		Codec:       apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion),
		Transformer: encryption,
	}
	s, destroy, err := newETCD3Storage(*cfg.ForResource(podsGR),
		func() runtime.Object { return &example.Pod{} },
		func() runtime.Object { return &example.PodList{} },
		nil, "/pods")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(destroy)
	return s, client
}

func compressiblePod() *example.Pod {
	return &example.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace:   "ns",
		Name:        "foo",
		Annotations: map[string]string{"note": strings.Repeat("compress me ", 200)},
	}}
}

func encodePod(t *testing.T, pod *example.Pod) []byte {
	t.Helper()
	data, err := runtime.Encode(apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion), pod)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// compressedValue returns pod as an apiserver with compression turned on would
// store it at key, through encryption when it isn't nil.
func compressedValue(t *testing.T, encryption value.Transformer, key string, pod *example.Pod) []byte {
	t.Helper()
	if encryption == nil {
		encryption = identity.NewEncryptCheckTransformer()
	}
	stored, err := compression.WithCompression(encryption, compressAll, podsGR).
		TransformToStorage(context.Background(), encodePod(t, pod), value.DefaultContext(key))
	if err != nil {
		t.Fatal(err)
	}
	return stored
}

func put(t *testing.T, client *kubernetes.Client, key string, v []byte) {
	t.Helper()
	if _, err := client.KV.Put(context.Background(), key, string(v)); err != nil {
		t.Fatal(err)
	}
}

// compressionMetricsText returns the storage compression metrics in text format.
// They're process-wide, so a test compares them before and after rather than
// expecting none.
func compressionMetricsText(t *testing.T) string {
	t.Helper()
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, mf := range families {
		if slices.Contains(compressionMetrics, mf.GetName()) {
			if _, err := testutil.MetricFamilyToText(&b, mf); err != nil {
				t.Fatal(err)
			}
		}
	}
	return b.String()
}

// TestStoresWithoutARuleAreInert checks that a store built with no compression
// Rule stores the encoded object unchanged, and that neither the write nor
// reading the value back records a compression metric.
func TestStoresWithoutARuleAreInert(t *testing.T) {
	ctx := context.Background()
	s, client := newPodStorage(t, nil)
	before := compressionMetricsText(t)

	pod := compressiblePod()
	key := "/pods/ns/foo"
	if err := s.Create(ctx, key, pod, nil, 0); err != nil {
		t.Fatal(err)
	}
	resp, err := client.KV.Get(ctx, key)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Kvs) != 1 {
		t.Fatalf("got %d values at %s, want 1", len(resp.Kvs), key)
	}
	if want := encodePod(t, pod); !bytes.Equal(resp.Kvs[0].Value, want) {
		t.Errorf("stored value isn't the encoded object:\ngot  %q\nwant %q", resp.Kvs[0].Value, want)
	}
	if err := s.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); err != nil {
		t.Fatal(err)
	}

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(before), compressionMetrics...); err != nil {
		t.Errorf("a store with no Rule recorded compression metrics: %v", err)
	}
}

// TestStoresReadCompressedValues checks that every store reads a compressed
// value through Get, GetList and Watch without any configuration. With AES-GCM
// configured, the stored value is an encrypted frame, so reading it back shows
// that decompression runs after decryption.
func TestStoresReadCompressedValues(t *testing.T) {
	aes, _, err := aestransformer.NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}
	aesgcm := value.NewPrefixTransformers(nil, value.PrefixTransformer{Prefix: []byte("k8s:enc:aesgcm:v1:key1:"), Transformer: aes})

	for name, encryption := range map[string]value.Transformer{"no encryption": nil, "AES-GCM": aesgcm} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			s, client := newPodStorage(t, encryption)
			list := &example.PodList{}
			if err := s.GetList(ctx, "/pods/", storage.ListOptions{Predicate: storage.Everything, Recursive: true}, list); err != nil {
				t.Fatal(err)
			}
			w, err := s.Watch(ctx, "/pods/", storage.ListOptions{ResourceVersion: list.ResourceVersion, Predicate: storage.Everything, Recursive: true})
			if err != nil {
				t.Fatal(err)
			}
			defer w.Stop()

			pod := compressiblePod()
			key := "/pods/ns/foo"
			put(t, client, key, compressedValue(t, encryption, key, pod))
			want := pod.Annotations["note"]

			select {
			case e := <-w.ResultChan():
				if got, ok := e.Object.(*example.Pod); e.Type != watch.Added || !ok || got.Annotations["note"] != want {
					t.Errorf("Watch: got %s event %#v, want the pod added", e.Type, e.Object)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("Watch: no event for the compressed value")
			}
			got := &example.Pod{}
			if err := s.Get(ctx, key, storage.GetOptions{}, got); err != nil || got.Annotations["note"] != want {
				t.Errorf("Get: got %v, %#v", err, got)
			}
			list = &example.PodList{}
			if err := s.GetList(ctx, "/pods/", storage.ListOptions{Predicate: storage.Everything, Recursive: true}, list); err != nil ||
				len(list.Items) != 1 || list.Items[0].Annotations["note"] != want {
				t.Errorf("GetList: got %v, %#v", err, list.Items)
			}
		})
	}
}

// TestOnlyAFailedChecksumIsACorruptObject checks that a frame that fails its
// checksum is offered for unsafe deletion and an intact frame from a newer
// apiserver isn't. The checksum case fails if compression is installed outside
// the corrupt object wrapper.
func TestOnlyAFailedChecksumIsACorruptObject(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
	ctx := context.Background()
	s, client := newPodStorage(t, nil)
	unsafeDeletionStore := etcd3.NewStoreWithUnsafeCorruptObjectDeletion(s, podsGR)

	damaged := compressedValue(t, nil, "/pods/ns/damaged", compressiblePod())
	damaged[len(damaged)-1] ^= 0xff
	put(t, client, "/pods/ns/damaged", damaged)
	err := unsafeDeletionStore.Get(ctx, "/pods/ns/damaged", storage.GetOptions{}, &example.Pod{})
	if !storage.IsCorruptObject(err) {
		t.Errorf("a frame with a bad checksum: got %v, want a corrupt object error", err)
	}

	// No apiserver writes algorithm 0xfe, so the test builds the frame by hand from
	// the layout in compression's format.go: magic, algorithm, then a CRC32C of
	// every other byte, with an empty body.
	future := []byte{0x00, 'k', '8', 's', 0xfe, 0, 0, 0, 0}
	binary.LittleEndian.PutUint32(future[5:], crc32.Checksum(future[:5], crc32.MakeTable(crc32.Castagnoli)))
	put(t, client, "/pods/ns/future", future)
	err = unsafeDeletionStore.Get(ctx, "/pods/ns/future", storage.GetOptions{}, &example.Pod{})
	if storage.IsCorruptObject(err) || !storage.IsInternalError(err) {
		t.Errorf("a frame with an unknown algorithm: got %v, want an internal error that isn't a corrupt object", err)
	}
}
