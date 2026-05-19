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

package cbor

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math/rand"
	"slices"
	"sort"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/klog/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"

	"github.com/fxamacker/cbor/v2"
)

func streamEncodeCollections(obj runtime.Object, w io.Writer, mode modes.EncMode) (bool, error) {
	list, ok := obj.(*unstructured.UnstructuredList)
	if ok {
		return true, streamingEncodeUnstructuredList(w, list, mode)
	}
	if _, ok := obj.(cbor.Marshaler); ok {
		return false, nil
	}
	if _, ok := obj.(json.Marshaler); ok {
		return false, nil
	}
	typeMeta, listMeta, items, err := getListMeta(obj)
	if err == nil {
		return true, streamingEncodeList(w, typeMeta, listMeta, items, mode)
	}
	klog.ErrorS(err, "getListMeta err", "obj", obj)
	return false, nil
}

// getListMeta implements list extraction logic for cbor stream serialization.
func getListMeta(list runtime.Object) (metav1.TypeMeta, metav1.ListMeta, []runtime.Object, error) {
	listValue, err := conversion.EnforcePtr(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	listType := listValue.Type()
	if listType.NumField() != 3 {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListType to have 3 fields")
	}
	// TypeMeta
	typeMeta, ok := listValue.Field(0).Interface().(metav1.TypeMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field to have TypeMeta type")
	}
	if !listType.Field(0).Anonymous {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag to be embedded`)
	}
	if jsonTag, jsonTagExists := listType.Field(0).Tag.Lookup("json"); !jsonTagExists {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag`)
	} else if jsonTag != "" && jsonTag != ",inline" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag to be "" or ",inline"`)
	}
	// ListMeta
	listMeta, ok := listValue.Field(1).Interface().(metav1.ListMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field to have ListMeta type")
	}
	if listType.Field(1).Tag.Get("json") != "metadata,omitempty" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected ListMeta json field tag to be "metadata,omitempty"`)
	}
	// Items
	items, err := meta.ExtractList(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	if listType.Field(2).Tag.Get("json") != "items" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected Items json field tag to be "items"`)
	}
	return typeMeta, listMeta, items, nil
}

type cborMapEntry struct {
	key   string
	write func() error
}

func streamingEncodeList(w io.Writer, typeMeta metav1.TypeMeta, listMeta metav1.ListMeta, items []runtime.Object, mode modes.EncMode) error {
	var entries []cborMapEntry

	if typeMeta.Kind != "" {
		entries = append(entries, cborMapEntry{
			key: "kind",
			write: func() error {
				return encodeKeyValuePair(w, "kind", typeMeta.Kind, mode)
			},
		})
	}
	entries = append(entries, cborMapEntry{
		key: "items",
		write: func() error {
			if err := mode.MarshalTo("items", w); err != nil {
				return err
			}
			if items == nil {
				_, err := w.Write([]byte{0xf6}) // CBOR null
				return err
			}
			if err := writeArrayHead(w, len(items)); err != nil {
				return err
			}
			for _, item := range items {
				if err := mode.MarshalTo(item, w); err != nil {
					return err
				}
			}
			return nil
		},
	})
	entries = append(entries, cborMapEntry{
		key: "metadata",
		write: func() error {
			return encodeKeyValuePair(w, "metadata", listMeta, mode)
		},
	})
	if typeMeta.APIVersion != "" {
		entries = append(entries, cborMapEntry{
			key: "apiVersion",
			write: func() error {
				return encodeKeyValuePair(w, "apiVersion", typeMeta.APIVersion, mode)
			},
		})
	}

	// For nondeterministic modes (SortFastShuffle), randomize the initial offset
	// of the encoding for-loop.
	start := 0
	if !mode.IsDeterministic() && len(entries) > 0 {
		start = rand.Intn(len(entries))
	}

	if err := writeMapHead(w, len(entries)); err != nil {
		return err
	}

	for i := 0; i < len(entries); i++ {
		entry := entries[(start+i)%len(entries)]
		if err := entry.write(); err != nil {
			return err
		}
	}
	return nil
}

func streamingEncodeUnstructuredList(w io.Writer, list *unstructured.UnstructuredList, mode modes.EncMode) error {
	keys := slices.Collect(maps.Keys(list.Object))
	if _, exists := list.Object["items"]; !exists {
		keys = append(keys, "items")
	}
	// Sort keys only for deterministic modes (SortBytewiseLexical):
	// shorter lengths come first, then lexicographic by content.
	// For nondeterministic modes (SortFastShuffle), randomize the initial offset
	// of the encoding for-loop (essentially what SortFastShuffle does for structs).
	start := 0
	if mode.IsDeterministic() {
		sort.Slice(keys, func(i, j int) bool {
			if len(keys[i]) != len(keys[j]) {
				return len(keys[i]) < len(keys[j])
			}
			return keys[i] < keys[j]
		})
	}

	if err := writeMapHead(w, len(keys)); err != nil {
		return err
	}

	for i := 0; i < len(keys); i++ {
		key := keys[(start+i)%len(keys)]
		if err := mode.MarshalTo(key, w); err != nil {
			return err
		}
		if key == "items" {
			if err := writeArrayHead(w, len(list.Items)); err != nil {
				return err
			}
			for _, item := range list.Items {
				if err := mode.MarshalTo(item.Object, w); err != nil {
					return err
				}
			}
		} else {
			if err := mode.MarshalTo(list.Object[key], w); err != nil {
				return err
			}
		}
	}
	return nil
}

func encodeKeyValuePair(w io.Writer, key string, value interface{}, mode modes.EncMode) error {
	if err := mode.MarshalTo(key, w); err != nil {
		return err
	}
	if err := mode.MarshalTo(value, w); err != nil {
		return err
	}
	return nil
}

// writeMapHead writes a CBOR map header for a map with n entries.
// Uses major type 5 (0xa0 base), following RFC 8949 Section 3.1.
func writeMapHead(w io.Writer, n int) error {
	return writeCollectionHead(w, 0xa0, int64(n))
}

// writeArrayHead writes a CBOR array header for an array with n elements.
// Uses major type 4 (0x80 base), following RFC 8949 Section 3.1.
func writeArrayHead(w io.Writer, n int) error {
	return writeCollectionHead(w, 0x80, int64(n))
}

// writeCollectionHead writes a CBOR collection (array or map) header encoding
// the number of elements n, following RFC 8949 Section 3 additional info rules:
//
//   - base: the prefix byte for the collection type.
//     For maps: 0xa0 (major type 5), for arrays: 0x80 (major type 4).
//
// The extended form prefixes are derived from base using bitwise OR:
//   - base|24: 1-byte length follows (additional info 24)
//   - base|25: 2-byte length follows (additional info 25)
//   - base|26: 4-byte length follows (additional info 26)
//   - base|27: 8-byte length follows (additional info 27)
//
// Encoding table (map example, base=0xa0):
//
//	n <= 23:         1 byte  — 0xa0|n
//	n <= 0xFF:        2 bytes — 0xb8 (0xa0|24), n
//	n <= 0xFFFF:      3 bytes — 0xb9 (0xa0|25), n>>8, n
//	n <= 0xFFFFFFFF:  5 bytes — 0xba (0xa0|26), n>>24..n
//	n > 0xFFFFFFFF:   9 bytes — 0xbb (0xa0|27), n>>56..n
func writeCollectionHead(w io.Writer, base byte, n int64) error {
	switch {
	case n <= 23:
		// Additional info 0–23: length is encoded directly in the low 5 bits.
		_, err := w.Write([]byte{base + byte(n)})
		return err
	case n <= 0xFF:
		// Additional info 24: one additional byte carries the length.
		_, err := w.Write([]byte{base | 24, byte(n)})
		return err
	case n <= 0xFFFF:
		// Additional info 25: two additional bytes carry the length (big-endian).
		_, err := w.Write([]byte{base | 25, byte(n >> 8), byte(n)})
		return err
	case n <= 0xFFFFFFFF:
		// Additional info 26: four additional bytes carry the length (big-endian).
		_, err := w.Write([]byte{base | 26, byte(n >> 24), byte(n >> 16), byte(n >> 8), byte(n)})
		return err
	default:
		// Additional info 27: eight additional bytes carry the length (big-endian).
		_, err := w.Write([]byte{
			base | 27, byte(n >> 56), byte(n >> 48), byte(n >> 40), byte(n >> 32), byte(n >> 24), byte(n >> 16),
			byte(n >> 8), byte(n),
		})
		return err
	}
}
