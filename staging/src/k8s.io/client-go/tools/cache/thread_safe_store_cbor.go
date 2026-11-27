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

package cache

import (
	"fmt"
	"reflect"

	"github.com/fxamacker/cbor/v2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/klog/v2"
)

// cborEncoded is the at-rest wrapper for an object that has been CBOR-encoded
// by the store.
type cborEncoded struct{ data []byte }

// cborDecMode decodes all CBOR maps (including nested ones) as
// map[string]interface{}, matching the shape that *unstructured.Unstructured
// field accessors expect.
var cborDecMode cbor.DecMode

func init() {
	dm, err := cbor.DecOptions{
		DefaultMapType: reflect.TypeOf(map[string]interface{}{}),
	}.DecMode()
	if err != nil {
		panic(fmt.Sprintf("cache: failed to create CBOR DecMode: %v", err))
	}
	cborDecMode = dm
}

// cborEncodeObj encodes obj to a cborEncoded blob when obj is a
// *unstructured.Unstructured. Any other type is returned unchanged so the
// store remains compatible with typed objects.
func cborEncodeObj(key string, obj interface{}) interface{} {
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return obj
	}
	b, err := cbor.Marshal(u.Object)
	if err != nil {
		klog.Warningf("ThreadSafeStore: CBOR encode failed for key %q, storing decoded: %v", key, err)
		return obj
	}
	return cborEncoded{data: b}
}

// cborDecodeObj decodes a cborEncoded blob back to *unstructured.Unstructured.
// Any other value is returned unchanged.
func cborDecodeObj(key string, stored interface{}) interface{} {
	enc, ok := stored.(cborEncoded)
	if !ok {
		return stored
	}
	var m map[string]interface{}
	if err := cborDecMode.Unmarshal(enc.data, &m); err != nil {
		klog.Warningf("ThreadSafeStore: CBOR decode failed for key %q, returning stored: %v", key, err)
		return stored
	}
	return &unstructured.Unstructured{Object: m}
}
