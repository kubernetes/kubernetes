/*
Copyright 2019 The Kubernetes Authors.

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

package resource

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

// hold a single instance of the case-sensitive decoder
var caseSensitiveJsonIterator = json.CaseSensitiveJSONIterator()

// metadataValidatingDecoder wraps a decoder and additionally ensures metadata schema fields decode before returning an unstructured object
type metadataValidatingDecoder struct {
	decoder runtime.Decoder
}

func (m *metadataValidatingDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := m.decoder.Decode(data, defaults, into)

	// if we already errored, return
	if err != nil {
		return obj, gvk, err
	}

	// if we're not unstructured, return
	if _, isUnstructured := obj.(runtime.Unstructured); !isUnstructured {
		return obj, gvk, err
	}

	// make sure the data can decode into ObjectMeta before we return,
	// so we don't silently truncate schema errors in metadata later with accesser get/set calls
	v := &metadataOnlyObject{}
	if typedErr := caseSensitiveJsonIterator.Unmarshal(data, v); typedErr != nil {
		return obj, gvk, typedErr
	}
	return obj, gvk, err
}

type metadataOnlyObject struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
}
