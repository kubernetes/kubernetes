/*
Copyright 2015 The Kubernetes Authors.

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

package metaonly

import (
	gojson "encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/json"
)

// MetaOnly allows decoding only the apiVersion, kind, and metadata fields of
// JSON data.
// TODO: enable meta-only decoding for protobuf.
type MetaOnly struct {
	unversioned.TypeMeta `json:",inline"`
	v1.ObjectMeta        `json:"metadata,omitempty"`
}

// MetaOnlyList allows decoding from JSON data only the typemeta and metadata of
// a list, and those of the enclosing objects.
// TODO: enable meta-only decoding for protobuf.
type MetaOnlyList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty"`

	Items []MetaOnly `json:"items"`
}

func (obj *MetaOnly) GetObjectKind() unversioned.ObjectKind     { return obj }
func (obj *MetaOnlyList) GetObjectKind() unversioned.ObjectKind { return obj }

// MetaOnlyJSONScheme is capable of converting JSON data into the MetaOnly and
// MetaOnlyList type, which can be used for generic access to objects without a
// predefined scheme.
var MetaOnlyJSONScheme runtime.Codec = metaOnlyJSONScheme{}

type metaOnlyJSONScheme struct{}

func (s metaOnlyJSONScheme) Decode(data []byte, _ *unversioned.GroupVersionKind, obj runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	var err error
	if obj != nil {
		err = s.decodeInto(data, obj)
	} else {
		obj, err = s.decode(data)
	}

	if err != nil {
		return nil, nil, err
	}

	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, &gvk, runtime.NewMissingKindErr(string(data))
	}

	return obj, &gvk, nil
}

func (s metaOnlyJSONScheme) Encode(obj runtime.Object, w io.Writer) error {
	return fmt.Errorf("metaOnlyJSONScheme doesn't implement Encode")
}

func (metaOnlyJSONScheme) decode(data []byte) (runtime.Object, error) {
	type detector struct {
		Kind  string
		Items gojson.RawMessage
	}
	var det detector
	if err := json.Unmarshal(data, &det); err != nil {
		return nil, err
	}
	if det.Items != nil || (len(det.Kind) >= 4 && det.Kind[len(det.Kind)-4:] == "List") {
		list := &MetaOnlyList{}
		err := json.Unmarshal(data, list)
		// For typed lists, e.g., a PodList, API server doesn't set each item's
		// APIVersion and Kind. We need to set it.
		itemAPIVersion := list.APIVersion
		itemKind := strings.TrimSuffix(list.Kind, "List")
		for i, _ := range list.Items {
			list.Items[i].APIVersion = itemAPIVersion
			list.Items[i].Kind = itemKind
		}
		return list, err
	}

	// No Items field, so it wasn't a list.
	metaOnly := &MetaOnly{}
	err := json.Unmarshal(data, metaOnly)
	return metaOnly, err
}

func (metaOnlyJSONScheme) decodeInto(data []byte, obj runtime.Object) error {
	return json.Unmarshal(data, obj)
}

// String converts a MetaOnly to a human-readable string.
func (metaOnly MetaOnly) String() string {
	return fmt.Sprintf("%s/%s, name: %s, DeletionTimestamp:%v", metaOnly.TypeMeta.APIVersion, metaOnly.TypeMeta.Kind, metaOnly.ObjectMeta.Name, metaOnly.ObjectMeta.DeletionTimestamp)
}

// PrintAsMetaOnly is a helper function that converts an interface{} to
// *MetaOnly and then convert it to a human-readable string.
func PrintAsMetaOnly(obj interface{}) string {
	metaOnly, ok := obj.(*MetaOnly)
	if !ok {
		return fmt.Sprintf("expected MetaOnly, got %s", reflect.TypeOf(obj))
	}
	return metaOnly.String()
}
