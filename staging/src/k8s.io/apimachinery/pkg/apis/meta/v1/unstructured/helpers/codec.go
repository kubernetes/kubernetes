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

package helpers

import (
	gojson "encoding/json"
	"io"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

// Codec is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
// TODO: move into serializer/json.
var Codec runtime.Codec = codec{}

type codec struct{}

func (s codec) Decode(data []byte, _ *schema.GroupVersionKind, obj runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
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

func (codec) Encode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case *unstructured.Unstructured:
		buf, err := t.MarshalJSON()
		if err != nil {
			return err
		}
		_, err = w.Write(buf)
		return err
	case *unstructured.UnstructuredList:
		buf, err := t.MarshalJSON()
		if err != nil {
			return err
		}
		_, err = w.Write(buf)
		return err
	case *runtime.Unknown:
		// TODO: Unstructured needs to deal with ContentType.
		_, err := w.Write(t.Raw)
		return err
	default:
		return json.NewEncoder(w).Encode(t)
	}
}

func (s codec) decode(data []byte) (runtime.Object, error) {
	type detector struct {
		Items gojson.RawMessage
	}
	var det detector
	if err := json.Unmarshal(data, &det); err != nil {
		return nil, err
	}

	if det.Items != nil {
		list := &unstructured.UnstructuredList{}
		err := list.UnmarshalJSON(data)
		return list, err
	}

	// No Items field, so it wasn't a list.
	unstruct := &unstructured.Unstructured{}
	err := unstruct.UnmarshalJSON(data)
	return unstruct, err
}

func (s codec) decodeInto(data []byte, obj runtime.Object) error {
	switch x := obj.(type) {
	case *unstructured.Unstructured:
		return x.UnmarshalJSON(data)
	case *unstructured.UnstructuredList:
		return x.UnmarshalJSON(data)
	case *runtime.VersionedObjects:
		o, err := s.decode(data)
		if err == nil {
			x.Objects = []runtime.Object{o}
		}
		return err
	default:
		return json.Unmarshal(data, x)
	}
}
