/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"errors"

	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema"
)

type encodable struct {
	E        Encoder `json:"-"`
	obj      Object
	versions []schema.GroupVersion
}

func (e encodable) GetObjectKind() schema.ObjectKind { return e.obj.GetObjectKind() }
func (e encodable) DeepCopyObject() Object {
	out := e
	out.obj = e.obj.DeepCopyObject()
	copy(out.versions, e.versions)
	return out
}

// NewEncodable creates an object that will be encoded with the provided codec on demand.
// Provided as a convenience for test cases dealing with internal objects.
func NewEncodable(e Encoder, obj Object, versions ...schema.GroupVersion) Object {
	if _, ok := obj.(*Unknown); ok {
		return obj
	}
	return encodable{e, obj, versions}
}

func (e encodable) UnmarshalJSON(in []byte) error {
	return errors.New("runtime.encodable cannot be unmarshalled from JSON")
}

// Marshal may get called on pointers or values, so implement MarshalJSON on value.
// http://stackoverflow.com/questions/21390979/custom-marshaljson-never-gets-called-in-go
func (e encodable) MarshalJSON() ([]byte, error) {
	return Encode(e.E, e.obj)
}

// NewEncodableList creates an object that will be encoded with the provided codec on demand.
// Provided as a convenience for test cases dealing with internal objects.
func NewEncodableList(e Encoder, objects []Object, versions ...schema.GroupVersion) []Object {
	out := make([]Object, len(objects))
	for i := range objects {
		if _, ok := objects[i].(*Unknown); ok {
			out[i] = objects[i]
			continue
		}
		out[i] = NewEncodable(e, objects[i], versions...)
	}
	return out
}

func (e *Unknown) UnmarshalJSON(in []byte) error {
	if e == nil {
		return errors.New("runtime.Unknown: UnmarshalJSON on nil pointer")
	}
	e.TypeMeta = TypeMeta{}
	e.Raw = append(e.Raw[0:0], in...)
	e.ContentEncoding = ""
	e.ContentType = ContentTypeJSON
	return nil
}

// Marshal may get called on pointers or values, so implement MarshalJSON on value.
// http://stackoverflow.com/questions/21390979/custom-marshaljson-never-gets-called-in-go
func (e Unknown) MarshalJSON() ([]byte, error) {
	// If ContentType is unset, we assume this is JSON.
	if e.ContentType != "" && e.ContentType != ContentTypeJSON {
		return nil, errors.New("runtime.Unknown: MarshalJSON on non-json data")
	}
	if e.Raw == nil {
		return []byte("null"), nil
	}
	return e.Raw, nil
}

func Convert_runtime_Object_To_runtime_RawExtension(in *Object, out *RawExtension, s conversion.Scope) error {
	if in == nil {
		out.Raw = []byte("null")
		return nil
	}
	obj := *in
	if unk, ok := obj.(*Unknown); ok {
		if unk.Raw != nil {
			out.Raw = unk.Raw
			return nil
		}
		obj = out.Object
	}
	if obj == nil {
		out.Raw = nil
		return nil
	}
	out.Object = obj
	return nil
}

func Convert_runtime_RawExtension_To_runtime_Object(in *RawExtension, out *Object, s conversion.Scope) error {
	if in.Object != nil {
		*out = in.Object
		return nil
	}
	data := in.Raw
	if len(data) == 0 || (len(data) == 4 && string(data) == "null") {
		*out = nil
		return nil
	}
	*out = &Unknown{
		Raw: data,
		// TODO: Set ContentEncoding and ContentType appropriately.
		// Currently we set ContentTypeJSON to make tests passing.
		ContentType: ContentTypeJSON,
	}
	return nil
}

func DefaultEmbeddedConversions() []interface{} {
	return []interface{}{
		Convert_runtime_Object_To_runtime_RawExtension,
		Convert_runtime_RawExtension_To_runtime_Object,
	}
}
