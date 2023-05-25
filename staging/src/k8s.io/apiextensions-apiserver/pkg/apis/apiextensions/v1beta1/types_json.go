/*
Copyright 2023 The Kubernetes Authors.

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

package v1beta1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
)

// JSON represents any valid JSON value.
// These types are supported: bool, int64, float64, string, []interface{}, map[string]interface{} and nil.
//
// +protobuf=true
// +protobuf.options.marshal=false
// +protobuf.as=ProtoJSON
// +protobuf.options.(gogoproto.goproto_stringer)=false
// +k8s:conversion-gen=false
type JSON struct {
	Object interface{} `json:"-" protobuf:"-"`
}

// OpenAPISchemaType is used by the kube-openapi generator when constructing
// the OpenAPI spec of this type.
//
// See: https://github.com/kubernetes/kube-openapi/tree/master/pkg/generators
func (_ JSON) OpenAPISchemaType() []string {
	// TODO: return actual types when anyOf is supported
	return nil
}

// OpenAPISchemaFormat is used by the kube-openapi generator when constructing
// the OpenAPI spec of this type.
func (_ JSON) OpenAPISchemaFormat() string { return "" }

func (j *JSON) DeepCopy() *JSON {
	if j == nil {
		return nil
	}
	return &JSON{Object: runtime.DeepCopyJSONValue(j.Object)}
}

func (j *JSON) DeepCopyInto(target *JSON) {
	if target == nil {
		return
	}
	if j == nil {
		target.Object = nil // shouldn't happen
	}
	target.Object = runtime.DeepCopyJSONValue(j.Object)
}

func (j JSON) MarshalJSON() ([]byte, error) {
	return json.Marshal(j.Object)
}

func (j *JSON) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &j.Object)
}

func (j *JSON) String() string {
	bs, _ := json.Marshal(j) // no way to handle error here
	return string(bs)
}

// ProtoJSON is a wrapper for JSON data, but intended for
// protobuf marshalling/unmarshalling. It is generated into a serialization
// that matches JSON. Do not use in Go structs.
type ProtoJSON struct {
	Raw []byte `json:"-" protobuf:"bytes,1,opt,name=raw"`
}

// Timestamp returns the Time as a new Timestamp value.
func (j *JSON) ProtoJSON() (*ProtoJSON, error) {
	if j == nil {
		return nil, nil
	}
	bs, err := json.Marshal(j.Object)
	if err != nil {
		return nil, err
	}
	return &ProtoJSON{
		Raw: bs,
	}, nil
}

// Size implements the protobuf marshalling interface.
func (j *JSON) Size() (n int) {
	if j == nil {
		return 0
	}
	pj, _ := j.ProtoJSON() // no way to handle error here
	return pj.Size()
}

// Unmarshal implements the protobuf marshalling interface.
func (j *JSON) Unmarshal(data []byte) error {
	if len(data) == 0 {
		j.Object = nil // TODO(sttts): test that nil JSON and null JSON roundtrip
		return nil
	}
	p := ProtoJSON{}
	if err := p.Unmarshal(data); err != nil {
		return err
	}
	return json.Unmarshal(p.Raw, &j.Object)
}

// Marshal implements the protobuf marshaling interface.
func (j *JSON) Marshal() (data []byte, err error) {
	if j == nil {
		return nil, nil
	}
	pj, err := j.ProtoJSON()
	if err != nil {
		return nil, err
	}
	return pj.Marshal()
}

// MarshalTo implements the protobuf marshaling interface.
func (j *JSON) MarshalTo(data []byte) (int, error) {
	if j == nil {
		return 0, nil
	}
	pj, err := j.ProtoJSON()
	if err != nil {
		return 0, err
	}
	return pj.MarshalTo(data)
}

// MarshalToSizedBuffer implements the protobuf reverse marshaling interface.
func (j *JSON) MarshalToSizedBuffer(data []byte) (int, error) {
	if j == nil {
		return 0, nil
	}
	pj, err := j.ProtoJSON()
	if err != nil {
		return 0, err
	}
	return pj.MarshalToSizedBuffer(data)
}
