/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/runtime"
)

type compressingCodec struct {
	dynamic.Codec
}

func NewCompressingCodec() compressingCodec {
	return compressingCodec{dynamic.Codec{}}
}

// Decode calls the embedded Codec to decode data. If the decoded object is
// runtime.Unstructured or runtime.UnstructuredList, Decode only keep the
// APIVersion, Kind, and ObjectMeta fields.
func (c compressingCodec) Decode(data []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	ret, gvk, err := c.Codec.Decode(data, gvk, into)
	if err != nil || ret == nil {
		return ret, gvk, err
	}
	switch x := ret.(type) {
	case *runtime.Unstructured:
		x.ExtractMeta()
		ret = x
		if into != nil {
			u, ok := into.(*runtime.Unstructured)
			if !ok {
				return nil, nil, fmt.Errorf("expect into to be *runtime.Unstructured, got %v", reflect.TypeOf(into))
			}
			u.Object = x.Object
		}
	case *runtime.UnstructuredList:
		x.ExtractMeta()
		ret = x
		if into != nil {
			u, ok := into.(*runtime.UnstructuredList)
			if !ok {
				return nil, nil, fmt.Errorf("expect into to be *runtime.UnstructuredList, got %v", reflect.TypeOf(into))
			}
			u.Items = x.Items
		}
	}
	return ret, gvk, err
}
