/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

type InternalSubtype struct {
	String string
}

type Internal struct {
	TypeMeta
	Bool    bool
	Complex InternalSubtype
}

type ExternalSubtype struct {
	String string
}

type External struct {
	TypeMeta
	Complex ExternalSubtype
}

func (obj *Internal) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *External) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }

func TestGenerateConversionsForStruct(t *testing.T) {
	internalGV := unversioned.GroupVersion{Group: "test.group", Version: APIVersionInternal}
	externalGV := unversioned.GroupVersion{Group: "test.group", Version: "external"}

	scheme := NewScheme()
	scheme.Log(t)
	scheme.AddKnownTypeWithName(internalGV.WithKind("Complex"), &Internal{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Complex"), &External{})

	generator := NewConversionGenerator(scheme, "foo")
	typedGenerator, ok := generator.(*conversionGenerator)
	if !ok {
		t.Fatalf("error converting to conversionGenerator")
	}

	internalType := reflect.TypeOf(Internal{})
	externalType := reflect.TypeOf(External{})
	err := typedGenerator.generateConversionsForStruct(internalType, externalType)

	if err == nil {
		t.Errorf("expected error for asymmetrical field")
	}

	// we are expecting Convert_runtime_InternalSubtype_To_runtime_ExternalSubtype to be generated
	// even though the conversion for the parent type cannot be auto generated
	if len(typedGenerator.publicFuncs) != 1 {
		t.Errorf("expected to find one public conversion for the Complex type but found: %v", typedGenerator.publicFuncs)
	}
}
