/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type TestListOptions struct {
	TypeMeta `json:",inline"`
}

func (o *TestListOptions) GetObjectKind() schema.ObjectKind {
	return &o.TypeMeta
}

func TestParameterCodec_EncodeParameters(t *testing.T) {
	scheme := NewScheme()
	parameterCodec := NewParameterCodec(scheme)

	fooGroupVersion := schema.GroupVersion{Group: "foo", Version: "v1"}
	scheme.AddKnownTypes(fooGroupVersion,
		&TestListOptions{},
	)

	barGroupVersion := schema.GroupVersion{Group: "bar", Version: "v1"}
	scheme.AddKnownTypes(barGroupVersion,
		&TestListOptions{},
	)

	options := TestListOptions{}
	_, err := parameterCodec.EncodeParameters(&options, fooGroupVersion)
	if err != nil {
		t.Errorf("cannot encode parameter to %v: %v", fooGroupVersion, err)
	}
	_, err = parameterCodec.EncodeParameters(&options, barGroupVersion)
	if err != nil {
		t.Errorf("cannot encode parameter to %v: %v", barGroupVersion, err)
	}
}

