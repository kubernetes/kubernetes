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

package openapi

import (
	"testing"

	"github.com/go-openapi/spec"
	assert_pkg "github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type TestType struct {
}

func (t TestType) GetObjectKind() schema.ObjectKind {
	return t
}

func (t TestType) SetGroupVersionKind(kind schema.GroupVersionKind) {
}

func (t TestType) GroupVersionKind() schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   "test",
		Version: "v1",
		Kind:    "TestType",
	}
}

func TestGetDefinitionName(t *testing.T) {
	assert := assert_pkg.New(t)
	testType := TestType{}
	s := runtime.NewScheme()
	s.AddKnownTypeWithName(testType.GroupVersionKind(), &testType)
	namer := NewDefinitionNamer(s)
	n, e := namer.GetDefinitionName("", "k8s.io/kubernetes/pkg/genericapiserver/endpoints/openapi.TestType")
	assert.Equal("io.k8s.kubernetes.pkg.genericapiserver.endpoints.openapi.TestType", n)
	assert.Equal(e["x-kubernetes-group-version-kind"], []v1.GroupVersionKind{
		{
			Group:   "test",
			Version: "v1",
			Kind:    "TestType",
		},
	})
	n, e2 := namer.GetDefinitionName("", "test.com/another.Type")
	assert.Equal("com.test.another.Type", n)
	assert.Equal(e2, spec.Extensions(nil))
}
