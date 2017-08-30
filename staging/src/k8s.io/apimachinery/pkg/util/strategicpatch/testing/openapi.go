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

package testing

import (
	"path/filepath"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

var (
	mergeItemGVK = schema.GroupVersionKind{
		Group:   "fake-group",
		Version: "some-version",
		Kind:    "mergeItem",
	}
	precisionItemGVK = schema.GroupVersionKind{
		Group:   "fake-group",
		Version: "some-version",
		Kind:    "precisionItem",
	}
	fakeMergeItemSchema     = testing.Fake{Path: filepath.Join("testing", "swagger-merge-item.json")}
	fakePrecisionItemSchema = testing.Fake{Path: filepath.Join("testing", "swagger-precision-item.json")}
)

func getMergeItemSchema() (openapi.Schema, error) {
	s, err := fakeMergeItemSchema.OpenAPISchema()
	if err != nil {
		return nil, err
	}
	r, err := openapi.NewOpenAPIData(s)
	if err != nil {
		return nil, err
	}
	return r.LookupResource(mergeItemGVK), nil
}

// GetMergeItemSchemaOrDie returns returns the openapi schema for merge item.
func GetMergeItemSchemaOrDie() openapi.Schema {
	s, err := getMergeItemSchema()
	if err != nil {
		panic(err)
	}
	return s
}

func getPrecisionItemSchema() (openapi.Schema, error) {
	s, err := fakePrecisionItemSchema.OpenAPISchema()
	if err != nil {
		return nil, err
	}
	r, err := openapi.NewOpenAPIData(s)
	if err != nil {
		return nil, err
	}
	return r.LookupResource(precisionItemGVK), nil
}

// GetPrecisionItemSchemaOrDie returns returns the openapi schema for precision item.
func GetPrecisionItemSchemaOrDie() openapi.Schema {
	s, err := getPrecisionItemSchema()
	if err != nil {
		panic(err)
	}
	return s
}
