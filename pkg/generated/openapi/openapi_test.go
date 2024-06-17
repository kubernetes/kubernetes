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

package openapi

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestOpenAPIRoundtrip(t *testing.T) {
	dummyRef := func(name string) spec.Ref { return spec.MustCreateRef("#/definitions/dummy") }
	for name, value := range GetOpenAPIDefinitions(dummyRef) {
		t.Run(name, func(t *testing.T) {
			// TODO(kubernetes/gengo#193): We currently round-trip ints to floats.
			value.Schema = *handler.PruneDefaultsSchema(&value.Schema)
			data, err := json.Marshal(value.Schema)
			if err != nil {
				t.Error(err)
				return
			}

			roundTripped := spec.Schema{}
			if err := json.Unmarshal(data, &roundTripped); err != nil {
				t.Error(err)
				return
			}

			// Remove the embedded v2 schema if it presents.
			// The v2 schema either become the schema (when serving v2) or get pruned (v3)
			// and it is never round-tripped.
			delete(roundTripped.Extensions, common.ExtensionV2Schema)
			delete(value.Schema.Extensions, common.ExtensionV2Schema)

			if !reflect.DeepEqual(value.Schema, roundTripped) {
				t.Errorf("unexpected diff (a=expected,b=roundtripped):\n%s", cmp.Diff(value.Schema, roundTripped))
				return
			}
		})
	}
}
