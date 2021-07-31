/*
Copyright 2021 The Kubernetes Authors.

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

package patch

import (
	"k8s.io/apimachinery/pkg/types"
	"reflect"
	"testing"
)

func TestGeneratePatchBytesForDelete(t *testing.T) {
	tests := []struct {
		name         string
		ownerUID     []types.UID
		dependentUID types.UID
		finalizers   []string
		want         []byte
	}{
		{
			name:         "check the structure of patch bytes",
			ownerUID:     []types.UID{"ss1"},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"}]}}`),
		},
		{
			name:         "check if parent uid is escaped",
			ownerUID:     []types.UID{`ss1"hello`},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1\"hello"}]}}`),
		},
		{
			name:         "check if revision uid uid is escaped",
			ownerUID:     []types.UID{`ss1`},
			dependentUID: `ss2"hello`,
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2\"hello","ownerReferences":[{"$patch":"delete","uid":"ss1"}]}}`),
		},
		{
			name:         "check the structure of patch bytes with multiple owners",
			ownerUID:     []types.UID{"ss1", "ss2"},
			dependentUID: "ss2",
			finalizers:   []string{},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"},{"$patch":"delete","uid":"ss2"}]}}`),
		},
		{
			name:         "check the structure of patch bytes with a finalizer and multiple owners",
			ownerUID:     []types.UID{"ss1", "ss2"},
			dependentUID: "ss2",
			finalizers:   []string{"f1"},
			want:         []byte(`{"metadata":{"uid":"ss2","ownerReferences":[{"$patch":"delete","uid":"ss1"},{"$patch":"delete","uid":"ss2"}],"$deleteFromPrimitiveList/finalizers":["f1"]}}`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := GenerateDeleteOwnerRefStrategicMergeBytes(tt.dependentUID, tt.ownerUID, tt.finalizers...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("generatePatchBytesForDelete() got = %s, want %s", got, tt.want)
			}
		})
	}
}
