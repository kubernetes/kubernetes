/*
Copyright The Kubernetes Authors.

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

package strategicpatch

import (
	"encoding/json"
	"reflect"
	"testing"
)

type unchangedListItem struct {
	Name string `json:"name,omitempty"`
}

type unchangedListObject struct {
	Replicas int                 `json:"replicas,omitempty"`
	Items    []unchangedListItem `json:"items,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
}

func TestCreateTwoWayMergePatchUnchangedListWithoutMergeKey(t *testing.T) {
	tests := []struct {
		name     string
		original string
		modified string
		want     string
	}{
		{
			name:     "empty item, unrelated change",
			original: `{"replicas":1,"items":[{}]}`,
			modified: `{"replicas":2,"items":[{}]}`,
			want:     `{"replicas":2}`,
		},
		{
			name:     "empty item next to named item, unrelated change",
			original: `{"replicas":1,"items":[{"name":"a"},{}]}`,
			modified: `{"replicas":2,"items":[{"name":"a"},{}]}`,
			want:     `{"replicas":2}`,
		},
		{
			name:     "empty item, no change",
			original: `{"replicas":1,"items":[{}]}`,
			modified: `{"replicas":1,"items":[{}]}`,
			want:     `{}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			patch, err := CreateTwoWayMergePatch([]byte(tc.original), []byte(tc.modified), unchangedListObject{})
			if err != nil {
				t.Fatalf("unexpected error creating patch: %v", err)
			}
			if string(patch) != tc.want {
				t.Fatalf("patch = %s, want %s", patch, tc.want)
			}
			result, err := StrategicMergePatch([]byte(tc.original), patch, unchangedListObject{})
			if err != nil {
				t.Fatalf("unexpected error applying patch: %v", err)
			}
			var got, want interface{}
			if err := json.Unmarshal(result, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.modified), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("applied result = %s, want %s", result, tc.modified)
			}
		})
	}
}
