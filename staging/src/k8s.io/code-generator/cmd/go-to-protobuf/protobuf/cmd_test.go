/*
Copyright 2020 The Kubernetes Authors.

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

package protobuf

import (
	"reflect"
	"testing"
)

func TestImportOrder(t *testing.T) {
	testcases := []struct {
		Name      string
		Input     map[string][]string
		Expect    []string
		ExpectErr bool
	}{
		{
			Name:   "empty",
			Input:  nil,
			Expect: []string{},
		},
		{
			Name:   "simple",
			Input:  map[string][]string{"apps": {"core", "extensions", "meta"}, "extensions": {"core", "meta"}, "core": {"meta"}},
			Expect: []string{"meta", "core", "extensions", "apps"},
		},
		{
			Name:      "cycle",
			Input:     map[string][]string{"apps": {"core", "extensions", "meta"}, "extensions": {"core", "meta"}, "core": {"meta", "apps"}},
			ExpectErr: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			order, err := importOrder(tc.Input)
			if err != nil {
				if !tc.ExpectErr {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if tc.ExpectErr {
				t.Fatalf("expected error, got none")
			}
			if !reflect.DeepEqual(order, tc.Expect) {
				t.Fatalf("expected %v, got %v", tc.Expect, order)
			}
		})
	}
}
