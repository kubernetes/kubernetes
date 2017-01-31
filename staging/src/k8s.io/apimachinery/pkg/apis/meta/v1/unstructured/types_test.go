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

package unstructured

import (
	"reflect"
	"testing"
)

func TestUnstructured_DeepCopy(t *testing.T) {
	x := Unstructured{
		Object: map[string]interface{}{
			"a": int(1),
			"b": float64(1.5),
			"c": "foo",
			"d": []interface{}{1, 1.5, "foo"},
			"e": map[string]interface{}{
				"A": 1,
				"B": 1.5,
				"C": "foo",
				"D": []interface{}{1, 1.5, "foo"},
				"E": map[string]interface{}{
					"1": 1,
				},
			},
		},
	}

	y := x.DeepCopy()
	if !reflect.DeepEqual(x, y) {
		t.Fatalf("Expected an equal object: x=%+v, y=%+v", x, y)
	}

	x.Object["a"] = 2
	if y.Object["a"] != 1 {
		t.Errorf("Expected a deep copy, but y.Object[\"a\"] changed.")
	}

	x.Object["d"].([]interface{})[0] = 2
	if y.Object["d"].([]interface{})[0] != 1 {
		t.Errorf("Expected a deep copy, but y.Object[\"d\"][0] changed.")
	}

	x.Object["e"].(map[string]interface{})["E"].(map[string]interface{})["1"] = 2
	if y.Object["e"].(map[string]interface{})["E"].(map[string]interface{})["1"] != 1 {
		t.Errorf("Expected a deep copy, but y.Object[\"e\"][\"E\"][\"1\"] changed.")
	}
}
