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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNilUnstructuredContent(t *testing.T) {
	var u Unstructured
	content := u.UnstructuredContent()
	expContent := make(map[string]interface{})
	assert.EqualValues(t, expContent, content)
}

// TestUnstructuredContentCopy validates that UnstructuredContent returns a deep copy of the contents
func TestUnstructuredContentCopy(t *testing.T) {
	expMap := map[string]interface{}{
		"a": "b",
	}
	u := &Unstructured{
		Object: map[string]interface{}{
			"map": expMap,
		},
	}
	content := u.UnstructuredContent()
	cMap := content["map"].(map[string]interface{})
	cMap["a"] = "c"
	assert.EqualValues(t, expMap, u.Object["map"])
}
