/*
Copyright 2014 The Kubernetes Authors.

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

package runtime_test

import (
	"encoding/json"
	"testing"

	"k8s.io/kubernetes/pkg/runtime"
)

func TestEmbeddedRawExtensionMarshal(t *testing.T) {
	type test struct {
		Ext runtime.RawExtension
	}

	extension := test{Ext: runtime.RawExtension{Raw: []byte(`{"foo":"bar"}`)}}
	data, err := json.Marshal(extension)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(data) != `{"Ext":{"foo":"bar"}}` {
		t.Errorf("unexpected data: %s", string(data))
	}
}
