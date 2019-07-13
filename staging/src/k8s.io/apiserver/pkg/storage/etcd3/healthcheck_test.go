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

package etcd3

import (
	"testing"
)

func TestEtcdHealthCheck(t *testing.T) {
	tests := []struct {
		data      string
		expectErr bool
	}{
		{
			data:      "{\"health\": \"true\"}",
			expectErr: false,
		},
		{
			data:      "{\"health\": \"false\"}",
			expectErr: true,
		},
		{
			data:      "invalid json",
			expectErr: true,
		},
	}
	for _, test := range tests {
		err := EtcdHealthCheck([]byte(test.data))
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectErr {
			t.Error("unexpected non-error")
		}
	}
}
