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

package etcd

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestValidateStorageData(t *testing.T) {
	for _, tc := range []struct {
		version   string
		wantPanic bool
	}{
		{version: "v1alpha1"},
		{version: "v1beta1", wantPanic: true},
	} {
		t.Run(tc.version+" without removal version", func(t *testing.T) {
			validate := func() {
				validateStorageData(map[schema.GroupVersionResource]StorageData{
					gvr("example.com", tc.version, "examples"): {IntroducedVersion: "1.32"},
				})
			}
			if tc.wantPanic {
				assert.Panics(t, validate)
			} else {
				assert.NotPanics(t, validate)
			}
		})
	}
}
