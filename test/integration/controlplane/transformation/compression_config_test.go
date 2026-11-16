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

package transformation

import (
	"strconv"
	"testing"
)

// TestStorageCompressionConfigStartup starts kube-apiserver with files at the edges
// of what it accepts: Deflate at 1.38 with minSize at both ends of its range, and
// only None entries below 1.38, which write no frames. The unit tests for
// compressionconfig and EtcdOptions cover the files it refuses.
func TestStorageCompressionConfigStartup(t *testing.T) {
	// minSize can be as low as 64 and as high as one byte under the ceiling.
	deflate := compressionConfig(
		compressionEntry("configmaps", "Deflate", "64"),
		compressionEntry("podtemplates", "Deflate", strconv.Itoa(maxPlaintextBytes-1)),
	)
	noneOnly := compressionConfig(
		compressionEntry("configmaps", "None", ""),
		compressionEntry("podtemplates", "None", ""),
	)
	tests := []struct {
		name             string
		config           string
		minCompatibility string
		// wantForm is how etcd stores a compressible configmap once the server is up.
		wantForm storedForm
	}{
		{name: "Deflate at 1.38", config: deflate, minCompatibility: "1.38", wantForm: formDeflate},
		{name: "only None at 1.37", config: noneOnly, minCompatibility: "1.37", wantForm: formUnframed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			test, err := newTransformTest(t, transformTestConfig{extraFlags: compressionFlags(t, tc.config, tc.minCompatibility)})
			if err != nil {
				t.Fatalf("failed to start kube-apiserver: %v", err)
			}
			defer test.cleanUp()
			cm := compressibleConfigMap("compressible")
			test.mustCreateConfigMap(t, cm)
			test.wantForm(t, configMapsGVR, cm.Name, tc.wantForm)
		})
	}
}
