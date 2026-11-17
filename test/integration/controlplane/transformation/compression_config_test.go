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
	"os"
	"strconv"
	"testing"

	genericapiserver "k8s.io/apiserver/pkg/server"
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

// TestStorageCompressionConfigHashAcrossFleet checks that apiservers loading
// identical files report the same hash, and one with a different file doesn't.
func TestStorageCompressionConfigHashAcrossFleet(t *testing.T) {
	hostname, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	// The apiserver ID is derived from the hostname, which every apiserver in
	// this process shares. Set it per replica, and put the real one back after.
	t.Cleanup(func() { genericapiserver.SetHostnameFuncForTests(hostname) })

	// last_config_info only shows the apiserver that started last, because one
	// provider serves the whole process. So the replicas run one at a time, and
	// each one's series is read while it's up.
	report := func(t *testing.T, host, config string) metricSample {
		t.Helper()
		genericapiserver.SetHostnameFuncForTests(host)
		test, err := newTransformTest(t, transformTestConfig{extraFlags: compressionFlags(t, config, "1.38")})
		if err != nil {
			t.Fatalf("failed to start kube-apiserver: %v", err)
		}
		defer test.cleanUp()
		samples := gatherSamples(t, lastConfigInfoMetric)
		if len(samples) != 1 {
			t.Fatalf("%s has %d series after %s started, want 1", lastConfigInfoMetric, len(samples), host)
		}
		if samples[0].labels["apiserver_id_hash"] == "" || samples[0].labels["hash"] == "" {
			t.Fatalf("%s series %v is missing a label", lastConfigInfoMetric, samples[0].labels)
		}
		return samples[0]
	}
	policy := compressionConfig(compressionEntry("configmaps", "Deflate", ""))
	first := report(t, "compression-replica-1", policy)
	other := report(t, "compression-replica-2", compressionConfig(compressionEntry("configmaps", "Deflate", "2Ki")))
	same := report(t, "compression-replica-3", policy)

	ids := map[string]bool{first.labels["apiserver_id_hash"]: true, other.labels["apiserver_id_hash"]: true, same.labels["apiserver_id_hash"]: true}
	if len(ids) != 3 {
		t.Errorf("three replicas reported %d apiserver_id_hash values, want 3", len(ids))
	}
	if first.labels["hash"] == other.labels["hash"] {
		t.Errorf("replicas with different policies both reported hash %s", first.labels["hash"])
	}
	if first.labels["hash"] != same.labels["hash"] {
		t.Errorf("replicas with identical files reported hashes %s and %s, want them equal", first.labels["hash"], same.labels["hash"])
	}
}
