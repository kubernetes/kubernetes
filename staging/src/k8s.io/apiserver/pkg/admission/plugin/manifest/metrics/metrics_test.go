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

package metrics

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
)

func TestGetHash(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string // empty means expect empty return
	}{
		{
			name:  "empty string returns empty",
			input: "",
			want:  "",
		},
		{
			name:  "non-empty string returns sha256 prefix",
			input: "test-data",
		},
		{
			name:  "different inputs produce different hashes",
			input: "other-data",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := getHash(tc.input)
			if tc.want == "" && tc.input == "" {
				if result != "" {
					t.Errorf("getHash(%q) = %q, want empty", tc.input, result)
				}
				return
			}
			if !strings.HasPrefix(result, "sha256:") {
				t.Errorf("getHash(%q) = %q, want sha256: prefix", tc.input, result)
			}
		})
	}

	// Verify determinism
	hash1 := getHash("same-input")
	hash2 := getHash("same-input")
	if hash1 != hash2 {
		t.Errorf("getHash is not deterministic: %q != %q", hash1, hash2)
	}

	// Verify different inputs produce different hashes
	hashA := getHash("input-a")
	hashB := getHash("input-b")
	if hashA == hashB {
		t.Errorf("different inputs produced same hash: %q", hashA)
	}
}

func TestRecordAutomaticReloadSuccess(t *testing.T) {
	RegisterMetrics()
	ResetMetricsForTest()

	RecordAutomaticReloadSuccess(ValidatingWebhookManifestType, "test-server-id", "test-config-data")

	// Verify config info was recorded
	provider := configHashProviders[ValidatingWebhookManifestType]
	hashes := provider.GetCurrentHashes()
	if len(hashes) < 2 {
		t.Fatalf("expected at least 2 hashes, got %d", len(hashes))
	}
	if hashes[0] == "" {
		t.Error("expected non-empty apiserver ID hash")
	}
	if hashes[1] == "" {
		t.Error("expected non-empty config hash")
	}
}

func TestRecordAutomaticReloadFailure(t *testing.T) {
	RegisterMetrics()
	ResetMetricsForTest()

	// Should not panic
	RecordAutomaticReloadFailure(MutatingWebhookManifestType, "test-server-id")
}

func TestRecordLastConfigInfo(t *testing.T) {
	RegisterMetrics()

	tests := []struct {
		name         string
		manifestType ManifestType
		apiServerID  string
		configData   string
	}{
		{
			name:         "validating webhook",
			manifestType: ValidatingWebhookManifestType,
			apiServerID:  "server-1",
			configData:   "config-1",
		},
		{
			name:         "mutating webhook",
			manifestType: MutatingWebhookManifestType,
			apiServerID:  "server-2",
			configData:   "config-2",
		},
		{
			name:         "VAP",
			manifestType: VAPManifestType,
			apiServerID:  "server-3",
			configData:   "config-3",
		},
		{
			name:         "MAP",
			manifestType: MAPManifestType,
			apiServerID:  "server-4",
			configData:   "config-4",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			RecordLastConfigInfo(tc.manifestType, tc.apiServerID, tc.configData)

			provider := configHashProviders[tc.manifestType]
			hashes := provider.GetCurrentHashes()
			if len(hashes) < 2 {
				t.Fatalf("expected at least 2 hashes, got %d", len(hashes))
			}
			expectedAPIHash := getHash(tc.apiServerID)
			if hashes[0] != expectedAPIHash {
				t.Errorf("apiserver hash = %q, want %q", hashes[0], expectedAPIHash)
			}
			// configData is already a hash, stored directly
			if hashes[1] != tc.configData {
				t.Errorf("config hash = %q, want %q", hashes[1], tc.configData)
			}
		})
	}
}

func TestRecordLastConfigInfoUnknownType(t *testing.T) {
	// Should not panic for unknown manifest type
	RecordLastConfigInfo(ManifestType("Unknown"), "server", "data")
}

func TestMultiTypeConfigInfoCollectorDescribe(t *testing.T) {
	collector := &multiTypeConfigInfoCollector{desc: admissionManifestLastConfigInfo}
	ch := make(chan *metrics.Desc, 1)
	collector.DescribeWithStability(ch)
	close(ch)

	var descs []*metrics.Desc
	for d := range ch {
		descs = append(descs, d)
	}
	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}
	if descs[0] != admissionManifestLastConfigInfo {
		t.Error("descriptor does not match admissionManifestLastConfigInfo")
	}
}

func TestMultiTypeConfigInfoCollectorCollectEmpty(t *testing.T) {
	// Reset all providers to empty state
	for _, provider := range configHashProviders {
		provider.SetHashes()
	}

	collector := &multiTypeConfigInfoCollector{desc: admissionManifestLastConfigInfo}
	ch := make(chan metrics.Metric, 10)
	collector.CollectWithStability(ch)
	close(ch)

	var collected []metrics.Metric
	for m := range ch {
		collected = append(collected, m)
	}
	if len(collected) != 0 {
		t.Errorf("expected 0 metrics when no hashes are set, got %d", len(collected))
	}
}

func TestMultiTypeConfigInfoCollectorCollectWithData(t *testing.T) {
	// Reset all providers
	for _, provider := range configHashProviders {
		provider.SetHashes()
	}

	// Set data for two types
	RecordLastConfigInfo(ValidatingWebhookManifestType, "server-a", "config-a")
	RecordLastConfigInfo(VAPManifestType, "server-b", "config-b")

	collector := &multiTypeConfigInfoCollector{desc: admissionManifestLastConfigInfo}
	ch := make(chan metrics.Metric, 10)
	collector.CollectWithStability(ch)
	close(ch)

	var collected []metrics.Metric
	for m := range ch {
		collected = append(collected, m)
	}
	if len(collected) != 2 {
		t.Fatalf("expected 2 metrics for 2 types with data, got %d", len(collected))
	}
}

func TestMultiTypeConfigInfoCollectorCollectAllTypes(t *testing.T) {
	// Set data for all four types
	RecordLastConfigInfo(ValidatingWebhookManifestType, "s1", "c1")
	RecordLastConfigInfo(MutatingWebhookManifestType, "s2", "c2")
	RecordLastConfigInfo(VAPManifestType, "s3", "c3")
	RecordLastConfigInfo(MAPManifestType, "s4", "c4")

	collector := &multiTypeConfigInfoCollector{desc: admissionManifestLastConfigInfo}
	ch := make(chan metrics.Metric, 10)
	collector.CollectWithStability(ch)
	close(ch)

	var collected []metrics.Metric
	for m := range ch {
		collected = append(collected, m)
	}
	if len(collected) != 4 {
		t.Fatalf("expected 4 metrics for all types, got %d", len(collected))
	}
}
