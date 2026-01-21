/*
Copyright 2025 The Kubernetes Authors.

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
	"k8s.io/component-base/metrics/testutil"
)

// TestVolumeOperationError tests recording metric when operation has error
func TestVolumeOperationError(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(volumeOperationErrorMetric)

	RecordVolumeOperationErrorMetric("kubernetes.io/fake-plugin", "deletion")

	want := `# HELP volume_operation_errors_total [ALPHA] Total volume operation errors
# TYPE volume_operation_errors_total counter
volume_operation_errors_total{operation_name="deletion",plugin_name="kubernetes.io/fake-plugin"} 1
`

	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "volume_operation_errors_total"); err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
