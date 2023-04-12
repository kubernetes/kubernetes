/*
Copyright 2023 The Kubernetes Authors.

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

func TestRecordDecodeError(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.Register(decodeErrorCounts)
	resourceName := "pods"
	testedMetrics := "apiserver_storage_decode_errors_total"
	testCases := []struct {
		desc     string
		resource string
		want     string
	}{
		{
			desc:     "test success",
			resource: resourceName,
			want: `
		# HELP apiserver_storage_decode_errors_total [ALPHA] Number of stored object decode errors split by object type
		# TYPE apiserver_storage_decode_errors_total counter
		apiserver_storage_decode_errors_total{resource="pods"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			RecordDecodeError(test.resource)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), testedMetrics); err != nil {
				t.Fatal(err)
			}
		})
	}
}
