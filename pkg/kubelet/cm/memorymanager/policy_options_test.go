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

package memorymanager

import (
	"testing"

	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestMaxMemoryDriftFromOptions(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description string
		options     map[string]string
		want        uint64
		wantAuto    bool
		wantErr     bool
	}{
		{description: "no options", options: nil, wantAuto: true},
		{description: "auto", options: map[string]string{MemoryDriftTolerance: "auto"}, wantAuto: true},
		{description: "off", options: map[string]string{MemoryDriftTolerance: "off"}, want: 0},
		{description: "explicit quantity", options: map[string]string{MemoryDriftTolerance: "128Mi"}, want: 128 * mb},
		{description: "zero", options: map[string]string{MemoryDriftTolerance: "0"}, want: 0},
		{description: "negative", options: map[string]string{MemoryDriftTolerance: "-1Mi"}, wantErr: true},
		{description: "not a quantity", options: map[string]string{MemoryDriftTolerance: "lots"}, wantErr: true},
		{description: "unknown option", options: map[string]string{"no-such-option": "true"}, wantErr: true},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			got, err := maxMemoryDriftFromOptions(logger, tc.options)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if tc.wantAuto {
				if want, _ := memoryDriftFromKernelImage(logger, procIomemPath); got != want {
					t.Fatalf("auto bound = %d, want %d", got, want)
				}
				return
			}
			if got != tc.want {
				t.Fatalf("bound = %d, want %d", got, tc.want)
			}
		})
	}
}
