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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNewPolicyOptions(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	autoBound, _ := memoryDriftFromKernelImage(logger, procIomemPath)
	testCases := []struct {
		description string
		gate        bool
		options     map[string]string
		want        uint64
		wantErr     bool
	}{
		{description: "no options without the gate", options: nil},
		{description: "option without the gate", options: map[string]string{MemoryDriftTolerance: "off"}, wantErr: true},
		{description: "no options", gate: true, options: nil},
		{description: "auto", gate: true, options: map[string]string{MemoryDriftTolerance: "auto"}, want: autoBound},
		{description: "off", gate: true, options: map[string]string{MemoryDriftTolerance: "off"}},
		{description: "explicit quantity", gate: true, options: map[string]string{MemoryDriftTolerance: "128Mi"}, want: 128 * mb},
		{description: "zero", gate: true, options: map[string]string{MemoryDriftTolerance: "0"}},
		{description: "negative", gate: true, options: map[string]string{MemoryDriftTolerance: "-1Mi"}, wantErr: true},
		{description: "not a quantity", gate: true, options: map[string]string{MemoryDriftTolerance: "lots"}, wantErr: true},
		{description: "unknown option", gate: true, options: map[string]string{"no-such-option": "true"}, wantErr: true},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MemoryManagerDriftTolerance, tc.gate)
			opts, err := NewPolicyOptions(logger, tc.options)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if err == nil && opts.MaxMemoryDrift != tc.want {
				t.Fatalf("MaxMemoryDrift = %d, want %d", opts.MaxMemoryDrift, tc.want)
			}
		})
	}
}
