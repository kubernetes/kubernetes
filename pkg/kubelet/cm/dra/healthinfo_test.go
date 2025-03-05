/*
Copyright 2024 The Kubernetes Authors.

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

package dra

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
)

const (
	testDriver    = "test-driver"
	testPool      = "test-pool"
	testDevice    = "test-device"
	testNamespace = "test-namespace"
	testClaim     = "test-claim"
)

var (
	testDeviceHealth = state.DeviceHealth{
		PoolName:   testPool,
		DeviceName: testDevice,
		Health:     "Healthy",
	}
)

// `TestNewHealthInfoCache tests cache creation and checkpoint loading.
func TestNewHealthInfoCache(t *testing.T) {
	tests := []struct {
		description string
		stateFile   string
		wantErr     bool
	}{
		{
			description: "successfully created cache",
			stateFile:   "/tmp/health_checkpoint",
		},
		{
			description: "empty state file",
			stateFile:   "",
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			cache, err := newHealthInfoCache(test.stateFile)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.NotNil(t, cache)
			if test.stateFile != "" {
				os.Remove(test.stateFile)
			}
		})
	}
}
