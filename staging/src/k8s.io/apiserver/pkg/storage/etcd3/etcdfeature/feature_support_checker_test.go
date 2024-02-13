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

package etcdfeature

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	fgt "k8s.io/component-base/featuregate/testing"
)

// MockEtcdClient is a mock implementation of the EtcdClientInterface interface.
type MockEtcdClient struct {
	Version string
	// Indicates whether to return an error on Status
	ReturnError  bool
	ErrorMessage error
}

func (m *MockEtcdClient) Endpoints() []string {
	return []string{"localhost:2390"}
}

// Status returns a mock status response.
func (m *MockEtcdClient) Status(ctx context.Context, endpoint string) (*clientv3.StatusResponse, error) {
	if m.ReturnError {
		return nil, m.ErrorMessage
	}
	// Return a mock status response
	return &clientv3.StatusResponse{
		Version: m.Version,
	}, nil
}

func TestCheckIsSupportedRequestWatchProgress(t *testing.T) {
	tests := []struct {
		name                  string
		mockClientVersion     string
		mockClientEndpoint    string
		expectedSupportResult bool
		featureDefaultValue   bool
		expectedError         error
		statusReturnError     bool
		statusErrorMessage    error
	}{
		{
			name:                  "Feature enabled and version not deprecated",
			mockClientVersion:     "3.4.26",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: true,
			featureDefaultValue:   true,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
		{
			name:                  "Feature disabled and version not deprecated",
			mockClientVersion:     "3.4.26",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: true,
			featureDefaultValue:   false,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
		{
			name:                  "Feature enabled and version deprecated, bounds",
			mockClientVersion:     "3.4.24",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: false,
			featureDefaultValue:   true,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
		{
			name:                  "Feature enabled and version deprecated, bounds",
			mockClientVersion:     "3.5.0",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: false,
			featureDefaultValue:   true,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
		{
			name:                  "Feature enabled and version deprecated, bounds",
			mockClientVersion:     "3.5.7",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: false,
			featureDefaultValue:   true,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
		{
			name:                  "Status return error",
			mockClientVersion:     "3.5.10",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: false,
			featureDefaultValue:   true,
			expectedError:         fmt.Errorf("failed checking etcd version, endpoint: %q: %w", "localhost:2380", errors.New("")),
			statusReturnError:     true,
			statusErrorMessage:    errors.New(""),
		},
		{
			name:                  "Malformed version",
			mockClientVersion:     "3.5.--a",
			mockClientEndpoint:    "localhost:2380",
			expectedSupportResult: false,
			featureDefaultValue:   true,
			expectedError:         nil,
			statusReturnError:     false,
			statusErrorMessage:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock Etcd client
			mockClient := &MockEtcdClient{Version: tt.mockClientVersion, ReturnError: tt.statusReturnError, ErrorMessage: tt.statusErrorMessage}

			// Mock feature gate
			defer fgt.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentListFromCache, tt.featureDefaultValue)()
			ctx := context.Background()

			// Call the function being tested
			supported, err := endpointSupportsRequestWatchProgress(ctx, mockClient, tt.mockClientEndpoint)

			// Assertions
			assert.Equal(t, tt.expectedSupportResult, supported)
			assert.Equal(t, tt.expectedError, err)
		})
	}
}

// func TestSetFeatureSupportCheckerDuringTest(t *testing.T) {
// 	supports, _ := DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress)
// 	assert.Equal(t, true, supports)
// 	defer DefaultFeatureSupportChecker.SetFeatureSupportCheckerDuringTest("localhost:2390", false)()
// 	supports, _ = DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress)
// 	assert.Equal(t, false, supports)
// }
