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

package kuberuntime

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

type ulimitsRuntimeService struct {
	internalapi.RuntimeService
	features    *runtimeapi.RuntimeFeatures
	statusError error
	statusCalls int
}

func (r *ulimitsRuntimeService) Status(ctx context.Context, verbose bool) (*runtimeapi.StatusResponse, error) {
	r.statusCalls++
	return &runtimeapi.StatusResponse{Features: r.features}, r.statusError
}

func TestDetermineEffectiveSecurityContextUlimits(t *testing.T) {
	for _, tc := range []struct {
		name        string
		features    *runtimeapi.RuntimeFeatures
		ulimits     *v1.Ulimits
		statusError error
		wantError   string
	}{
		{name: "supported", features: &runtimeapi.RuntimeFeatures{ContainerUlimits: true}, ulimits: &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](1024), Hard: ptr.To[int64](2048)}}},
		{name: "unsupported", features: &runtimeapi.RuntimeFeatures{}, ulimits: &v1.Ulimits{Nofile: &v1.Ulimit{Soft: ptr.To[int64](1024), Hard: ptr.To[int64](2048)}}, wantError: "runtime does not support ContainerUlimits"},
		{name: "old runtime", ulimits: &v1.Ulimits{Core: &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)}}, wantError: "runtime does not support ContainerUlimits"},
		{name: "status unavailable", ulimits: &v1.Ulimits{Core: &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)}}, statusError: errors.New("runtime unavailable"), wantError: "check runtime support for container ulimits: runtime unavailable"},
		{name: "unset"},
		{name: "empty", ulimits: &v1.Ulimits{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerUlimits, true)
			ctx := ktesting.Init(t)
			_, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			runtime := &ulimitsRuntimeService{RuntimeService: m.runtimeService, features: tc.features, statusError: tc.statusError}
			m.runtimeService = runtime
			container := &v1.Container{Name: "debug", SecurityContext: &v1.SecurityContext{Ulimits: tc.ulimits}}
			got, err := m.determineEffectiveSecurityContext(ctx, &v1.Pod{}, container, nil, "")
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				require.Nil(t, got)
			} else {
				require.NoError(t, err)
				require.Equal(t, convertToRuntimeUlimits(tc.ulimits), got.Ulimits)
			}
			if tc.ulimits == nil || *tc.ulimits == (v1.Ulimits{}) {
				require.Zero(t, runtime.statusCalls)
			} else {
				require.Equal(t, 1, runtime.statusCalls)
			}
		})
	}
}
