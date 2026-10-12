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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestConvertToRuntimeSecurityContextMsgqueue(t *testing.T) {
	for _, tc := range []struct {
		name      string
		enabled   bool
		ulimits   *v1.Ulimits
		wantSoft  int64
		wantHard  int64
		wantLimit bool
	}{
		{name: "unset", enabled: true},
		{name: "empty", enabled: true, ulimits: &v1.Ulimits{}},
		{name: "disabled", ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](819200), Hard: ptr.To[int64](1048576)}}},
		{name: "finite", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](819200), Hard: ptr.To[int64](1048576)}}, wantSoft: 819200, wantHard: 1048576, wantLimit: true},
		{name: "zero", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)}}, wantLimit: true},
		{name: "maximum", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](17179869184), Hard: ptr.To[int64](17179869184)}}, wantSoft: 17179869184, wantHard: 17179869184, wantLimit: true},
		{name: "unlimited", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](-1), Hard: ptr.To[int64](-1)}}, wantSoft: -1, wantHard: -1, wantLimit: true},
		{name: "missing soft", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Hard: ptr.To[int64](819200)}}},
		{name: "missing hard", enabled: true, ulimits: &v1.Ulimits{Msgqueue: &v1.Ulimit{Soft: ptr.To[int64](819200)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerUlimits, tc.enabled)
			got := convertToRuntimeSecurityContext(&v1.SecurityContext{Ulimits: tc.ulimits})
			require.NotNil(t, got)
			if !tc.wantLimit {
				assert.Nil(t, got.Ulimits)
				return
			}
			require.Len(t, got.Ulimits, 1)
			limit := got.Ulimits[0]
			assert.Equal(t, "msgqueue", limit.Name)
			require.NotNil(t, limit.Soft)
			require.NotNil(t, limit.Hard)
			assert.Equal(t, tc.wantSoft, limit.Soft.Value)
			assert.Equal(t, tc.wantHard, limit.Hard.Value)
		})
	}
}
