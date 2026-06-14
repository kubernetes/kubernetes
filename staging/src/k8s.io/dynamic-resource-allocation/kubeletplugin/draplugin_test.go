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

package kubeletplugin

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
)

// stubPlugin implements the minimal [DRAPlugin] interface.
type stubPlugin struct{}

func (s *stubPlugin) PrepareResourceClaims(ctx context.Context, claims []*resourceapi.ResourceClaim) (map[types.UID]PrepareResult, error) {
	return nil, nil
}

func (s *stubPlugin) UnprepareResourceClaims(ctx context.Context, claims []NamespacedObject) (map[types.UID]error, error) {
	return nil, nil
}

func (s *stubPlugin) HandleError(ctx context.Context, err error, msg string) {}

// v1alpha1HealthPlugin simulates a DRA driver which still implements the
// obsolete v1alpha1 DRAResourceHealth service instead of v1beta1.
type v1alpha1HealthPlugin struct {
	stubPlugin
	drahealthv1alpha1.UnimplementedDRAResourceHealthServer
}

// TestStartRejectsV1Alpha1HealthServer ensures that a driver implementing only
// the obsolete v1alpha1 health service fails loudly at startup instead of
// silently losing health reporting.
func TestStartRejectsV1Alpha1HealthServer(t *testing.T) {
	ctx := t.Context()

	_, err := Start(ctx, &v1alpha1HealthPlugin{},
		DriverName("test-driver"),
		KubeClient(fake.NewClientset()),
	)
	require.ErrorContains(t, err, "implement the v1beta1 DRAResourceHealth interface from k8s.io/kubelet/pkg/apis/dra-health/v1beta1")

	// HealthService(false) is the escape hatch: the same driver may start
	// with health reporting explicitly disabled.
	helper, err := Start(ctx, &v1alpha1HealthPlugin{},
		DriverName("test-driver"),
		KubeClient(fake.NewClientset()),
		HealthService(false),
		// Don't start any sockets, this test only cares about the
		// health service validation.
		RegistrationService(false),
		DRAService(false),
	)
	require.NoError(t, err)
	helper.Stop()
}
