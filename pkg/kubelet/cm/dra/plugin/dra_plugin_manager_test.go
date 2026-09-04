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

package plugin

import (
	"context"
	"fmt"
	"math/rand/v2"
	"path"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPickHealthService(t *testing.T) {
	for name, tc := range map[string]struct {
		supportedServices []string
		want              string
	}{
		"none": {
			supportedServices: []string{"v1beta1.DRAPlugin"},
			want:              "",
		},
		"empty": {
			supportedServices: nil,
			want:              "",
		},
		"v1alpha1": {
			supportedServices: []string{"v1beta1.DRAPlugin", drahealthv1alpha1.DRAResourceHealth_ServiceDesc.ServiceName},
			want:              drahealthv1alpha1.DRAResourceHealth_ServiceDesc.ServiceName,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := pickHealthService(tc.supportedServices); got != tc.want {
				t.Errorf("pickHealthService(%v) = %q, want %q", tc.supportedServices, got, tc.want)
			}
		})
	}
}

type unimplementedStreamHandler struct {
	calls  atomic.Int32
	called chan struct{}
}

func (h *unimplementedStreamHandler) HandleWatchResourcesStream(context.Context, drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesClient, string) error {
	h.calls.Add(1)
	select {
	case h.called <- struct{}{}:
	default:
	}
	return status.Error(codes.Unimplemented, "device health reporting is not supported by this driver")
}

func TestHealthStreamRequiresAdvertisedService(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ResourceHealthStatus, true)
	tCtx := ktesting.Init(t)
	handler := &unimplementedStreamHandler{called: make(chan struct{}, 1)}
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, handler, 0)
	t.Cleanup(draPlugins.Stop)

	const driverName = "dummy-driver"
	tCtx.ExpectNoError(draPlugins.RegisterPlugin(driverName, "dra.sock", []string{drapbv1.DRAPluginService}, nil), "register plugin")
	t.Cleanup(func() { draPlugins.remove(driverName, "dra.sock") })

	plugin := draPlugins.get(driverName)
	require.NotNil(t, plugin)
	require.Never(t, func() bool {
		plugin.mutex.Lock()
		defer plugin.mutex.Unlock()
		return plugin.healthStreamCtx != nil
	}, time.Second, 10*time.Millisecond, "health stream should not start when the plugin did not advertise the service")
	require.Zero(t, handler.calls.Load())
}

func TestUnimplementedHealthStreamIsNotRetried(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ResourceHealthStatus, true)
	tCtx := ktesting.Init(t)

	addr := path.Join(t.TempDir(), "dra.sock")
	teardown, err := setupFakeGRPCServer(tCtx, "", addr)
	require.NoError(t, err)
	t.Cleanup(teardown)

	handler := &unimplementedStreamHandler{called: make(chan struct{}, 1)}
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, handler, 0)
	t.Cleanup(draPlugins.Stop)

	const driverName = "dummy-driver"
	tCtx.ExpectNoError(draPlugins.RegisterPlugin(driverName, addr, []string{drapbv1.DRAPluginService, drahealthv1alpha1.DRAResourceHealth_ServiceDesc.ServiceName}, nil), "register plugin")
	t.Cleanup(func() { draPlugins.remove(driverName, addr) })

	select {
	case <-handler.called:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for health stream handler")
	}

	plugin := draPlugins.get(driverName)
	require.NotNil(t, plugin)
	plugin.mutex.Lock()
	healthStreamCtx := plugin.healthStreamCtx
	plugin.mutex.Unlock()
	require.NotNil(t, healthStreamCtx)
	select {
	case <-healthStreamCtx.Done():
	case <-time.After(10 * time.Second):
		t.Fatal("health stream was not stopped after an Unimplemented error")
	}
	require.EqualValues(t, 1, handler.calls.Load(), "health stream handler should only be called once")
}

func TestAddSameName(t *testing.T) {
	tCtx := ktesting.Init(t)
	// name will have a random value to avoid conflicts
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, nil, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, "old.sock", "", "", defaultClientCallTimeout), "add first plugin")
	p, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get first plugin")

	// Same name, same endpoint -> error.
	require.Error(tCtx, draPlugins.add(driverName, "old.sock", "", "", defaultClientCallTimeout))

	tCtx.ExpectNoError(draPlugins.add(driverName, "new.sock", "", "", defaultClientCallTimeout), "add second plugin")
	p2, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get second plugin")
	if p == p2 {
		tCtx.Fatal("expected to get second plugin, got first one again")
	}

	// Remove old plugin.
	draPlugins.remove(p.driverName, p.endpoint)
	plugin, err := draPlugins.GetPlugin(driverName)

	// Now we should have p2 left.
	tCtx.ExpectNoError(err, "get plugin")
	if p2 != plugin {
		tCtx.Fatal("expected to get second plugin again, got something else")
	}
}

func TestDelete(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))
	socketFile := "dra.sock"

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, "dra.sock", "", "", defaultClientCallTimeout), "add plugin")

	draPlugins.remove(driverName, socketFile)

	_, err := draPlugins.GetPlugin(driverName)
	require.Error(t, err, "plugin should not exist after being removed")
}
