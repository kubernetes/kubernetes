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
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
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
		// Drivers which shipped before v1 existed only serve v1alpha1.
		// The kubelet keeps consuming it for three releases of transition,
		// see healthServicesSupportedByKubelet.
		"only-v1alpha1": {
			supportedServices: []string{"v1beta1.DRAPlugin", drahealthv1alpha1.DRAResourceHealthService},
			want:              drahealthv1alpha1.DRAResourceHealthService,
		},
		"only-v1": {
			supportedServices: []string{"v1beta1.DRAPlugin", drahealthv1.DRAResourceHealthService},
			want:              drahealthv1.DRAResourceHealthService,
		},
		"both-picks-v1": {
			supportedServices: []string{drahealthv1alpha1.DRAResourceHealthService, drahealthv1.DRAResourceHealthService},
			want:              drahealthv1.DRAResourceHealthService,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := pickHealthService(tc.supportedServices); got != tc.want {
				t.Errorf("pickHealthService(%v) = %q, want %q", tc.supportedServices, got, tc.want)
			}
		})
	}
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

func TestHealthStreamUsesLatestPlugin(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))
	streamHandler := &mockStreamHandler{}
	manager := NewDRAPluginManager(tCtx, nil, nil, streamHandler, 0)
	defer manager.Stop()

	activeStream := func() (string, uint64) {
		manager.mutex.RLock()
		defer manager.mutex.RUnlock()
		stream := manager.healthStreams[driverName]
		if stream == nil {
			return "", 0
		}
		return stream.plugin.endpoint, stream.generation
	}

	tCtx.ExpectNoError(manager.add(driverName, "old.sock", "", drahealthv1.DRAResourceHealthService, defaultClientCallTimeout), "add old plugin")
	endpoint, oldGeneration := activeStream()
	require.Equal(t, "old.sock", endpoint)
	require.NotZero(t, oldGeneration)

	tCtx.ExpectNoError(manager.add(driverName, "new.sock", "", drahealthv1.DRAResourceHealthService, defaultClientCallTimeout), "add new plugin")
	endpoint, newGeneration := activeStream()
	require.Equal(t, "new.sock", endpoint)
	require.Equal(t, oldGeneration+1, newGeneration)
	require.Equal(t, []healthStreamLifecycleEvent{
		{action: "activate", driverName: driverName, generation: oldGeneration},
		{action: "activate", driverName: driverName, generation: newGeneration},
	}, streamHandler.lifecycleEvents(), "handover must activate the replacement without deactivating health")

	manager.remove(driverName, "old.sock")
	endpoint, generation := activeStream()
	require.Equal(t, "new.sock", endpoint)
	require.Equal(t, newGeneration, generation, "removing the old endpoint must not replace the active stream")

	manager.remove(driverName, "new.sock")
	endpoint, generation = activeStream()
	require.Empty(t, endpoint)
	require.Zero(t, generation)
	require.Equal(t, []healthStreamLifecycleEvent{
		{action: "activate", driverName: driverName, generation: oldGeneration},
		{action: "activate", driverName: driverName, generation: newGeneration},
		{action: "deactivate", driverName: driverName, generation: newGeneration},
	}, streamHandler.lifecycleEvents())
}

func TestHealthStreamDoesNotFallBackFromPreferredPlugin(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))
	streamHandler := &mockStreamHandler{}
	manager := NewDRAPluginManager(tCtx, nil, nil, streamHandler, 0)
	defer manager.Stop()

	tCtx.ExpectNoError(manager.add(driverName, "old.sock", "", drahealthv1.DRAResourceHealthService, defaultClientCallTimeout), "add old plugin")
	tCtx.ExpectNoError(manager.add(driverName, "new.sock", "", "", defaultClientCallTimeout), "add new plugin without health service")

	manager.mutex.RLock()
	stream := manager.healthStreams[driverName]
	manager.mutex.RUnlock()
	require.NotNil(t, stream)
	require.Equal(t, "new.sock", stream.plugin.endpoint)
	require.Zero(t, stream.generation)
	require.Equal(t, []healthStreamLifecycleEvent{
		{action: "activate", driverName: driverName, generation: 1},
		{action: "deactivate", driverName: driverName, generation: 1},
	}, streamHandler.lifecycleEvents())
}

func TestHealthStreamUnimplemented(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))
	streamHandler := &mockStreamHandler{}
	manager := NewDRAPluginManager(tCtx, nil, nil, streamHandler, 0)
	defer manager.Stop()

	tCtx.ExpectNoError(manager.add(driverName, "plugin.sock", "", drahealthv1.DRAResourceHealthService, defaultClientCallTimeout), "add plugin")
	manager.mutex.RLock()
	stream := manager.healthStreams[driverName]
	manager.mutex.RUnlock()
	require.NotNil(t, stream)
	require.NotZero(t, stream.generation)
	generation := stream.generation

	require.False(t, manager.healthStreamTerminated(tCtx, stream.plugin, generation, errors.New("temporary stream failure")))
	require.Equal(t, []healthStreamLifecycleEvent{
		{action: "activate", driverName: driverName, generation: generation},
	}, streamHandler.lifecycleEvents())

	require.True(t, manager.healthStreamTerminated(tCtx, stream.plugin, generation, status.Error(codes.Unimplemented, "not supported")))
	require.Equal(t, []healthStreamLifecycleEvent{
		{action: "activate", driverName: driverName, generation: generation},
		{action: "deactivate", driverName: driverName, generation: generation},
	}, streamHandler.lifecycleEvents())

	manager.mutex.RLock()
	defer manager.mutex.RUnlock()
	require.Zero(t, manager.healthStreams[driverName].generation)
	require.Nil(t, manager.healthStreams[driverName].cancel)
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
