/*
Copyright 2020 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/pluginmanagerfakes"
)

func TestPluginHandlerRegistration(t *testing.T) {
	// Setup
	testKubelet := newTestKubeletWithImageList(t, nil /*imageList*/, false /*controllerAttachDetachEnabled*/, defaultVolumePlugins())
	defer testKubelet.Cleanup()

	pluginManager := &pluginmanagerfakes.FakePluginManager{}
	kubelet := testKubelet.kubelet
	kubelet.pluginManager = pluginManager

	// Run
	// kubelet.initializeRuntimeDependentModules() calls kubelet.registerCSIPluginHandler()
	kubelet.initializeRuntimeDependentModules()

	// Assert
	addHandlerCallData := pluginManager.Invocations()["AddHandler"]
	registeredPlugins := []string{}
	for i := 0; i < len(addHandlerCallData); i++ {
		// the first arguments to AddHandler is the plugin name
		registeredPlugins = append(registeredPlugins, addHandlerCallData[i][0].(string))
	}

	require.Containsf(t, registeredPlugins, "CSIPlugin", "Expected the CSIPlugin to have been registered as a PluginHandler")
	require.Containsf(t, registeredPlugins, "DevicePlugin", "Expected the Devicelugin to have been registered as a PluginHandler")
}

func TestRegisterCSIPluginHandler(t *testing.T) {
	tests := map[string]struct {
		errors         []error
		expectedCalls  int
		expectedEvents []string
	}{
		"successful on first registration attempt": {
			errors:        []error{nil},
			expectedCalls: 1,
		},
		"successful on later registration attempt": {
			errors:        []error{fmt.Errorf("not yet"), fmt.Errorf("still not yet"), nil, fmt.Errorf("we should never hit that")},
			expectedCalls: 3,
			expectedEvents: []string{
				"Warning KubeletSetupFailed not yet",
				"Warning KubeletSetupFailed still not yet",
			},
		},
	}

	for name, test := range tests {
		name, test := name, test

		t.Run(name, func(t *testing.T) {
			// Setup
			done := make(chan error, 1)

			k := &Kubelet{}

			pluginManager := &pluginmanagerfakes.FakePluginManager{}
			k.pluginManager = pluginManager

			// we should, worst case, generate as many events as we trigger registration errors
			events := make(chan string, len(test.errors))
			k.recorder = &record.FakeRecorder{Events: events}

			pluginManager.AddHandlerCalls(func(name string, _ cache.PluginHandler) {
				var err error
				if a, e := name, "CSIPlugin"; e != a {
					err = fmt.Errorf("Expected to be called with plugin name %q, got called with %q", e, a)
					t.Error(err)
				}
				done <- err
			})

			fpg := &fakePluginGetter{done: done, errors: test.errors}

			// Run
			k.registerCSIPluginHandler(fpg.get, time.Duration(1))

			// Assert
			require.NoError(t, <-done, "Expected no error to occur")
			require.Equal(t, test.expectedCalls, fpg.count, "Expected number of calls to the plugin handler getter")
			require.EqualValues(t, test.expectedEvents, getEvents(t, events))
		})
	}
}

func getEvents(t *testing.T, ch <-chan string) []string {
	t.Helper()

	var events []string

	for {
		select {
		case event, ok := <-ch:
			if ok {
				events = append(events, event)
			} else {
				t.Errorf("Unexpected channel closure detected")
				return events
			}
		default: // nothing ready to read (anymore)
			return events
		}
	}
}

type fakePluginGetter struct {
	sync.Mutex
	count  int
	errors []error
	done   chan error
}

func (g *fakePluginGetter) get() (cache.PluginHandler, error) {
	g.Lock()
	defer g.Unlock()

	if g.count >= len(g.errors) {
		err := fmt.Errorf("no more fake returns")
		g.done <- err
		return nil, err
	}

	e := g.errors[g.count]

	g.count++
	return nil, e
}
