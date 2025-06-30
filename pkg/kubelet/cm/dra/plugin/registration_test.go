/*
Copyright 2023 The Kubernetes Authors.

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
	"path"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	cgotesting "k8s.io/client-go/testing"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	timedworkers "k8s.io/kubernetes/pkg/controller/tainteviction"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	nodeName = "worker"
	pluginA  = "pluginA"
	pluginB  = "pluginB"
)

func getFakeNode() (*v1.Node, error) {
	return &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}, nil
}

func getSlice(name string) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
		},
	}
}

func getFakeClient(t *testing.T, nodeName, driverName string, slice *resourceapi.ResourceSlice) kubernetes.Interface {
	expectedSliceFields := fields.Set{"spec.nodeName": nodeName}
	fakeClient := fake.NewClientset(slice)
	fakeClient.AddReactor("delete-collection", "resourceslices", func(action cgotesting.Action) (bool, runtime.Object, error) {
		deleteAction := action.(cgotesting.DeleteCollectionAction)
		restrictions := deleteAction.GetListRestrictions()
		fieldsSelector := fields.SelectorFromSet(expectedSliceFields)
		// The order of field requirements is random because it comes
		// from a map. We need to sort.
		normalize := func(selector string) string {
			requirements := strings.Split(selector, ",")
			sort.Strings(requirements)
			return strings.Join(requirements, ",")
		}
		assert.Empty(t, restrictions.Labels.String(), "label selector in DeleteCollection")
		assert.Equal(t, normalize(fieldsSelector.String()), normalize(restrictions.Fields.String()), "field selector in DeleteCollection")

		// There's only one object that could get matched, so delete it.
		// Delete doesn't return an error if already deleted, which is what
		// we need here (no error when nothing to delete).
		err := fakeClient.Tracker().Delete(resourceapi.SchemeGroupVersion.WithResource("resourceslices"), "", slice.Name)

		// Set expected slice fields for the next call of this reactor.
		// The reactor will be called next time when resourceslices object is deleted
		// by the kubelet after plugin deregistration.
		switch len(expectedSliceFields) {
		case 1:
			// Startup cleanup done, now expect cleanup for test plugin.
			expectedSliceFields = fields.Set{"spec.nodeName": nodeName, "spec.driver": driverName}
		case 2:
			// Test plugin cleanup done, now expect cleanup for the other plugin.
			otherPlugin := pluginA
			if otherPlugin == driverName {
				otherPlugin = pluginB
			}
			expectedSliceFields = fields.Set{"spec.nodeName": nodeName, "spec.driver": otherPlugin}
		}
		return true, nil, err
	})
	return fakeClient
}

func requireNoSlices(tCtx ktesting.TContext) {
	tCtx.Helper()
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
		slices, err := tCtx.Client().ResourceV1beta1().ResourceSlices().List(tCtx, metav1.ListOptions{})
		if err != nil {
			return err
		}
		assert.Empty(tCtx, slices.Items, "slices")
		return nil
	}).Should(gomega.Succeed(), "there should be no slices")
}

func TestRegistrationHandler(t *testing.T) {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "test-slice"},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
		},
	}

	socketFileA := "a.sock"
	socketFileB := "b.sock"

	for _, test := range []struct {
		description       string
		driverName        string
		socketFile        string
		withClient        bool
		supportedServices []string
		shouldError       bool
		chosenService     string
	}{
		{
			description: "no-services",
			driverName:  pluginB,
			socketFile:  socketFileB,
			shouldError: true,
		},
		{
			description:       "current-service",
			driverName:        pluginB,
			socketFile:        socketFileB,
			supportedServices: []string{drapb.DRAPluginService},
			chosenService:     drapb.DRAPluginService,
		},
		{
			description:       "two-services",
			driverName:        pluginB,
			socketFile:        socketFileB,
			supportedServices: []string{drapb.DRAPluginService /* TODO: add v1 here once we have it */},
			chosenService:     drapb.DRAPluginService,
		},
		// TODO: use v1beta1 here once we have v1
		// {
		// 	description:       "old-service",
		// 	driverName:        pluginB,
		// 	socketFile:        socketFileB,
		// 	supportedServices: []string{drapbv1alpha4.NodeService},
		// 	chosenService:     drapbv1alpha4.NodeService,
		// },
		{
			// Legacy behavior of picking v1alpha4 is no longer supported.
			description:       "version",
			driverName:        pluginB,
			socketFile:        socketFileB,
			supportedServices: []string{"1.0.0"},
			shouldError:       true,
		},
		{
			description:       "replace",
			driverName:        pluginA,
			socketFile:        socketFileB,
			supportedServices: []string{drapb.DRAPluginService},
			chosenService:     drapb.DRAPluginService,
		},
		{
			description:       "manage-resource-slices",
			withClient:        true,
			driverName:        pluginB,
			socketFile:        socketFileB,
			supportedServices: []string{drapb.DRAPluginService},
			chosenService:     drapb.DRAPluginService,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			// Run GRPC services for both plugins.
			//
			// This is necessary because otherwise connection
			// monitoring will start wiping slices, regardless
			// of whether the plugin is registered or not.
			//
			// Here we are only interested in registration.
			// In TestConnectionHandling we check detection
			// of the connection state.

			service := drapb.DRAPluginService
			tmp := t.TempDir()
			endpointA := path.Join(tmp, socketFileA)
			teardownA, err := setupFakeGRPCServer(service, endpointA)
			require.NoError(t, err)
			tCtx.Cleanup(teardownA)

			endpoint := path.Join(tmp, test.socketFile)
			teardown, err := setupFakeGRPCServer(service, endpoint)
			require.NoError(t, err)
			tCtx.Cleanup(teardown)

			// Stand-alone kubelet has no connection to an
			// apiserver, so faking one is optional.
			var client kubernetes.Interface
			if test.withClient {
				fakeClient := getFakeClient(t, nodeName, test.driverName, getSlice("test-slice"))
				client = fakeClient
				tCtx = ktesting.WithClients(tCtx, nil, nil, client, nil, nil)
			}

			// The DRAPluginManager wipes all slices at startup.
			draPlugins := NewDRAPluginManager(tCtx, client, getFakeNode, time.Second /* very short wiping delay for testing */)
			tCtx.Cleanup(draPlugins.Stop)
			if test.withClient {
				requireNoSlices(tCtx)
			}

			// Simulate one existing plugin A.
			err = draPlugins.RegisterPlugin(pluginA, endpointA, []string{drapb.DRAPluginService}, nil)
			require.NoError(t, err)
			t.Cleanup(func() {
				tCtx.Logf("Removing plugin %s", pluginA)
				draPlugins.DeRegisterPlugin(pluginA, endpointA)
			})

			err = draPlugins.ValidatePlugin(test.driverName, endpoint, test.supportedServices)
			if test.shouldError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if err != nil {
				return
			}
			if test.driverName != pluginA {
				require.Nil(t, draPlugins.get(test.driverName), "not registered yet")
			}

			// Add plugin for the first time.
			err = draPlugins.RegisterPlugin(test.driverName, endpoint, test.supportedServices, nil)
			if test.shouldError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			plugin := draPlugins.get(test.driverName)
			require.NotNil(t, plugin, "plugin should be registered")
			t.Cleanup(func() {
				if client != nil {
					// Create the slice as if the plugin had done that while it runs.
					_, err := client.ResourceV1beta1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
					assert.NoError(t, err, "recreate slice")
				}

				tCtx.Logf("Removing plugin %s", test.driverName)
				draPlugins.DeRegisterPlugin(test.driverName, endpoint)
				// Nop.
				draPlugins.DeRegisterPlugin(test.driverName, endpoint)
				if test.withClient {
					requireNoSlices(tCtx)
				}
			})
			// Which plugin was chosen is random in this test: it depends on which plugin was detected as connected,
			// which can be both, one, or none at this point. Some attributes are common to both.
			assert.Equal(t, test.driverName, plugin.driverName, "DRA driver driver name")
			assert.Equal(t, test.chosenService, plugin.chosenService, "chosen service")
		})
	}
}

// TestConnectionHandling checks the reaction to state changes of the service connection.
func TestConnectionHandling(t *testing.T) {
	t.Parallel()
	for description, test := range map[string]struct {
		delay               time.Duration
		requireSliceRemoval bool
	}{
		"wipe-on-disconnect": {
			delay:               time.Second, // very short wiping delay for testing
			requireSliceRemoval: true,
		},
		"no-wipe-on-reconnect": {
			delay:               time.Hour, // long delay to avoid wiping while the test runs
			requireSliceRemoval: false,
		},
	} {
		t.Run(description, func(t *testing.T) {
			t.Parallel()
			tCtx := ktesting.Init(t)

			service := drapb.DRAPluginService
			driverName := "test-plugin"
			sliceName := "test-slice"

			slice := getSlice(sliceName)
			client := getFakeClient(t, nodeName, driverName, slice)
			tCtx = ktesting.WithClients(tCtx, nil, nil, client, nil, nil)

			// The handler wipes all slices at startup.
			draPlugins := NewDRAPluginManager(tCtx, client, getFakeNode, test.delay)
			tCtx.Cleanup(draPlugins.Stop)
			requireNoSlices(tCtx)

			// Run GRPC service.
			endpoint := path.Join(t.TempDir(), "dra.sock")
			teardown, err := setupFakeGRPCServer(service, endpoint)
			require.NoError(t, err)
			defer teardown()

			err = draPlugins.RegisterPlugin(driverName, endpoint, []string{service}, nil)
			require.NoError(t, err)

			plugin := draPlugins.get(driverName)
			assert.NotNil(t, plugin, "plugin should be present in the plugin store")

			// Create the slice as if the plugin had done that while it runs.
			_, err = client.ResourceV1beta1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
			require.NoError(t, err, "recreate slice")

			// Stop gRPC server.
			tCtx.Log("Stopping plugin gRPC server")
			teardown()

			if test.requireSliceRemoval {
				// Slice should get removed.
				requireNoSlices(tCtx)
			} else {
				wipingIsPending := func() bool {
					return draPlugins.pendingWipes.GetWorkerUnsafe(timedworkers.NewWorkArgs(driverName, "").KeyFromWorkArgs()) != nil
				}
				require.Eventuallyf(t, wipingIsPending, time.Minute, time.Second, "wiping should be queued for plugin %s", driverName)

				// Start up gRPC server again.
				tCtx.Log("Restarting plugin gRPC server")
				teardown, err = setupFakeGRPCServer(service, endpoint)
				require.NoError(t, err)
				defer teardown()

				// There shouldn't be any pending wipes for the plugin.
				require.Eventuallyf(t, func() bool {
					return !wipingIsPending()
				}, time.Minute, time.Second, "wiping should be stopped for plugin %s", driverName)

				// Slice should still be there
				slices, err := client.ResourceV1beta1().ResourceSlices().List(tCtx, metav1.ListOptions{})
				require.NoError(t, err, "list slices")
				assert.Len(t, slices.Items, 1, "slices")
			}
		})
	}
}
