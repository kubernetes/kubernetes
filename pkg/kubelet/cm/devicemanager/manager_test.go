/*
Copyright 2017 The Kubernetes Authors.

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

package devicemanager

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	goruntime "runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	watcherapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
	plugin "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/plugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/resourceupdates"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	testResourceName = "fake-domain/resource"
)

func newWrappedManagerImpl(logger klog.Logger, socketPath string, manager *ManagerImpl) *wrappedManagerImpl {
	w := &wrappedManagerImpl{
		ManagerImpl: manager,
		callback:    manager.genericDeviceUpdateCallback,
	}
	w.socketdir, _ = filepath.Split(socketPath)
	w.server, _ = plugin.NewServer(logger, socketPath, w, w)
	return w
}

type wrappedManagerImpl struct {
	*ManagerImpl
	socketdir string
	callback  func(klog.Logger, string, []*pluginapi.Device)
}

func (m *wrappedManagerImpl) PluginListAndWatchReceiver(logger klog.Logger, r string, resp *pluginapi.ListAndWatchResponse) {
	m.callback(logger, r, resp.Devices)
}

func tmpSocketDir() (socketDir, socketName, pluginSocketName string, err error) {
	socketDir, err = os.MkdirTemp("", "device_plugin")
	if err != nil {
		return
	}
	socketName = filepath.Join(socketDir, "server.sock")
	pluginSocketName = filepath.Join(socketDir, "device-plugin.sock")
	os.MkdirAll(socketDir, 0755)
	return
}

func TestNewManagerImpl(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	topologyStore := topologymanager.NewFakeManager()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	_, err = newManagerImpl(logger, socketName, nil, topologyStore)
	require.NoError(t, err)
	os.RemoveAll(socketDir)
}

func TestNewManagerImplStart(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	m, _, p := setup(tCtx, t, []*pluginapi.Device{}, func(_ klog.Logger, n string, d []*pluginapi.Device) {}, socketName, pluginSocketName)
	err = cleanup(logger, m, p)
	require.NoError(t, err)
	// Stop should tolerate being called more than once.
	err = cleanup(logger, m, p)
	require.NoError(t, err)
}

func TestNewManagerImplStartProbeMode(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	m, _, p, _ := setupInProbeMode(tCtx, t, []*pluginapi.Device{}, func(_ klog.Logger, n string, d []*pluginapi.Device) {}, socketName, pluginSocketName)
	err = cleanup(logger, m, p)
	require.NoError(t, err)
}

// Tests that the device plugin manager correctly handles registration and re-registration by
// making sure that after registration, devices are correctly updated and if a re-registration
// happens, we will NOT delete devices; and no orphaned devices left.
func TestDevicePluginReRegistration(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	// TODO: Remove skip once https://github.com/kubernetes/kubernetes/pull/115269 merges.
	if goruntime.GOOS == "windows" {
		t.Skip("Skipping test on Windows.")
	}
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	devs := []*pluginapi.Device{
		{ID: "Dev1", Health: pluginapi.Healthy},
		{ID: "Dev2", Health: pluginapi.Healthy},
	}
	devsForRegistration := []*pluginapi.Device{
		{ID: "Dev3", Health: pluginapi.Healthy},
	}
	for _, preStartContainerFlag := range []bool{false, true} {
		for _, getPreferredAllocationFlag := range []bool{false, true} {
			m, ch, p1 := setup(tCtx, t, devs, nil, socketName, pluginSocketName)
			err = p1.Register(tCtx, socketName, testResourceName, "")
			require.NoError(t, err)

			select {
			case <-ch:
			case <-time.After(5 * time.Second):
				t.Fatalf("timeout while waiting for manager update")
			}
			capacity, allocatable, _ := m.GetCapacity()
			resourceCapacity := capacity[v1.ResourceName(testResourceName)]
			resourceAllocatable := allocatable[v1.ResourceName(testResourceName)]
			require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
			require.Equal(t, int64(2), resourceAllocatable.Value(), "Devices are not updated.")

			p2 := plugin.NewDevicePluginStub(logger, devs, pluginSocketName+".new", testResourceName, preStartContainerFlag, getPreferredAllocationFlag)
			err = p2.Start(tCtx)
			require.NoError(t, err)
			err = p2.Register(tCtx, socketName, testResourceName, "")
			require.NoError(t, err)

			select {
			case <-ch:
			case <-time.After(5 * time.Second):
				t.Fatalf("timeout while waiting for manager update")
			}
			capacity, allocatable, _ = m.GetCapacity()
			resourceCapacity = capacity[v1.ResourceName(testResourceName)]
			resourceAllocatable = allocatable[v1.ResourceName(testResourceName)]
			require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
			require.Equal(t, int64(2), resourceAllocatable.Value(), "Devices shouldn't change.")

			// Test the scenario that a plugin re-registers with different devices.
			p3 := plugin.NewDevicePluginStub(logger, devsForRegistration, pluginSocketName+".third", testResourceName, preStartContainerFlag, getPreferredAllocationFlag)
			err = p3.Start(tCtx)
			require.NoError(t, err)
			err = p3.Register(tCtx, socketName, testResourceName, "")
			require.NoError(t, err)

			select {
			case <-ch:
			case <-time.After(5 * time.Second):
				t.Fatalf("timeout while waiting for manager update")
			}
			capacity, allocatable, _ = m.GetCapacity()
			resourceCapacity = capacity[v1.ResourceName(testResourceName)]
			resourceAllocatable = allocatable[v1.ResourceName(testResourceName)]
			require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
			require.Equal(t, int64(1), resourceAllocatable.Value(), "Devices of plugin previously registered should be removed.")
			err = p2.Stop(logger)
			require.NoError(t, err)
			err = p3.Stop(logger)
			require.NoError(t, err)
			err = cleanup(logger, m, p1)
			require.NoError(t, err)
		}
	}
}

// Tests that the device plugin manager correctly handles registration and re-registration by
// making sure that after registration, devices are correctly updated and if a re-registration
// happens, we will NOT delete devices; and no orphaned devices left.
// While testing above scenario, plugin discovery and registration will be done using
// Kubelet probe based mechanism
func TestDevicePluginReRegistrationProbeMode(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	// TODO: Remove skip once https://github.com/kubernetes/kubernetes/pull/115269 merges.
	if goruntime.GOOS == "windows" {
		t.Skip("Skipping test on Windows.")
	}
	socketDir, socketName, pluginSocketName, err := tmpSocketDir()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	devs := []*pluginapi.Device{
		{ID: "Dev1", Health: pluginapi.Healthy},
		{ID: "Dev2", Health: pluginapi.Healthy},
	}
	devsForRegistration := []*pluginapi.Device{
		{ID: "Dev3", Health: pluginapi.Healthy},
	}

	m, ch, p1, _ := setupInProbeMode(tCtx, t, devs, nil, socketName, pluginSocketName)

	// Wait for the first callback to be issued.
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.FailNow()
	}
	capacity, allocatable, _ := m.GetCapacity()
	resourceCapacity := capacity[v1.ResourceName(testResourceName)]
	resourceAllocatable := allocatable[v1.ResourceName(testResourceName)]
	require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
	require.Equal(t, int64(2), resourceAllocatable.Value(), "Devices are not updated.")

	p2 := plugin.NewDevicePluginStub(logger, devs, pluginSocketName+".new", testResourceName, false, false)
	err = p2.Start(tCtx)
	require.NoError(t, err)
	// Wait for the second callback to be issued.
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.FailNow()
	}

	capacity, allocatable, _ = m.GetCapacity()
	resourceCapacity = capacity[v1.ResourceName(testResourceName)]
	resourceAllocatable = allocatable[v1.ResourceName(testResourceName)]
	require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
	require.Equal(t, int64(2), resourceAllocatable.Value(), "Devices are not updated.")

	// Test the scenario that a plugin re-registers with different devices.
	p3 := plugin.NewDevicePluginStub(logger, devsForRegistration, pluginSocketName+".third", testResourceName, false, false)
	err = p3.Start(tCtx)
	require.NoError(t, err)
	// Wait for the third callback to be issued.
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.FailNow()
	}

	capacity, allocatable, _ = m.GetCapacity()
	resourceCapacity = capacity[v1.ResourceName(testResourceName)]
	resourceAllocatable = allocatable[v1.ResourceName(testResourceName)]
	require.Equal(t, resourceCapacity.Value(), resourceAllocatable.Value(), "capacity should equal to allocatable")
	require.Equal(t, int64(1), resourceAllocatable.Value(), "Devices of previous registered should be removed")
	err = p2.Stop(logger)
	require.NoError(t, err)
	err = p3.Stop(logger)
	require.NoError(t, err)
	err = cleanup(logger, m, p1)
	require.NoError(t, err)
}

func setupDeviceManager(t *testing.T, devs []*pluginapi.Device, callback monitorCallback, socketName string,
	topology []cadvisorapi.Node, logger klog.Logger) (Manager, <-chan interface{}) {
	topologyStore := topologymanager.NewFakeManager()
	m, err := newManagerImpl(logger, socketName, topology, topologyStore)
	require.NoError(t, err)
	updateChan := make(chan interface{})

	w := newWrappedManagerImpl(logger, socketName, m)
	if callback != nil {
		w.callback = callback
	}

	originalCallback := w.callback
	w.callback = func(logger klog.Logger, resourceName string, devices []*pluginapi.Device) {
		originalCallback(logger, resourceName, devices)
		updateChan <- new(interface{})
	}
	activePods := func() []*v1.Pod {
		return []*v1.Pod{}
	}

	// test steady state, initialization where sourcesReady, containerMap and containerRunningSet
	// are relevant will be tested with a different flow
	err = w.Start(logger, activePods, &sourcesReadyStub{}, containermap.NewContainerMap(), sets.New[string]())
	require.NoError(t, err)

	return w, updateChan
}

func setupDevicePlugin(ctx context.Context, t *testing.T, devs []*pluginapi.Device, pluginSocketName string) *plugin.Stub {
	p := plugin.NewDevicePluginStub(klog.FromContext(ctx), devs, pluginSocketName, testResourceName, false, false)
	err := p.Start(ctx)
	require.NoError(t, err)
	return p
}

func setupPluginManager(t *testing.T, pluginSocketName string, m Manager) pluginmanager.PluginManager {
	pluginManager := pluginmanager.NewPluginManager(
		filepath.Dir(pluginSocketName), /* sockDir */
		&record.FakeRecorder{},
	)

	tCtx := ktesting.Init(t)
	runPluginManager(tCtx, pluginManager)
	pluginManager.AddHandler(watcherapi.DevicePlugin, m.GetWatcherHandler())
	return pluginManager
}

func runPluginManager(ctx context.Context, pluginManager pluginmanager.PluginManager) {
	// FIXME: Replace sets.Set[string] with sets.Set[string]
	sourcesReady := config.NewSourcesReady(func(_ sets.Set[string]) bool { return true })
	go pluginManager.Run(ctx, sourcesReady, wait.NeverStop)
}

func setup(ctx context.Context, t *testing.T, devs []*pluginapi.Device, callback monitorCallback, socketName string, pluginSocketName string) (Manager, <-chan interface{}, *plugin.Stub) {
	logger := klog.FromContext(ctx)
	m, updateChan := setupDeviceManager(t, devs, callback, socketName, nil, logger)
	p := setupDevicePlugin(ctx, t, devs, pluginSocketName)
	return m, updateChan, p
}

func setupInProbeMode(ctx context.Context, t *testing.T, devs []*pluginapi.Device, callback monitorCallback, socketName string, pluginSocketName string) (Manager, <-chan interface{}, *plugin.Stub, pluginmanager.PluginManager) {
	logger := klog.FromContext(ctx)
	m, updateChan := setupDeviceManager(t, devs, callback, socketName, nil, logger)
	p := setupDevicePlugin(ctx, t, devs, pluginSocketName)
	pm := setupPluginManager(t, pluginSocketName, m)
	return m, updateChan, p, pm
}

func cleanup(logger klog.Logger, m Manager, p *plugin.Stub) error {
	if err := p.Stop(logger); err != nil {
		return err
	}
	return m.Stop(logger)
}

func TestUpdateCapacityAllocatable(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	topologyStore := topologymanager.NewFakeManager()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	testManager, err := newManagerImpl(logger, socketName, nil, topologyStore)
	as := assert.New(t)
	as.NotNil(testManager)
	as.NoError(err)

	devs := []*pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}
	callback := testManager.genericDeviceUpdateCallback

	// Adds three devices for resource1, two healthy and one unhealthy.
	// Expects capacity for resource1 to be 2.
	resourceName1 := "domain1.com/resource1"
	e1 := &endpointImpl{}
	testManager.endpoints[resourceName1] = endpointInfo{e: e1, opts: nil}
	callback(logger, resourceName1, devs)
	capacity, allocatable, removedResources := testManager.GetCapacity()
	resource1Capacity, ok := capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok := allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(3), resource1Capacity.Value())
	as.Equal(int64(2), resource1Allocatable.Value())
	as.Empty(removedResources)

	// Deletes an unhealthy device should NOT change allocatable but change capacity.
	devs1 := devs[:len(devs)-1]
	callback(logger, resourceName1, devs1)
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(2), resource1Capacity.Value())
	as.Equal(int64(2), resource1Allocatable.Value())
	as.Empty(removedResources)

	// Updates a healthy device to unhealthy should reduce allocatable by 1.
	devs[1].Health = pluginapi.Unhealthy
	callback(logger, resourceName1, devs)
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(3), resource1Capacity.Value())
	as.Equal(int64(1), resource1Allocatable.Value())
	as.Empty(removedResources)

	// Deletes a healthy device should reduce capacity and allocatable by 1.
	devs2 := devs[1:]
	callback(logger, resourceName1, devs2)
	capacity, allocatable, removedResources = testManager.GetCapacity()
	resource1Capacity, ok = capacity[v1.ResourceName(resourceName1)]
	as.True(ok)
	resource1Allocatable, ok = allocatable[v1.ResourceName(resourceName1)]
	as.True(ok)
	as.Equal(int64(0), resource1Allocatable.Value())
	as.Equal(int64(2), resource1Capacity.Value())
	as.Empty(removedResources)

	// Tests adding another resource.
	resourceName2 := "resource2"
	e2 := &endpointImpl{socket: socketName}
	e2.client = plugin.NewPluginClient(resourceName2, socketName, testManager)
	eInfo := endpointInfo{e: e2, opts: nil}
	testManager.endpoints[resourceName2] = eInfo
	testManager.endpointStore[resourceName2] = map[string]*endpointInfo{socketName: &eInfo}
	callback(logger, resourceName2, devs)
	capacity, allocatable, removedResources = testManager.GetCapacity()
	as.Len(capacity, 2)
	resource2Capacity, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	resource2Allocatable, ok := allocatable[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(3), resource2Capacity.Value())
	as.Equal(int64(1), resource2Allocatable.Value())
	as.Empty(removedResources)

	// Expires resourceName1 endpoint. Verifies testManager.GetCapacity() reports that resourceName1
	// is removed from capacity and it no longer exists in healthyDevices after the call.
	e1.setStopTime(time.Now().Add(-1*endpointStopGracePeriod - time.Duration(10)*time.Second))
	capacity, allocatable, removed := testManager.GetCapacity()
	as.Equal([]string{resourceName1}, removed)
	as.NotContains(capacity, v1.ResourceName(resourceName1))
	as.NotContains(allocatable, v1.ResourceName(resourceName1))
	val, ok := capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(3), val.Value())
	as.NotContains(testManager.healthyDevices, resourceName1)
	as.NotContains(testManager.unhealthyDevices, resourceName1)
	as.NotContains(testManager.endpoints, resourceName1)
	as.Len(testManager.endpoints, 1)

	// Stops resourceName2 endpoint. Verifies its stopTime is set, allocate and
	// preStartContainer calls return errors.
	err = e2.client.Disconnect(logger)
	require.NoError(t, err)
	as.False(e2.stopTime.IsZero())
	_, err = e2.allocate(tCtx, []string{"Device1"})
	reflect.DeepEqual(err, fmt.Errorf(errEndpointStopped, e2))
	_, err = e2.preStartContainer(tCtx, []string{"Device1"})
	reflect.DeepEqual(err, fmt.Errorf(errEndpointStopped, e2))
	// Marks resourceName2 unhealthy and verifies its capacity/allocatable are
	// correctly updated.
	testManager.markResourceUnhealthy(logger, resourceName2)
	capacity, allocatable, removed = testManager.GetCapacity()
	val, ok = capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(3), val.Value())
	val, ok = allocatable[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(0), val.Value())
	as.Empty(removed)
	// Writes and re-reads checkpoints. Verifies we create a stopped endpoint
	// for resourceName2, its capacity is set to zero, and we still consider
	// it as a DevicePlugin resource. This makes sure any pod that was scheduled
	// during the time of propagating capacity change to the scheduler will be
	// properly rejected instead of being incorrectly started.
	err = testManager.writeCheckpoint(logger)
	as.NoError(err)
	testManager.healthyDevices = make(map[string]sets.Set[string])
	testManager.unhealthyDevices = make(map[string]sets.Set[string])
	err = testManager.readCheckpoint(logger)
	as.NoError(err)
	as.Len(testManager.endpoints, 1)
	as.Contains(testManager.endpoints, resourceName2)
	capacity, allocatable, removed = testManager.GetCapacity()
	val, ok = capacity[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(0), val.Value())
	val, ok = allocatable[v1.ResourceName(resourceName2)]
	as.True(ok)
	as.Equal(int64(0), val.Value())
	as.Empty(removed)
	as.True(testManager.isDevicePluginResource(resourceName2))
}

func TestGetAllocatableDevicesMultipleResources(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	topologyStore := topologymanager.NewFakeManager()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	testManager, err := newManagerImpl(logger, socketName, nil, topologyStore)
	as := assert.New(t)
	as.NotNil(testManager)
	as.NoError(err)

	resource1Devs := []*pluginapi.Device{
		{ID: "R1Device1", Health: pluginapi.Healthy},
		{ID: "R1Device2", Health: pluginapi.Healthy},
		{ID: "R1Device3", Health: pluginapi.Unhealthy},
	}
	resourceName1 := "domain1.com/resource1"
	e1 := &endpointImpl{}
	testManager.endpoints[resourceName1] = endpointInfo{e: e1, opts: nil}
	testManager.genericDeviceUpdateCallback(logger, resourceName1, resource1Devs)

	resource2Devs := []*pluginapi.Device{
		{ID: "R2Device1", Health: pluginapi.Healthy},
	}
	resourceName2 := "other.domain2.org/resource2"
	e2 := &endpointImpl{}
	testManager.endpoints[resourceName2] = endpointInfo{e: e2, opts: nil}
	testManager.genericDeviceUpdateCallback(logger, resourceName2, resource2Devs)

	allocatableDevs := testManager.GetAllocatableDevices()
	as.Len(allocatableDevs, 2)

	devInstances1, ok := allocatableDevs[resourceName1]
	as.True(ok)
	checkAllocatableDevicesConsistsOf(as, devInstances1, []string{"R1Device1", "R1Device2"})

	devInstances2, ok := allocatableDevs[resourceName2]
	as.True(ok)
	checkAllocatableDevicesConsistsOf(as, devInstances2, []string{"R2Device1"})

}

func TestGetAllocatableDevicesHealthTransition(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	topologyStore := topologymanager.NewFakeManager()
	require.NoError(t, err)
	defer os.RemoveAll(socketDir)
	testManager, err := newManagerImpl(logger, socketName, nil, topologyStore)
	as := assert.New(t)
	as.NotNil(testManager)
	as.NoError(err)

	resource1Devs := []*pluginapi.Device{
		{ID: "R1Device1", Health: pluginapi.Healthy},
		{ID: "R1Device2", Health: pluginapi.Healthy},
		{ID: "R1Device3", Health: pluginapi.Unhealthy},
	}

	// Adds three devices for resource1, two healthy and one unhealthy.
	// Expects allocatable devices for resource1 to be 2.
	resourceName1 := "domain1.com/resource1"
	e1 := &endpointImpl{}
	testManager.endpoints[resourceName1] = endpointInfo{e: e1, opts: nil}

	testManager.genericDeviceUpdateCallback(logger, resourceName1, resource1Devs)

	allocatableDevs := testManager.GetAllocatableDevices()
	as.Len(allocatableDevs, 1)
	devInstances, ok := allocatableDevs[resourceName1]
	as.True(ok)
	checkAllocatableDevicesConsistsOf(as, devInstances, []string{"R1Device1", "R1Device2"})

	// Unhealthy device becomes healthy
	resource1Devs = []*pluginapi.Device{
		{ID: "R1Device1", Health: pluginapi.Healthy},
		{ID: "R1Device2", Health: pluginapi.Healthy},
		{ID: "R1Device3", Health: pluginapi.Healthy},
	}
	testManager.genericDeviceUpdateCallback(logger, resourceName1, resource1Devs)

	allocatableDevs = testManager.GetAllocatableDevices()
	as.Len(allocatableDevs, 1)
	devInstances, ok = allocatableDevs[resourceName1]
	as.True(ok)
	checkAllocatableDevicesConsistsOf(as, devInstances, []string{"R1Device1", "R1Device2", "R1Device3"})
}

func checkAllocatableDevicesConsistsOf(as *assert.Assertions, devInstances DeviceInstances, expectedDevs []string) {
	as.Equal(len(expectedDevs), len(devInstances))
	for _, deviceID := range expectedDevs {
		_, ok := devInstances[deviceID]
		as.True(ok)
	}
}

func constructDevices(devices []string) checkpoint.DevicesPerNUMA {
	ret := checkpoint.DevicesPerNUMA{}
	for _, dev := range devices {
		ret[0] = append(ret[0], dev)
	}
	return ret
}

// containerAllocateResponseBuilder is a helper to build a ContainerAllocateResponse
type containerAllocateResponseBuilder struct {
	devices    map[string]string
	mounts     map[string]string
	envs       map[string]string
	cdiDevices []string
}

// containerAllocateResponseBuilderOption defines a functional option for a containerAllocateResponseBuilder
type containerAllocateResponseBuilderOption func(*containerAllocateResponseBuilder)

// withDevices sets the devices for the containerAllocateResponseBuilder
func withDevices(devices map[string]string) containerAllocateResponseBuilderOption {
	return func(b *containerAllocateResponseBuilder) {
		b.devices = devices
	}
}

// withMounts sets the mounts for the containerAllocateResponseBuilder
func withMounts(mounts map[string]string) containerAllocateResponseBuilderOption {
	return func(b *containerAllocateResponseBuilder) {
		b.mounts = mounts
	}
}

// withEnvs sets the envs for the containerAllocateResponseBuilder
func withEnvs(envs map[string]string) containerAllocateResponseBuilderOption {
	return func(b *containerAllocateResponseBuilder) {
		b.envs = envs
	}
}

// withCDIDevices sets the cdiDevices for the containerAllocateResponseBuilder
func withCDIDevices(cdiDevices ...string) containerAllocateResponseBuilderOption {
	return func(b *containerAllocateResponseBuilder) {
		b.cdiDevices = cdiDevices
	}
}

// newContainerAllocateResponse creates a ContainerAllocateResponse with the given options.
func newContainerAllocateResponse(opts ...containerAllocateResponseBuilderOption) *pluginapi.ContainerAllocateResponse {
	b := &containerAllocateResponseBuilder{}
	for _, opt := range opts {
		opt(b)
	}

	return b.Build()
}

// Build uses the configured builder to create a ContainerAllocateResponse.
func (b *containerAllocateResponseBuilder) Build() *pluginapi.ContainerAllocateResponse {
	resp := &pluginapi.ContainerAllocateResponse{}
	for k, v := range b.devices {
		resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
			HostPath:      k,
			ContainerPath: v,
			Permissions:   "mrw",
		})
	}
	for k, v := range b.mounts {
		resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
			ContainerPath: k,
			HostPath:      v,
			ReadOnly:      true,
		})
	}
	resp.Envs = make(map[string]string)
	for k, v := range b.envs {
		resp.Envs[k] = v
	}

	var cdiDevices []*pluginapi.CDIDevice
	for _, dev := range b.cdiDevices {
		cdiDevice := pluginapi.CDIDevice{
			Name: dev,
		}
		cdiDevices = append(cdiDevices, &cdiDevice)
	}
	resp.CdiDevices = cdiDevices

	return resp
}

func TestCheckpoint(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	resourceName1 := "domain1.com/resource1"
	resourceName2 := "domain2.com/resource2"
	resourceName3 := "domain2.com/resource3"
	as := assert.New(t)
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)
	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	as.NoError(err)
	testManager := &ManagerImpl{
		endpoints:         make(map[string]endpointInfo),
		healthyDevices:    make(map[string]sets.Set[string]),
		unhealthyDevices:  make(map[string]sets.Set[string]),
		allocatedDevices:  make(map[string]sets.Set[string]),
		podDevices:        newPodDevices(),
		checkpointManager: ckm,
	}

	testManager.podDevices.insert("pod1", "con1", resourceName1,
		constructDevices([]string{"dev1", "dev2"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
			withCDIDevices("domain1.com/resource1=dev1", "domain1.com/resource1=dev2"),
		),
	)
	testManager.podDevices.insert("pod1", "con1", resourceName2,
		constructDevices([]string{"dev1", "dev2"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r2dev1": "/dev/r2dev1", "/dev/r2dev2": "/dev/r2dev2"}),
			withMounts(map[string]string{"/home/r2lib1": "/usr/r2lib1"}),
			withEnvs(map[string]string{"r2devices": "dev1 dev2"}),
		),
	)
	testManager.podDevices.insert("pod1", "con2", resourceName1,
		constructDevices([]string{"dev3"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev3": "/dev/r1dev3"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)
	testManager.podDevices.insert("pod2", "con1", resourceName1,
		constructDevices([]string{"dev4"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev4": "/dev/r1dev4"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)
	testManager.podDevices.insert("pod3", "con3", resourceName3,
		checkpoint.DevicesPerNUMA{nodeWithoutTopology: []string{"dev5"}},
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r3dev5": "/dev/r3dev5"}),
			withMounts(map[string]string{"/home/r3lib1": "/usr/r3lib1"}),
		),
	)

	testManager.healthyDevices[resourceName1] = sets.New[string]()
	testManager.healthyDevices[resourceName1].Insert("dev1")
	testManager.healthyDevices[resourceName1].Insert("dev2")
	testManager.healthyDevices[resourceName1].Insert("dev3")
	testManager.healthyDevices[resourceName1].Insert("dev4")
	testManager.healthyDevices[resourceName1].Insert("dev5")
	testManager.healthyDevices[resourceName2] = sets.New[string]()
	testManager.healthyDevices[resourceName2].Insert("dev1")
	testManager.healthyDevices[resourceName2].Insert("dev2")
	testManager.healthyDevices[resourceName3] = sets.New[string]()
	testManager.healthyDevices[resourceName3].Insert("dev5")

	expectedPodDevices := testManager.podDevices
	expectedAllocatedDevices := testManager.podDevices.devices()
	expectedAllDevices := testManager.healthyDevices

	err = testManager.writeCheckpoint(logger)

	as.NoError(err)
	testManager.podDevices = newPodDevices()
	err = testManager.readCheckpoint(logger)
	as.NoError(err)

	as.Equal(expectedPodDevices.size(), testManager.podDevices.size())
	for podUID, containerDevices := range expectedPodDevices.devs {
		for conName, resources := range containerDevices {
			for resource := range resources {
				expDevices := expectedPodDevices.containerDevices(podUID, conName, resource)
				testDevices := testManager.podDevices.containerDevices(podUID, conName, resource)
				as.True(reflect.DeepEqual(expDevices, testDevices))
				opts1 := expectedPodDevices.deviceRunContainerOptions(logger, podUID, conName)
				opts2 := testManager.podDevices.deviceRunContainerOptions(logger, podUID, conName)
				as.Equal(len(opts1.Envs), len(opts2.Envs))
				as.Equal(len(opts1.Mounts), len(opts2.Mounts))
				as.Equal(len(opts1.Devices), len(opts2.Devices))
			}
		}
	}
	as.True(reflect.DeepEqual(expectedAllocatedDevices, testManager.allocatedDevices))
	as.True(reflect.DeepEqual(expectedAllDevices, testManager.healthyDevices))
}

type activePodsStub struct {
	activePods []*v1.Pod
}

func (a *activePodsStub) getActivePods() []*v1.Pod {
	return a.activePods
}

func (a *activePodsStub) updateActivePods(newPods []*v1.Pod) {
	a.activePods = newPods
}

type MockEndpoint struct {
	getPreferredAllocationFunc func(available, mustInclude []string, size int) (*pluginapi.PreferredAllocationResponse, error)
	allocateFunc               func(devs []string) (*pluginapi.AllocateResponse, error)
	initChan                   chan []string
	socket                     string
}

func (m *MockEndpoint) preStartContainer(_ context.Context, devs []string) (*pluginapi.PreStartContainerResponse, error) {
	m.initChan <- devs
	return &pluginapi.PreStartContainerResponse{}, nil
}

func (m *MockEndpoint) getPreferredAllocation(_ context.Context, available, mustInclude []string, size int) (*pluginapi.PreferredAllocationResponse, error) {
	if m.getPreferredAllocationFunc != nil {
		return m.getPreferredAllocationFunc(available, mustInclude, size)
	}
	return nil, nil
}

func (m *MockEndpoint) allocate(ctx context.Context, devs []string) (*pluginapi.AllocateResponse, error) {
	if m.allocateFunc != nil {
		return m.allocateFunc(devs)
	}
	return nil, nil
}

func (m *MockEndpoint) setStopTime(t time.Time) {}

func (m *MockEndpoint) isStopped() bool { return false }

func (m *MockEndpoint) stopGracePeriodExpired() bool { return false }

func (m *MockEndpoint) socketPath() string {
	return m.socket
}

func makePod(limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: limits,
					},
				},
			},
		},
	}
}

func getTestManager(tmpDir string, activePods ActivePodsFunc, testRes []TestResource) (*wrappedManagerImpl, error) {
	monitorCallback := func(logger klog.Logger, resourceName string, devices []*pluginapi.Device) {}
	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	if err != nil {
		return nil, err
	}
	m := &ManagerImpl{
		healthyDevices:        make(map[string]sets.Set[string]),
		unhealthyDevices:      make(map[string]sets.Set[string]),
		allocatedDevices:      make(map[string]sets.Set[string]),
		endpoints:             make(map[string]endpointInfo),
		podDevices:            newPodDevices(),
		devicesToReuse:        make(PodReusableDevices),
		topologyAffinityStore: topologymanager.NewFakeManager(),
		activePods:            activePods,
		sourcesReady:          &sourcesReadyStub{},
		checkpointManager:     ckm,
		allDevices:            NewResourceDeviceInstances(),
	}
	testManager := &wrappedManagerImpl{
		ManagerImpl: m,
		socketdir:   tmpDir,
		callback:    monitorCallback,
	}

	for _, res := range testRes {
		testManager.healthyDevices[res.resourceName] = sets.New[string](res.devs.Devices().UnsortedList()...)
		if res.resourceName == "domain1.com/resource1" {
			testManager.endpoints[res.resourceName] = endpointInfo{
				e:    &MockEndpoint{allocateFunc: allocateStubFunc()},
				opts: nil,
			}
		}
		if res.resourceName == "domain2.com/resource2" {
			testManager.endpoints[res.resourceName] = endpointInfo{
				e: &MockEndpoint{
					allocateFunc: func(devs []string) (*pluginapi.AllocateResponse, error) {
						resp := new(pluginapi.ContainerAllocateResponse)
						resp.Envs = make(map[string]string)
						for _, dev := range devs {
							switch dev {
							case "dev3":
								resp.Envs["key2"] = "val2"

							case "dev4":
								resp.Envs["key2"] = "val3"
							}
						}
						resps := new(pluginapi.AllocateResponse)
						resps.ContainerResponses = append(resps.ContainerResponses, resp)
						return resps, nil
					},
				},
				opts: nil,
			}
		}
		testManager.allDevices[res.resourceName] = makeDevice(res.devs, res.topology)

	}
	return testManager, nil
}

type TestResource struct {
	resourceName     string
	resourceQuantity resource.Quantity
	devs             checkpoint.DevicesPerNUMA
	topology         bool
}

func TestFilterByAffinity(t *testing.T) {
	as := require.New(t)
	allDevices := ResourceDeviceInstances{
		"res1": map[string]*pluginapi.Device{
			"dev1": {
				ID: "dev1",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 1,
						},
					},
				},
			},
			"dev2": {
				ID: "dev2",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 1,
						},
						{
							ID: 2,
						},
					},
				},
			},
			"dev3": {
				ID: "dev3",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 2,
						},
					},
				},
			},
			"dev4": {
				ID: "dev4",
				Topology: &pluginapi.TopologyInfo{
					Nodes: []*pluginapi.NUMANode{
						{
							ID: 2,
						},
					},
				},
			},
			"devwithouttopology": {
				ID: "dev5",
			},
		},
	}

	fakeAffinity, _ := bitmask.NewBitMask(2)
	fakeHint := topologymanager.TopologyHint{
		NUMANodeAffinity: fakeAffinity,
		Preferred:        true,
	}
	testManager := ManagerImpl{
		topologyAffinityStore: topologymanager.NewFakeManagerWithHint(&fakeHint),
		allDevices:            allDevices,
	}

	testCases := []struct {
		available               sets.Set[string]
		fromAffinityExpected    sets.Set[string]
		notFromAffinityExpected sets.Set[string]
		withoutTopologyExpected sets.Set[string]
	}{
		{
			available:               sets.New[string]("dev1", "dev2"),
			fromAffinityExpected:    sets.New[string]("dev2"),
			notFromAffinityExpected: sets.New[string]("dev1"),
			withoutTopologyExpected: sets.New[string](),
		},
		{
			available:               sets.New[string]("dev1", "dev2", "dev3", "dev4"),
			fromAffinityExpected:    sets.New[string]("dev2", "dev3", "dev4"),
			notFromAffinityExpected: sets.New[string]("dev1"),
			withoutTopologyExpected: sets.New[string](),
		},
	}

	for _, testCase := range testCases {
		fromAffinity, notFromAffinity, withoutTopology := testManager.filterByAffinity("", "", "res1", testCase.available)
		as.Truef(fromAffinity.Equal(testCase.fromAffinityExpected), "expect devices from affinity to be %v but got %v", testCase.fromAffinityExpected, fromAffinity)
		as.Truef(notFromAffinity.Equal(testCase.notFromAffinityExpected), "expect devices not from affinity to be %v but got %v", testCase.notFromAffinityExpected, notFromAffinity)
		as.Truef(withoutTopology.Equal(testCase.withoutTopologyExpected), "expect devices without topology to be %v but got %v", testCase.notFromAffinityExpected, notFromAffinity)
	}
}

func TestPodContainerDeviceAllocation(t *testing.T) {
	tCtx := ktesting.Init(t)
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev1", "dev2"}},
		topology:         true,
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev3", "dev4"}},
		topology:         false,
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)
	testManager, err := getTestManager(tmpDir, podsStub.getActivePods, testResources)
	as.NoError(err)

	testPods := []*v1.Pod{
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res1.resourceQuantity,
			v1.ResourceName("cpu"):             res1.resourceQuantity,
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res1.resourceName): res2.resourceQuantity}),
		makePod(v1.ResourceList{
			v1.ResourceName(res2.resourceName): res2.resourceQuantity}),
	}
	testCases := []struct {
		description               string
		testPod                   *v1.Pod
		expectedContainerOptsLen  []int
		expectedAllocatedResName1 int
		expectedAllocatedResName2 int
		expErr                    error
	}{
		{
			description:               "Successful allocation of two Res1 resources and one Res2 resource",
			testPod:                   testPods[0],
			expectedContainerOptsLen:  []int{3, 2, 2},
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr:                    nil,
		},
		{
			description:               "Requesting to create a pod without enough resources should fail",
			testPod:                   testPods[1],
			expectedContainerOptsLen:  nil,
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 1,
			expErr:                    fmt.Errorf("requested number of devices unavailable for domain1.com/resource1. Requested: 1, Available: 0"),
		},
		{
			description:               "Successful allocation of all available Res1 resources and Res2 resources",
			testPod:                   testPods[2],
			expectedContainerOptsLen:  []int{0, 0, 1},
			expectedAllocatedResName1: 2,
			expectedAllocatedResName2: 2,
			expErr:                    nil,
		},
	}
	activePods := []*v1.Pod{}
	for _, testCase := range testCases {
		pod := testCase.testPod
		activePods = append(activePods, pod)
		podsStub.updateActivePods(activePods)
		err := testManager.Allocate(pod, &pod.Spec.Containers[0])
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("DevicePluginManager error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
		runContainerOpts, err := testManager.GetDeviceRunContainerOptions(tCtx, pod, &pod.Spec.Containers[0])
		if testCase.expErr == nil {
			as.NoError(err)
		}
		if testCase.expectedContainerOptsLen == nil {
			as.Nil(runContainerOpts)
		} else {
			as.Len(runContainerOpts.Devices, testCase.expectedContainerOptsLen[0])
			as.Len(runContainerOpts.Mounts, testCase.expectedContainerOptsLen[1])
			as.Len(runContainerOpts.Envs, testCase.expectedContainerOptsLen[2])
		}
		as.Equal(testCase.expectedAllocatedResName1, testManager.allocatedDevices[res1.resourceName].Len())
		as.Equal(testCase.expectedAllocatedResName2, testManager.allocatedDevices[res2.resourceName].Len())
	}

}

func TestPodContainerDeviceToAllocate(t *testing.T) {
	tCtx := ktesting.Init(t)
	resourceName1 := "domain1.com/resource1"
	resourceName2 := "domain2.com/resource2"
	resourceName3 := "domain2.com/resource3"
	as := require.New(t)
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	testManager := &ManagerImpl{
		endpoints:        make(map[string]endpointInfo),
		healthyDevices:   make(map[string]sets.Set[string]),
		unhealthyDevices: make(map[string]sets.Set[string]),
		allocatedDevices: make(map[string]sets.Set[string]),
		podDevices:       newPodDevices(),
		activePods:       func() []*v1.Pod { return []*v1.Pod{} },
		sourcesReady:     &sourcesReadyStub{},
	}

	testManager.podDevices.insert("pod1", "con1", resourceName1,
		constructDevices([]string{"dev1", "dev2"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r2dev1": "/dev/r2dev1", "/dev/r2dev2": "/dev/r2dev2"}),
			withMounts(map[string]string{"/home/r2lib1": "/usr/r2lib1"}),
			withEnvs(map[string]string{"r2devices": "dev1 dev2"}),
		),
	)
	testManager.podDevices.insert("pod2", "con2", resourceName2,
		checkpoint.DevicesPerNUMA{nodeWithoutTopology: []string{"dev5"}},
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev5": "/dev/r1dev5"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)
	testManager.podDevices.insert("pod3", "con3", resourceName3,
		checkpoint.DevicesPerNUMA{nodeWithoutTopology: []string{"dev5"}},
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev5": "/dev/r1dev5"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)

	// no healthy devices for resourceName1 and devices corresponding to
	// resource2 are intentionally omitted to simulate that the resource
	// hasn't been registered.
	testManager.healthyDevices[resourceName1] = sets.New[string]()
	testManager.healthyDevices[resourceName3] = sets.New[string]()
	// dev5 is no longer in the list of healthy devices
	testManager.healthyDevices[resourceName3].Insert("dev7")
	testManager.healthyDevices[resourceName3].Insert("dev8")

	testCases := []struct {
		description              string
		podUID                   string
		contName                 string
		resource                 string
		required                 int
		reusableDevices          sets.Set[string]
		expectedAllocatedDevices sets.Set[string]
		expErr                   error
	}{
		{
			description:              "Admission error in case no healthy devices to allocate present",
			podUID:                   "pod1",
			contName:                 "con1",
			resource:                 resourceName1,
			required:                 2,
			reusableDevices:          sets.New[string](),
			expectedAllocatedDevices: nil,
			expErr:                   fmt.Errorf("no healthy devices present; cannot allocate unhealthy devices %s", resourceName1),
		},
		{
			description:              "Admission error in case resource is not registered",
			podUID:                   "pod2",
			contName:                 "con2",
			resource:                 resourceName2,
			required:                 1,
			reusableDevices:          sets.New[string](),
			expectedAllocatedDevices: nil,
			expErr:                   fmt.Errorf("cannot allocate unregistered device %s", resourceName2),
		},
		{
			description:              "Admission error in case resource not devices previously allocated no longer healthy",
			podUID:                   "pod3",
			contName:                 "con3",
			resource:                 resourceName3,
			required:                 1,
			reusableDevices:          sets.New[string](),
			expectedAllocatedDevices: nil,
			expErr:                   fmt.Errorf("previously allocated devices are no longer healthy; cannot allocate unhealthy devices %s", resourceName3),
		},
	}

	for _, testCase := range testCases {
		allocDevices, err := testManager.devicesToAllocate(tCtx, testCase.podUID, testCase.contName, testCase.resource, testCase.required, testCase.reusableDevices)
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("devicePluginManager error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
		if !reflect.DeepEqual(allocDevices, testCase.expectedAllocatedDevices) {
			t.Errorf("devicePluginManager error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expectedAllocatedDevices, allocDevices)
		}
	}

}

func TestDevicesToAllocateConflictWithUpdateAllocatedDevices(t *testing.T) {
	tCtx := ktesting.Init(t)
	podToAllocate := "podToAllocate"
	containerToAllocate := "containerToAllocate"
	podToRemove := "podToRemove"
	containerToRemove := "containerToRemove"
	deviceID := "deviceID"
	resourceName := "domain1.com/resource"

	socket := filepath.Join(os.TempDir(), esocketName())
	devs := []*pluginapi.Device{
		{ID: deviceID, Health: pluginapi.Healthy},
	}
	p, e := esetup(tCtx, t, devs, socket, resourceName, func(logger klog.Logger, n string, d []*pluginapi.Device) {})

	waitUpdateAllocatedDevicesChan := make(chan struct{})
	waitSetGetPreferredAllocChan := make(chan struct{})

	p.SetGetPreferredAllocFunc(func(r *pluginapi.PreferredAllocationRequest, devs map[string]*pluginapi.Device) (*pluginapi.PreferredAllocationResponse, error) {
		waitSetGetPreferredAllocChan <- struct{}{}
		<-waitUpdateAllocatedDevicesChan
		return &pluginapi.PreferredAllocationResponse{
			ContainerResponses: []*pluginapi.ContainerPreferredAllocationResponse{
				{
					DeviceIDs: []string{deviceID},
				},
			},
		}, nil
	})

	testManager := &ManagerImpl{
		endpoints:             make(map[string]endpointInfo),
		healthyDevices:        make(map[string]sets.Set[string]),
		unhealthyDevices:      make(map[string]sets.Set[string]),
		allocatedDevices:      make(map[string]sets.Set[string]),
		podDevices:            newPodDevices(),
		activePods:            func() []*v1.Pod { return []*v1.Pod{} },
		sourcesReady:          &sourcesReadyStub{},
		topologyAffinityStore: topologymanager.NewFakeManager(),
	}

	testManager.endpoints[resourceName] = endpointInfo{
		e: e,
		opts: &pluginapi.DevicePluginOptions{
			GetPreferredAllocationAvailable: true,
		},
	}
	testManager.healthyDevices[resourceName] = sets.New[string](deviceID)
	testManager.podDevices.insert(podToRemove, containerToRemove, resourceName, nil, nil)

	go func() {
		<-waitSetGetPreferredAllocChan
		testManager.UpdateAllocatedDevices()
		waitUpdateAllocatedDevicesChan <- struct{}{}
	}()

	set, err := testManager.devicesToAllocate(tCtx, podToAllocate, containerToAllocate, resourceName, 1, sets.New[string]())
	assert.NoError(t, err)
	assert.Equal(t, sets.New[string](deviceID), set)
}

func TestGetDeviceRunContainerOptions(t *testing.T) {
	tCtx := ktesting.Init(t)
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev1", "dev2"}},
		topology:         true,
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev3", "dev4"}},
		topology:         false,
	}

	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)

	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	as := require.New(t)

	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	testManager, err := getTestManager(tmpDir, podsStub.getActivePods, testResources)
	as.NoError(err)

	pod1 := makePod(v1.ResourceList{
		v1.ResourceName(res1.resourceName): res1.resourceQuantity,
		v1.ResourceName(res2.resourceName): res2.resourceQuantity,
	})
	pod2 := makePod(v1.ResourceList{
		v1.ResourceName(res2.resourceName): res2.resourceQuantity,
	})

	activePods := []*v1.Pod{pod1, pod2}
	podsStub.updateActivePods(activePods)

	err = testManager.Allocate(pod1, &pod1.Spec.Containers[0])
	as.NoError(err)
	err = testManager.Allocate(pod2, &pod2.Spec.Containers[0])
	as.NoError(err)

	// when pod is in activePods, GetDeviceRunContainerOptions should return
	runContainerOpts, err := testManager.GetDeviceRunContainerOptions(tCtx, pod1, &pod1.Spec.Containers[0])
	as.NoError(err)
	as.Len(runContainerOpts.Devices, 3)
	as.Len(runContainerOpts.Mounts, 2)
	as.Len(runContainerOpts.Envs, 2)

	activePods = []*v1.Pod{pod2}
	podsStub.updateActivePods(activePods)
	testManager.UpdateAllocatedDevices()

	// when pod is removed from activePods,G etDeviceRunContainerOptions should return error
	runContainerOpts, err = testManager.GetDeviceRunContainerOptions(tCtx, pod1, &pod1.Spec.Containers[0])
	as.NoError(err)
	as.Nil(runContainerOpts)
}

func TestInitContainerDeviceAllocation(t *testing.T) {
	// Requesting to create a pod that requests resourceName1 in init containers and normal containers
	// should succeed with devices allocated to init containers reallocated to normal containers.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev1", "dev2"}},
		topology:         false,
	}
	res2 := TestResource{
		resourceName:     "domain2.com/resource2",
		resourceQuantity: *resource.NewQuantity(int64(1), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev3", "dev4"}},
		topology:         true,
	}
	testResources := make([]TestResource, 2)
	testResources = append(testResources, res1)
	testResources = append(testResources, res2)
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	testManager, err := getTestManager(tmpDir, podsStub.getActivePods, testResources)
	as.NoError(err)

	podWithPluginResourcesInInitContainers := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res1.resourceQuantity,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): res2.resourceQuantity,
							v1.ResourceName(res2.resourceName): res2.resourceQuantity,
						},
					},
				},
			},
		},
	}
	podsStub.updateActivePods([]*v1.Pod{podWithPluginResourcesInInitContainers})
	for _, container := range podWithPluginResourcesInInitContainers.Spec.InitContainers {
		err = testManager.Allocate(podWithPluginResourcesInInitContainers, &container)
	}
	for _, container := range podWithPluginResourcesInInitContainers.Spec.Containers {
		err = testManager.Allocate(podWithPluginResourcesInInitContainers, &container)
	}
	as.NoError(err)
	podUID := string(podWithPluginResourcesInInitContainers.UID)
	initCont1 := podWithPluginResourcesInInitContainers.Spec.InitContainers[0].Name
	initCont2 := podWithPluginResourcesInInitContainers.Spec.InitContainers[1].Name
	normalCont1 := podWithPluginResourcesInInitContainers.Spec.Containers[0].Name
	normalCont2 := podWithPluginResourcesInInitContainers.Spec.Containers[1].Name
	initCont1Devices := testManager.podDevices.containerDevices(podUID, initCont1, res1.resourceName)
	initCont2Devices := testManager.podDevices.containerDevices(podUID, initCont2, res1.resourceName)
	normalCont1Devices := testManager.podDevices.containerDevices(podUID, normalCont1, res1.resourceName)
	normalCont2Devices := testManager.podDevices.containerDevices(podUID, normalCont2, res1.resourceName)
	as.Equal(1, initCont1Devices.Len())
	as.Equal(2, initCont2Devices.Len())
	as.Equal(1, normalCont1Devices.Len())
	as.Equal(1, normalCont2Devices.Len())
	as.True(initCont2Devices.IsSuperset(initCont1Devices))
	as.True(initCont2Devices.IsSuperset(normalCont1Devices))
	as.True(initCont2Devices.IsSuperset(normalCont2Devices))
	as.Equal(0, normalCont1Devices.Intersection(normalCont2Devices).Len())
}

func TestRestartableInitContainerDeviceAllocation(t *testing.T) {
	// Requesting to create a pod that requests resourceName1 in restartable
	// init containers and normal containers should succeed with devices
	// allocated to init containers not reallocated to normal containers.
	oneDevice := resource.NewQuantity(int64(1), resource.DecimalSI)
	twoDevice := resource.NewQuantity(int64(2), resource.DecimalSI)
	threeDevice := resource.NewQuantity(int64(3), resource.DecimalSI)
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(6), resource.DecimalSI),
		devs: checkpoint.DevicesPerNUMA{
			0: []string{"dev1", "dev2", "dev3", "dev4", "dev5", "dev6"},
		},
		topology: false,
	}
	testResources := []TestResource{
		res1,
	}
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	testManager, err := getTestManager(tmpDir, podsStub.getActivePods, testResources)
	as.NoError(err)

	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	podWithPluginResourcesInRestartableInitContainers := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): *threeDevice,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): *oneDevice,
						},
					},
					RestartPolicy: &containerRestartPolicyAlways,
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): *twoDevice,
						},
					},
					RestartPolicy: &containerRestartPolicyAlways,
				},
			},
			Containers: []v1.Container{
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): *oneDevice,
						},
					},
				},
				{
					Name: string(uuid.NewUUID()),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(res1.resourceName): *twoDevice,
						},
					},
				},
			},
		},
	}
	podsStub.updateActivePods([]*v1.Pod{podWithPluginResourcesInRestartableInitContainers})
	for _, container := range podWithPluginResourcesInRestartableInitContainers.Spec.InitContainers {
		err = testManager.Allocate(podWithPluginResourcesInRestartableInitContainers, &container)
	}
	for _, container := range podWithPluginResourcesInRestartableInitContainers.Spec.Containers {
		err = testManager.Allocate(podWithPluginResourcesInRestartableInitContainers, &container)
	}
	as.NoError(err)
	podUID := string(podWithPluginResourcesInRestartableInitContainers.UID)
	regularInitCont1 := podWithPluginResourcesInRestartableInitContainers.Spec.InitContainers[0].Name
	restartableInitCont2 := podWithPluginResourcesInRestartableInitContainers.Spec.InitContainers[1].Name
	restartableInitCont3 := podWithPluginResourcesInRestartableInitContainers.Spec.InitContainers[2].Name
	normalCont1 := podWithPluginResourcesInRestartableInitContainers.Spec.Containers[0].Name
	normalCont2 := podWithPluginResourcesInRestartableInitContainers.Spec.Containers[1].Name
	regularInitCont1Devices := testManager.podDevices.containerDevices(podUID, regularInitCont1, res1.resourceName)
	restartableInitCont2Devices := testManager.podDevices.containerDevices(podUID, restartableInitCont2, res1.resourceName)
	restartableInitCont3Devices := testManager.podDevices.containerDevices(podUID, restartableInitCont3, res1.resourceName)
	normalCont1Devices := testManager.podDevices.containerDevices(podUID, normalCont1, res1.resourceName)
	normalCont2Devices := testManager.podDevices.containerDevices(podUID, normalCont2, res1.resourceName)
	as.Equal(3, regularInitCont1Devices.Len())
	as.Equal(1, restartableInitCont2Devices.Len())
	as.Equal(2, restartableInitCont3Devices.Len())
	as.Equal(1, normalCont1Devices.Len())
	as.Equal(2, normalCont2Devices.Len())
	as.True(regularInitCont1Devices.IsSuperset(restartableInitCont2Devices))
	as.True(regularInitCont1Devices.IsSuperset(restartableInitCont3Devices))
	// regularInitCont1Devices are sharable with other containers

	dedicatedContainerDevices := []sets.Set[string]{
		restartableInitCont2Devices,
		restartableInitCont3Devices,
		normalCont1Devices,
		normalCont2Devices,
	}

	for i := 0; i < len(dedicatedContainerDevices)-1; i++ {
		for j := i + 1; j < len(dedicatedContainerDevices); j++ {
			t.Logf("containerDevices[%d] = %v", i, dedicatedContainerDevices[i])
			t.Logf("containerDevices[%d] = %v", j, dedicatedContainerDevices[j])
			as.Empty(dedicatedContainerDevices[i].Intersection(dedicatedContainerDevices[j]))
		}
	}
}

func TestUpdatePluginResources(t *testing.T) {
	pod := &v1.Pod{}
	pod.UID = types.UID("testPod")

	resourceName1 := "domain1.com/resource1"
	devID1 := "dev1"

	resourceName2 := "domain2.com/resource2"
	devID2 := "dev2"

	as := assert.New(t)
	monitorCallback := func(logger klog.Logger, resourceName string, devices []*pluginapi.Device) {}
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	as.NoError(err)
	m := &ManagerImpl{
		allocatedDevices:  make(map[string]sets.Set[string]),
		healthyDevices:    make(map[string]sets.Set[string]),
		podDevices:        newPodDevices(),
		checkpointManager: ckm,
	}
	testManager := wrappedManagerImpl{
		ManagerImpl: m,
		callback:    monitorCallback,
	}
	testManager.podDevices.devs[string(pod.UID)] = make(containerDevices)

	// require one of resource1 and one of resource2
	testManager.allocatedDevices[resourceName1] = sets.New[string]()
	testManager.allocatedDevices[resourceName1].Insert(devID1)
	testManager.allocatedDevices[resourceName2] = sets.New[string]()
	testManager.allocatedDevices[resourceName2].Insert(devID2)

	cachedNode := &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				// has no resource1 and two of resource2
				v1.ResourceName(resourceName2): *resource.NewQuantity(int64(2), resource.DecimalSI),
			},
		},
	}
	nodeInfo := &schedulerframework.NodeInfo{}
	nodeInfo.SetNode(cachedNode)

	testManager.UpdatePluginResources(nodeInfo, &lifecycle.PodAdmitAttributes{Pod: pod})

	allocatableScalarResources := nodeInfo.Allocatable.ScalarResources
	// allocatable in nodeInfo is less than needed, should update
	as.Equal(1, int(allocatableScalarResources[v1.ResourceName(resourceName1)]))
	// allocatable in nodeInfo is more than needed, should skip updating
	as.Equal(2, int(allocatableScalarResources[v1.ResourceName(resourceName2)]))
}

func TestDevicePreStartContainer(t *testing.T) {
	tCtx := ktesting.Init(t)
	// Ensures that if device manager is indicated to invoke `PreStartContainer` RPC
	// by device plugin, then device manager invokes PreStartContainer at endpoint interface.
	// Also verifies that final allocation of mounts, envs etc is same as expected.
	res1 := TestResource{
		resourceName:     "domain1.com/resource1",
		resourceQuantity: *resource.NewQuantity(int64(2), resource.DecimalSI),
		devs:             checkpoint.DevicesPerNUMA{0: []string{"dev1", "dev2"}},
		topology:         false,
	}
	as := require.New(t)
	podsStub := activePodsStub{
		activePods: []*v1.Pod{},
	}
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)

	testManager, err := getTestManager(tmpDir, podsStub.getActivePods, []TestResource{res1})
	as.NoError(err)

	ch := make(chan []string, 1)
	testManager.endpoints[res1.resourceName] = endpointInfo{
		e: &MockEndpoint{
			initChan:     ch,
			allocateFunc: allocateStubFunc(),
		},
		opts: &pluginapi.DevicePluginOptions{PreStartRequired: true},
	}
	pod := makePod(v1.ResourceList{
		v1.ResourceName(res1.resourceName): res1.resourceQuantity})
	activePods := []*v1.Pod{}
	activePods = append(activePods, pod)
	podsStub.updateActivePods(activePods)
	err = testManager.Allocate(pod, &pod.Spec.Containers[0])
	as.NoError(err)
	runContainerOpts, err := testManager.GetDeviceRunContainerOptions(tCtx, pod, &pod.Spec.Containers[0])
	as.NoError(err)
	var initializedDevs []string
	select {
	case <-time.After(time.Second):
		t.Fatalf("Timed out while waiting on channel for response from PreStartContainer RPC stub")
	case initializedDevs = <-ch:
		break
	}

	as.Contains(initializedDevs, "dev1")
	as.Contains(initializedDevs, "dev2")
	as.Len(initializedDevs, res1.devs.Devices().Len())

	expectedResps, err := allocateStubFunc()([]string{"dev1", "dev2"})
	as.NoError(err)
	as.Len(expectedResps.ContainerResponses, 1)
	expectedResp := expectedResps.ContainerResponses[0]
	as.Len(runContainerOpts.Devices, len(expectedResp.Devices))
	as.Len(runContainerOpts.Mounts, len(expectedResp.Mounts))
	as.Len(runContainerOpts.Envs, len(expectedResp.Envs))

	pod2 := makePod(v1.ResourceList{
		v1.ResourceName(res1.resourceName): *resource.NewQuantity(int64(0), resource.DecimalSI)})
	activePods = append(activePods, pod2)
	podsStub.updateActivePods(activePods)
	err = testManager.Allocate(pod2, &pod2.Spec.Containers[0])
	as.NoError(err)
	_, err = testManager.GetDeviceRunContainerOptions(tCtx, pod2, &pod2.Spec.Containers[0])
	as.NoError(err)
	select {
	case <-time.After(time.Millisecond):
		t.Log("When pod resourceQuantity is 0,  PreStartContainer RPC stub will be skipped")
	case <-ch:
		break
	}
}

func TestResetExtendedResource(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	as := assert.New(t)
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	as.NoError(err)
	defer os.RemoveAll(tmpDir)
	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	as.NoError(err)
	testManager := &ManagerImpl{
		endpoints:         make(map[string]endpointInfo),
		healthyDevices:    make(map[string]sets.Set[string]),
		unhealthyDevices:  make(map[string]sets.Set[string]),
		allocatedDevices:  make(map[string]sets.Set[string]),
		podDevices:        newPodDevices(),
		checkpointManager: ckm,
	}

	extendedResourceName := "domain.com/resource"
	testManager.podDevices.insert("pod", "con", extendedResourceName,
		constructDevices([]string{"dev1"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/dev1": "/dev/dev1"}),
			withMounts(map[string]string{"/home/lib1": "/usr/lib1"}),
		),
	)

	testManager.healthyDevices[extendedResourceName] = sets.New[string]()
	testManager.healthyDevices[extendedResourceName].Insert("dev1")
	// checkpoint is present, indicating node hasn't been recreated
	err = testManager.writeCheckpoint(logger)
	require.NoError(t, err)

	as.False(testManager.ShouldResetExtendedResourceCapacity())

	// checkpoint is absent, representing node recreation
	ckpts, err := ckm.ListCheckpoints()
	as.NoError(err)
	for _, ckpt := range ckpts {
		err = ckm.RemoveCheckpoint(ckpt)
		as.NoError(err)
	}
	as.True(testManager.ShouldResetExtendedResourceCapacity())
}

func allocateStubFunc() func(devs []string) (*pluginapi.AllocateResponse, error) {
	return func(devs []string) (*pluginapi.AllocateResponse, error) {
		resp := new(pluginapi.ContainerAllocateResponse)
		resp.Envs = make(map[string]string)
		for _, dev := range devs {
			switch dev {
			case "dev1":
				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/aaa",
					HostPath:      "/dev/aaa",
					Permissions:   "mrw",
				})

				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/bbb",
					HostPath:      "/dev/bbb",
					Permissions:   "mrw",
				})

				resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
					ContainerPath: "/container_dir1/file1",
					HostPath:      "host_dir1/file1",
					ReadOnly:      true,
				})

			case "dev2":
				resp.Devices = append(resp.Devices, &pluginapi.DeviceSpec{
					ContainerPath: "/dev/ccc",
					HostPath:      "/dev/ccc",
					Permissions:   "mrw",
				})

				resp.Mounts = append(resp.Mounts, &pluginapi.Mount{
					ContainerPath: "/container_dir1/file2",
					HostPath:      "host_dir1/file2",
					ReadOnly:      true,
				})

				resp.Envs["key1"] = "val1"
			}
		}
		resps := new(pluginapi.AllocateResponse)
		resps.ContainerResponses = append(resps.ContainerResponses, resp)
		return resps, nil
	}
}

func makeDevice(devOnNUMA checkpoint.DevicesPerNUMA, topology bool) map[string]*pluginapi.Device {
	res := make(map[string]*pluginapi.Device)
	var topologyInfo *pluginapi.TopologyInfo
	for node, devs := range devOnNUMA {
		if topology {
			topologyInfo = &pluginapi.TopologyInfo{Nodes: []*pluginapi.NUMANode{{ID: node}}}
		} else {
			topologyInfo = nil
		}
		for idx := range devs {
			res[devs[idx]] = &pluginapi.Device{ID: devs[idx], Topology: topologyInfo}
		}
	}
	return res
}

func TestGetTopologyHintsWithUpdates(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	defer os.RemoveAll(socketDir)
	require.NoError(t, err)

	devs := []*pluginapi.Device{}
	for i := 0; i < 1000; i++ {
		devs = append(devs, &pluginapi.Device{
			ID:     fmt.Sprintf("dev-%d", i),
			Health: pluginapi.Healthy,
			Topology: &pluginapi.TopologyInfo{
				Nodes: []*pluginapi.NUMANode{
					{ID: 0},
				},
			}})
	}
	testPod := makePod(v1.ResourceList{
		testResourceName: *resource.NewQuantity(int64(1), resource.DecimalSI),
	})
	topology := []cadvisorapi.Node{
		{Id: 0},
	}
	testCases := []struct {
		description string
		count       int
		devices     []*pluginapi.Device
		testfunc    func(manager *wrappedManagerImpl)
	}{
		{
			description: "GetTopologyHints data race when update device",
			count:       10,
			devices:     devs,
			testfunc: func(manager *wrappedManagerImpl) {
				manager.GetTopologyHints(testPod, &testPod.Spec.Containers[0])
			},
		},
		{
			description: "GetPodTopologyHints data race when update device",
			count:       10,
			devices:     devs,
			testfunc: func(manager *wrappedManagerImpl) {
				manager.GetPodTopologyHints(testPod)
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.description, func(t *testing.T) {
			m, _ := setupDeviceManager(t, nil, nil, socketName, topology, logger)
			defer func() {
				err := m.Stop(logger)
				require.NoError(t, err)
			}()
			mimpl := m.(*wrappedManagerImpl)

			wg := sync.WaitGroup{}
			wg.Add(2)

			updated := atomic.Bool{}
			updated.Store(false)
			go func() {
				defer wg.Done()
				for i := 0; i < test.count; i++ {
					// simulate the device plugin to send device updates
					mimpl.genericDeviceUpdateCallback(logger, testResourceName, devs)
				}
				updated.Store(true)
			}()
			go func() {
				defer wg.Done()
				for !updated.Load() {
					// When a data race occurs, golang will throw an error, and recover() cannot catch this error,
					// Such as: `throw("Concurrent map iteration and map writing")`.
					// When this test ends quietly, no data race error occurs.
					// Otherwise, the test process exits automatically and prints all goroutine call stacks.
					test.testfunc(mimpl)
				}
			}()
			wg.Wait()
		})
	}
}
func TestUpdateAllocatedResourcesStatus(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	podUID := "test-pod-uid"
	containerName := "test-container"
	resourceName := "test-resource"

	tmpDir, err := os.MkdirTemp("", "checkpoint")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}

	defer func() {
		err = os.RemoveAll(tmpDir)
		if err != nil {
			t.Fatalf("Fail to remove tmpdir: %v", err)
		}
	}()
	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	if err != nil {
		t.Fatalf("failed to create checkpoint manager: %v", err)
	}

	testManager := &ManagerImpl{
		endpoints:         make(map[string]endpointInfo),
		healthyDevices:    make(map[string]sets.Set[string]),
		unhealthyDevices:  make(map[string]sets.Set[string]),
		allocatedDevices:  make(map[string]sets.Set[string]),
		allDevices:        make(map[string]DeviceInstances),
		podDevices:        newPodDevices(),
		checkpointManager: ckm,
	}

	testManager.podDevices.insert(podUID, containerName, resourceName,
		constructDevices([]string{"dev1", "dev2"}),
		newContainerAllocateResponse(
			withDevices(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"}),
			withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
		),
	)

	testManager.genericDeviceUpdateCallback(logger, resourceName, []*pluginapi.Device{
		{ID: "dev1", Health: pluginapi.Healthy},
		{ID: "dev2", Health: pluginapi.Unhealthy},
	})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(podUID),
		},
	}
	status := &v1.PodStatus{
		ContainerStatuses: []v1.ContainerStatus{
			{
				Name: containerName,
			},
		},
	}
	testManager.UpdateAllocatedResourcesStatus(pod, status)

	expectedStatus := v1.ResourceStatus{
		Name: v1.ResourceName(resourceName),
		Resources: []v1.ResourceHealth{
			{
				ResourceID: "dev1",
				Health:     pluginapi.Healthy,
			},
			{
				ResourceID: "dev2",
				Health:     pluginapi.Unhealthy,
			},
		},
	}
	expectedContainerStatuses := []v1.ContainerStatus{
		{
			Name:                     containerName,
			AllocatedResourcesStatus: []v1.ResourceStatus{expectedStatus},
		},
	}

	// Sort the resources for the expected status and actual status
	sortContainerStatuses(status.ContainerStatuses)
	sortContainerStatuses(expectedContainerStatuses)

	if !reflect.DeepEqual(status.ContainerStatuses, expectedContainerStatuses) {
		t.Errorf("UpdateAllocatedResourcesStatus failed, expected: %v, got: %v", expectedContainerStatuses, status.ContainerStatuses)
	}
}

// Helper function to sort ResourceHealth slices
func sortResourceHealth(resources []v1.ResourceHealth) {
	sort.SliceStable(resources, func(i, j int) bool {
		return resources[i].ResourceID < resources[j].ResourceID
	})
}

// Helper function to sort ContainerStatus slices
func sortContainerStatuses(statuses []v1.ContainerStatus) {
	for i := range statuses {
		for j := range statuses[i].AllocatedResourcesStatus {
			sortResourceHealth(statuses[i].AllocatedResourcesStatus[j].Resources)
		}
	}
}

func TestFeatureGateResourceHealthStatus(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tmpDir, err := os.MkdirTemp("", "checkpoint")
	require.NoError(t, err, "err should be nil")
	defer func() {
		err = os.RemoveAll(tmpDir)
		if err != nil {
			t.Fatalf("Fail to remove tmpdir: %v", err)
		}
	}()
	ckm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	require.NoError(t, err, "err should be nil")
	resourceName := "domain1.com/resource1"
	existDevices := map[string]DeviceInstances{}
	resourceNameMap := make(map[string]*pluginapi.Device)
	deviceUpdateNumber, deviceUpdateChanBuffer := 200, 100
	for i := 0; i < deviceUpdateNumber; i++ {
		resourceNameMap[fmt.Sprintf("dev%d", i)] = &pluginapi.Device{
			ID:     fmt.Sprintf("dev%d", i),
			Health: pluginapi.Healthy,
		}
	}
	existDevices[resourceName] = resourceNameMap

	testManager := &ManagerImpl{
		allDevices:        ResourceDeviceInstances(existDevices),
		endpoints:         make(map[string]endpointInfo),
		healthyDevices:    make(map[string]sets.Set[string]),
		unhealthyDevices:  make(map[string]sets.Set[string]),
		allocatedDevices:  make(map[string]sets.Set[string]),
		podDevices:        newPodDevices(),
		checkpointManager: ckm,
		update:            make(chan resourceupdates.Update, deviceUpdateChanBuffer),
	}

	for i := 0; i < deviceUpdateNumber; i++ {
		podID := fmt.Sprintf("pod%d", i)
		contID := fmt.Sprintf("con%d", i)
		devices := checkpoint.DevicesPerNUMA{0: []string{fmt.Sprintf("dev%d", i)}}
		testManager.podDevices.insert(podID, contID, resourceName,
			devices,
			newContainerAllocateResponse(
				withDevices(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"}),
				withMounts(map[string]string{"/home/r1lib1": "/usr/r1lib1"}),
			),
		)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ResourceHealthStatus, true)

	for i := 0; i < deviceUpdateNumber; i++ {
		testManager.genericDeviceUpdateCallback(logger, resourceName, []*pluginapi.Device{
			{ID: "dev1", Health: pluginapi.Healthy},
		})
	}
	// update chan no data
	assert.Empty(t, testManager.update)

	// update device status, assume all device unhealthy.
	for i := 0; i < deviceUpdateNumber; i++ {
		testManager.genericDeviceUpdateCallback(logger, resourceName, []*pluginapi.Device{
			{ID: fmt.Sprintf("dev%d", i), Health: pluginapi.Unhealthy},
		})
	}
	for i := 0; i < deviceUpdateChanBuffer; i++ {
		u := <-testManager.update
		assert.Equal(t, resourceupdates.Update{
			PodUIDs: []string{fmt.Sprintf("pod%d", i)},
		}, u)
	}
}

// TestAdmitPodWithDRAResources verifies the behavior of admission
// of the pods referring DRA extended resources depending on whether
// the DRAExtendedResource feature gate is enabled or disabled.
func TestAdmitPodWithDRAResources(t *testing.T) {
	testCases := map[string]struct {
		enableFeatureGate bool
		checkError        func(t require.TestingT, err error, msgAndArgs ...interface{})
	}{
		"DRAExtendedResource enabled": {
			enableFeatureGate: true,
			checkError:        require.NoError,
		},
		"DRAExtendedResource disabled": {
			enableFeatureGate: false,
			checkError:        require.Error,
		},
	}

	containerName := "container1"
	resourceName := "domain1.com/resource1"

	for description, test := range testCases {
		t.Run(description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, test.enableFeatureGate)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: uuid.NewUUID(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: containerName,
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceName(resourceName): resource.MustParse("1"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						RequestMappings: []v1.ContainerExtendedResourceRequest{
							{
								ContainerName: containerName,
								ResourceName:  resourceName,
							},
						},
					},
				},
			}

			require.True(t, isDRAExtendedResource(pod, containerName, resourceName))

			testManager := &ManagerImpl{
				devicesToReuse: make(PodReusableDevices),
				podDevices:     newPodDevices(),
				allocatedDevices: map[string]sets.Set[string]{
					resourceName: sets.New("Dev"),
				},
				activePods:   func() []*v1.Pod { return nil },
				sourcesReady: &sourcesReadyStub{},
			}

			err := testManager.Allocate(pod, &pod.Spec.Containers[0])
			test.checkError(t, err)
		})
	}
}

// TestEndpointSyncOnDisconnect verifies that when a device plugin disconnects,
// the device manager correctly updates its internal state by marking all
// devices from that endpoint as unhealthy. It ensures that the healthyDevices
// and unhealthyDevices maps are in sync with the plugin endpoint info.
func TestEndpointSyncOnDisconnect(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	require.NoError(t, err)
	defer func() {
		if err := os.RemoveAll(socketDir); err != nil {
			logger.Error(err, "unable to remove socket directory", "dir", socketDir)
		}
	}()

	manager, err := newManagerImpl(logger, socketName, nil, nil)
	require.NoError(t, err)

	resourceName := "domain1.com/resource1"
	ep := &endpointImpl{
		resourceName: resourceName,
		client:       plugin.NewPluginClient(resourceName, socketName, manager),
		stopTime:     time.Now().Add(-endpointStopGracePeriod * 2), // make the grace period expired
		socket:       socketName,
	}

	eInfo := endpointInfo{e: ep, opts: nil}
	manager.endpoints[resourceName] = eInfo
	manager.endpointStore[resourceName] = map[string]*endpointInfo{socketName: &eInfo}
	devs := []*pluginapi.Device{
		{ID: "Device1", Health: pluginapi.Healthy},
		{ID: "Device2", Health: pluginapi.Healthy},
		{ID: "Device3", Health: pluginapi.Unhealthy},
	}
	manager.genericDeviceUpdateCallback(logger, resourceName, devs)

	// Disconnect should result in all devices for this resource
	// moved to the unhealthy set.
	err = ep.client.Disconnect(logger)
	require.NoError(t, err)

	require.Contains(t, manager.endpoints, resourceName)
	require.Contains(t, manager.healthyDevices, resourceName)
	require.Contains(t, manager.unhealthyDevices, resourceName)
	require.Len(t, manager.endpoints, 1)
	require.Empty(t, manager.healthyDevices[resourceName])
	require.Equal(t, len(devs), manager.unhealthyDevices[resourceName].Len())

	// Expire endpoint to shorten the test
	ep.stopTime = time.Now().Add(-endpointStopGracePeriod * 2)
	// Call GetCapacity to trigger https://github.com/kubernetes/kubernetes/issues/133702
	manager.GetCapacity()

	require.Empty(t, manager.endpoints)
	require.Empty(t, manager.healthyDevices)
	require.Empty(t, manager.unhealthyDevices)
}

// --- Socket-level endpoint lifecycle tests ---
//
// The tests below exercise PluginConnected / PluginDisconnected at the
// socket-path level, bypassing real gRPC connections. They use
// fakeDevicePluginAPI (a minimal pluginapi.DevicePluginClient) and operate
// directly on the manager's endpointStore to verify connect/disconnect
// semantics, race conditions, and endpoint identity.

// fakeDevicePluginAPI satisfies pluginapi.DevicePluginClient enough for
// ManagerImpl.PluginConnected to capture device-plugin options without a
// real gRPC connection. PluginConnected only calls GetDevicePluginOptions
// before storing the endpoint; the other methods are not exercised by the
// manager's connect/disconnect paths and so are left to the embedded nil
// interface to panic if a future change starts calling them — that panic
// is intentional, a signal that the test needs updating.
type fakeDevicePluginAPI struct {
	pluginapi.DevicePluginClient
	opts           *pluginapi.DevicePluginOptions
	allocateCalled int
}

func (f *fakeDevicePluginAPI) GetDevicePluginOptions(_ context.Context, _ *pluginapi.Empty, _ ...grpc.CallOption) (*pluginapi.DevicePluginOptions, error) {
	return f.opts, nil
}

func (f *fakeDevicePluginAPI) Allocate(_ context.Context, _ *pluginapi.AllocateRequest, _ ...grpc.CallOption) (*pluginapi.AllocateResponse, error) {
	f.allocateCalled++
	return &pluginapi.AllocateResponse{
		ContainerResponses: []*pluginapi.ContainerAllocateResponse{{}},
	}, nil
}

// fakeDevicePlugin is a minimal plugin.DevicePlugin that the manager treats
// as a device-plugin handle. Each test owns its own fakes so identity
// comparisons (api pointer equality) are meaningful.
type fakeDevicePlugin struct {
	api      pluginapi.DevicePluginClient
	resource string
	socket   string
}

func (f *fakeDevicePlugin) API() pluginapi.DevicePluginClient { return f.api }
func (f *fakeDevicePlugin) Resource() string                  { return f.resource }
func (f *fakeDevicePlugin) SocketPath() string                { return f.socket }

func newFakeDevicePlugin(resource, socket string) *fakeDevicePlugin {
	return &fakeDevicePlugin{
		api:      &fakeDevicePluginAPI{opts: &pluginapi.DevicePluginOptions{}},
		resource: resource,
		socket:   socket,
	}
}

// newSameSocketTestManager builds a ManagerImpl in the same way the existing
// TestEndpointSyncOnDisconnect does (real mutex / endpointStore behavior),
// but parameterised so each same-socket test gets a fresh scratch directory.
func newSameSocketTestManager(t *testing.T) (*ManagerImpl, func()) {
	t.Helper()
	logger, _ := ktesting.NewTestContext(t)
	socketDir, socketName, _, err := tmpSocketDir()
	require.NoError(t, err)
	manager, err := newManagerImpl(logger, socketName, nil, nil)
	require.NoError(t, err)
	cleanup := func() {
		if err := os.RemoveAll(socketDir); err != nil {
			logger.Error(err, "unable to remove socket directory", "dir", socketDir)
		}
	}
	return manager, cleanup
}

// makeEndpointAt is the hand-rolled endpointImpl helper that the plan calls
// for — mirrors the construction inside TestEndpointSyncOnDisconnect so the
// PluginDisconnected path operates on a real endpointImpl whose socketPath()
// returns the expected value.
func makeEndpointAt(resourceName, socketPath string) *endpointImpl {
	return &endpointImpl{
		resourceName: resourceName,
		socket:       socketPath,
	}
}

// installEndpoint atomically populates both the primary endpoints map and
// the per-endpoint store for one (resource, socket) pair, matching what
// PluginConnected would do without exercising the connect path.
func installEndpoint(m *ManagerImpl, resourceName string, ep *endpointImpl) {
	info := endpointInfo{e: ep, opts: &pluginapi.DevicePluginOptions{}}
	m.endpoints[resourceName] = info
	if m.endpointStore[resourceName] == nil {
		m.endpointStore[resourceName] = map[string]*endpointInfo{}
	}
	m.endpointStore[resourceName][ep.socketPath()] = &endpointInfo{e: ep, opts: info.opts}
}

// TestPluginConnected_SameResourceSameSocketRejected verifies that a second
// PluginConnected for the same (resource, socket) is rejected without mutating
// the existing endpoint state.
func TestPluginConnected_SameResourceSameSocketRejected(t *testing.T) {
	_, tCtx := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"
	p1 := newFakeDevicePlugin(resourceName, socketA)
	p2 := newFakeDevicePlugin(resourceName, socketA)

	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p1),
		"first PluginConnected for (resource, socketA) must succeed")

	err := manager.PluginConnected(tCtx, resourceName, p2)
	require.Error(t, err, "second PluginConnected at the same socket must be rejected (I1)")
	require.Contains(t, err.Error(), "device plugin already connected",
		"rejection error must use the documented prefix from manager.go (I1)")
	require.Contains(t, err.Error(), socketA,
		"rejection error must include the offending socket path (I1)")

	require.Len(t, manager.endpointStore[resourceName], 1,
		"endpointStore must still contain exactly one entry after rejection (I1)")
	stored, ok := manager.endpointStore[resourceName][socketA]
	require.True(t, ok, "stored entry must be at socketA")
	storedImpl, ok := stored.e.(*endpointImpl)
	require.True(t, ok, "stored endpoint must be *endpointImpl")
	require.Same(t, p1.api, storedImpl.api,
		"first endpoint's api pointer must survive the rejected second register (I1)")
	primaryImpl, ok := manager.endpoints[resourceName].e.(*endpointImpl)
	require.True(t, ok)
	require.Same(t, p1.api, primaryImpl.api,
		"primary m.endpoints slot must also still reference the first endpoint (I1)")
}

// TestPluginConnected_SameResourceDifferentSocketsCoexist verifies that two
// endpoints for the same resource at different socket paths can coexist.
func TestPluginConnected_SameResourceDifferentSocketsCoexist(t *testing.T) {
	_, tCtx := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"
	const socketB = "/var/lib/kubelet/plugins/socketB.sock"
	pA := newFakeDevicePlugin(resourceName, socketA)
	pB := newFakeDevicePlugin(resourceName, socketB)

	require.NoError(t, manager.PluginConnected(tCtx, resourceName, pA))
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, pB),
		"two endpoints at different socket paths for the same resource must coexist")

	require.Len(t, manager.endpointStore[resourceName], 2,
		"endpointStore must record both endpoints (I3 setup)")
	require.Contains(t, manager.endpointStore[resourceName], socketA)
	require.Contains(t, manager.endpointStore[resourceName], socketB)

	// The manager does not promise which sibling is the primary when more
	// than one is registered — Go map iteration order is unspecified
	// (manager.go's promote-survivor branch in PluginDisconnected), and
	// PluginConnected's last-write-wins is an implementation detail that
	// callers must not depend on. Assert only that the primary is one of
	// the two registered plugins.
	primaryImpl, ok := manager.endpoints[resourceName].e.(*endpointImpl)
	require.True(t, ok)
	require.True(t, primaryImpl.api == pA.api || primaryImpl.api == pB.api,
		"primary endpoint must be one of the two registered plugins")
}

// TestPluginDisconnected_WrongSocketIsNoop verifies that PluginDisconnected
// for a non-matching or unknown socket path does not mutate any state.
func TestPluginDisconnected_WrongSocketIsNoop(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"
	const socketB = "/var/lib/kubelet/plugins/socketB.sock"
	ep := makeEndpointAt(resourceName, socketA)
	installEndpoint(manager, resourceName, ep)
	manager.healthyDevices[resourceName] = sets.New("dev1", "dev2")

	manager.PluginDisconnected(logger, resourceName, socketB)

	require.Len(t, manager.endpointStore[resourceName], 1,
		"PluginDisconnected for a non-matching socket must not remove the existing entry (I2)")
	require.Contains(t, manager.endpointStore[resourceName], socketA,
		"existing socketA entry must survive (I2)")
	require.True(t, manager.endpoints[resourceName].e.(*endpointImpl).stopTime.IsZero(),
		"a no-op disconnect must not call setStopTime on the surviving endpoint (I2)")
	require.Equal(t, 2, manager.healthyDevices[resourceName].Len(),
		"healthy devices must remain healthy when no endpoint actually disconnected (I2)")

	// Also exercise the resourceName-unknown branch.
	manager.PluginDisconnected(logger, "unknown.com/resource", socketA)
	require.Len(t, manager.endpointStore[resourceName], 1,
		"PluginDisconnected for an unknown resource must not touch unrelated state (I2)")
}

// TestPluginDisconnected_PromotesSurvivor verifies that when one of multiple
// endpoints for a resource disconnects, an arbitrary survivor is promoted to
// the primary slot and healthy devices remain healthy.
func TestPluginDisconnected_PromotesSurvivor(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"
	const socketB = "/var/lib/kubelet/plugins/socketB.sock"
	epA := makeEndpointAt(resourceName, socketA)
	epB := makeEndpointAt(resourceName, socketB)
	installEndpoint(manager, resourceName, epA)
	installEndpoint(manager, resourceName, epB)
	// installEndpoint sets the primary to whichever was inserted last; pin
	// the primary to epA so the test specifically removes the primary and
	// observes the survivor being promoted into m.endpoints.
	manager.endpoints[resourceName] = endpointInfo{e: epA, opts: &pluginapi.DevicePluginOptions{}}
	manager.healthyDevices[resourceName] = sets.New("dev1")

	manager.PluginDisconnected(logger, resourceName, socketA)

	require.Len(t, manager.endpointStore[resourceName], 1,
		"the disconnected endpoint must be removed from endpointStore (I3)")
	require.NotContains(t, manager.endpointStore[resourceName], socketA,
		"socketA must be gone (I3)")
	require.Contains(t, manager.endpointStore[resourceName], socketB,
		"the surviving sibling at socketB must remain in endpointStore (I3)")
	primary, ok := manager.endpoints[resourceName].e.(*endpointImpl)
	require.True(t, ok)
	require.Equal(t, socketB, primary.socketPath(),
		"the surviving sibling must be promoted into m.endpoints (I3)")
	require.Equal(t, 1, manager.healthyDevices[resourceName].Len(),
		"healthy devices must stay healthy when this was NOT the last endpoint (I3)")
	require.False(t, epA.stopTime.IsZero(),
		"the removed endpoint must have setStopTime called (I3)")
}

// TestPluginDisconnected_LastEndpointMarksUnhealthy verifies that when the
// last endpoint for a resource disconnects, all its devices are marked
// unhealthy and the resource is removed from endpointStore.
func TestPluginDisconnected_LastEndpointMarksUnhealthy(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"
	ep := makeEndpointAt(resourceName, socketA)
	installEndpoint(manager, resourceName, ep)
	manager.healthyDevices[resourceName] = sets.New("dev1", "dev2")

	manager.PluginDisconnected(logger, resourceName, socketA)

	require.NotContains(t, manager.endpointStore, resourceName,
		"the resource entry must be fully removed from endpointStore on last disconnect")
	require.Equal(t, 0, manager.healthyDevices[resourceName].Len(),
		"markResourceUnhealthy must zero healthyDevices on last disconnect")
	require.Equal(t, 2, manager.unhealthyDevices[resourceName].Len(),
		"markResourceUnhealthy must migrate the previously healthy IDs into unhealthyDevices")
	require.False(t, ep.stopTime.IsZero(),
		"setStopTime must have been called on the removed endpoint")
}

// TestSameSocketRace_LateDisconnectAfterReconnect verifies that a stale
// PluginDisconnected callback for an old endpoint evicts a fresh endpoint
// that reused the same socket path. This is current behavior, but it is not
// the best practice to reuse socket paths. It is also a recoverable
// state as ListAndWatch will fail and the plugin will be reconnecting.
func TestSameSocketRace_LateDisconnectAfterReconnect(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"

	// Step 1: register e1 at socketA via the real connect path.
	p1 := newFakeDevicePlugin(resourceName, socketA)
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p1))
	manager.healthyDevices[resourceName] = sets.New("dev1")

	// Step 2: disconnect e1 — this is the last endpoint so the resource
	// transitions to unhealthy and the resource key is removed from
	// endpointStore.
	manager.PluginDisconnected(logger, resourceName, socketA)
	require.NotContains(t, manager.endpointStore, resourceName,
		"after disconnecting the only endpoint, the resource key must be gone")

	// Step 3: register e2 at the same socketA — the store is empty for
	// this socket so the connect succeeds.
	p2 := newFakeDevicePlugin(resourceName, socketA)
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p2),
		"reconnecting at the same socket after a clean disconnect must succeed")

	// e2's connect doesn't repopulate healthyDevices on its own (that
	// would normally come from the device plugin's ListAndWatch stream).
	// Repopulate it so the assertion below ("devices remain healthy")
	// has something to observe.
	manager.healthyDevices[resourceName] = sets.New("dev1")

	// Step 4: a *second*, late PluginDisconnected for socketA arrives —
	// this is the stale callback for e1 from the kubelet plugin server.
	// Snapshot state, fire the callback, and assert what happens.
	primaryBefore, ok := manager.endpoints[resourceName].e.(*endpointImpl)
	require.True(t, ok)
	require.Same(t, p2.api, primaryBefore.api,
		"sanity: e2 must be the primary endpoint before the late callback")

	manager.PluginDisconnected(logger, resourceName, socketA)

	// this is mostly for documentation purposes. Plugin will be rediscovered if
	// socket stays open. And it is also not the best practice to reuse socket path.
	require.NotContains(t, manager.endpointStore, resourceName,
		"current behavior: late disconnect callback keyed on socket path evicts the fresh endpoint (I4 regression risk)")
	require.Equal(t, 0, manager.healthyDevices[resourceName].Len(),
		"current behavior: late callback also drives the resource unhealthy because it counts as the last-endpoint disconnect")
}

// TestSameSocketRace_DisconnectBeforeReconnectAttempt verifies that a connect
// attempt at an occupied socket is rejected, but succeeds after the old
// endpoint disconnects — the expected recovery path.
func TestSameSocketRace_DisconnectBeforeReconnectAttempt(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"

	p1 := newFakeDevicePlugin(resourceName, socketA)
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p1))

	// Attempt to reconnect WITHOUT first disconnecting — must be rejected.
	p2 := newFakeDevicePlugin(resourceName, socketA)
	err := manager.PluginConnected(tCtx, resourceName, p2)
	require.Error(t, err, "second connect without an intervening disconnect must be rejected")
	require.Contains(t, err.Error(), "device plugin already connected",
		"rejection error must match the documented prefix (I1)")

	// Sanity: the rejected register must not corrupt state.
	stored, ok := manager.endpointStore[resourceName][socketA]
	require.True(t, ok)
	require.Same(t, p1.api, stored.e.(*endpointImpl).api,
		"rejected register must leave the original endpoint untouched (I1)")

	// Now disconnect the original endpoint and try again — must succeed.
	manager.PluginDisconnected(logger, resourceName, socketA)
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p2),
		"after disconnect, reconnect at the same socket must succeed")

	primary, ok := manager.endpoints[resourceName].e.(*endpointImpl)
	require.True(t, ok)
	require.Same(t, p2.api, primary.api,
		"the fresh p2 endpoint must be installed as the primary after a clean reconnect")
}

// TestSameSocketRace_FastTakeoverMayResultInInfiniteRetries emulates a fast
// socket takeover: plugin1 triggers registration, but by the time its
// Connect() dials the socket, plugin2 is already listening there. Plugin1's
// PluginConnected succeeds first (it connected to plugin2's server). When
// plugin2 then tries to register itself at the same (resourceName,
// socketPath), it is rejected — the duplicate check in PluginConnected
// prevents two endpoints from coexisting at the same socket path.
//
// This is not ideal behaviour: plugin2 is the rightful owner of the socket,
// yet it cannot register. Plugin1 holds the endpointStore slot and runs
// ListAndWatch against plugin2's gRPC server — which is alive and serving —
// so plugin1's stream never breaks and PluginDisconnected is never called to
// clear the slot. Plugin2's retries will be rejected indefinitely. Recovery
// requires plugin2 to restart its gRPC server (which severs plugin1's stream,
// triggering disconnect and clearing the slot) or external intervention
// (e.g. kubelet restart).
func TestSameSocketRace_FastTakeoverMayResultInInfiniteRetries(t *testing.T) {
	_, tCtx := ktesting.NewTestContext(t)
	manager, cleanup := newSameSocketTestManager(t)
	defer cleanup()

	const resourceName = "domain1.com/resource1"
	const socketA = "/var/lib/kubelet/plugins/socketA.sock"

	// p1 represents plugin1 whose dial connected to plugin2's server
	// (fast takeover — plugin2 took over the socket before plugin1 dialed).
	p1 := newFakeDevicePlugin(resourceName, socketA)
	require.NoError(t, manager.PluginConnected(tCtx, resourceName, p1))

	// p2 represents plugin2 — the rightful owner of the socket — trying to
	// register. It is rejected because p1 already occupies that slot.
	p2 := newFakeDevicePlugin(resourceName, socketA)
	err := manager.PluginConnected(tCtx, resourceName, p2)
	require.Error(t, err, "plugin2 is rejected even though it owns the socket")
	require.Contains(t, err.Error(), "device plugin already connected")

	// Simulate plugin2 retrying registration — it keeps failing because
	// plugin1's entry is never cleared (its ListAndWatch stream is alive).
	for retry := range 3 {
		p2Retry := newFakeDevicePlugin(resourceName, socketA)
		err = manager.PluginConnected(tCtx, resourceName, p2Retry)
		require.Error(t, err, "retry %d: plugin2 still cannot register", retry)
		require.Contains(t, err.Error(), "device plugin already connected")
	}

	// Plugin1's stale registration remains — this is the problematic state.
	require.Len(t, manager.endpointStore[resourceName], 1)
	stored := manager.endpointStore[resourceName][socketA]
	require.Same(t, p1.api, stored.e.(*endpointImpl).api,
		"plugin1's stale endpoint persists; plugin2 cannot take over")

	// Verify that Allocate calls go to plugin1's API — which in the real
	// scenario is plugin2's gRPC server (since plugin1's dial connected
	// to plugin2's listener). The kubelet thinks it's talking to plugin1,
	// but the RPC actually reaches plugin2.
	p1API := p1.api.(*fakeDevicePluginAPI)
	p2API := p2.api.(*fakeDevicePluginAPI)
	ep := stored.e.(*endpointImpl)
	resp, err := ep.allocate(tCtx, []string{"dev1"})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Equal(t, 1, p1API.allocateCalled,
		"Allocate must reach plugin1's API (which is really plugin2's server in the real race)")
	require.Equal(t, 0, p2API.allocateCalled,
		"plugin2's own API is never called — it was never registered")
}
