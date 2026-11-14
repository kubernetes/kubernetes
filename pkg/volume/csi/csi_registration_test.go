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

package csi

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	storageconversion "k8s.io/kubernetes/pkg/apis/storage/v1"
	storagevalidation "k8s.io/kubernetes/pkg/apis/storage/validation"
	"k8s.io/kubernetes/pkg/features"
	csifake "k8s.io/kubernetes/pkg/volume/csi/fake"
	"k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	registrationDriverName = "com.example.csi.driver"
	registrationNodeName   = "node1"
	registrationNodeID     = "i-instance1"
	registrationVersion    = "1.0.0"
	registrationZoneKey    = "topology.example.com/zone"
)

type registrationNodeServer struct {
	csipbv1.UnimplementedNodeServer
	client *csifake.NodeClient
}

func (s *registrationNodeServer) NodeGetInfo(ctx context.Context, req *csipbv1.NodeGetInfoRequest) (*csipbv1.NodeGetInfoResponse, error) {
	return s.client.NodeGetInfo(ctx, req)
}

func (s *registrationNodeServer) NodeGetCapabilities(ctx context.Context, req *csipbv1.NodeGetCapabilitiesRequest) (*csipbv1.NodeGetCapabilitiesResponse, error) {
	return s.client.NodeGetCapabilities(ctx, req)
}

func listenTestUnixSocket(t *testing.T) net.Listener {
	t.Helper()

	// t.TempDir() paths can exceed the Unix socket limit.
	dir, err := os.MkdirTemp("", "csi")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, os.RemoveAll(dir)) })

	listener, err := net.Listen("unix", filepath.Join(dir, "csi.sock"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	return listener
}

func startRegistrationNodeService(t *testing.T, nodeClient *csifake.NodeClient) string {
	t.Helper()

	listener := listenTestUnixSocket(t)
	server := grpc.NewServer()
	csipbv1.RegisterNodeServer(server, &registrationNodeServer{client: nodeClient})
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(server.Stop)
	return listener.Addr().String()
}

func registrationNodeClient(capable bool) *csifake.NodeClient {
	nodeClient := csifake.NewNodeClient(false)
	nodeClient.SetNodeInfoFromControllerCapability(capable)
	nodeClient.SetNodeGetInfoResp(&csipbv1.NodeGetInfoResponse{
		NodeId:            registrationNodeID,
		MaxVolumesPerNode: 10,
		AccessibleTopology: &csipbv1.Topology{
			Segments: map[string]string{registrationZoneKey: "zone-a"},
		},
	})
	return nodeClient
}

type registrationFixture struct {
	*RegistrationHandler
	client *fakeclient.Clientset
}

func newRegistrationFixture(t *testing.T, driver *storagev1.CSIDriver) *registrationFixture {
	t.Helper()

	client := fakeclient.NewSimpleClientset(&api.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: registrationNodeName,
			UID:  "node-uid",
		},
	})
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if driver != nil {
		require.NoError(t, client.Tracker().Add(driver))
		require.NoError(t, indexer.Add(driver))
	}

	host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
		t.TempDir(),
		client,
		nil,
		registrationNodeName,
		nil,
		nil,
	)
	plugin := &csiPlugin{
		host:            host,
		csiDriverLister: storagelisters.NewCSIDriverLister(indexer),
	}

	oldNIM := nim
	oldDrivers := csiDrivers
	oldUpdater := csiNodeUpdaterVar
	nim = nodeinfomanager.NewNodeInfoManager(host.GetNodeName(), host, nil)
	csiDrivers = &DriversStore{}
	csiNodeUpdaterVar = nil
	t.Cleanup(func() {
		nim = oldNIM
		csiDrivers = oldDrivers
		csiNodeUpdaterVar = oldUpdater
	})
	require.NoError(t, nim.InitializeCSINodeWithAnnotation())

	// The object tracker does not enforce the API's node-ID consistency check.
	client.PrependReactor("update", "csinodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		csiNode := action.(clienttesting.UpdateAction).GetObject().(*storagev1.CSINode)
		var internal storageapi.CSINode
		if err := storageconversion.Convert_v1_CSINode_To_storage_CSINode(csiNode, &internal, nil); err != nil {
			return true, nil, err
		}
		if errs := storagevalidation.ValidateCSINode(&internal); len(errs) > 0 {
			kind := storagev1.SchemeGroupVersion.WithKind("CSINode").GroupKind()
			return true, nil, apierrors.NewInvalid(kind, csiNode.Name, errs)
		}
		return false, nil, nil
	})

	return &registrationFixture{
		RegistrationHandler: &RegistrationHandler{csiPlugin: plugin},
		client:              client,
	}
}

func testCSIDriver(attachRequired *bool) *storagev1.CSIDriver {
	return &storagev1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{Name: registrationDriverName},
		Spec:       storagev1.CSIDriverSpec{AttachRequired: attachRequired},
	}
}

func (f *registrationFixture) registerPlugin(ctx context.Context, endpoint string) error {
	return f.RegisterPlugin(ctx, registrationDriverName, endpoint, []string{registrationVersion}, nil)
}

func (f *registrationFixture) getCSINode(t *testing.T) *storagev1.CSINode {
	t.Helper()

	csiNode, err := f.client.StorageV1().CSINodes().Get(t.Context(), registrationNodeName, metav1.GetOptions{})
	require.NoError(t, err)
	return csiNode
}

func (f *registrationFixture) setCSINodeSpec(t *testing.T, spec storagev1.CSINodeSpec) {
	t.Helper()

	csiNode := f.getCSINode(t)
	csiNode.Spec = spec
	require.NoError(t, f.client.Tracker().Update(storagev1.SchemeGroupVersion.WithResource("csinodes"), csiNode, ""))
}

func TestRegisterPluginNodeInfo(t *testing.T) {
	testCases := []struct {
		name           string
		disableGate    bool
		capable        bool
		noLister       bool
		emptyID        bool
		driver         *storagev1.CSIDriver
		infoErr        error
		capabilityErr  error
		wantController bool
		wantErr        string
	}{
		{
			name:          "feature disabled",
			disableGate:   true,
			capable:       true,
			noLister:      true,
			capabilityErr: errors.New("capability query must be skipped"),
		},
		{
			name:           "controller flow",
			capable:        true,
			driver:         testCSIDriver(new(true)),
			wantController: true,
		},
		{
			name:           "AttachRequired unset",
			capable:        true,
			driver:         testCSIDriver(nil),
			wantController: true,
		},
		{
			name:    "AttachRequired false",
			capable: true,
			driver:  testCSIDriver(new(false)),
		},
		{
			name:     "capability absent skips CSIDriver lookup",
			noLister: true,
		},
		{
			name:           "CSIDriver absent",
			capable:        true,
			wantController: true,
		},
		{
			name:           "NodeGetInfo failure",
			capable:        true,
			infoErr:        errors.New("info failed"),
			wantController: true,
			wantErr:        "info failed",
		},
		{
			name:           "empty node ID",
			capable:        true,
			emptyID:        true,
			wantController: true,
			wantErr:        "driverNodeID must not be empty",
		},
		{
			name:          "NodeGetCapabilities failure",
			capabilityErr: errors.New("capabilities failed"),
			wantErr:       "capabilities failed",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, !tc.disableGate)
			f := newRegistrationFixture(t, tc.driver)
			if tc.noLister {
				f.csiPlugin.csiDriverLister = nil
			}

			nodeClient := registrationNodeClient(tc.capable)
			nodeClient.SetNextError(tc.infoErr)
			nodeClient.SetGetCapabilitiesErr(tc.capabilityErr)
			if tc.emptyID {
				nodeClient.SetNodeGetInfoResp(&csipbv1.NodeGetInfoResponse{})
			}
			endpoint := startRegistrationNodeService(t, nodeClient)

			err := f.registerPlugin(t.Context(), endpoint)

			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			requests := nodeClient.NodeGetInfoRequests()
			if tc.capabilityErr != nil && !tc.disableGate {
				assert.Empty(t, requests)
			} else {
				require.Len(t, requests, 1)
				assert.Equal(t, tc.wantController, requests[0].GetControllerGetNodeInfo())
			}

			csiNode := f.getCSINode(t)
			driver, registered := csiDrivers.Get(registrationDriverName)
			assert.Equal(t, tc.wantErr == "", registered)

			switch {
			case tc.wantErr != "":
				assert.Empty(t, csiNode.Spec.Drivers)
				assert.Empty(t, csiNode.Spec.DriverRegistrations)
			case tc.wantController:
				wantRegistrations := []storagev1.CSINodeDriverRegistration{
					{
						Name:   registrationDriverName,
						NodeID: registrationNodeID,
					},
				}
				assert.Empty(t, csiNode.Spec.Drivers)
				assert.Equal(t, wantRegistrations, csiNode.Spec.DriverRegistrations)
				assert.True(t, driver.infoFromController)
			default:
				wantDrivers := []storagev1.CSINodeDriver{
					{
						Name:         registrationDriverName,
						NodeID:       registrationNodeID,
						TopologyKeys: []string{registrationZoneKey},
						Allocatable:  &storagev1.VolumeNodeResources{Count: new(int32(10))},
					},
				}
				assert.Equal(t, wantDrivers, csiNode.Spec.Drivers)
				assert.Empty(t, csiNode.Spec.DriverRegistrations)
				assert.False(t, driver.infoFromController)
			}

			kubeNode, err := f.client.CoreV1().Nodes().Get(t.Context(), registrationNodeName, metav1.GetOptions{})
			require.NoError(t, err)
			if tc.wantErr == "" && !tc.wantController {
				assert.Equal(t, "zone-a", kubeNode.Labels[registrationZoneKey])
			} else {
				assert.NotContains(t, kubeNode.Labels, registrationZoneKey)
			}
		})
	}
}

func TestRegisterPluginLifecycle(t *testing.T) {
	testCases := []struct {
		name               string
		infoFromController bool
	}{
		{name: "node-side flow"},
		{name: "controller-side flow", infoFromController: true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, tc.infoFromController)
			f := newRegistrationFixture(t, testCSIDriver(nil))

			published := storagev1.CSINodeDriver{
				Name:         registrationDriverName,
				NodeID:       registrationNodeID,
				TopologyKeys: []string{registrationZoneKey},
				Allocatable:  &storagev1.VolumeNodeResources{Count: new(int32(6))},
			}
			registrations := []storagev1.CSINodeDriverRegistration{
				{
					Name:   registrationDriverName,
					NodeID: registrationNodeID,
				},
			}
			spec := storagev1.CSINodeSpec{Drivers: []storagev1.CSINodeDriver{published}}
			if tc.infoFromController {
				spec.DriverRegistrations = registrations
			}
			f.setCSINodeSpec(t, spec)

			nodeClient := registrationNodeClient(tc.infoFromController)
			endpoint := startRegistrationNodeService(t, nodeClient)

			require.NoError(t, f.registerPlugin(t.Context(), endpoint))

			if !tc.infoFromController {
				published.Allocatable.Count = new(int32(10))
			}
			csiNode := f.getCSINode(t)
			assert.Equal(t, []storagev1.CSINodeDriver{published}, csiNode.Spec.Drivers)
			if tc.infoFromController {
				assert.Equal(t, registrations, csiNode.Spec.DriverRegistrations)
			} else {
				assert.Empty(t, csiNode.Spec.DriverRegistrations)
			}

			nodeClient.SetNodeGetInfoResp(&csipbv1.NodeGetInfoResponse{
				NodeId:            registrationNodeID,
				MaxVolumesPerNode: 20,
				AccessibleTopology: &csipbv1.Topology{
					Segments: map[string]string{registrationZoneKey: "zone-a"},
				},
			})
			f.client.ClearActions()

			require.NoError(t, updateCSIDriver(registrationDriverName))

			if tc.infoFromController {
				assert.Len(t, nodeClient.NodeGetInfoRequests(), 1)
				assert.Empty(t, f.client.Actions())
			} else {
				assert.Len(t, nodeClient.NodeGetInfoRequests(), 2)
				published.Allocatable.Count = new(int32(20))
			}
			assert.Equal(t, []storagev1.CSINodeDriver{published}, f.getCSINode(t).Spec.Drivers)

			f.DeRegisterPlugin(t.Context(), registrationDriverName, endpoint)

			csiNode = f.getCSINode(t)
			assert.Empty(t, csiNode.Spec.Drivers)
			assert.Empty(t, csiNode.Spec.DriverRegistrations)
			assert.Empty(t, csiDrivers.List())
			require.ErrorContains(t, updateCSIDriver(registrationDriverName), "not found in the list of registered CSI drivers")
		})
	}
}

func TestRegisterPluginAPIFailures(t *testing.T) {
	testCases := []struct {
		name         string
		wantErr      string
		dropField    bool
		mismatchedID bool
	}{
		{
			name:      "API server drops field",
			dropField: true,
			wantErr:   "ensure CSIControllerGetNodeInfo is enabled on kube-apiserver",
		},
		{
			name:         "node ID changed",
			mismatchedID: true,
			wantErr:      "must match the nodeID of the drivers entry",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, true)
			f := newRegistrationFixture(t, testCSIDriver(nil))
			if tc.mismatchedID {
				f.setCSINodeSpec(t, storagev1.CSINodeSpec{
					Drivers: []storagev1.CSINodeDriver{
						{
							Name:   registrationDriverName,
							NodeID: "i-old",
						},
					},
					DriverRegistrations: []storagev1.CSINodeDriverRegistration{
						{
							Name:   registrationDriverName,
							NodeID: "i-old",
						},
					},
				})
			}

			dropField := tc.dropField
			f.client.PrependReactor("update", "csinodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if dropField {
					csiNode := action.(clienttesting.UpdateAction).GetObject().(*storagev1.CSINode)
					csiNode.Spec.DriverRegistrations = nil
				}
				return false, nil, nil
			})
			endpoint := startRegistrationNodeService(t, registrationNodeClient(true))

			require.ErrorContains(t, f.registerPlugin(t.Context(), endpoint), tc.wantErr)

			csiNode := f.getCSINode(t)
			assert.Empty(t, csiNode.Spec.Drivers)
			assert.Empty(t, csiNode.Spec.DriverRegistrations)
			assert.Empty(t, csiDrivers.List())

			dropField = false
			require.NoError(t, f.registerPlugin(t.Context(), endpoint))

			wantRegistrations := []storagev1.CSINodeDriverRegistration{
				{
					Name:   registrationDriverName,
					NodeID: registrationNodeID,
				},
			}
			assert.Equal(t, wantRegistrations, f.getCSINode(t).Spec.DriverRegistrations)
			assert.Equal(t, []string{registrationDriverName}, csiDrivers.List())
		})
	}
}

func TestRegisterPluginConflict(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, true)
	f := newRegistrationFixture(t, testCSIDriver(nil))

	otherDriver := storagev1.CSINodeDriver{
		Name:        "other.csi.example.com",
		NodeID:      "other-node",
		Allocatable: &storagev1.VolumeNodeResources{Count: new(int32(7))},
	}
	otherRegistration := storagev1.CSINodeDriverRegistration{
		Name:   otherDriver.Name,
		NodeID: otherDriver.NodeID,
	}
	attempts := 0
	f.client.PrependReactor("update", "csinodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempts++
		if attempts == 1 {
			// Simulate another writer updating CSINode after the registration's GET.
			csiNode := action.(clienttesting.UpdateAction).GetObject().(*storagev1.CSINode).DeepCopy()
			csiNode.Spec = storagev1.CSINodeSpec{
				Drivers:             []storagev1.CSINodeDriver{otherDriver},
				DriverRegistrations: []storagev1.CSINodeDriverRegistration{otherRegistration},
			}
			if err := f.client.Tracker().Update(storagev1.SchemeGroupVersion.WithResource("csinodes"), csiNode, ""); err != nil {
				return true, nil, err
			}
			return true, nil, apierrors.NewConflict(storagev1.Resource("csinodes"), csiNode.Name, errors.New("concurrent update"))
		}
		return false, nil, nil
	})
	endpoint := startRegistrationNodeService(t, registrationNodeClient(true))

	require.NoError(t, f.registerPlugin(t.Context(), endpoint))

	assert.Equal(t, 2, attempts)
	csiNode := f.getCSINode(t)
	wantRegistrations := []storagev1.CSINodeDriverRegistration{
		otherRegistration,
		{
			Name:   registrationDriverName,
			NodeID: registrationNodeID,
		},
	}
	assert.Equal(t, []storagev1.CSINodeDriver{otherDriver}, csiNode.Spec.Drivers)
	assert.ElementsMatch(t, wantRegistrations, csiNode.Spec.DriverRegistrations)
}

func TestRegisterPluginRegistersOnlyAfterSuccess(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, true)
	f := newRegistrationFixture(t, testCSIDriver(nil))

	entered := make(chan struct{})
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	markEntered := sync.OnceFunc(func() { close(entered) })
	ctx, cancel := context.WithTimeout(t.Context(), wait.ForeverTestTimeout)
	defer cancel()

	f.client.PrependReactor("update", "csinodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
		markEntered()
		select {
		case <-release:
			return false, nil, nil
		case <-ctx.Done():
			return true, nil, ctx.Err()
		}
	})
	endpoint := startRegistrationNodeService(t, registrationNodeClient(true))

	result := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		result <- f.registerPlugin(ctx, endpoint)
	}()
	defer func() {
		unblock()
		<-done
	}()

	select {
	case <-entered:
	case err := <-result:
		t.Fatalf("registration returned before the API write: %v", err)
	case <-ctx.Done():
		t.Fatal("registration did not reach the API write")
	}

	assert.Empty(t, csiDrivers.List())
	_, err := newCsiDriverClient(csiDriverName(registrationDriverName))
	require.ErrorContains(t, err, "not found in the list of registered CSI drivers")

	unblock()
	require.NoError(t, <-result)

	driver, registered := csiDrivers.Get(registrationDriverName)
	require.True(t, registered)
	assert.Equal(t, endpoint, driver.endpoint)
	assert.True(t, driver.infoFromController)
}

func TestRegisterPluginTimeout(t *testing.T) {
	testCases := []struct {
		name             string
		enableController bool
	}{
		{name: "NodeGetInfo timeout"},
		{name: "NodeGetCapabilities timeout", enableController: true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, tc.enableController)
			f := newRegistrationFixture(t, testCSIDriver(nil))
			listener := listenTestUnixSocket(t)
			timeout := 100 * time.Millisecond

			err := f.RegisterPlugin(t.Context(), registrationDriverName, listener.Addr().String(), []string{registrationVersion}, &timeout)

			require.ErrorContains(t, err, "context deadline exceeded")
			assert.Empty(t, csiDrivers.List())
		})
	}
}
