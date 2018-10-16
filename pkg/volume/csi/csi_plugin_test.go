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

package csi

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	fakecsi "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// create a plugin mgr to load plugins and setup a fake client
func newTestPlugin(t *testing.T, client *fakeclient.Clientset, csiClient *fakecsi.Clientset) (*CSIPlugin, string) {
	err := utilfeature.DefaultFeatureGate.Set("CSIBlockVolume=true")
	if err != nil {
		t.Fatalf("Failed to enable feature gate for CSIBlockVolume: %v", err)
	}

	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	if client == nil {
		client = fakeclient.NewSimpleClientset()
	}
	if csiClient == nil {
		csiClient = fakecsi.NewSimpleClientset()
	}
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		client,
		csiClient,
		nil,
		"fakeNode",
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName(PluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", PluginName)
	}

	csiPlug, ok := plug.(*CSIPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type CSIPlugin")
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		// Wait until the informer in CSI volume plugin has all CSIDrivers.
		wait.PollImmediate(testInformerSyncPeriod, testInformerSyncTimeout, func() (bool, error) {
			return csiPlug.csiDriverInformer.Informer().HasSynced(), nil
		})
	}

	return csiPlug, tmpDir
}

func makeTestPV(name string, sizeGig int, driverName, volID string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", sizeGig),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volID,
					ReadOnly:     false,
				},
			},
		},
	}
}

func TestPluginGetPluginName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	if plug.GetPluginName() != "kubernetes.io/csi" {
		t.Errorf("unexpected plugin name %v", plug.GetPluginName())
	}
}

func TestPluginGetVolumeName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		driverName string
		volName    string
		shouldFail bool
	}{
		{"alphanum names", "testdr", "testvol", false},
		{"mixchar driver", "test.dr.cc", "testvol", false},
		{"mixchar volume", "testdr", "test-vol-name", false},
		{"mixchars all", "test-driver", "test.vol.name", false},
	}

	for _, tc := range testCases {
		t.Logf("testing: %s", tc.name)
		pv := makeTestPV("test-pv", 10, tc.driverName, tc.volName)
		spec := volume.NewSpecFromPersistentVolume(pv, false)
		name, err := plug.GetVolumeName(spec)
		if tc.shouldFail && err == nil {
			t.Fatal("GetVolumeName should fail, but got err=nil")
		}
		if name != fmt.Sprintf("%s%s%s", tc.driverName, volNameSep, tc.volName) {
			t.Errorf("unexpected volume name %s", name)
		}
	}
}

func TestPluginCanSupport(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, false)

	if !plug.CanSupport(spec) {
		t.Errorf("should support CSI spec")
	}
}

func TestPluginConstructVolumeSpec(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name       string
		specVolID  string
		data       map[string]string
		shouldFail bool
	}{
		{
			name:      "valid spec name",
			specVolID: "test.vol.id",
			data:      map[string]string{volDataKey.specVolID: "test.vol.id", volDataKey.volHandle: "test-vol0", volDataKey.driverName: "test-driver0"},
		},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		dir := getTargetPath(testPodUID, tc.specVolID, plug.host)

		// create the data file
		if tc.data != nil {
			mountDir := path.Join(getTargetPath(testPodUID, tc.specVolID, plug.host), "/mount")
			if err := os.MkdirAll(mountDir, 0755); err != nil && !os.IsNotExist(err) {
				t.Errorf("failed to create dir [%s]: %v", mountDir, err)
			}
			if err := saveVolumeData(path.Dir(mountDir), volDataFileName, tc.data); err != nil {
				t.Fatal(err)
			}
		}

		// rebuild spec
		spec, err := plug.ConstructVolumeSpec("test-pv", dir)
		if tc.shouldFail {
			if err == nil {
				t.Fatal("expecting ConstructVolumeSpec to fail, but got nil error")
			}
			continue
		}

		volHandle := spec.PersistentVolume.Spec.CSI.VolumeHandle
		if volHandle != tc.data[volDataKey.volHandle] {
			t.Errorf("expected volID %s, got volID %s", tc.data[volDataKey.volHandle], volHandle)
		}

		if spec.PersistentVolume.Spec.VolumeMode == nil {
			t.Fatalf("Volume mode has not been set.")
		}

		if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeFilesystem {
			t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
		}

		if spec.Name() != tc.specVolID {
			t.Errorf("Unexpected spec name %s", spec.Name())
		}
	}
}

func TestPluginNewMounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	mounter, err := plug.NewMounter(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if mounter == nil {
		t.Fatal("failed to create CSI mounter")
	}
	csiMounter := mounter.(*csiMountMgr)

	// validate mounter fields
	if csiMounter.driverName != testDriver {
		t.Error("mounter driver name not set")
	}
	if csiMounter.volumeID != testVol {
		t.Error("mounter volume id not set")
	}
	if csiMounter.pod == nil {
		t.Error("mounter pod not set")
	}
	if csiMounter.podUID == types.UID("") {
		t.Error("mounter podUID not set")
	}
	if csiMounter.csiClient == nil {
		t.Error("mounter csiClient is nil")
	}

	// ensure data file is created
	dataDir := path.Dir(mounter.GetPath())
	dataFile := filepath.Join(dataDir, volDataFileName)
	if _, err := os.Stat(dataFile); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("data file not created %s", dataFile)
		} else {
			t.Fatal(err)
		}
	}
}

func TestPluginNewUnmounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file to re-create client
	dir := path.Join(getTargetPath(testPodUID, pv.ObjectMeta.Name, plug.host), "/mount")
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		path.Dir(dir),
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	// test unmounter
	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	csiUnmounter := unmounter.(*csiMountMgr)

	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}

	if csiUnmounter == nil {
		t.Fatal("failed to create CSI Unmounter")
	}

	if csiUnmounter.podUID != testPodUID {
		t.Error("podUID not set")
	}

	if csiUnmounter.csiClient == nil {
		t.Error("unmounter csiClient is nil")
	}
}

func TestPluginNewAttacher(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}

	csiAttacher := attacher.(*csiAttacher)
	if csiAttacher.plugin == nil {
		t.Error("plugin not set for attacher")
	}
	if csiAttacher.k8s == nil {
		t.Error("Kubernetes client not set for attacher")
	}
}

func TestPluginNewDetacher(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	detacher, err := plug.NewDetacher()
	if err != nil {
		t.Fatalf("failed to create new detacher: %v", err)
	}

	csiDetacher := detacher.(*csiAttacher)
	if csiDetacher.plugin == nil {
		t.Error("plugin not set for detacher")
	}
	if csiDetacher.k8s == nil {
		t.Error("Kubernetes client not set for detacher")
	}
}

func TestPluginNewBlockMapper(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-block-pv", 10, testDriver, testVol)
	mounter, err := plug.NewBlockVolumeMapper(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new BlockMapper: %v", err)
	}

	if mounter == nil {
		t.Fatal("failed to create CSI BlockMapper, mapper is nill")
	}
	csiMapper := mounter.(*csiBlockMapper)

	// validate mounter fields
	if csiMapper.driverName != testDriver {
		t.Error("CSI block mapper missing driver name")
	}
	if csiMapper.volumeID != testVol {
		t.Error("CSI block mapper missing volumeID")
	}

	if csiMapper.podUID == types.UID("") {
		t.Error("CSI block mapper missing pod.UID")
	}
	if csiMapper.csiClient == nil {
		t.Error("mapper csiClient is nil")
	}

	// ensure data file is created
	dataFile := getVolumeDeviceDataDir(csiMapper.spec.Name(), plug.host)
	if _, err := os.Stat(dataFile); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("data file not created %s", dataFile)
		} else {
			t.Fatal(err)
		}
	}
}

func TestPluginNewUnmapper(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file to re-create client
	dir := getVolumeDeviceDataDir(pv.ObjectMeta.Name, plug.host)
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		dir,
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	// test unmounter
	unmapper, err := plug.NewBlockVolumeUnmapper(pv.ObjectMeta.Name, testPodUID)
	csiUnmapper := unmapper.(*csiBlockMapper)

	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}

	if csiUnmapper == nil {
		t.Fatal("failed to create CSI Unmounter")
	}

	if csiUnmapper.podUID != testPodUID {
		t.Error("podUID not set")
	}

	if csiUnmapper.specName != pv.ObjectMeta.Name {
		t.Error("specName not set")
	}

	if csiUnmapper.csiClient == nil {
		t.Error("unmapper csiClient is nil")
	}

	// test loaded vol data
	if csiUnmapper.driverName != testDriver {
		t.Error("unmapper driverName not set")
	}
	if csiUnmapper.volumeID != testVol {
		t.Error("unmapper volumeHandle not set")
	}
}

func TestPluginConstructBlockVolumeSpec(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name       string
		specVolID  string
		data       map[string]string
		shouldFail bool
	}{
		{
			name:      "valid spec name",
			specVolID: "test.vol.id",
			data:      map[string]string{volDataKey.specVolID: "test.vol.id", volDataKey.volHandle: "test-vol0", volDataKey.driverName: "test-driver0"},
		},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		deviceDataDir := getVolumeDeviceDataDir(tc.specVolID, plug.host)

		// create data file in csi plugin dir
		if tc.data != nil {
			if err := os.MkdirAll(deviceDataDir, 0755); err != nil && !os.IsNotExist(err) {
				t.Errorf("failed to create dir [%s]: %v", deviceDataDir, err)
			}
			if err := saveVolumeData(deviceDataDir, volDataFileName, tc.data); err != nil {
				t.Fatal(err)
			}
		}

		// rebuild spec
		spec, err := plug.ConstructBlockVolumeSpec("test-podUID", tc.specVolID, getVolumeDevicePluginDir(tc.specVolID, plug.host))
		if tc.shouldFail {
			if err == nil {
				t.Fatal("expecting ConstructVolumeSpec to fail, but got nil error")
			}
			continue
		}

		if spec.PersistentVolume.Spec.VolumeMode == nil {
			t.Fatalf("Volume mode has not been set.")
		}

		if *spec.PersistentVolume.Spec.VolumeMode != api.PersistentVolumeBlock {
			t.Errorf("Unexpected volume mode %q", *spec.PersistentVolume.Spec.VolumeMode)
		}

		volHandle := spec.PersistentVolume.Spec.CSI.VolumeHandle
		if volHandle != tc.data[volDataKey.volHandle] {
			t.Errorf("expected volID %s, got volID %s", tc.data[volDataKey.volHandle], volHandle)
		}

		if spec.Name() != tc.specVolID {
			t.Errorf("Unexpected spec name %s", spec.Name())
		}
	}
}

//go:generate $KUBE_ROOT/hack/generate-mocks.sh ./nodeinfomanager/nodeinfomanager.go Interface ./fake/fake_node_info_manager.go FakeNodeInfoManager
//go:generate $KUBE_ROOT/hack/generate-mocks.sh $KUBE_ROOT/vendor/github.com/container-storage-interface/spec/lib/go/csi/v0/csi.pb.go NodeServer ./fake/fake_node_server.go
func TestRegisterPlugin(t *testing.T) {
	testcases := map[string]struct {
		// setup
		sockPath            string
		nodeGetInfoResponse *csipb.NodeGetInfoResponse
		nodeGetInfoErr      error
		addNodeInfoErr      error
		// expectations
		expectedRegisterPluginErrMsg string
		expectedNimAddCalls          int
		expectedNimRemoveCalls       int
		expectedDriverEntry          bool
	}{
		"happy path": {
			sockPath:            "/tmp/some.csi.sock",
			nodeGetInfoResponse: &csipb.NodeGetInfoResponse{NodeId: "some node id"},
			expectedNimAddCalls: 1,
			expectedDriverEntry: true,
		},
		"when node server returns an error": {
			sockPath:                     "/tmp/some.csi.sock",
			nodeGetInfoErr:               errors.New("some random node get info error"),
			expectedNimRemoveCalls:       1,
			expectedRegisterPluginErrMsg: "some random node get info error",
			expectedDriverEntry:          false,
		},
		"when node info manager returns an error": {
			sockPath:                     "/tmp/some.csi.sock",
			nodeGetInfoResponse:          &csipb.NodeGetInfoResponse{NodeId: "some node id"},
			addNodeInfoErr:               errors.New("some random node info add error"),
			expectedNimAddCalls:          1,
			expectedNimRemoveCalls:       1,
			expectedRegisterPluginErrMsg: "some random node info add error",
			expectedDriverEntry:          false,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			nodeServer := &fake.FakeNodeServer{}
			nodeServer.NodeGetInfoReturns(tc.nodeGetInfoResponse, tc.nodeGetInfoErr)

			nodeInfoManager := &fake.FakeNodeInfoManager{}
			nodeInfoManager.AddNodeInfoReturns(tc.addNodeInfoErr)

			registrationHandler := &RegistrationHandler{
				csiDrivers: &csiDriversStore{
					driversMap: map[string]csiDriver{},
				},
				nim: nodeInfoManager,
			}

			driverStarter, driverStopper := newCsiDriverServer(t, tc.sockPath, nodeServer)
			go driverStarter()
			defer driverStopper()

			err := registrationHandler.RegisterPlugin("some driver name", tc.sockPath)
			checkErrWithMessage(t, err, tc.expectedRegisterPluginErrMsg)

			if a, e := nodeInfoManager.AddNodeInfoCallCount(), tc.expectedNimAddCalls; e != a {
				t.Errorf("Expected nim.AddNodeInfo to be called %d times, got called %d times", e, a)
			}
			if a, e := nodeInfoManager.RemoveNodeInfoCallCount(), tc.expectedNimRemoveCalls; e != a {
				t.Errorf("Expected nim.RemoveNodeInfo to be called %d times, got called %d times", e, a)
			}

			driversEntry, foundDriver := registrationHandler.csiDrivers.driversMap["some driver name"]
			if foundDriver != tc.expectedDriverEntry {
				if foundDriver {
					t.Errorf("Expected to find the driver registered")
				} else {
					t.Errorf("Expected not to find the driver registered, but found: %#v", driversEntry)
				}
			}
		})
	}
}

func TestCSIPluginInit(t *testing.T) {
	testcases := map[string]struct {
		expectedErrMsg  string
		registryEnabled bool
	}{
		"with registry disabled": {},
		"with registry enabled":  {registryEnabled: true},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			plugin := CSIPlugin{}

			csiClient := fakecsi.NewSimpleClientset(
				getCSIDriver("fake csi driver name", nil, nil),
			)
			volumeHost := volumetest.NewFakeVolumeHostWithCSINodeName(
				"fakeTmpDir", // root dirctory
				nil,          // kube clientset
				csiClient,    // csi clientset
				nil,          // plugins
				"fake node",  // node name
			)

			if tc.registryEnabled {
				defer utilfeaturetesting.SetFeatureGateDuringTest(
					t, utilfeature.DefaultFeatureGate, features.CSIDriverRegistry, true,
				)()
			}

			err := plugin.Init(volumeHost)
			checkErrWithMessage(t, err, tc.expectedErrMsg)

			if plugin.host != volumeHost {
				t.Errorf("Expected plugin.host to be the volume host")
			}

			if plugin.RegistrationHandler.csiDrivers.driversMap == nil {
				t.Errorf("Expected csiDrivers to be initialized")
			}
			if plugin.RegistrationHandler.nim == nil {
				t.Errorf("Expected nodeInfoManager to be initialized")
			}

			if !utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
				// with the csi driver registry disabled, there is not much more we can
				// check for
				return
			}

			informer := plugin.csiDriverInformer
			lister := plugin.csiDriverLister

			if lister == nil {
				t.Errorf("Expected plugin.csiDriverLister to be set up")
			}
			if informer == nil {
				t.Errorf("Expected plugin.csiDriverInformer to be set up")
			}

			waitForInformerSync(t, informer.Informer())

			csiDriverList, err := lister.List(labels.Everything())
			if err != nil {
				t.Errorf("Expected no error on plugin.csiDriverLister.List(), got: %#v", err)
			}
			if csiDriverList[0].ObjectMeta.Name != "fake csi driver name" {
				t.Errorf("Expected to get only the fake csi driver from the lister, got those drivers: %#v", csiDriverList)
			}
		})
	}
}

func waitForInformerSync(t *testing.T, informer cache.SharedInformer) {
	t.Helper()
	err := wait.PollImmediate(testInformerSyncPeriod, testInformerSyncTimeout, func() (bool, error) {
		return informer.HasSynced(), nil
	})
	if err != nil {
		t.Fatal("Timeout waiting for informer to sync")
	}
}

func checkErrWithMessage(t *testing.T, actualErr error, expectedErrMsg string) {
	t.Helper()
	if expectedErrMsg == "" {
		if actualErr != nil {
			t.Errorf("Expected no error, got: %v", actualErr)
		}
	} else {
		if !strings.Contains(actualErr.Error(), expectedErrMsg) {
			t.Errorf("Expected an error containing '%s', got: '%#v'", expectedErrMsg, actualErr)
		}
	}
}

func newCsiDriverServer(t *testing.T, sockPath string, nodeServer csipb.NodeServer) (func(), func()) {
	t.Helper()

	listener, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	csipb.RegisterNodeServer(grpcServer, nodeServer)

	starter := func() {
		if err := grpcServer.Serve(listener); err != nil {
			nestedErr := ""
			if opErr, ok := err.(*net.OpError); ok {
				nestedErr = fmt.Sprintf(" (nested Error: %#v)", opErr.Err)
			}
			t.Logf("Error on Serve(): %#v"+nestedErr, err)
		}
	}

	stopper := func() {
		// TODO(hoegaarden): How can we get rid of that stupid sleep without making
		//                   grpcServer.Serve() return an error?
		time.Sleep(time.Millisecond)
		grpcServer.GracefulStop()
	}

	return starter, stopper
}
