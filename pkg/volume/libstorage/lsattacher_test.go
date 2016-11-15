/*
Copyright 2016 The Kubernetes Authors.

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

package libstorage

/* --- Bypassing Attacher for now
// TODO Uncomment when Attacher API reactivated
import (
	"errors"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/libstorage/lstypes"
)

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				LibStorage: &api.LibStorageVolumeSource{
					Host:       "tcp://:1234",
					Service:    "ls-service",
					VolumeName: name,
					ReadOnly:   readOnly,
				},
			},
		},
	}
}

func createPVSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					LibStorage: &api.LibStorageVolumeSource{
						Host:       "tcp://:1234",
						Service:    "ls-service",
						VolumeName: name,
						ReadOnly:   readOnly,
					},
				},
			},
		},
	}
}

func TestGetDeviceName_Volume(t *testing.T) {
	plugMgr, tmpDir := makePlugMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", lsPluginName)
	}

	lsPlug := plug.(*lsPlugin)

	name := "my-pd-volume"
	spec := createVolSpec(name, false)

	deviceName, err := lsPlug.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != name {
		t.Errorf("GetDeviceName error: expected %s, got %s", name, deviceName)
	}
}

func TestGetDeviceName_PersistentVolume(t *testing.T) {
	plugMgr, tmpDir := makePlugMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", lsPluginName)
	}

	lsPlug := plug.(*lsPlugin)

	name := "my-pd-volume"
	spec := createPVSpec(name, false)

	deviceName, err := lsPlug.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != name {
		t.Errorf("GetDeviceName error: expected %s, got %s", name, deviceName)
	}
}

type retParams struct {
	device string
	err    error
}

// table-driven structur for attacher/detacher
type testCase struct {
	t              *testing.T
	name           string
	plugMgr        *volume.VolumePluginMgr
	tmpDir         string
	plugin         *lsPlugin
	attach         func(test *testCase) (string, error)
	detach         func(test *testCase) error
	retParams      retParams
	expectedDevice string
	expectedError  error
}

var attachError = errors.New("attach error")

// make testCase implement lsMgr
func (m *testCase) createVolume(name string, size int64) (*lstypes.Volume, error) {
	return &lstypes.Volume{Name: "vol-0001"}, nil
}
func (m *testCase) attachVolume(volName string) (string, error) {
	rets := m.retParams
	if volName != "vol-0001" {
		m.t.Logf("expecting volume name %s, got %s", "vol-0001", volName)
		return rets.device, rets.err
	}
	return rets.device, rets.err
}
func (m *testCase) isAttached(volName string) (bool, error) {
	rets := m.retParams
	if volName != "vol-0001" {
		return false, rets.err
	}
	return true, rets.err
}
func (m *testCase) detachVolume(volName string) error {
	rets := m.retParams
	if volName != "vol-0001" {
		m.t.Logf("expecting volume name %s, got %s", "vol-0001", volName)
		return rets.err
	}

	return rets.err
}
func (m *testCase) deleteVolume(volName string) error {
	return errors.New("not implemented")
}
func (m *testCase) getHost() string {
	return "tcp://:1234"
}
func (m *testCase) getService() string {
	return "ls-service"
}

func TestAttachDetach(t *testing.T) {
	volName := "vol-0001"
	spec := createVolSpec(volName, true)
	host := "localhost"
	testCases := []testCase{
		{
			name:      "Attach_OK",
			t:         t,
			retParams: retParams{"/dev/sdb123", nil},
			attach: func(test *testCase) (string, error) {
				attacher := test.newAttacher(t)
				return attacher.Attach(spec, types.NodeName(host))
			},
			expectedDevice: "/dev/sdb123",
		},
		{
			name:      "Attach_Failure",
			t:         t,
			retParams: retParams{"", attachError},
			attach: func(test *testCase) (string, error) {
				spec.Volume.LibStorage.VolumeName = ""
				attacher := test.newAttacher(t)
				_, err := attacher.Attach(spec, types.NodeName(host))
				if err != nil {
					return "", attachError
				}
				return "", errors.New("expected attach failure, but didn't")
			},
			expectedDevice: "",
			expectedError:  attachError,
		},
		{
			name:      "Detach_OK",
			t:         t,
			retParams: retParams{"", nil},
			detach: func(test *testCase) error {
				attacher := test.newAttacher(t)
				return attacher.Detach(volName, types.NodeName(host))
			},
			expectedDevice: "",
			expectedError:  nil,
		},
		{
			name:      "Detach_Failure",
			t:         t,
			retParams: retParams{"", attachError},
			detach: func(test *testCase) error {
				attacher := test.newAttacher(t)
				_, err := attacher.Attach(spec, types.NodeName(host))
				if err != nil {
					return attachError
				}
				return errors.New("expected attach failure, but didn't")
			},
			expectedDevice: "",
			expectedError:  attachError,
		},
	}
	for _, test := range testCases {
		t.Logf("testing case %v", test.name)
		var device string
		var err error
		if test.attach != nil {
			device, err = test.attach(&test)
		}
		if test.detach != nil {
			err = test.detach(&test)
		}
		if device != test.expectedDevice {
			t.Errorf(
				"testcase %s filed: expecting device %s, got %s",
				test.name, test.expectedDevice, device,
			)
		}
		if err != test.expectedError {
			t.Errorf("testcase %s failed: expected err=%q, got %q",
				test.name, test.expectedError.Error(), err.Error(),
			)
		}

	}
}

func (m *testCase) newAttacher(t *testing.T) *lsVolume {
	if m.plugMgr == nil {
		plugMgr, tmpDir := makePlugMgr(t)
		m.plugMgr = plugMgr
		m.tmpDir = tmpDir
		plug, err := plugMgr.FindPluginByName(lsPluginName)
		if err != nil {
			t.Errorf("Can't find the plugin %v", lsPluginName)
		}
		m.plugin = plug.(*lsPlugin)
		m.plugin.lsMgr = m
	}
	attacher, err := m.plugin.NewAttacher()
	if err != nil {
		t.Errorf("failed to create attacher %v", err)
	}
	lsAttacher := attacher.(*lsVolume)
	lsAttacher.mounter = &mount.FakeMounter{}
	lsAttacher.plugin = m.plugin
	return lsAttacher
}
*/
