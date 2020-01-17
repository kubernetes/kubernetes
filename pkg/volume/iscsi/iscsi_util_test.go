/*
Copyright 2015 The Kubernetes Authors.

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

package iscsi

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	testingexec "k8s.io/utils/exec/testing"

	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	TestIface = "192.168.1.10:pv0001"
)

func TestExtractDeviceAndPrefix(t *testing.T) {
	devicePath := "127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00"
	mountPrefix := "/var/lib/kubelet/plugins/kubernetes.io/iscsi/iface-default/" + devicePath
	lun := "-lun-0"
	device, prefix, err := extractDeviceAndPrefix(mountPrefix + lun)
	if err != nil || device != (devicePath+lun) || prefix != mountPrefix {
		t.Errorf("extractDeviceAndPrefix: expected %s and %s, got %v %s and %s", devicePath+lun, mountPrefix, err, device, prefix)
	}
}

func TestExtractIface(t *testing.T) {
	ifaceName := "default"
	devicePath := "127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"
	iface, found := extractIface("/var/lib/kubelet/plugins/kubernetes.io/iscsi/iface-" + ifaceName + "/" + devicePath)
	if !found || iface != ifaceName {
		t.Errorf("extractIface: expected %s and %t, got %s and %t", ifaceName, true, iface, found)
	}
	iface, found = extractIface("/var/lib/kubelet/plugins/kubernetes.io/iscsi/" + devicePath)
	if found || iface != "" {
		t.Errorf("extractIface: expected %s and %t, got %s and %t", "", false, iface, found)
	}
}

func TestExtractPortalAndIqn(t *testing.T) {
	devicePath := "127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"
	portal, iqn, err := extractPortalAndIqn(devicePath)
	if err != nil || portal != "127.0.0.1:3260" || iqn != "iqn.2014-12.com.example:test.tgt00" {
		t.Errorf("extractPortalAndIqn: got %v %s %s", err, portal, iqn)
	}
	devicePath = "127.0.0.1:3260-eui.02004567A425678D-lun-0"
	portal, iqn, err = extractPortalAndIqn(devicePath)
	if err != nil || portal != "127.0.0.1:3260" || iqn != "eui.02004567A425678D" {
		t.Errorf("extractPortalAndIqn: got %v %s %s", err, portal, iqn)
	}
}

func TestRemoveDuplicate(t *testing.T) {
	dupPortals := []string{"127.0.0.1:3260", "127.0.0.1:3260", "127.0.0.100:3260"}
	portals := removeDuplicate(dupPortals)
	want := []string{"127.0.0.1:3260", "127.0.0.100:3260"}
	if reflect.DeepEqual(portals, want) == false {
		t.Errorf("removeDuplicate: want: %s, got: %s", want, portals)
	}
}

func fakeOsStat(devicePath string) (fi os.FileInfo, err error) {
	var cmd os.FileInfo
	return cmd, nil
}

func fakeFilepathGlob(devicePath string) (globs []string, err error) {
	return []string{devicePath}, nil
}

func fakeFilepathGlob2(devicePath string) (globs []string, err error) {
	return []string{
		"/dev/disk/by-path/pci-0000:00:00.0-ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0",
	}, nil
}

func TestExtractTransportname(t *testing.T) {
	fakeIscsiadmOutput := []string{
		"# BEGIN RECORD 2.0-873\n" +
			"iface.iscsi_ifacename = default\n" +
			"iface.transport_name = tcp\n" +
			"iface.initiatorname = <empty>\n" +
			"# END RECORD",
		"# BEGIN RECORD 2.0-873\n" +
			"iface.iscsi_ifacename = default\n" +
			"iface.transport_name = cxgb4i\n" +
			"iface.initiatorname = <empty>\n" +
			"# END RECORD",
		"# BEGIN RECORD 2.0-873\n" +
			"iface.iscsi_ifacename = default\n" +
			"iface.transport_name = <empty>\n" +
			"iface.initiatorname = <empty>\n" +
			"# END RECORD",
		"# BEGIN RECORD 2.0-873\n" +
			"iface.iscsi_ifacename = default\n" +
			"iface.initiatorname = <empty>\n" +
			"# END RECORD"}
	transportName := extractTransportname(fakeIscsiadmOutput[0])
	if transportName != "tcp" {
		t.Errorf("extractTransportname: Could not extract correct iface.transport_name 'tcp', got %s", transportName)
	}
	transportName = extractTransportname(fakeIscsiadmOutput[1])
	if transportName != "cxgb4i" {
		t.Errorf("extractTransportname: Could not extract correct iface.transport_name 'cxgb4i', got %s", transportName)
	}
	transportName = extractTransportname(fakeIscsiadmOutput[2])
	if transportName != "tcp" {
		t.Errorf("extractTransportname: Could not extract correct iface.transport_name 'tcp', got %s", transportName)
	}
	transportName = extractTransportname(fakeIscsiadmOutput[3])
	if transportName != "" {
		t.Errorf("extractTransportname: Could not extract correct iface.transport_name '', got %s", transportName)
	}
}

func TestWaitForPathToExist(t *testing.T) {
	devicePath := []string{"/dev/disk/by-path/ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0",
		"/dev/disk/by-path/pci-*-ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"}
	fpath := "/dev/disk/by-path/pci-0000:00:00.0-ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"

	exist := waitForPathToExistInternal(&devicePath[0], 1, "tcp", fakeOsStat, filepath.Glob)
	if exist == false {
		t.Errorf("waitForPathToExist: could not find path %s", devicePath[0])
	}
	exist = waitForPathToExistInternal(&devicePath[0], 1, "fake_iface", fakeOsStat, filepath.Glob)
	if exist != false {
		t.Errorf("waitForPathToExist: wrong code path called for %s", devicePath[0])
	}

	exist = waitForPathToExistInternal(&devicePath[1], 1, "fake_iface", os.Stat, fakeFilepathGlob)
	if exist == false {
		t.Errorf("waitForPathToExist: could not find path %s", devicePath[1])
	}
	exist = waitForPathToExistInternal(&devicePath[1], 1, "tcp", os.Stat, fakeFilepathGlob)
	if exist != false {
		t.Errorf("waitForPathToExist: wrong code path called for %s", devicePath[1])
	}

	exist = waitForPathToExistInternal(&devicePath[1], 1, "fake_iface", os.Stat, fakeFilepathGlob2)
	if devicePath[1] != fpath {
		t.Errorf("waitForPathToExist: wrong code path called for %s", devicePath[1])
	}
}

func TestParseIscsiadmShow(t *testing.T) {
	fakeIscsiadmOutput1 := "# BEGIN RECORD 2.0-873\n" +
		"iface.iscsi_ifacename = default\n" +
		"iface.transport_name = tcp\n" +
		"iface.initiatorname = <empty>\n" +
		"iface.mtu = 0\n" +
		"# END RECORD"

	fakeIscsiadmOutput2 := "# BEGIN RECORD 2.0-873\n" +
		"iface.iscsi_ifacename = default\n" +
		"iface.transport_name = cxgb4i\n" +
		"iface.initiatorname = <empty>\n" +
		"iface.mtu = 0\n" +
		"# END RECORD"

	fakeIscsiadmOutput3 := "# BEGIN RECORD 2.0-873\n" +
		"iface.iscsi_ifacename = custom\n" +
		"iface.transport_name = <empty>\n" +
		"iface.initiatorname = <empty>\n" +
		"iface.mtu = 0\n" +
		"# END RECORD"

	fakeIscsiadmOutput4 := "iface.iscsi_ifacename=error"
	fakeIscsiadmOutput5 := "iface.iscsi_ifacename + error"

	expectedIscsiadmOutput1 := map[string]string{
		"iface.transport_name": "tcp",
		"iface.mtu":            "0"}

	expectedIscsiadmOutput2 := map[string]string{
		"iface.transport_name": "cxgb4i",
		"iface.mtu":            "0"}

	expectedIscsiadmOutput3 := map[string]string{
		"iface.mtu": "0"}

	params, _ := parseIscsiadmShow(fakeIscsiadmOutput1)
	if !reflect.DeepEqual(params, expectedIscsiadmOutput1) {
		t.Errorf("parseIscsiadmShow: Fail to parse iface record: %s", params)
	}
	params, _ = parseIscsiadmShow(fakeIscsiadmOutput2)
	if !reflect.DeepEqual(params, expectedIscsiadmOutput2) {
		t.Errorf("parseIscsiadmShow: Fail to parse iface record: %s", params)
	}
	params, _ = parseIscsiadmShow(fakeIscsiadmOutput3)
	if !reflect.DeepEqual(params, expectedIscsiadmOutput3) {
		t.Errorf("parseIscsiadmShow: Fail to parse iface record: %s", params)
	}
	_, err := parseIscsiadmShow(fakeIscsiadmOutput4)
	if err == nil {
		t.Errorf("parseIscsiadmShow: Fail to handle invalid record: iface %s", fakeIscsiadmOutput4)
	}
	_, err = parseIscsiadmShow(fakeIscsiadmOutput5)
	if err == nil {
		t.Errorf("parseIscsiadmShow: Fail to handle invalid record: iface %s", fakeIscsiadmOutput5)
	}
}

func TestClonedIface(t *testing.T) {
	fakeExec := &testingexec.FakeExec{}
	scripts := []volumetest.CommandScript{
		{
			Cmd:    "iscsiadm",
			Args:   []string{"-m", "iface", "-I", "", "-o", "show"},
			Output: "iface.ipaddress = <empty>\niface.transport_name = tcp\niface.initiatorname = <empty>\n",
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "new"},
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "update", "-n", "iface.initiatorname", "-v", ""},
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "update", "-n", "iface.transport_name", "-v", "tcp"},
		},
	}
	volumetest.ScriptCommands(fakeExec, scripts)
	fakeExec.ExactOrder = true
	plugins := []volume.VolumePlugin{
		&iscsiPlugin{
			host: nil,
		},
	}
	plugin := plugins[0]
	fakeMounter := iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			Iface:  TestIface,
			plugin: plugin.(*iscsiPlugin)},
		exec: fakeExec,
	}
	err := cloneIface(fakeMounter)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeExec.CommandCalls != len(scripts) {
		t.Errorf("expected 4 CombinedOutput() calls, got %d", fakeExec.CommandCalls)
	}
}

func TestClonedIfaceShowError(t *testing.T) {
	fakeExec := &testingexec.FakeExec{}
	scripts := []volumetest.CommandScript{
		{
			Cmd:        "iscsiadm",
			Args:       []string{"-m", "iface", "-I", "", "-o", "show"},
			Output:     "test error",
			ReturnCode: 1,
		},
	}
	volumetest.ScriptCommands(fakeExec, scripts)
	fakeExec.ExactOrder = true

	plugins := []volume.VolumePlugin{
		&iscsiPlugin{
			host: nil,
		},
	}
	plugin := plugins[0]
	fakeMounter := iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			Iface:  TestIface,
			plugin: plugin.(*iscsiPlugin)},
		exec: fakeExec,
	}
	err := cloneIface(fakeMounter)
	if err == nil {
		t.Errorf("expect to receive error, nil received")
	}
	if fakeExec.CommandCalls != len(scripts) {
		t.Errorf("expected 1 CombinedOutput() calls, got %d", fakeExec.CommandCalls)
	}

}

func TestClonedIfaceUpdateError(t *testing.T) {
	fakeExec := &testingexec.FakeExec{}
	scripts := []volumetest.CommandScript{
		{
			Cmd:    "iscsiadm",
			Args:   []string{"-m", "iface", "-I", "", "-o", "show"},
			Output: "iface.ipaddress = <empty>\niface.transport_name = tcp\niface.initiatorname = <empty>\n",
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "new"},
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "update", "-n", "iface.initiatorname", "-v", ""},
		},
		{
			Cmd:        "iscsiadm",
			Args:       []string{"-m", "iface", "-I", TestIface, "-o", "update", "-n", "iface.transport_name", "-v", "tcp"},
			ReturnCode: 1,
		},
		{
			Cmd:  "iscsiadm",
			Args: []string{"-m", "iface", "-I", TestIface, "-o", "delete"},
		},
	}
	volumetest.ScriptCommands(fakeExec, scripts)
	fakeExec.ExactOrder = true

	plugins := []volume.VolumePlugin{
		&iscsiPlugin{
			host: nil,
		},
	}
	plugin := plugins[0]
	fakeMounter := iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			Iface:  TestIface,
			plugin: plugin.(*iscsiPlugin)},
		exec: fakeExec,
	}
	err := cloneIface(fakeMounter)
	if err == nil {
		t.Errorf("expect to receive error, nil received")
	}
	if fakeExec.CommandCalls != len(scripts) {
		t.Errorf("expected 5 CombinedOutput() calls, got %d", fakeExec.CommandCalls)
	}

}

func TestGetVolCount(t *testing.T) {
	// This will create a dir structure like this:
	// /tmp/refcounter555814673
	// ├── iface-127.0.0.1:3260:pv1
	// │   └── 127.0.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-3
	// └── iface-127.0.0.1:3260:pv2
	// │   ├── 127.0.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-2
	// │   └── 192.168.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-1
	// └── volumeDevices
	//     └── 192.168.0.2:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-4
	//     └── 192.168.0.3:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-5

	baseDir, err := createFakePluginDirs()
	if err != nil {
		t.Errorf("error creating fake plugin dir: %v", err)
	}

	defer os.RemoveAll(baseDir)

	testCases := []struct {
		name    string
		baseDir string
		portal  string
		iqn     string
		count   int
	}{
		{
			name:    "wrong portal, no volumes",
			baseDir: baseDir,
			portal:  "192.168.0.2:3260", // incorrect IP address
			iqn:     "iqn.2003-01.io.k8s:e2e.volume-1",
			count:   0,
		},
		{
			name:    "wrong iqn, no volumes",
			baseDir: baseDir,
			portal:  "127.0.0.1:3260",
			iqn:     "iqn.2003-01.io.k8s:e2e.volume-3", // incorrect volume
			count:   0,
		},
		{
			name:    "single volume",
			baseDir: baseDir,
			portal:  "192.168.0.1:3260",
			iqn:     "iqn.2003-01.io.k8s:e2e.volume-1",
			count:   1,
		},
		{
			name:    "two volumes",
			baseDir: baseDir,
			portal:  "127.0.0.1:3260",
			iqn:     "iqn.2003-01.io.k8s:e2e.volume-1",
			count:   2,
		},
		{
			name:    "volumeDevices (block) volume",
			baseDir: filepath.Join(baseDir, config.DefaultKubeletVolumeDevicesDirName),
			portal:  "192.168.0.2:3260",
			iqn:     "iqn.2003-01.io.k8s:e2e.volume-1-lun-4",
			count:   1,
		},
		{
			name:    "nonexistent path",
			baseDir: filepath.Join(baseDir, "this_path_should_not_exist"),
			portal:  "127.0.0.1:3260",
			iqn:     "iqn.2003-01.io.k8s:e2e.unknown",
			count:   0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			count, err := getVolCount(tc.baseDir, tc.portal, tc.iqn)
			if err != nil {
				t.Errorf("expected no error, got %v", err)
			}
			if count != tc.count {
				t.Errorf("expected %d volumes, got %d", tc.count, count)
			}
		})
	}
}

func createFakePluginDirs() (string, error) {
	dir, err := ioutil.TempDir("", "refcounter")
	if err != nil {
		return "", err
	}

	subdirs := []string{
		"iface-127.0.0.1:3260:pv1/127.0.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-3",
		"iface-127.0.0.1:3260:pv2/127.0.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-2",
		"iface-127.0.0.1:3260:pv2/192.168.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-1",
		filepath.Join(config.DefaultKubeletVolumeDevicesDirName, "iface-127.0.0.1:3260/192.168.0.2:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-4"),
		filepath.Join(config.DefaultKubeletVolumeDevicesDirName, "iface-127.0.0.1:3260/192.168.0.3:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-5"),
	}

	for _, d := range subdirs {
		if err := os.MkdirAll(filepath.Join(dir, d), os.ModePerm); err != nil {
			return dir, err
		}
	}

	return dir, err
}
