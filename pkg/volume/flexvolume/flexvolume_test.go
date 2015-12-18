/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package flexvolume

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"testing"
	"text/template"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// The temp dir where test plugins will be stored.
const testPluginPath = "/tmp/fake/plugins/volume"

const execScriptTempl1 = `#!/bin/bash
if [ "$1" == "init" -a $# -eq 1 ]; then
  echo -n '{
    "status": "Success"
  }'
  exit 0
fi

PATH=$2
if [ "$1" == "attach" -a $# -eq 2 ]; then
  echo -n '{
    "device": "{{.DevicePath}}",
    "status": "Success"
  }'
  exit 0
elif [ "$1" == "detach" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Success"
  }'
  exit 0
elif [ "$1" == "mount" -a $# -eq 4 ]; then
  echo -n '{
    "status": "Not supported"
  }'
  exit 0
elif [ "$1" == "unmount" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Not supported"
  }'
  exit 0
fi

echo -n '{
  "status": "Failure",
  "reason": "Invalid usage"
}'
exit 1

# Direct the arguments to a file to be tested against later
echo -n $@ &> {{.OutputFile}}
`

const execScriptTempl2 = `#!/bin/bash
if [ "$1" == "init" -a $# -eq 1 ]; then
  echo -n '{
    "status": "Success"
  }'
  exit 0
fi

if [ "$1" == "attach" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Not supported"
  }'
  exit 0
elif [ "$1" == "detach" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Not supported"
  }'
  exit 0
elif [ "$1" == "mount" -a $# -eq 4 ]; then
  PATH=$2
  /bin/mkdir -p $PATH
  if [ $? -ne 0 ]; then
    echo -n '{
      "status": "Failure",
      "reason": "Failed to create $PATH"
    }'
    exit 1
  fi
  echo -n '{
    "status": "Success"
  }'
  exit 0
elif [ "$1" == "unmount" -a $# -eq 2 ]; then
  PATH=$2
  /bin/rm -r $PATH
  if [ $? -ne 0 ]; then
    echo -n '{
      "status": "Failure",
      "reason": "Failed to cleanup $PATH"
    }'
    exit 1
  fi
  echo -n '{
    "status": "Success"
  }'
  exit 0
fi

echo -n '{
  "status": "Failure",
  "reason": "Invalid usage"
}'
exit 1

# Direct the arguments to a file to be tested against later
echo -n $@ &> {{.OutputFile}}
`

func installPluginUnderTest(t *testing.T, vendorName string, plugName string, execScriptTempl string, execTemplateData *map[string]interface{}) {
	vendoredName := plugName
	if vendorName != "" {
		vendoredName = fmt.Sprintf("%s~%s", vendorName, plugName)
	}
	pluginDir := path.Join(testPluginPath, vendoredName)
	err := os.MkdirAll(pluginDir, 0777)
	if err != nil {
		t.Errorf("Failed to create plugin: %v", err)
	}
	pluginExec := path.Join(pluginDir, plugName)
	f, err := os.Create(pluginExec)
	if err != nil {
		t.Errorf("Failed to install plugin")
	}
	err = f.Chmod(0777)
	if err != nil {
		t.Errorf("Failed to set exec perms on plugin")
	}
	if execTemplateData == nil {
		execTemplateData = &map[string]interface{}{
			"DevicePath": "/dev/sdx",
			"OutputFile": path.Join(pluginDir, plugName+".out"),
		}
	}

	tObj := template.Must(template.New("test").Parse(execScriptTempl))
	buf := &bytes.Buffer{}
	if err := tObj.Execute(buf, *execTemplateData); err != nil {
		t.Errorf("Error in executing script template - %v", err)
	}
	execScript := buf.String()
	_, err = f.WriteString(execScript)
	if err != nil {
		t.Errorf("Failed to write plugin exec")
	}
	f.Close()
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	installPluginUnderTest(t, "kubernetes.io", "fakeAttacher", execScriptTempl1, nil)
	plugMgr.InitPlugins(ProbeVolumePlugins(testPluginPath), volume.NewFakeVolumeHost("fake", nil, nil))
	plugin, err := plugMgr.FindPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.Name() != "kubernetes.io/fakeAttacher" {
		t.Errorf("Wrong name: %s", plugin.Name())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}) {
		t.Errorf("Expected true")
	}
	if !plugin.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}}) {
		t.Errorf("Expected true")
	}
	if plugin.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(testPluginPath), volume.NewFakeVolumeHost("fake", nil, nil))

	plugin, err := plugMgr.FindPersistentPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plugin.GetAccessModes(), api.ReadWriteOnce) || !contains(plugin.GetAccessModes(), api.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", api.ReadWriteOnce, api.ReadOnlyMany)
	}
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func doTestPluginAttachDetach(t *testing.T, spec *volume.Spec) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(testPluginPath), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	plugin, err := plugMgr.FindPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fake := &mount.FakeMounter{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plugin.(*flexVolumePlugin).newBuilderInternal(spec, pod, &flexVolumeUtil{}, fake, exec.New(), "")
	volumePath := builder.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}
	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~fakeAttacher/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}
	if err := builder.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	t.Logf("Setup successful")
	if builder.(*flexVolumeBuilder).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}

	if len(fake.Log) != 1 {
		t.Errorf("Mount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionMount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}
	fake.ResetLog()

	cleaner, err := plugin.(*flexVolumePlugin).newCleanerInternal("vol1", types.UID("poduid"), &flexVolumeUtil{}, fake, exec.New())
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}
	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(fake.Log) != 1 {
		t.Errorf("Unmount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}

	fake.ResetLog()
}

func doTestPluginMountUnmount(t *testing.T, spec *volume.Spec) {
	plugMgr := volume.VolumePluginMgr{}
	installPluginUnderTest(t, "kubernetes.io", "fakeMounter", execScriptTempl2, nil)
	plugMgr.InitPlugins(ProbeVolumePlugins(testPluginPath), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	plugin, err := plugMgr.FindPluginByName("kubernetes.io/fakeMounter")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fake := &mount.FakeMounter{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plugin.(*flexVolumePlugin).newBuilderInternal(spec, pod, &flexVolumeUtil{}, fake, exec.New(), "")
	volumePath := builder.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}
	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~fakeMounter/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}
	if err := builder.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	t.Logf("Setup successful")
	if builder.(*flexVolumeBuilder).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}

	cleaner, err := plugin.(*flexVolumePlugin).newCleanerInternal("vol1", types.UID("poduid"), &flexVolumeUtil{}, fake, exec.New())
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}
	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestPluginVolumeAttacher(t *testing.T) {
	vol := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher", ReadOnly: false}},
	}
	doTestPluginAttachDetach(t, volume.NewSpecFromVolume(vol))
}

func TestPluginVolumeMounter(t *testing.T) {
	vol := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/fakeMounter", ReadOnly: false}},
	}
	doTestPluginMountUnmount(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolume(t *testing.T) {
	vol := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "vol1",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher", ReadOnly: false},
			},
		},
	}

	doTestPluginAttachDetach(t, volume.NewSpecFromPersistentVolume(vol, false))
}
