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

package flexvolume

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"testing"
	"text/template"

	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

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
elif [ "$1" == "getvolumename" -a $# -eq 4 ]; then
  echo -n '{
    "status": "Success",
    "volume": "fakevolume"
  }'
  exit 0
elif [ "$1" == "isattached" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Success",
    "attached": true
  }'
  exit 0
fi

echo -n '{
  "status": "Not supported"
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

if [ "$1" == "getvolumename" -a $# -eq 2 ]; then
  echo -n '{
    "status": "Success",
    "volumeName": "fakevolume"
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
  "status": "Not Supported"
}'
exit 1

# Direct the arguments to a file to be tested against later
echo -n $@ &> {{.OutputFile}}
`

func installPluginUnderTest(t *testing.T, vendorName, plugName, tmpDir string, execScriptTempl string, execTemplateData *map[string]interface{}) {
	vendoredName := plugName
	if vendorName != "" {
		vendoredName = fmt.Sprintf("%s~%s", vendorName, plugName)
	}
	pluginDir := path.Join(tmpDir, vendoredName)
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
	tmpDir, err := utiltesting.MkTmpdir("flexvolume_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	installPluginUnderTest(t, "kubernetes.io", "fakeAttacher", tmpDir, execScriptTempl1, nil)
	plugMgr.InitPlugins(ProbeVolumePlugins(tmpDir), volumetest.NewFakeVolumeHost("fake", nil, nil))
	plugin, err := plugMgr.FindPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != "kubernetes.io/fakeAttacher" {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{FlexVolume: &v1.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}) {
		t.Errorf("Expected true")
	}
	if !plugin.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{FlexVolume: &v1.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}}) {
		t.Errorf("Expected true")
	}
	if plugin.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("flexvolume_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	installPluginUnderTest(t, "kubernetes.io", "fakeAttacher", tmpDir, execScriptTempl1, nil)
	plugMgr.InitPlugins(ProbeVolumePlugins(tmpDir), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plugin, err := plugMgr.FindPersistentPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Fatalf("Can't find the plugin by name")
	}
	if !contains(plugin.GetAccessModes(), v1.ReadWriteOnce) || !contains(plugin.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
	}
}

func contains(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}
