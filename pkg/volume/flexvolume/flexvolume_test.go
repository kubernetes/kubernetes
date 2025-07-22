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
	"path/filepath"
	goruntime "runtime"
	"testing"
	"text/template"

	"k8s.io/api/core/v1"
	utiltesting "k8s.io/client-go/util/testing"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/exec"
)

const execScriptTempl1 = `#!/usr/bin/env bash
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

// NOTE: Typically, Windows requires file extensions for executable files. If a file does not
// have a file extension, Windows will check if there is a file with the given name + one of the
// extensions from $env:PATHEXT (in order) and run that file with that extension.
// For example, if we have the file C:\\foo.bat, we can run C:\\foo.
// For these tests, .bat was chosen since it's one of the default values in $env.PATHEXT. .ps1 is
// not in that list, but it might be useful for flexvolumes to be able to handle powershell scripts.
// There's no argument count variable in batch. Instead, we can check that the n-th argument
// is an empty string.
const execScriptTemplBat = `
@echo off

if "%1"=="init" if "%2"=="" (
    echo {"status": "Success"}
    exit 0
)
if "%1"=="attach" if "%3"=="" (
    echo {"device": "{{.DevicePath}}", "status": "Success"}
    exit 0
)

if "%1"=="detach" if "%3"=="" (
    echo {"status": "Success"}
    exit 0
)

if "%1"=="getvolumename" if "%5"=="" (
    echo {"status": "Success", "volume": "fakevolume"}
    exit 0
)

if "%1"=="isattached" if "%3"=="" (
    echo {"status": "Success", "attached": true}
    exit 0
)

echo {"status": "Not supported"}
exit 1
`

func installPluginUnderTest(t *testing.T, vendorName, plugName, tmpDir string, execScriptTempl string, execTemplateData *map[string]interface{}) {
	vendoredName := plugName
	if vendorName != "" {
		vendoredName = fmt.Sprintf("%s~%s", vendorName, plugName)
	}
	pluginDir := filepath.Join(tmpDir, vendoredName)
	err := os.MkdirAll(pluginDir, 0777)
	if err != nil {
		t.Errorf("Failed to create plugin: %v", err)
	}
	pluginExec := filepath.Join(pluginDir, plugName)
	if goruntime.GOOS == "windows" {
		pluginExec = pluginExec + ".bat"
	}
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
			"OutputFile": filepath.Join(pluginDir, plugName+".out"),
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
	runner := exec.New()
	execScriptTempl := execScriptTempl1
	if goruntime.GOOS == "windows" {
		execScriptTempl = execScriptTemplBat
	}
	installPluginUnderTest(t, "kubernetes.io", "fakeAttacher", tmpDir, execScriptTempl, nil)
	if err := plugMgr.InitPlugins(nil, GetDynamicPluginProberWithoutWatcher(tmpDir, runner), volumetest.NewFakeVolumeHost(t, "fake", nil, nil)); err != nil {
		t.Fatalf("Could not initialize plugins: %v", err)
	}
	plugin, err := plugMgr.FindPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Fatalf("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != "kubernetes.io/fakeAttacher" {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{FlexVolume: &v1.FlexVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}) {
		t.Errorf("Expected true")
	}
	if !plugin.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{FlexVolume: &v1.FlexPersistentVolumeSource{Driver: "kubernetes.io/fakeAttacher"}}}}}) {
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
	runner := exec.New()
	execScriptTempl := execScriptTempl1
	if goruntime.GOOS == "windows" {
		execScriptTempl = execScriptTemplBat
	}
	installPluginUnderTest(t, "kubernetes.io", "fakeAttacher", tmpDir, execScriptTempl, nil)
	if err := plugMgr.InitPlugins(nil, GetDynamicPluginProberWithoutWatcher(tmpDir, runner), volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil)); err != nil {
		t.Fatalf("Could not initialize plugins: %v", err)
	}
	plugin, err := plugMgr.FindPersistentPluginByName("kubernetes.io/fakeAttacher")
	if err != nil {
		t.Fatalf("Can't find the plugin by name")
	}
	if !volumetest.ContainsAccessMode(plugin.GetAccessModes(), v1.ReadWriteOnce) || !volumetest.ContainsAccessMode(plugin.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
	}
}

func GetDynamicPluginProberWithoutWatcher(pluginDir string, runner exec.Interface) volume.DynamicPluginProber {
	return &flexVolumeProber{
		pluginDir: pluginDir,
		watcher:   newFakeWatcher(),
		factory:   pluginFactory{},
		runner:    runner,
		fs:        &utilfs.DefaultFs{},
	}
}
