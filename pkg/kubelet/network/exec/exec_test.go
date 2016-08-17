// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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

package exec

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path"
	"sync"
	"testing"
	"text/template"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/util/sets"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func tmpDirOrDie() string {
	dir, err := utiltesting.MkTmpdir("exec-test")
	if err != nil {
		panic(fmt.Sprintf("error creating tmp dir: %v", err))
	}
	return path.Join(dir, "fake", "plugins", "net")
}

var lock sync.Mutex
var namesInUse = sets.NewString()

func selectName() string {
	lock.Lock()
	defer lock.Unlock()
	for {
		pluginName := fmt.Sprintf("test%d", rand.Intn(1000))
		if !namesInUse.Has(pluginName) {
			namesInUse.Insert(pluginName)
			return pluginName
		}
	}
}

func releaseName(name string) {
	lock.Lock()
	defer lock.Unlock()
	namesInUse.Delete(name)
}

func installPluginUnderTest(t *testing.T, vendorName, testPluginPath, plugName string, execTemplateData *map[string]interface{}) {
	vendoredName := plugName
	if vendorName != "" {
		vendoredName = fmt.Sprintf("%s~%s", vendorName, plugName)
	}
	pluginDir := path.Join(testPluginPath, vendoredName)
	err := os.MkdirAll(pluginDir, 0777)
	if err != nil {
		t.Errorf("Failed to create plugin dir %q: %v", pluginDir, err)
	}
	pluginExec := path.Join(pluginDir, plugName)
	f, err := os.Create(pluginExec)
	if err != nil {
		t.Errorf("Failed to install plugin %q: %v", pluginExec, err)
	}
	defer f.Close()
	err = f.Chmod(0777)
	if err != nil {
		t.Errorf("Failed to set exec perms on plugin %q: %v", pluginExec, err)
	}
	const execScriptTempl = `#!/bin/bash

# If status hook is called print the expected json to stdout
if [ "$1" == "status" ]; then
  echo -n '{
	"ip" : "{{.IPAddress}}"
}'
fi

# Direct the arguments to a file to be tested against later
echo -n "$@" &> {{.OutputFile}}
`
	if execTemplateData == nil {
		execTemplateData = &map[string]interface{}{
			"IPAddress":  "10.20.30.40",
			"OutputFile": path.Join(pluginDir, plugName+".out"),
		}
	}

	tObj := template.Must(template.New("test").Parse(execScriptTempl))
	buf := &bytes.Buffer{}
	if err := tObj.Execute(buf, *execTemplateData); err != nil {
		t.Errorf("Error in executing script template: %v", err)
	}
	execScript := buf.String()
	_, err = f.WriteString(execScript)
	if err != nil {
		t.Errorf("Failed to write plugin %q: %v", pluginExec, err)
	}
}

func tearDownPlugin(testPluginPath string) {
	err := os.RemoveAll(testPluginPath)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
}

func TestSelectPlugin(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")
	if err != nil {
		t.Errorf("Failed to select the desired plugin: %v", err)
	}
	if plug.Name() != pluginName {
		t.Errorf("Wrong plugin selected, chose %s, got %s\n", pluginName, plug.Name())
	}
}

func TestSelectVendoredPlugin(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	vendor := "mycompany"
	installPluginUnderTest(t, vendor, testPluginPath, pluginName, nil)

	vendoredPluginName := fmt.Sprintf("%s/%s", vendor, pluginName)
	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), vendoredPluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")
	if err != nil {
		t.Errorf("Failed to select the desired plugin: %v", err)
	}
	if plug.Name() != vendoredPluginName {
		t.Errorf("Wrong plugin selected, chose %s, got %s\n", vendoredPluginName, plug.Name())
	}
}

func TestSelectWrongPlugin(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	wrongPlugin := "abcd"
	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), wrongPlugin, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")
	if plug != nil || err == nil {
		t.Errorf("Expected to see an error. Wrong plugin selected.")
	}
}

func TestPluginValidation(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	// modify the perms of the pluginExecutable
	f, err := os.Open(path.Join(testPluginPath, pluginName, pluginName))
	if err != nil {
		t.Errorf("Nil value expected.")
	}
	err = f.Chmod(0444)
	if err != nil {
		t.Errorf("Failed to set perms on plugin exec")
	}
	f.Close()

	_, err = network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")
	if err == nil {
		// we expected an error here because validation would have failed
		t.Errorf("Expected non-nil value.")
	}
}

func TestPluginSetupHook(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")

	err = plug.SetUpPod("podNamespace", "podName", kubecontainer.ContainerID{Type: "docker", ID: "dockerid2345"})
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	// check output of setup hook
	output, err := ioutil.ReadFile(path.Join(testPluginPath, pluginName, pluginName+".out"))
	if err != nil {
		t.Errorf("Expected nil")
	}
	expectedOutput := "setup podNamespace podName dockerid2345"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}
}

func TestPluginTearDownHook(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")

	err = plug.TearDownPod("podNamespace", "podName", kubecontainer.ContainerID{Type: "docker", ID: "dockerid2345"})
	if err != nil {
		t.Errorf("Expected nil")
	}
	// check output of setup hook
	output, err := ioutil.ReadFile(path.Join(testPluginPath, pluginName, pluginName+".out"))
	if err != nil {
		t.Errorf("Expected nil")
	}
	expectedOutput := "teardown podNamespace podName dockerid2345"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for teardown hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}
}

func TestPluginStatusHook(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	installPluginUnderTest(t, "", testPluginPath, pluginName, nil)

	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")

	ip, err := plug.GetPodNetworkStatus("namespace", "name", kubecontainer.ContainerID{Type: "docker", ID: "dockerid2345"})
	if err != nil {
		t.Errorf("Expected nil got %v", err)
	}
	// check output of status hook
	output, err := ioutil.ReadFile(path.Join(testPluginPath, pluginName, pluginName+".out"))
	if err != nil {
		t.Errorf("Expected nil")
	}
	expectedOutput := "status namespace name dockerid2345"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for status hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}
	if ip.IP.String() != "10.20.30.40" {
		t.Errorf("Mismatch in expected output for status hook. Expected '10.20.30.40', got '%s'", ip.IP.String())
	}
}

func TestPluginStatusHookIPv6(t *testing.T) {
	// The temp dir where test plugins will be stored.
	testPluginPath := tmpDirOrDie()

	// install some random plugin under testPluginPath
	pluginName := selectName()
	defer tearDownPlugin(testPluginPath)
	defer releaseName(pluginName)

	pluginDir := path.Join(testPluginPath, pluginName)
	execTemplate := &map[string]interface{}{
		"IPAddress":  "fe80::e2cb:4eff:fef9:6710",
		"OutputFile": path.Join(pluginDir, pluginName+".out"),
	}
	installPluginUnderTest(t, "", testPluginPath, pluginName, execTemplate)

	plug, err := network.InitNetworkPlugin(ProbeNetworkPlugins(testPluginPath), pluginName, nettest.NewFakeHost(nil), componentconfig.HairpinNone, "10.0.0.0/8")
	if err != nil {
		t.Errorf("InitNetworkPlugin() failed: %v", err)
	}

	ip, err := plug.GetPodNetworkStatus("namespace", "name", kubecontainer.ContainerID{Type: "docker", ID: "dockerid2345"})
	if err != nil {
		t.Errorf("Status() failed: %v", err)
	}
	// check output of status hook
	outPath := path.Join(testPluginPath, pluginName, pluginName+".out")
	output, err := ioutil.ReadFile(outPath)
	if err != nil {
		t.Errorf("ReadFile(%q) failed: %v", outPath, err)
	}
	expectedOutput := "status namespace name dockerid2345"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for status hook. Expected %q, got %q", expectedOutput, string(output))
	}
	if ip.IP.String() != "fe80::e2cb:4eff:fef9:6710" {
		t.Errorf("Mismatch in expected output for status hook. Expected 'fe80::e2cb:4eff:fef9:6710', got '%s'", ip.IP.String())
	}
}
