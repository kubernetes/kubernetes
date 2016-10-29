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

package cni

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"path"
	"testing"
	"text/template"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	cnitypes "github.com/containernetworking/cni/pkg/types"
	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni/testing"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func installPluginUnderTest(t *testing.T, testVendorCNIDirPrefix, testNetworkConfigPath, vendorName string, plugName string) {
	pluginDir := path.Join(testNetworkConfigPath, plugName)
	err := os.MkdirAll(pluginDir, 0777)
	if err != nil {
		t.Fatalf("Failed to create plugin config dir: %v", err)
	}
	pluginConfig := path.Join(pluginDir, plugName+".conf")
	f, err := os.Create(pluginConfig)
	if err != nil {
		t.Fatalf("Failed to install plugin")
	}
	networkConfig := fmt.Sprintf("{ \"name\": \"%s\", \"type\": \"%s\" }", plugName, vendorName)

	_, err = f.WriteString(networkConfig)
	if err != nil {
		t.Fatalf("Failed to write network config file (%v)", err)
	}
	f.Close()

	vendorCNIDir := fmt.Sprintf(VendorCNIDirTemplate, testVendorCNIDirPrefix, vendorName)
	err = os.MkdirAll(vendorCNIDir, 0777)
	if err != nil {
		t.Fatalf("Failed to create plugin dir: %v", err)
	}
	pluginExec := path.Join(vendorCNIDir, vendorName)
	f, err = os.Create(pluginExec)

	const execScriptTempl = `#!/bin/bash
read ignore
env > {{.OutputEnv}}
echo "%@" >> {{.OutputEnv}}
export $(echo ${CNI_ARGS} | sed 's/;/ /g') &> /dev/null
mkdir -p {{.OutputDir}} &> /dev/null
echo -n "$CNI_COMMAND $CNI_NETNS $K8S_POD_NAMESPACE $K8S_POD_NAME $K8S_POD_INFRA_CONTAINER_ID" >& {{.OutputFile}}
echo -n "{ \"ip4\": { \"ip\": \"10.1.0.23/24\" } }"
`
	execTemplateData := &map[string]interface{}{
		"OutputFile": path.Join(pluginDir, plugName+".out"),
		"OutputEnv":  path.Join(pluginDir, plugName+".env"),
		"OutputDir":  pluginDir,
	}

	tObj := template.Must(template.New("test").Parse(execScriptTempl))
	buf := &bytes.Buffer{}
	if err := tObj.Execute(buf, *execTemplateData); err != nil {
		t.Fatalf("Error in executing script template - %v", err)
	}
	execScript := buf.String()
	_, err = f.WriteString(execScript)
	if err != nil {
		t.Fatalf("Failed to write plugin exec - %v", err)
	}

	err = f.Chmod(0777)
	if err != nil {
		t.Fatalf("Failed to set exec perms on plugin")
	}

	f.Close()
}

func tearDownPlugin(tmpDir string) {
	err := os.RemoveAll(tmpDir)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
}

type fakeNetworkHost struct {
	kubeClient clientset.Interface
	runtime    kubecontainer.Runtime
}

func NewFakeHost(kubeClient clientset.Interface, pods []*containertest.FakePod) *fakeNetworkHost {
	host := &fakeNetworkHost{
		kubeClient: kubeClient,
		runtime: &containertest.FakeRuntime{
			AllPodList: pods,
		},
	}
	return host
}

func (fnh *fakeNetworkHost) GetPodByName(name, namespace string) (*api.Pod, bool) {
	return nil, false
}

func (fnh *fakeNetworkHost) GetKubeClient() clientset.Interface {
	return fnh.kubeClient
}

func (fnh *fakeNetworkHost) GetRuntime() kubecontainer.Runtime {
	return fnh.runtime
}

func (fnh *fakeNetworkHost) GetNetNS(containerID string) (string, error) {
	return fnh.GetRuntime().GetNetNS(kubecontainer.ContainerID{Type: "test", ID: containerID})
}

func (fnh *fakeNetworkHost) SupportsLegacyFeatures() bool {
	return true
}

func TestCNIPlugin(t *testing.T) {
	// install some random plugin
	pluginName := fmt.Sprintf("test%d", rand.Intn(1000))
	vendorName := fmt.Sprintf("test_vendor%d", rand.Intn(1000))

	podIP := "10.0.0.2"
	podIPOutput := fmt.Sprintf("4: eth0    inet %s/24 scope global dynamic eth0\\       valid_lft forever preferred_lft forever", podIP)
	fakeCmds := []utilexec.FakeCommandAction{
		func(cmd string, args ...string) utilexec.Cmd {
			return utilexec.InitFakeCmd(&utilexec.FakeCmd{
				CombinedOutputScript: []utilexec.FakeCombinedOutputAction{
					func() ([]byte, error) {
						return []byte(podIPOutput), nil
					},
				},
			}, cmd, args...)
		},
	}

	fexec := &utilexec.FakeExec{
		CommandScript: fakeCmds,
		LookPathFunc: func(file string) (string, error) {
			return fmt.Sprintf("/fake-bin/%s", file), nil
		},
	}

	mockLoCNI := &mock_cni.MockCNI{}
	// TODO mock for the test plugin too

	tmpDir := utiltesting.MkTmpdirOrDie("cni-test")
	testNetworkConfigPath := path.Join(tmpDir, "plugins", "net", "cni")
	testVendorCNIDirPrefix := tmpDir
	defer tearDownPlugin(tmpDir)
	installPluginUnderTest(t, testVendorCNIDirPrefix, testNetworkConfigPath, vendorName, pluginName)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "test_infra_container"}
	pods := []*containertest.FakePod{{
		Pod: &kubecontainer.Pod{
			Containers: []*kubecontainer.Container{
				{ID: containerID},
			},
		},
		NetnsPath: "/proc/12345/ns/net",
	}}

	plugins := probeNetworkPluginsWithVendorCNIDirPrefix(path.Join(testNetworkConfigPath, pluginName), "", testVendorCNIDirPrefix)
	if len(plugins) != 1 {
		t.Fatalf("Expected only one network plugin, got %d", len(plugins))
	}
	if plugins[0].Name() != "cni" {
		t.Fatalf("Expected CNI network plugin, got %q", plugins[0].Name())
	}

	cniPlugin, ok := plugins[0].(*cniNetworkPlugin)
	if !ok {
		t.Fatalf("Not a CNI network plugin!")
	}
	cniPlugin.execer = fexec
	cniPlugin.loNetwork.CNIConfig = mockLoCNI

	mockLoCNI.On("AddNetwork", cniPlugin.loNetwork.NetworkConfig, mock.AnythingOfType("*libcni.RuntimeConf")).Return(&cnitypes.Result{IP4: &cnitypes.IPConfig{IP: net.IPNet{IP: []byte{127, 0, 0, 1}}}}, nil)

	plug, err := network.InitNetworkPlugin(plugins, "cni", NewFakeHost(nil, pods), componentconfig.HairpinNone, "10.0.0.0/8", network.UseDefaultMTU)
	if err != nil {
		t.Fatalf("Failed to select the desired plugin: %v", err)
	}

	// Set up the pod
	err = plug.SetUpPod("podNamespace", "podName", containerID)
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	outputEnv := path.Join(testNetworkConfigPath, pluginName, pluginName+".env")
	eo, eerr := ioutil.ReadFile(outputEnv)
	outputFile := path.Join(testNetworkConfigPath, pluginName, pluginName+".out")
	output, err := ioutil.ReadFile(outputFile)
	if err != nil {
		t.Errorf("Failed to read output file %s: %v (env %s err %v)", outputFile, err, eo, eerr)
	}
	expectedOutput := "ADD /proc/12345/ns/net podNamespace podName test_infra_container"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}

	// Get its IP address
	status, err := plug.GetPodNetworkStatus("podNamespace", "podName", containerID)
	if err != nil {
		t.Errorf("Failed to read pod network status: %v", err)
	}
	if status.IP.String() != podIP {
		t.Errorf("Expected pod IP %q but got %q", podIP, status.IP.String())
	}

	// Tear it down
	err = plug.TearDownPod("podNamespace", "podName", containerID)
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	output, err = ioutil.ReadFile(path.Join(testNetworkConfigPath, pluginName, pluginName+".out"))
	expectedOutput = "DEL /proc/12345/ns/net podNamespace podName test_infra_container"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}

	mockLoCNI.AssertExpectations(t)
}

func TestLoNetNonNil(t *testing.T) {
	if conf := getLoNetwork("", ""); conf == nil {
		t.Error("Expected non-nil lo network")
	}
}
