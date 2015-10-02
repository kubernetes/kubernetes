// +build linux

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"os"
	"path"
	"testing"
	"text/template"

	docker "github.com/fsouza/go-dockerclient"
	cadvisorApi "github.com/google/cadvisor/info/v1"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/kubernetes/pkg/util/sets"
)

// The temp dir where test plugins will be stored.
const testNetworkConfigPath = "/tmp/fake/plugins/net/cni"
const testVendorCNIDirPrefix = "/tmp"

func installPluginUnderTest(t *testing.T, vendorName string, plugName string) {
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
export $(echo ${CNI_ARGS} | sed 's/;/ /g') &> /dev/null
mkdir -p {{.OutputDir}} &> /dev/null
echo -n "$CNI_COMMAND $CNI_NETNS $K8S_POD_NAMESPACE $K8S_POD_NAME $K8S_POD_INFRA_CONTAINER_ID" >& {{.OutputFile}}
echo -n "{ \"ip4\": { \"ip\": \"10.1.0.23/24\" } }"
`
	execTemplateData := &map[string]interface{}{
		"OutputFile": path.Join(pluginDir, plugName+".out"),
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

func tearDownPlugin(plugName string, vendorName string) {
	err := os.RemoveAll(testNetworkConfigPath)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
	vendorCNIDir := fmt.Sprintf(VendorCNIDirTemplate, testVendorCNIDirPrefix, vendorName)
	err = os.RemoveAll(vendorCNIDir)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
}

type fakeNetworkHost struct {
	kubeClient client.Interface
}

func NewFakeHost(kubeClient client.Interface) *fakeNetworkHost {
	host := &fakeNetworkHost{kubeClient: kubeClient}
	return host
}

func (fnh *fakeNetworkHost) GetPodByName(name, namespace string) (*api.Pod, bool) {
	return nil, false
}

func (fnh *fakeNetworkHost) GetKubeClient() client.Interface {
	return nil
}

func (nh *fakeNetworkHost) GetRuntime() kubecontainer.Runtime {
	dm, fakeDockerClient := newTestDockerManager()
	fakeDockerClient.Container = &docker.Container{
		ID:    "foobar",
		State: docker.State{Pid: 12345},
	}
	return dm
}

func newTestDockerManager() (*dockertools.DockerManager, *dockertools.FakeDockerClient) {
	fakeDocker := &dockertools.FakeDockerClient{VersionInfo: docker.Env{"Version=1.1.3", "ApiVersion=1.15"}, Errors: make(map[string]error), RemovedImages: sets.String{}}
	fakeRecorder := &record.FakeRecorder{}
	containerRefManager := kubecontainer.NewRefManager()
	networkPlugin, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	dockerManager := dockertools.NewFakeDockerManager(
		fakeDocker,
		fakeRecorder,
		prober.FakeProber{},
		containerRefManager,
		&cadvisorApi.MachineInfo{},
		dockertools.PodInfraContainerImage,
		0, 0, "",
		kubecontainer.FakeOS{},
		networkPlugin,
		nil,
		nil,
		nil)

	return dockerManager, fakeDocker
}

func TestCNIPlugin(t *testing.T) {
	// install some random plugin
	pluginName := fmt.Sprintf("test%d", rand.Intn(1000))
	vendorName := fmt.Sprintf("test_vendor%d", rand.Intn(1000))
	defer tearDownPlugin(pluginName, vendorName)
	installPluginUnderTest(t, vendorName, pluginName)

	np := probeNetworkPluginsWithVendorCNIDirPrefix(path.Join(testNetworkConfigPath, pluginName), testVendorCNIDirPrefix)
	plug, err := network.InitNetworkPlugin(np, "cni", NewFakeHost(nil))
	if err != nil {
		t.Fatalf("Failed to select the desired plugin: %v", err)
	}

	err = plug.SetUpPod("podNamespace", "podName", "dockerid2345")
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	output, err := ioutil.ReadFile(path.Join(testNetworkConfigPath, pluginName, pluginName+".out"))
	expectedOutput := "ADD /proc/12345/ns/net podNamespace podName dockerid2345"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}
	err = plug.TearDownPod("podNamespace", "podName", "dockerid4545454")
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	output, err = ioutil.ReadFile(path.Join(testNetworkConfigPath, pluginName, pluginName+".out"))
	expectedOutput = "DEL /proc/12345/ns/net podNamespace podName dockerid4545454"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}
}
