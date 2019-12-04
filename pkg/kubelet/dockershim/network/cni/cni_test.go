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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"path"
	"reflect"
	"testing"
	"text/template"

	types020 "github.com/containernetworking/cni/pkg/types/020"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	utiltesting "k8s.io/client-go/util/testing"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/cni/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport"
	networktest "k8s.io/kubernetes/pkg/kubelet/dockershim/network/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

// Returns .in file path, .out file path, and .env file path
func installPluginUnderTest(t *testing.T, testBinDir, testConfDir, testDataDir, binName string, confName, podIP string) (string, string, string) {
	for _, dir := range []string{testBinDir, testConfDir, testDataDir} {
		err := os.MkdirAll(dir, 0777)
		if err != nil {
			t.Fatalf("Failed to create test plugin dir %s: %v", dir, err)
		}
	}

	const cniVersion = "0.2.0"

	confFile := path.Join(testConfDir, confName+".conf")
	f, err := os.Create(confFile)
	if err != nil {
		t.Fatalf("Failed to install plugin %s: %v", confFile, err)
	}
	networkConfig := fmt.Sprintf(`{ "cniVersion": "%s", "name": "%s", "type": "%s", "capabilities": {"portMappings": true, "bandwidth": true, "ipRanges": true}  }`, cniVersion, confName, binName)
	_, err = f.WriteString(networkConfig)
	if err != nil {
		t.Fatalf("Failed to write network config file (%v)", err)
	}
	f.Close()

	pluginExec := path.Join(testBinDir, binName)
	f, err = os.Create(pluginExec)
	require.NoError(t, err)

	// TODO: use mock instead of fake shell script plugin
	const execScriptTempl = `#!/usr/bin/env bash
echo -n "{ \"cniVersion\": \"{{.CNIVersion}}\", \"ip4\": { \"ip\": \"{{.PodIP}}/24\" } }"
if [ "$CNI_COMMAND" = "VERSION" ]; then
	exit
fi
cat > {{.InputFile}}
env > {{.OutputEnv}}
echo "%@" >> {{.OutputEnv}}
export $(echo ${CNI_ARGS} | sed 's/;/ /g') &> /dev/null
mkdir -p {{.OutputDir}} &> /dev/null
echo -n "$CNI_COMMAND $CNI_NETNS $K8S_POD_NAMESPACE $K8S_POD_NAME $K8S_POD_INFRA_CONTAINER_ID" >& {{.OutputFile}}`

	inputFile := path.Join(testDataDir, binName+".in")
	outputFile := path.Join(testDataDir, binName+".out")
	envFile := path.Join(testDataDir, binName+".env")
	execTemplateData := &map[string]interface{}{
		"InputFile":  inputFile,
		"OutputFile": outputFile,
		"OutputEnv":  envFile,
		"OutputDir":  testDataDir,
		"CNIVersion": cniVersion,
		"PodIP":      podIP,
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

	return inputFile, outputFile, envFile
}

func tearDownPlugin(tmpDir string) {
	err := os.RemoveAll(tmpDir)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
}

type FakeNetworkHost struct {
	networktest.FakePortMappingGetter
	kubeClient clientset.Interface
	pods       []*containertest.FakePod
}

func NewFakeHost(kubeClient clientset.Interface, pods []*containertest.FakePod, ports map[string][]*hostport.PortMapping) *FakeNetworkHost {
	host := &FakeNetworkHost{
		networktest.FakePortMappingGetter{PortMaps: ports},
		kubeClient,
		pods,
	}
	return host
}

func (fnh *FakeNetworkHost) GetPodByName(name, namespace string) (*v1.Pod, bool) {
	return nil, false
}

func (fnh *FakeNetworkHost) GetKubeClient() clientset.Interface {
	return fnh.kubeClient
}

func (fnh *FakeNetworkHost) GetNetNS(containerID string) (string, error) {
	for _, fp := range fnh.pods {
		for _, c := range fp.Pod.Containers {
			if c.ID.ID == containerID {
				return fp.NetnsPath, nil
			}
		}
	}
	return "", fmt.Errorf("container %q not found", containerID)
}

func (fnh *FakeNetworkHost) SupportsLegacyFeatures() bool {
	return true
}

func TestCNIPlugin(t *testing.T) {
	// install some random plugin
	netName := fmt.Sprintf("test%d", rand.Intn(1000))
	binName := fmt.Sprintf("test_vendor%d", rand.Intn(1000))

	podIP := "10.0.0.2"
	podIPOutput := fmt.Sprintf("4: eth0    inet %s/24 scope global dynamic eth0\\       valid_lft forever preferred_lft forever", podIP)
	fakeCmds := []fakeexec.FakeCommandAction{
		func(cmd string, args ...string) exec.Cmd {
			return fakeexec.InitFakeCmd(&fakeexec.FakeCmd{
				CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
					func() ([]byte, error) {
						return []byte(podIPOutput), nil
					},
				},
			}, cmd, args...)
		},
	}

	fexec := &fakeexec.FakeExec{
		CommandScript: fakeCmds,
		LookPathFunc: func(file string) (string, error) {
			return fmt.Sprintf("/fake-bin/%s", file), nil
		},
	}

	mockLoCNI := &mock_cni.MockCNI{}
	// TODO mock for the test plugin too

	tmpDir := utiltesting.MkTmpdirOrDie("cni-test")
	testConfDir := path.Join(tmpDir, "etc", "cni", "net.d")
	testBinDir := path.Join(tmpDir, "opt", "cni", "bin")
	testDataDir := path.Join(tmpDir, "output")
	testCacheDir := path.Join(tmpDir, "var", "lib", "cni", "cache")
	defer tearDownPlugin(tmpDir)
	inputFile, outputFile, outputEnv := installPluginUnderTest(t, testBinDir, testConfDir, testDataDir, binName, netName, podIP)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "test_infra_container"}
	pods := []*containertest.FakePod{{
		Pod: &kubecontainer.Pod{
			Containers: []*kubecontainer.Container{
				{ID: containerID},
			},
		},
		NetnsPath: "/proc/12345/ns/net",
	}}

	plugins := ProbeNetworkPlugins(testConfDir, testCacheDir, []string{testBinDir})
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

	mockLoCNI.On("AddNetworkList", mock.AnythingOfType("*context.timerCtx"), cniPlugin.loNetwork.NetworkConfig, mock.AnythingOfType("*libcni.RuntimeConf")).Return(&types020.Result{IP4: &types020.IPConfig{IP: net.IPNet{IP: []byte{127, 0, 0, 1}}}}, nil)
	mockLoCNI.On("DelNetworkList", mock.AnythingOfType("*context.timerCtx"), cniPlugin.loNetwork.NetworkConfig, mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

	// Check that status returns an error
	if err := cniPlugin.Status(); err == nil {
		t.Fatalf("cniPlugin returned non-err with no podCidr")
	}

	cniPlugin.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, map[string]interface{}{
		network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR: "10.0.2.0/24",
	})

	if err := cniPlugin.Status(); err != nil {
		t.Fatalf("unexpected status err: %v", err)
	}

	ports := map[string][]*hostport.PortMapping{
		containerID.ID: {
			{
				Name:          "name",
				HostPort:      8008,
				ContainerPort: 80,
				Protocol:      "UDP",
				HostIP:        "0.0.0.0",
			},
		},
	}
	fakeHost := NewFakeHost(nil, pods, ports)

	plug, err := network.InitNetworkPlugin(plugins, "cni", fakeHost, kubeletconfig.HairpinNone, "10.0.0.0/8", network.UseDefaultMTU)
	if err != nil {
		t.Fatalf("Failed to select the desired plugin: %v", err)
	}

	bandwidthAnnotation := make(map[string]string)
	bandwidthAnnotation["kubernetes.io/ingress-bandwidth"] = "1M"
	bandwidthAnnotation["kubernetes.io/egress-bandwidth"] = "1M"

	// Set up the pod
	err = plug.SetUpPod("podNamespace", "podName", containerID, bandwidthAnnotation, nil)
	if err != nil {
		t.Errorf("Expected nil: %v", err)
	}
	eo, eerr := ioutil.ReadFile(outputEnv)
	output, err := ioutil.ReadFile(outputFile)
	if err != nil || eerr != nil {
		t.Errorf("Failed to read output file %s: %v (env %s err %v)", outputFile, err, eo, eerr)
	}

	expectedOutput := "ADD /proc/12345/ns/net podNamespace podName test_infra_container"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}

	// Verify the correct network configuration was passed
	inputConfig := struct {
		RuntimeConfig struct {
			PortMappings []map[string]interface{}   `json:"portMappings"`
			Bandwidth    map[string]interface{}     `json:"bandwidth"`
			IPRanges     [][]map[string]interface{} `json:"IPRanges"`
		} `json:"runtimeConfig"`
	}{}
	inputBytes, inerr := ioutil.ReadFile(inputFile)
	parseerr := json.Unmarshal(inputBytes, &inputConfig)
	if inerr != nil || parseerr != nil {
		t.Errorf("failed to parse reported cni input config %s: (%v %v)", inputFile, inerr, parseerr)
	}
	expectedMappings := []map[string]interface{}{
		// hah, golang always unmarshals unstructured json numbers as float64
		{"hostPort": 8008.0, "containerPort": 80.0, "protocol": "udp", "hostIP": "0.0.0.0"},
	}
	if !reflect.DeepEqual(inputConfig.RuntimeConfig.PortMappings, expectedMappings) {
		t.Errorf("mismatch in expected port mappings. expected %v got %v", expectedMappings, inputConfig.RuntimeConfig.PortMappings)
	}
	expectedBandwidth := map[string]interface{}{
		"ingressRate": 1000000.0, "egressRate": 1000000.0,
		"ingressBurst": 2147483647.0, "egressBurst": 2147483647.0,
	}
	if !reflect.DeepEqual(inputConfig.RuntimeConfig.Bandwidth, expectedBandwidth) {
		t.Errorf("mismatch in expected bandwidth. expected %v got %v", expectedBandwidth, inputConfig.RuntimeConfig.Bandwidth)
	}

	expectedIPRange := [][]map[string]interface{}{
		{
			{"subnet": "10.0.2.0/24"},
		},
	}

	if !reflect.DeepEqual(inputConfig.RuntimeConfig.IPRanges, expectedIPRange) {
		t.Errorf("mismatch in expected ipRange. expected %v got %v", expectedIPRange, inputConfig.RuntimeConfig.IPRanges)
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
	output, err = ioutil.ReadFile(outputFile)
	require.NoError(t, err)
	expectedOutput = "DEL /proc/12345/ns/net podNamespace podName test_infra_container"
	if string(output) != expectedOutput {
		t.Errorf("Mismatch in expected output for setup hook. Expected '%s', got '%s'", expectedOutput, string(output))
	}

	mockLoCNI.AssertExpectations(t)
}

func TestLoNetNonNil(t *testing.T) {
	if conf := getLoNetwork(nil); conf == nil {
		t.Error("Expected non-nil lo network")
	}
}
