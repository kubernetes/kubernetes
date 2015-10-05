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

// Package exec scans and loads networking plugins that are installed
// under /usr/libexec/kubernetes/kubelet-plugins/net/exec/
// The layout convention for a plugin is:
//   plugin-name/            (plugins have to be directories first)
//   plugin-name/plugin-name (executable that will be called out, see Vendoring Note for more nuances)
//   plugin-name/<other-files>
//   where, 'executable' has the following requirements:
//     - should have exec permissions
//     - should give non-zero exit code on failure, and zero on success
//     - the arguments will be <action> <pod_namespace> <pod_name> <docker_id_of_infra_container>
//       whereupon, <action> will be one of:
//         - init, called when the kubelet loads the plugin
//         - setup, called after the infra container of a pod is
//                created, but before other containers of the pod are created
//         - teardown, called before the pod infra container is killed
//         - status, called at regular intervals and is supposed to return a json
//                formatted output indicating the pod's IPAddress(v4/v6). An empty string value or an erroneous output
//                will mean the container runtime (docker) will be asked for the PodIP
//                e.g. {
//                         "apiVersion" : "v1beta1",
//                         "kind" : "PodNetworkStatus",
//                         "ip" : "10.20.30.40"
//                     }
//                The fields "apiVersion" and "kind" are optional in version v1beta1
// As the executables are called, the file-descriptors stdin, stdout, stderr
// remain open. The combined output of stdout/stderr is captured and logged.
//
// Note: If the pod infra container self-terminates (e.g. crashes or is killed),
// the entire pod lifecycle will be restarted, but teardown will not be called.
//
// Vendoring Note:
//      Plugin Names can be vendored also. Use '~' as the escaped name for plugin directories.
//      And expect command line argument to call vendored plugins as 'vendor/pluginName'
//      e.g. pluginName = mysdn
//           vendorname = mycompany
//      then, plugin layout should be
//           mycompany~mysdn/
//           mycompany~mysdn/mysdn (this becomes the executable)
//           mycompany~mysdn/<other-files>
//      and, call the kubelet with '--network-plugin=mycompany/mysdn'
package exec

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"path"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/network"
	kubeletTypes "k8s.io/kubernetes/pkg/kubelet/types"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

type execNetworkPlugin struct {
	execName string
	execPath string
	host     network.Host
}

const (
	initCmd     = "init"
	setUpCmd    = "setup"
	tearDownCmd = "teardown"
	statusCmd   = "status"
)

func ProbeNetworkPlugins(pluginDir string) []network.NetworkPlugin {
	execPlugins := []network.NetworkPlugin{}

	files, _ := ioutil.ReadDir(pluginDir)
	for _, f := range files {
		// only directories are counted as plugins
		// and pluginDir/dirname/dirname should be an executable
		// unless dirname contains '~' for escaping namespace
		// e.g. dirname = vendor~ipvlan
		// then, executable will be pluginDir/dirname/ipvlan
		if f.IsDir() {
			execPath := path.Join(pluginDir, f.Name())
			execPlugins = append(execPlugins, &execNetworkPlugin{execName: network.UnescapePluginName(f.Name()), execPath: execPath})
		}
	}
	return execPlugins
}

func (plugin *execNetworkPlugin) Init(host network.Host) error {
	err := plugin.validate()
	if err != nil {
		return err
	}
	plugin.host = host
	// call the init script
	out, err := utilexec.New().Command(plugin.getExecutable(), initCmd).CombinedOutput()
	glog.V(5).Infof("Init 'exec' network plugin output: %s, %v", string(out), err)
	return err
}

func (plugin *execNetworkPlugin) getExecutable() string {
	parts := strings.Split(plugin.execName, "/")
	execName := parts[len(parts)-1]
	return path.Join(plugin.execPath, execName)
}

func (plugin *execNetworkPlugin) Name() string {
	return plugin.execName
}

func (plugin *execNetworkPlugin) validate() error {
	if !isExecutable(plugin.getExecutable()) {
		errStr := fmt.Sprintf("Invalid exec plugin. Executable '%s' does not have correct permissions.", plugin.execName)
		return errors.New(errStr)
	}
	return nil
}

func (plugin *execNetworkPlugin) SetUpPod(namespace string, name string, id kubeletTypes.DockerID, containerRuntime string) error {
	out, err := utilexec.New().Command(plugin.getExecutable(), setUpCmd, namespace, name, string(id), containerRuntime).CombinedOutput()
	glog.V(5).Infof("SetUpPod 'exec' network plugin output: %s, %v", string(out), err)
	return err
}

func (plugin *execNetworkPlugin) TearDownPod(namespace string, name string, id kubeletTypes.DockerID, containerRuntime string) error {
	out, err := utilexec.New().Command(plugin.getExecutable(), tearDownCmd, namespace, name, string(id), containerRuntime).CombinedOutput()
	glog.V(5).Infof("TearDownPod 'exec' network plugin output: %s, %v", string(out), err)
	return err
}

func (plugin *execNetworkPlugin) Status(namespace string, name string, id kubeletTypes.DockerID, containerRuntime string) (*network.PodNetworkStatus, error) {
	out, err := utilexec.New().Command(plugin.getExecutable(), statusCmd, namespace, name, string(id), containerRuntime).CombinedOutput()
	glog.V(5).Infof("Status 'exec' network plugin output: %s, %v", string(out), err)
	if err != nil {
		return nil, err
	}
	if string(out) == "" {
		return nil, nil
	}
	findVersion := struct {
		unversioned.TypeMeta `json:",inline"`
	}{}
	err = json.Unmarshal(out, &findVersion)
	if err != nil {
		return nil, err
	}

	// check kind and version
	if findVersion.Kind != "" && findVersion.Kind != "PodNetworkStatus" {
		errStr := fmt.Sprintf("Invalid 'kind' returned in network status for pod '%s'. Valid value is 'PodNetworkStatus', got '%s'.", name, findVersion.Kind)
		return nil, errors.New(errStr)
	}
	switch findVersion.APIVersion {
	case "":
		fallthrough
	case "v1beta1":
		networkStatus := &network.PodNetworkStatus{}
		err = json.Unmarshal(out, networkStatus)
		return networkStatus, err
	}
	errStr := fmt.Sprintf("Unknown version '%s' in network status for pod '%s'.", findVersion.APIVersion, name)
	return nil, errors.New(errStr)
}
