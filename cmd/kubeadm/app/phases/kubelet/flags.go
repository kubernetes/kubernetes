/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilsexec "k8s.io/utils/exec"
)

// WriteKubeletDynamicEnvFile writes a environment file with dynamic flags to the kubelet.
// Used at "kubeadm init" and "kubeadm join" time.
func WriteKubeletDynamicEnvFile(nodeRegOpts *kubeadmapi.NodeRegistrationOptions, registerTaintsUsingFlags bool) error {

	argList := kubeadmutil.BuildArgumentListFromMap(buildKubeletArgMap(nodeRegOpts, registerTaintsUsingFlags), nodeRegOpts.ExtraArgs)
	envFileContent := fmt.Sprintf("%s=%s\n", constants.KubeletEnvFileVariableName, strings.Join(argList, " "))

	return writeKubeletFlagBytesToDisk([]byte(envFileContent))
}

// buildKubeletArgMap takes a MasterConfiguration object and builds based on that a string-string map with flags
// that should be given to the local kubelet daemon.
func buildKubeletArgMap(nodeRegOpts *kubeadmapi.NodeRegistrationOptions, registerTaintsUsingFlags bool) map[string]string {
	kubeletFlags := map[string]string{}

	if nodeRegOpts.CRISocket == kubeadmapiv1alpha2.DefaultCRISocket {
		// These flags should only be set when running docker
		kubeletFlags["network-plugin"] = "cni"
		kubeletFlags["cni-conf-dir"] = "/etc/cni/net.d"
		kubeletFlags["cni-bin-dir"] = "/opt/cni/bin"
		execer := utilsexec.New()
		driver, err := kubeadmutil.GetCgroupDriverDocker(execer)
		if err != nil {
			glog.Warningf("cannot automatically assign a '--cgroup-driver' value when starting the Kubelet: %v\n", err)
		} else {
			kubeletFlags["cgroup-driver"] = driver
		}
	} else {
		kubeletFlags["container-runtime"] = "remote"
		kubeletFlags["container-runtime-endpoint"] = nodeRegOpts.CRISocket
	}

	if registerTaintsUsingFlags && nodeRegOpts.Taints != nil && len(nodeRegOpts.Taints) > 0 {
		taintStrs := []string{}
		for _, taint := range nodeRegOpts.Taints {
			taintStrs = append(taintStrs, taint.ToString())
		}

		kubeletFlags["register-with-taints"] = strings.Join(taintStrs, ",")
	}

	if pids, _ := procfs.PidOf("systemd-resolved"); len(pids) > 0 {
		// procfs.PidOf only returns an error if the regex is empty or doesn't compile, so we can ignore it
		kubeletFlags["resolv-conf"] = "/run/systemd/resolve/resolv.conf"
	}

	// Make sure the node name we're passed will work with Kubelet
	if nodeRegOpts.Name != "" && nodeRegOpts.Name != nodeutil.GetHostname("") {
		glog.V(1).Info("setting kubelet hostname-override to %q", nodeRegOpts.Name)
		kubeletFlags["hostname-override"] = nodeRegOpts.Name
	}

	// TODO: Conditionally set `--cgroup-driver` to either `systemd` or `cgroupfs` for CRI other than Docker

	return kubeletFlags
}

// writeKubeletFlagBytesToDisk writes a byte slice down to disk at the specific location of the kubelet flag overrides file
func writeKubeletFlagBytesToDisk(b []byte) error {
	fmt.Printf("[kubelet] Writing kubelet environment file with flags to file %q\n", constants.KubeletEnvFile)

	// creates target folder if not already exists
	if err := os.MkdirAll(filepath.Dir(constants.KubeletEnvFile), 0700); err != nil {
		return fmt.Errorf("failed to create directory %q: %v", filepath.Dir(constants.KubeletEnvFile), err)
	}
	if err := ioutil.WriteFile(constants.KubeletEnvFile, b, 0644); err != nil {
		return fmt.Errorf("failed to write kubelet configuration to the file %q: %v", constants.KubeletEnvFile, err)
	}
	return nil
}
