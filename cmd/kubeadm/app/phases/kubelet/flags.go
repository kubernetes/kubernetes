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
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilsexec "k8s.io/utils/exec"
)

type kubeletFlagsOpts struct {
	nodeRegOpts              *kubeadmapi.NodeRegistrationOptions
	featureGates             map[string]bool
	registerTaintsUsingFlags bool
	execer                   utilsexec.Interface
	pidOfFunc                func(string) ([]int, error)
	defaultHostname          string
}

// WriteKubeletDynamicEnvFile writes a environment file with dynamic flags to the kubelet.
// Used at "kubeadm init" and "kubeadm join" time.
func WriteKubeletDynamicEnvFile(nodeRegOpts *kubeadmapi.NodeRegistrationOptions, featureGates map[string]bool, registerTaintsUsingFlags bool, kubeletDir string) error {
	hostName, err := nodeutil.GetHostname("")
	if err != nil {
		return err
	}

	flagOpts := kubeletFlagsOpts{
		nodeRegOpts:              nodeRegOpts,
		featureGates:             featureGates,
		registerTaintsUsingFlags: registerTaintsUsingFlags,
		execer:          utilsexec.New(),
		pidOfFunc:       procfs.PidOf,
		defaultHostname: hostName,
	}
	stringMap := buildKubeletArgMap(flagOpts)
	argList := kubeadmutil.BuildArgumentListFromMap(stringMap, nodeRegOpts.KubeletExtraArgs)
	envFileContent := fmt.Sprintf("%s=%s\n", constants.KubeletEnvFileVariableName, strings.Join(argList, " "))

	return writeKubeletFlagBytesToDisk([]byte(envFileContent), kubeletDir)
}

// buildKubeletArgMap takes a InitConfiguration object and builds based on that a string-string map with flags
// that should be given to the local kubelet daemon.
func buildKubeletArgMap(opts kubeletFlagsOpts) map[string]string {
	kubeletFlags := map[string]string{}

	if opts.nodeRegOpts.CRISocket == kubeadmapiv1alpha3.DefaultCRISocket {
		// These flags should only be set when running docker
		kubeletFlags["network-plugin"] = "cni"
		driver, err := kubeadmutil.GetCgroupDriverDocker(opts.execer)
		if err != nil {
			glog.Warningf("cannot automatically assign a '--cgroup-driver' value when starting the Kubelet: %v\n", err)
		} else {
			kubeletFlags["cgroup-driver"] = driver
		}
	} else {
		kubeletFlags["container-runtime"] = "remote"
		kubeletFlags["container-runtime-endpoint"] = opts.nodeRegOpts.CRISocket
	}

	if opts.registerTaintsUsingFlags && opts.nodeRegOpts.Taints != nil && len(opts.nodeRegOpts.Taints) > 0 {
		taintStrs := []string{}
		for _, taint := range opts.nodeRegOpts.Taints {
			taintStrs = append(taintStrs, taint.ToString())
		}

		kubeletFlags["register-with-taints"] = strings.Join(taintStrs, ",")
	}

	if pids, _ := opts.pidOfFunc("systemd-resolved"); len(pids) > 0 {
		// procfs.PidOf only returns an error if the regex is empty or doesn't compile, so we can ignore it
		kubeletFlags["resolv-conf"] = "/run/systemd/resolve/resolv.conf"
	}

	// Make sure the node name we're passed will work with Kubelet
	if opts.nodeRegOpts.Name != "" && opts.nodeRegOpts.Name != opts.defaultHostname {
		glog.V(1).Infof("setting kubelet hostname-override to %q", opts.nodeRegOpts.Name)
		kubeletFlags["hostname-override"] = opts.nodeRegOpts.Name
	}

	// If the user enabled Dynamic Kubelet Configuration (which is disabled by default), set the directory
	// in the CLI flags so that the feature actually gets enabled
	if features.Enabled(opts.featureGates, features.DynamicKubeletConfig) {
		kubeletFlags["dynamic-config-dir"] = filepath.Join(constants.KubeletRunDirectory, constants.DynamicKubeletConfigurationDirectoryName)
	}

	// TODO: Conditionally set `--cgroup-driver` to either `systemd` or `cgroupfs` for CRI other than Docker

	return kubeletFlags
}

// writeKubeletFlagBytesToDisk writes a byte slice down to disk at the specific location of the kubelet flag overrides file
func writeKubeletFlagBytesToDisk(b []byte, kubeletDir string) error {
	kubeletEnvFilePath := filepath.Join(kubeletDir, constants.KubeletEnvFileName)
	fmt.Printf("[kubelet] Writing kubelet environment file with flags to file %q\n", kubeletEnvFilePath)

	// creates target folder if not already exists
	if err := os.MkdirAll(kubeletDir, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q: %v", kubeletDir, err)
	}
	if err := ioutil.WriteFile(kubeletEnvFilePath, b, 0644); err != nil {
		return fmt.Errorf("failed to write kubelet configuration to the file %q: %v", kubeletEnvFilePath, err)
	}
	return nil
}
