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
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"

	nodeutil "k8s.io/component-helpers/node/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

type kubeletFlagsOpts struct {
	nodeRegOpts              *kubeadmapi.NodeRegistrationOptions
	pauseImage               string
	registerTaintsUsingFlags bool
}

// GetNodeNameAndHostname obtains the name for this Node using the following precedence
// (from lower to higher):
// - actual hostname
// - NodeRegistrationOptions.Name (same as "--node-name" passed to "kubeadm init/join")
// - "hostname-override" flag in NodeRegistrationOptions.KubeletExtraArgs
// It also returns the hostname or an error if getting the hostname failed.
func GetNodeNameAndHostname(cfg *kubeadmapi.NodeRegistrationOptions) (string, string, error) {
	hostname, err := nodeutil.GetHostname("")
	nodeName := hostname
	if cfg.Name != "" {
		nodeName = cfg.Name
	}
	if name, idx := kubeadmapi.GetArgValue(cfg.KubeletExtraArgs, "hostname-override", -1); idx > -1 {
		nodeName = name
	}
	return nodeName, hostname, err
}

// WriteKubeletDynamicEnvFile writes an environment file with dynamic flags to the kubelet.
// Used at "kubeadm init" and "kubeadm join" time.
func WriteKubeletDynamicEnvFile(cfg *kubeadmapi.ClusterConfiguration, nodeReg *kubeadmapi.NodeRegistrationOptions, registerTaintsUsingFlags bool, kubeletDir string) error {
	flagOpts := kubeletFlagsOpts{
		nodeRegOpts:              nodeReg,
		pauseImage:               images.GetPauseImage(cfg),
		registerTaintsUsingFlags: registerTaintsUsingFlags,
	}
	stringMap := buildKubeletArgs(flagOpts)
	argList := kubeadmutil.ArgumentsToCommand(stringMap, nodeReg.KubeletExtraArgs)
	envFileContent := fmt.Sprintf("%s=%q\n", constants.KubeletEnvFileVariableName, strings.Join(argList, " "))

	return writeKubeletFlagBytesToDisk([]byte(envFileContent), kubeletDir)
}

// buildKubeletArgsCommon takes a kubeletFlagsOpts object and builds based on that a slice of arguments
// that are common to both Linux and Windows
func buildKubeletArgsCommon(opts kubeletFlagsOpts) []kubeadmapi.Arg {
	kubeletFlags := []kubeadmapi.Arg{}
	kubeletFlags = append(kubeletFlags, kubeadmapi.Arg{Name: "container-runtime-endpoint", Value: opts.nodeRegOpts.CRISocket})

	// This flag passes the pod infra container image (e.g. "pause" image) to the kubelet
	// and prevents its garbage collection
	if opts.pauseImage != "" {
		kubeletFlags = append(kubeletFlags, kubeadmapi.Arg{Name: "pod-infra-container-image", Value: opts.pauseImage})
	}

	if opts.registerTaintsUsingFlags && opts.nodeRegOpts.Taints != nil && len(opts.nodeRegOpts.Taints) > 0 {
		taintStrs := []string{}
		for _, taint := range opts.nodeRegOpts.Taints {
			taintStrs = append(taintStrs, taint.ToString())
		}
		kubeletFlags = append(kubeletFlags, kubeadmapi.Arg{Name: "register-with-taints", Value: strings.Join(taintStrs, ",")})
	}

	// Pass the "--hostname-override" flag to the kubelet only if it's different from the hostname
	nodeName, hostname, err := GetNodeNameAndHostname(opts.nodeRegOpts)
	if err != nil {
		klog.Warning(err)
	}
	if nodeName != hostname {
		klog.V(1).Infof("setting kubelet hostname-override to %q", nodeName)
		kubeletFlags = append(kubeletFlags, kubeadmapi.Arg{Name: "hostname-override", Value: nodeName})
	}

	return kubeletFlags
}

// writeKubeletFlagBytesToDisk writes a byte slice down to disk at the specific location of the kubelet flag overrides file
func writeKubeletFlagBytesToDisk(b []byte, kubeletDir string) error {
	kubeletEnvFilePath := filepath.Join(kubeletDir, constants.KubeletEnvFileName)
	fmt.Printf("[kubelet-start] Writing kubelet environment file with flags to file %q\n", kubeletEnvFilePath)

	// creates target folder if not already exists
	if err := os.MkdirAll(kubeletDir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q", kubeletDir)
	}
	if err := os.WriteFile(kubeletEnvFilePath, b, 0644); err != nil {
		return errors.Wrapf(err, "failed to write kubelet configuration to the file %q", kubeletEnvFilePath)
	}
	return nil
}

// buildKubeletArgs takes a kubeletFlagsOpts object and builds based on that a slice of arguments
// that should be given to the local kubelet daemon.
func buildKubeletArgs(opts kubeletFlagsOpts) []kubeadmapi.Arg {
	return buildKubeletArgsCommon(opts)
}
