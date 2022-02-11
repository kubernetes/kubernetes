/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"github.com/opencontainers/selinux/go-selinux"
	"github.com/opencontainers/selinux/go-selinux/label"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

// SELinuxOptionsToFileLabel returns SELinux file label for given options.
func SELinuxOptionsToFileLabel(opts *v1.SELinuxOptions) (string, error) {
	if opts == nil {
		return "", nil
	}

	args := contextOptions(opts)
	if len(args) == 0 {
		return "", nil
	}

	// TODO: use interface for InitLabels for unit tests.
	processLabel, fileLabel, err := label.InitLabels(args)
	if err != nil {
		// In theory, this should be unreachable. InitLabels can fail only when args contain an unknown option,
		// and all options returned by contextOptions are known.
		return "", err
	}
	// InitLabels() may allocate a new unique SELinux label in kubelet memory. The label is *not* allocated
	// in the container runtime. Clear it to avoid memory problems.
	// ReleaseLabel on non-allocated label is NOOP.
	selinux.ReleaseLabel(processLabel)

	return fileLabel, nil
}

// Convert SELinuxOptions to []string accepted by label.InitLabels
func contextOptions(opts *v1.SELinuxOptions) []string {
	if opts == nil {
		return nil
	}
	args := make([]string, 0, 3)
	if opts.User != "" {
		args = append(args, "user:"+opts.User)
	}
	if opts.Role != "" {
		args = append(args, "role:"+opts.Role)
	}
	if opts.Type != "" {
		args = append(args, "type:"+opts.Type)
	}
	if opts.Level != "" {
		args = append(args, "level:"+opts.Level)
	}
	return args
}

// SupportsSELinuxContextMount checks if the given volumeSpec supports with mount -o context
func SupportsSELinuxContextMount(volumeSpec *volume.Spec, volumePluginMgr *volume.VolumePluginMgr) (bool, error) {
	// This is cheap
	if !selinux.GetEnabled() {
		return false, nil
	}

	plugin, _ := volumePluginMgr.FindPluginBySpec(volumeSpec)
	if plugin != nil {
		return plugin.SupportsSELinuxContextMount(volumeSpec)
	}

	return false, nil
}

func IsRWOP(volumeSpec *volume.Spec) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ReadWriteOncePod) {
		return false
	}
	if volumeSpec.PersistentVolume == nil {
		return false
	}
	if len(volumeSpec.PersistentVolume.Spec.AccessModes) != 1 {
		return false
	}
	if !v1helper.ContainsAccessMode(volumeSpec.PersistentVolume.Spec.AccessModes, v1.ReadWriteOncePod) {
		return false
	}
	return true
}

// AddSELinuxMountOption adds -o context="XYZ" mount option to a given list
func AddSELinuxMountOption(options []string, seLinuxContext string) []string {
	if !utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		return options
	}
	// Use double quotes to support a comma "," in the SELinux context string.
	// For example: dirsync,context="system_u:object_r:container_file_t:s0:c15,c25",noatime
	return append(options, fmt.Sprintf("context=%q", seLinuxContext))
}
