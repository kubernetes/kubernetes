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

// SELinuxLabelTranslator translates v1.SELinuxOptions of a process to SELinux file label.
type SELinuxLabelTranslator interface {
	// SELinuxOptionsToFileLabel returns SELinux file label for given SELinuxOptions
	// of a container process.
	// When Role, User or Type are empty, they're read from the system defaults.
	// It returns "" and no error on platforms that do not have SELinux enabled
	// or don't support SELinux at all.
	SELinuxOptionsToFileLabel(opts *v1.SELinuxOptions) (string, error)

	// SELinuxEnabled returns true when the OS has enabled SELinux support.
	SELinuxEnabled() bool
}

// Real implementation of the interface.
// On Linux with SELinux enabled it translates. Otherwise it always returns an empty string and no error.
type translator struct{}

var _ SELinuxLabelTranslator = &translator{}

// NewSELinuxLabelTranslator returns new SELinuxLabelTranslator for the platform.
func NewSELinuxLabelTranslator() SELinuxLabelTranslator {
	return &translator{}
}

// SELinuxOptionsToFileLabel returns SELinux file label for given SELinuxOptions
// of a container process.
// When Role, User or Type are empty, they're read from the system defaults.
// It returns "" and no error on platforms that do not have SELinux enabled
// or don't support SELinux at all.
func (l *translator) SELinuxOptionsToFileLabel(opts *v1.SELinuxOptions) (string, error) {
	if opts == nil {
		return "", nil
	}

	args := contextOptions(opts)
	if len(args) == 0 {
		return "", nil
	}

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

func (l *translator) SELinuxEnabled() bool {
	return selinux.GetEnabled()
}

// Fake implementation of the interface for unit tests.
type fakeTranslator struct{}

var _ SELinuxLabelTranslator = &fakeTranslator{}

// NewFakeSELinuxLabelTranslator returns a fake translator for unit tests.
// It imitates a real translator on platforms that do not have SELinux enabled
// or don't support SELinux at all.
func NewFakeSELinuxLabelTranslator() SELinuxLabelTranslator {
	return &fakeTranslator{}
}

// SELinuxOptionsToFileLabel returns SELinux file label for given options.
func (l *fakeTranslator) SELinuxOptionsToFileLabel(opts *v1.SELinuxOptions) (string, error) {
	if opts == nil {
		return "", nil
	}
	// Fill empty values from "system defaults" (taken from Fedora Linux).
	user := opts.User
	if user == "" {
		user = "system_u"
	}

	role := opts.Role
	if role == "" {
		role = "object_r"
	}

	// opts is context of the *process* to run in a container. Translate
	// process type "container_t" to file label type "container_file_t".
	// (The rest of the context is the same for processes and files).
	fileType := opts.Type
	if fileType == "" || fileType == "container_t" {
		fileType = "container_file_t"
	}

	level := opts.Level
	if level == "" {
		// If empty, level is allocated randomly.
		level = "s0:c998,c999"
	}

	ctx := fmt.Sprintf("%s:%s:%s:%s", user, role, fileType, level)
	return ctx, nil
}

func (l *fakeTranslator) SELinuxEnabled() bool {
	return true
}

// SupportsSELinuxContextMount checks if the given volumeSpec supports with mount -o context
func SupportsSELinuxContextMount(volumeSpec *volume.Spec, volumePluginMgr *volume.VolumePluginMgr) (bool, error) {
	plugin, _ := volumePluginMgr.FindPluginBySpec(volumeSpec)
	if plugin != nil {
		return plugin.SupportsSELinuxContextMount(volumeSpec)
	}

	return false, nil
}

// VolumeSupportsSELinuxMount returns true if given volume access mode can support mount with SELinux mount options.
func VolumeSupportsSELinuxMount(volumeSpec *volume.Spec) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		return false
	}
	if volumeSpec.PersistentVolume == nil {
		return false
	}
	if len(volumeSpec.PersistentVolume.Spec.AccessModes) != 1 {
		return false
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMount) {
		return true
	}
	// Only SELinuxMountReadWriteOncePod feature enabled
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
