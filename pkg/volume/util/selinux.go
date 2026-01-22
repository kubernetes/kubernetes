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
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/selinux/go-selinux"
	"github.com/opencontainers/selinux/go-selinux/label"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
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
		return "", &SELinuxLabelTranslationError{msg: err.Error()}
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

type SELinuxLabelTranslationError struct {
	msg string
}

func (e *SELinuxLabelTranslationError) Error() string {
	return e.msg
}

func IsSELinuxLabelTranslationError(err error) bool {
	var seLinuxError *SELinuxLabelTranslationError
	return errors.As(err, &seLinuxError)
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
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMount) {
		return true
	}

	// Only SELinuxMountReadWriteOncePod feature is enabled
	if len(volumeSpec.PersistentVolume.Spec.AccessModes) != 1 {
		// RWOP volumes must be the only access mode of the volume
		return false
	}
	if !v1helper.ContainsAccessMode(volumeSpec.PersistentVolume.Spec.AccessModes, v1.ReadWriteOncePod) {
		// Not a RWOP volume
		return false
	}
	// RWOP volume
	return true
}

// MultipleSELinuxLabelsError tells that one volume in a pod is mounted in multiple containers and each has a different SELinux label.
type MultipleSELinuxLabelsError struct {
	labels []string
}

func (e *MultipleSELinuxLabelsError) Error() string {
	return fmt.Sprintf("multiple SELinux labels found: %s", strings.Join(e.labels, ","))
}

func (e *MultipleSELinuxLabelsError) Labels() []string {
	return e.labels
}

func IsMultipleSELinuxLabelsError(err error) bool {
	var multiError *MultipleSELinuxLabelsError
	return errors.As(err, &multiError)
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

// SELinuxLabelInfo contains information about SELinux labels that should be used to mount a volume for a Pod.
type SELinuxLabelInfo struct {
	// SELinuxMountLabel is the SELinux label that should be used to mount the volume.
	// The volume plugin supports SELinuxMount and the Pod did not opt out via SELinuxChangePolicy.
	// Empty string otherwise.
	SELinuxMountLabel string
	// SELinuxProcessLabel is the SELinux label that will the container runtime use for the Pod.
	// Regardless if the volume plugin supports SELinuxMount or the Pod opted out via SELinuxChangePolicy.
	SELinuxProcessLabel string
	// PluginSupportsSELinuxContextMount is true if the volume plugin supports SELinux mount.
	PluginSupportsSELinuxContextMount bool
}

// GetMountSELinuxLabel returns SELinux labels that should be used to mount the given volume volumeSpec and podSecurityContext.
// It expects effectiveSELinuxContainerLabels as returned by volumeutil.GetPodVolumeNames, i.e. with all SELinuxOptions
// from all containers that use the volume in the pod, potentially expanded with PodSecurityContext.SELinuxOptions,
// if container's SELinuxOptions are nil.
// It does not evaluate the volume access mode! It's up to the caller to check SELinuxMount feature gate,
// it may need to bump different metrics based on feature gates / access modes / label anyway.
func GetMountSELinuxLabel(volumeSpec *volume.Spec, effectiveSELinuxContainerLabels []*v1.SELinuxOptions, podSecurityContext *v1.PodSecurityContext, volumePluginMgr *volume.VolumePluginMgr, seLinuxTranslator SELinuxLabelTranslator) (SELinuxLabelInfo, error) {
	info := SELinuxLabelInfo{}
	if !utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		return info, nil
	}

	if !seLinuxTranslator.SELinuxEnabled() {
		return info, nil
	}

	pluginSupportsSELinuxContextMount, err := SupportsSELinuxContextMount(volumeSpec, volumePluginMgr)
	if err != nil {
		return info, err
	}

	info.PluginSupportsSELinuxContextMount = pluginSupportsSELinuxContextMount

	// Collect all SELinux options from all containers that use this volume.
	// A set will squash any duplicities.
	labels := sets.New[string]()
	for _, containerLabel := range effectiveSELinuxContainerLabels {
		lbl, err := seLinuxTranslator.SELinuxOptionsToFileLabel(containerLabel)
		if err != nil {
			fullErr := fmt.Errorf("failed to construct SELinux label from context %q: %w", containerLabel, err)
			return info, fullErr
		}
		labels.Insert(lbl)
	}

	// Ensure that all containers use the same SELinux label.
	if labels.Len() > 1 {
		// This volume is used with more than one SELinux label in the pod.
		return info, &MultipleSELinuxLabelsError{labels: labels.UnsortedList()}
	}
	if labels.Len() == 0 {
		return info, nil
	}

	lbl, _ := labels.PopAny()
	info.SELinuxProcessLabel = lbl
	info.SELinuxMountLabel = lbl

	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxChangePolicy) &&
		podSecurityContext != nil &&
		podSecurityContext.SELinuxChangePolicy != nil &&
		*podSecurityContext.SELinuxChangePolicy == v1.SELinuxChangePolicyRecursive {
		// The pod has opted into recursive SELinux label changes. Do not mount with -o context.
		info.SELinuxMountLabel = ""
	}

	if !pluginSupportsSELinuxContextMount {
		// The volume plugin does not support SELinux mount. Do not mount with -o context.
		info.SELinuxMountLabel = ""
	}

	return info, nil
}
