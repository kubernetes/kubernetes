//go:build windows

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

package validation_test

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/config/validation"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

var (
	// These config options are not supported on Windows.
	cgroupsPerQOS          = false
	enforceNodeAllocatable = []string{}
)

// TestValidateKubeletConfigurationWindowsInodeEviction verifies that inode-based
// eviction signals in EvictionHard/EvictionSoft/EvictionMinimumReclaim do not make
// kubelet configuration validation fail on Windows (they cannot be enforced because
// NTFS has no POSIX inodes), and that they are dropped from the maps with a warning.
func TestValidateKubeletConfigurationWindowsInodeEviction(t *testing.T) {
	featureGate := utilfeature.DefaultFeatureGate.DeepCopy()
	logsapi.AddFeatureGates(featureGate)

	kc := successConfig.DeepCopy()
	kc.EvictionHard = map[string]string{
		"memory.available":   "250Mi",
		"nodefs.available":   "10%",
		"nodefs.inodesFree":  "5%",
		"imagefs.inodesFree": "5%",
	}
	kc.EvictionSoft = map[string]string{
		"nodefs.available":       "10%",
		"nodefs.inodesFree":      "5%",
		"containerfs.inodesFree": "5%",
	}
	kc.EvictionMinimumReclaim = map[string]string{
		"nodefs.available":   "5%",
		"imagefs.inodesFree": "5%",
	}

	if err := validation.ValidateKubeletConfiguration(kc, featureGate); err != nil {
		t.Fatalf("inode-based eviction signals should be tolerated on Windows, got validation error: %v", err)
	}

	for _, tc := range []struct {
		name   string
		signal evictionapi.Signal
		m      map[string]string
	}{{
		name:   "hard nodefs.inodesFree",
		signal: evictionapi.SignalNodeFsInodesFree,
		m:      kc.EvictionHard,
	}, {
		name:   "hard imagefs.inodesFree",
		signal: evictionapi.SignalImageFsInodesFree,
		m:      kc.EvictionHard,
	}, {
		name:   "soft nodefs.inodesFree",
		signal: evictionapi.SignalNodeFsInodesFree,
		m:      kc.EvictionSoft,
	}, {
		name:   "soft containerfs.inodesFree",
		signal: evictionapi.SignalContainerFsInodesFree,
		m:      kc.EvictionSoft,
	}, {
		name:   "min reclaim imagefs.inodesFree",
		signal: evictionapi.SignalImageFsInodesFree,
		m:      kc.EvictionMinimumReclaim,
	}} {
		if _, ok := tc.m[string(tc.signal)]; ok {
			t.Errorf("%s should be dropped from the eviction %s after validation", tc.signal, tc.name)
		}
	}

	// The supported (non-inode) signals must be preserved.
	if kc.EvictionHard["nodefs.available"] != "10%" || kc.EvictionHard["memory.available"] != "250Mi" {
		t.Errorf("inode-drop must not remove supported eviction signals: %v", kc.EvictionHard)
	}
}
