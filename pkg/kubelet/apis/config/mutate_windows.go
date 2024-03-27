//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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

package config

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func (kc *KubeletConfiguration) Mutate() error {
	// This replicates configuration fix for Windows originally introduced
	// in https://github.com/kubernetes/kubernetes/blob/60d73ba75148483d150622a4d8b6e122e189ceab/cmd/kubeadm/app/componentconfigs/kubelet_windows.go#L31
	// TODO: fixups of other paths will be needed in future releases
	klog.V(2).Infoln("Adapting the paths in the KubeletConfiguration for Windows...")

	// Get the drive from where the kubelet binary was called.
	exe, err := os.Executable()
	if err != nil {
		return errors.Wrap(err, "could not obtain information about the kubelet executable")
	}

	drive := filepath.VolumeName(filepath.Dir(exe))

	// Mutate the paths in the config.
	mutatePaths(kc, drive)
	return nil
}

func mutatePaths(cfg *KubeletConfiguration, drive string) {
	mutateStringField := func(field *string) {
		// path.IsAbs() is not reliable here in the Windows runtime, so check if the
		// path starts with "/" instead. This means the path originated from a Unix node and
		// is an absolute path.
		if !strings.HasPrefix(*field, "/") {
			return
		}

		// Prepend the drive letter to the path and update the field.
		*field = filepath.Join(drive, *field)
	}

	mutateStringField(&cfg.PodLogsDir)
}
