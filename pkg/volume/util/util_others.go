//go:build !windows
// +build !windows

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
	utilexec "k8s.io/utils/exec"
)

//WriteVolumeCache flush disk data given the spcified mount path
func WriteVolumeCache(deviceMountPath string, exec utilexec.Interface) error {
	// For linux runtime, it skips because unmount will automatically flush disk data
	return nil
}
