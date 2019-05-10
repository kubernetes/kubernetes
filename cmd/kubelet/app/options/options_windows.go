// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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

package options

import (
	"os"
	"path/filepath"

	"k8s.io/klog"
)

const defaultRemoteRuntimeEndpoint = `npipe:////./pipe/dockershim`

var defaultRootDir, defaultCertDir, defaultVolumePluginDir string

// initOptions defaults the kubelet root, certificates and volume plugin directories
// using the SystemDrive environment variable.
func initOptions() {
	const (
		sysDrive        = "SystemDrive"
		defaultSysDrive = "c:"
	)
	drive := os.Getenv(defaultSysDrive)
	if drive == "" {
		klog.Warningf("the environment variable %q is not set. Defaulting to %q", sysDrive, defaultSysDrive)
		drive = defaultSysDrive
	}
	defaultRootDir = filepath.Join(drive, `var\lib\kubelet`)
	defaultCertDir = filepath.Join(drive, `var\lib\kubelet\pki`)
	defaultVolumePluginDir = filepath.Join(drive, `usr\libexec\kubernetes\kubelet-plugins\volume\exec\`)
}
