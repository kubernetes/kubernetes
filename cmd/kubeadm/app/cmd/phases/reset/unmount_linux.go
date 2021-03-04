// +build linux

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

package phases

import (
	"io/ioutil"
	"strings"
	"syscall"

	"k8s.io/klog/v2"
)

// unmountKubeletDirectory unmounts all paths that contain KubeletRunDirectory
func unmountKubeletDirectory(absoluteKubeletRunDirectory string) error {
	raw, err := ioutil.ReadFile("/proc/mounts")
	if err != nil {
		return err
	}

	if !strings.HasSuffix(absoluteKubeletRunDirectory, "/") {
		// trailing "/" is needed to ensure that possibly mounted /var/lib/kubelet is skipped
		absoluteKubeletRunDirectory += "/"
	}

	mounts := strings.Split(string(raw), "\n")
	for _, mount := range mounts {
		m := strings.Split(mount, " ")
		if len(m) < 2 || !strings.HasPrefix(m[1], absoluteKubeletRunDirectory) {
			continue
		}
		if err := syscall.Unmount(m[1], 0); err != nil {
			klog.Warningf("[reset] Failed to unmount mounted directory in %s: %s", absoluteKubeletRunDirectory, m[1])
		}
	}
	return nil
}
